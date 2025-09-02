import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import logging
import tweepy
from scheduler import CompetitorCommentData, CompetitorCommentManager, CommentManager
from agents.comment_scheduler_agent import create_competitor_comment_agent
from tools.comment_transfer_tool import CommentTransferTool
from rapid_tweepy import RapidTweepy


class CompetitorTweetCollector:
    def __init__(self, csv_file: str, tweepy_client: tweepy.Client, comment_manager: CommentManager,
                 competitor_comment_manager: CompetitorCommentManager,rapid_client:RapidTweepy,
                 lookback_hours: int = 24
                 ):
        """
        Args:
            csv_file (str): Path to the CSV file containing competitor data.
                Expected columns: "Name Company", "Website", "Twitter handle", and optionally "id".
            tweepy_client (tweepy.Client): A configured Tweepy client.
            comment_manager (CompetitorCommentManager): Manager instance for handling competitor comments.
            lookback_hours (int): How far back (in hours) to fetch tweets.
        """
        self.csv_file = csv_file
        self.client = tweepy_client
        self.lookback_hours = lookback_hours
        self.user_data = {}  # Dictionary mapping username to cached user info.
        self.error_usernames: List[str] = []
        self.competitor_comment_manager = competitor_comment_manager
        self.comment_manager = comment_manager
        self.rapid_client = rapid_client

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Initialize tools and agents
        self.comment_transfer_tool = CommentTransferTool(
            competitor_manager=competitor_comment_manager,
            scheduler_manager=comment_manager
        )
        self.agent = create_competitor_comment_agent(
            model="openai/gpt-4o-mini",
            comment_transfer_tool=self.comment_transfer_tool,
            exa_api_key=os.getenv("EXA_API_KEY"),
            markdown=True,
            show_tool_calls=True
        )

        self.load_users_data_from_csv()

    @staticmethod
    def ensure_timezone_aware(dt: datetime) -> datetime:
        """Ensure a datetime object is timezone-aware."""
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def load_users_data_from_csv(self) -> None:
        """Load and cache user data from CSV file."""
        try:
            df = pd.read_csv(self.csv_file, dtype={'id': str})
            if "id" not in df.columns:
                df["id"] = None

            updated = False
            for idx, row in df.iterrows():
                twitter_handle = str(row.get("Twitter handle", "")).strip()
                username = twitter_handle.lstrip("@") if twitter_handle else None

                if not username:
                    logging.warning(f"Could not determine username for row: {row}")
                    continue

                if pd.isna(row["id"]) or row["id"] in [None, ""]:
                    try:
                        user_response = self.rapid_client.get_user_info(username=username)
                        if user_response:
                            user_id = str(user_response.id)
                            df.at[idx, "id"] = user_id
                            updated = True
                            self.user_data[username] = {
                                "user_response": user_response,
                                "company_link": str(row.get("Website", "")).strip(),
                                "name_company": str(row.get("Name Company", "")).strip()
                            }
                            logging.info(f"Fetched and cached ID for {username}: {str(user_id)}")
                        else:
                            logging.error(f"User {username} not found.")
                            self.error_usernames.append(username)
                    except Exception as e:
                        logging.error(f"Error fetching data for {username}: {str(e)}")
                        self.error_usernames.append(username)
                else:
                    dummy_user = type("DummyUser", (), {})()
                    dummy_user.id = str(row["id"])
                    self.user_data[username] = {
                        "user_response": dummy_user,
                        "company_link": str(row.get("Website", "")).strip(),
                        "name_company": str(row.get("Name Company", "")).strip()
                    }
                    logging.info(f"Loaded cached ID for {username}: {str(row['id'])}")

            if updated:
                df.to_csv(self.csv_file, index=False)
                logging.info(f"CSV '{self.csv_file}' updated with new user_ids.")

            logging.info(f"Cached data for {len(self.user_data)} users.")

        except Exception as e:
            logging.error(f"Error loading CSV file: {str(e)}")
            raise

    def get_start_time(self) -> datetime:
        """Get timezone-aware start time."""
        return datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)

    def fetch_recent_tweets_for_user(self, username: str, max_results: int = 5) -> List[tweepy.Tweet]:
        """Fetch recent tweets for a specific user."""
        tweets_with_data = []
        user_info = self.user_data.get(username)

        if not user_info:
            self.error_usernames.append(username)
            logging.warning(f"No cached data for {username}")
            return tweets_with_data

        user_id = user_info["user_response"].id

        try:
            tweets = self.rapid_client.get_user_tweets(user_id, count=max_results)
            tweets_with_data.extend(tweets)

        except Exception as e:
            logging.error(f"Error fetching tweets for {username}: {str(e)}")
            self.error_usernames.append(username)

        return tweets_with_data

    def get_competitor_comment_data(self, tweet_id: str) -> Optional[CompetitorCommentData]:
        """Get competitor comment data for a specific tweet."""
        for username, info in self.user_data.items():
            tweets = self.fetch_recent_tweets_for_user(username, max_results=50)
            for tweet in tweets:
                if str(tweet.id) == tweet_id:
                    # Ensure created_at is timezone-aware
                    created_at = self.ensure_timezone_aware(tweet.created_at)
                    return CompetitorCommentData(
                        time_posted=created_at,
                        tweet_id=str(tweet.id),
                        comment_text=tweet.text,
                        company_link=info.get("company_link") or None
                    )
        return None

    def schedule_all_competitor_comments(self, max_results_per_user: int = 10, max_total_tweets: int = 100) -> str:
        """Schedule all competitor comments."""
        all_tweets_with_user = []

        logging.info("Starting to fetch competitor tweets")

        # Step 1: Gather tweets from all users
        for username in self.user_data.keys():
            logging.info(f"Fetching tweets for {username}")
            tweets = self.fetch_recent_tweets_for_user(username, max_results=max_results_per_user)

            for tweet in tweets:
                created_at = self.ensure_timezone_aware(tweet.created_at)
                all_tweets_with_user.append({
                    "tweet": tweet,
                    "created_at": created_at,
                    "username": username
                })

        logging.info(f"Total tweets fetched: {len(all_tweets_with_user)}")

        sorted_tweets = sorted(all_tweets_with_user, key=lambda x: x["created_at"], reverse=True)

        top_tweets = sorted_tweets[:max_total_tweets]

        total_scheduled = 0
        logging.info(f"Scheduling top {len(top_tweets)} tweets")

        for entry in top_tweets:
            tweet = entry["tweet"]
            created_at = entry["created_at"]
            username = entry["username"]

            comment_data = CompetitorCommentData(
                time_posted=created_at,
                tweet_id=str(tweet.id),
                comment_text=tweet.text,
                company_link=self.user_data[username].get("company_link") or None
            )
            try:
                self.competitor_comment_manager.add_comment(comment_data)
                total_scheduled += 1
                logging.debug(f"Scheduled tweet {tweet.id} from {username}")
            except Exception as e:
                logging.error(f"Error scheduling tweet {tweet.id} from {username}: {str(e)}")

        logging.info(f"Scheduled {total_scheduled} competitor comments from top tweets.")
        return f"Scheduled {total_scheduled} competitor comments from top tweets."

    def transfer_top_competitor_comments(self, num_posts: int = 2) -> None:
        """Transfer top competitor comments to scheduling system."""
        try:
            prompt = f"I want you to get me top {num_posts} tweets which will help in our company growth and more engagement."
            self.agent.run(prompt)
            scheduled_count = len(self.comment_manager.get_all_events())
            logging.info(f"Transferred comments. Total scheduled: {scheduled_count}")
        except Exception as e:
            logging.error(f"Error transferring competitor comments: {str(e)}")
            raise
