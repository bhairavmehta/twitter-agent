import os
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
from typing import List
import tweepy
from scheduler import RetweetManager, RetweetCandidate
from agents.retweet_agent import create_retweet_agent
from tools.retweet_transfer_tool import RetweetTransferTool
from phi.utils.log import logger
from rapid_tweepy import RapidTweepy


class RetweetPipeline:
    def __init__(
            self,
            csv_file: str,
            tweepy_client: tweepy.Client,
            retweet_manager: RetweetManager,
            rapid_client: RapidTweepy,
            lookback_hours: int = 24,
    ):
        """
        Initialize the retweet pipeline.

        Args:
            csv_file (str): Path to CSV with influencer data.
                Expected columns: "Name Company", "Twitter handle", "Account Type", "Priority", "id"
            tweepy_client (tweepy.Client): Configured Tweepy client
            retweet_manager (RetweetManager): Manager for handling retweets
            lookback_hours (int): How far back to look for tweets
        """
        self.csv_file = csv_file
        self.client = tweepy_client
        self.lookback_hours = lookback_hours
        self.user_data = {}  # Maps username to cached user info
        self.error_usernames: List[str] = []
        self.retweet_manager = retweet_manager
        self.retweeted_history: List[str] = []
        self.rapid_client = rapid_client

        # Set up transfer tool and agent
        self.retweet_transfer_tool = RetweetTransferTool(
            retweet_manager=retweet_manager
        )
        self.agent = create_retweet_agent(
            model="openai/gpt-4o-mini",
            retweet_transfer_tool=self.retweet_transfer_tool,
            exa_api_key=os.getenv("EXA_API_KEY"),
            markdown=True,
            show_tool_calls=True
        )

        # Load initial user data
        self.load_users_data_from_csv()

    @staticmethod
    def clean_twitter_handle(handle: str) -> str:
        """Clean Twitter handle by removing @ and any extra quotes or spaces."""
        if not handle:
            return ""
        handle = str(handle).strip().strip('"').strip("'")
        return handle.lstrip("@").strip()

    def load_users_data_from_csv(self) -> None:
        """Load and cache user data from CSV file with ID caching."""
        try:
            df = pd.read_csv("docs/csv_files/Key_People.csv", dtype={'id': str})

            if "id" not in df.columns:
                df["id"] = None

            required_columns = ["Person", "Twitter handle", "Twitter URL"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            logger.info(f"Loading data for {len(df)} users from CSV")
            updated = False

            for idx, row in df.iterrows():
                twitter_handle = self.clean_twitter_handle(row.get("Twitter handle", ""))
                username = twitter_handle if twitter_handle else None

                if not username:
                    logger.warning(f"Could not determine username for row: {row['Person']}")
                    continue
                if pd.isna(row["id"]) or row["id"] in [None, ""]:
                    try:
                        logger.info(f"Fetching new user data for {username}")
                        user_response = self.rapid_client.get_user_info(username=username)

                        if user_response :
                            user_id = user_response.id
                            df.at[idx, "id"] = str(user_id)
                            updated = True
                            self.user_data[username] = {
                                "user_response": user_response,
                                "person_name": str(row.get("Person", "")).strip(),
                                "twitter_url": str(row.get("Twitter URL", "")).strip(),
                                "id": str(user_id)
                            }
                            logger.info(f"Fetched and cached ID for {username}: {str(user_id)}")
                        else:
                            logger.error(f"User {username} not found.")
                            self.error_usernames.append(username)

                    except Exception as e:
                        logger.error(f"Error fetching data for {username}: {str(e)}")
                        self.error_usernames.append(username)

                else:
                    dummy_user = type("DummyUser", (), {})()
                    dummy_user.id = str(row["id"])
                    self.user_data[username] = {
                        "user_response": dummy_user,
                        "person_name": str(row.get("Person", "")).strip(),
                        "twitter_url": str(row.get("Twitter URL", "")).strip(),
                        "id" : dummy_user.id
                    }
                    logger.info(f"Using cached ID for {username}: {str(row['id'])}")
            if updated:
                df.to_csv(self.csv_file, index=False)
                logger.info(f"CSV '{self.csv_file}' updated with new user_ids.")

            logger.info(f"Successfully loaded {len(self.user_data)} users")
            if self.error_usernames:
                logger.warning(f"Failed to load {len(self.error_usernames)} users: {', '.join(self.error_usernames)}")

        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise

    def get_start_time(self) -> datetime:
        """Get timezone-aware start time based on lookback hours."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        logger.info(f"Looking for tweets after: {start_time.isoformat()}")
        return start_time

    def fetch_recent_tweets_for_user(self, username: str, max_results: int = 100) -> List[tweepy.Tweet]:
        """Fetch recent tweets for a specific user."""
        tweets_with_data = []
        user_info = self.user_data.get(username)

        if not user_info:
            self.error_usernames.append(username)
            logger.warning(f"No cached data for {username}")
            return tweets_with_data

        user_id = str(user_info["user_response"].id)
        start_time = self.get_start_time()

        try:
            logger.info(f"Fetching tweets for {username} (ID: {str(user_id)})")
            tweets = self.rapid_client.get_user_tweets(user_id=user_id, count=max_results)
            if tweets:
                logger.info(f"Fetched {len(tweets)} tweets for {username}")
                for tweet in tweets:
                    # As Rapid_Tweet doesn't include a timestamp, use current time or enhance _parse_tweets to add one.
                    tweet_time = datetime.now(timezone.utc)
                    logger.info(f"Found tweet at {tweet_time.isoformat()}: {tweet.text[:50]}...")
                    tweets_with_data.append(tweet)
            else:
                logger.warning(f"No tweets found for {username}")
        except Exception as e:
            logger.error(f"Error fetching tweets for {username}: {e}")
            self.error_usernames.append(username)
        logger.info(f"Found tweet at {len(tweets_with_data)}...")
        return tweets_with_data

    def collect_candidate_tweets(self, max_results: int = 100) -> str:
        """
        Collect tweets from all users and add them as candidates for retweeting.

        Returns:
            str: Summary message of collection results
        """
        total_collected = 0
        logger.info(f"Starting tweet collection for {len(self.user_data)} users")

        for username in self.user_data.keys():
            logger.info(f"Processing user: {username}")
            tweets = self.fetch_recent_tweets_for_user(username, max_results=max_results)

            for tweet in tweets:
                tweet_time = tweet.created_at.replace(
                    tzinfo=timezone.utc) if tweet.created_at.tzinfo is None else tweet.created_at

                # Add debug logging for tweet timestamps
                now = datetime.now(timezone.utc)
                age = now - tweet_time
                logger.info(f"Tweet age: {age.total_seconds() / 3600:.2f} hours")

                candidate = RetweetCandidate(
                    time_posted=tweet_time,
                    source_acc=username,
                    tweet_id=str(tweet.id),
                    tweet_text=tweet.text,
                    like_count=tweet.likes,
                    retweet_count=tweet.retweet
                )
                self.retweet_manager.add_candidate(candidate)
                total_collected += 1

        logger.info(f"Collection complete. Total tweets collected: {total_collected}")
        return f"Collected {total_collected} candidate tweets for potential retweeting."

    def schedule_retweets(self, num_retweets: int = 2) -> None:
        """
        Use the agent to select and schedule the specified number of retweets.

        Args:
            num_retweets (int): Number of tweets to select and schedule for retweeting
        """
        candidates = self.retweet_manager.get_all_candidates()
        logger.info(f"Scheduling from {len(candidates)} available candidates")

        if not candidates:
            logger.info("No candidates available, collecting tweets first")
            self.collect_candidate_tweets()
            candidates = self.retweet_manager.get_all_candidates()
            logger.info(f"After collection: {len(candidates)} candidates available")

        prompt = (
            f"Select the top {num_retweets} tweets that will provide the most value for our engagement and growth "
            f"when retweeted. Consider engagement metrics, content relevance, and source credibility."
            f"Also make sure to not take too many tweets from any one user Also check the timing when the tweet was posted"
            f"If the tweet is more than 24 hours old dont add it to the list "
            )
        result = self.agent.run(prompt)
        logger.info(f": Scheduled Retweets:{len(self.retweet_manager.get_all_pending())} and agent response is {result.content}")

    def process_retweets(self, num_retweets: int = 2) -> None:
        """
        Process any pending retweets that are ready to be posted.
        Makes the actual API calls to retweet the content.
        """
        overdue = self.retweet_manager.get_all_pending()
        logger.info(f"Processing {len(overdue)} pending retweets")
        processed_count = 0

        for retweet in overdue[:num_retweets]:
            if retweet.tweet_id in self.retweeted_history:
                logger.info(f"Already retweeted tweet {retweet.tweet_id}; skipping.")
                continue

            try:
                tweet_check = self.client.get_tweet(retweet.tweet_id)
                if not tweet_check.data:
                    logger.warning(f"Tweet {retweet.tweet_id} no longer exists; skipping.")
                    continue

                response = self.client.retweet(retweet.tweet_id)

                if response.data:
                    logger.info(f"Successfully retweeted tweet {retweet.tweet_id}")
                    self.retweeted_history.append(retweet.tweet_id)
                    self.retweet_manager.mark_completed(retweet)
                    processed_count += 1
                    time.sleep(30)

            except Exception as e:
                logger.error(f"Error retweeting tweet {retweet.tweet_id}: {e}")
                continue