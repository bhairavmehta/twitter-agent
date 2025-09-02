import logging
import tempfile
import requests
import os
from dotenv import load_dotenv
import tweepy
from datetime import datetime, timezone, timedelta
import pandas as pd
import json
from dataclasses import dataclass
from typing import Optional, Dict

from agents.post_category_agent import create_post_selector_agent
from agents.post_gen_with_url_agent import create_post_generator_w_agent
from agents.post_gen_agent import create_post_generator_agent
from agents.company_info_agent import create_company_info_agent
from tweet_tracker import TweetTracker
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TweetData:
    user_id: str
    tweet_id: str
    content: str
    category: str
    timestamp: datetime
    media_url: Optional[str] = None


class PostHandler:
    def __init__(self,
                 tweepy_client: tweepy.Client,
                 csv_file: str ="docs/csv_files/posts.csv",
                 lookback_hours: int = 1,
                 testing: bool = False,
                 tweepy_api :tweepy.api = None,
                 tweet_tracker: TweetTracker = None,
                 ):
        """
        Initializes the PostHandler.

        Args:
            tweepy_client: The Tweepy client instance.
            csv_file: Path to the CSV file containing user data.
            lookback_hours: Time window for fetching recent tweets.
        """
        self.tweepy_client = tweepy_client
        self.tweet_api = tweepy_api
        self.csv_file = csv_file
        self.lookback_hours = lookback_hours
        self.user_data: Dict[str, Dict[str, str]] = {}
        self.tweet_counts = {
            "WuBlockchain": 8,
            #"aixbt_agent": 2,
            "lookonchain": 2,
            "naiivememe": 1,
            "365X.ai": 1,
            #"crypto_url": 1
        }
        self.tweet_tracker = tweet_tracker
        self.categories = ["company","meme","crypto_url","crypto_only"]
        self.post_generator_agent = create_post_generator_agent(api_key=os.getenv("OPENROUTER_API_KEY"))
        self.post_selector_agent = create_post_selector_agent(api_key=os.getenv("OPENROUTER_API_KEY"))
        self.post_w_url_agent = create_post_generator_w_agent(api_key=os.getenv("OPENROUTER_API_KEY"))
        self.load_users_data_from_csv()
        self.company_agent = create_company_info_agent()
        self.last_tweet = "na"
        auth = tweepy.OAuth1UserHandler(
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
        )
        self.twitter_api_v1 = tweepy.API(auth)

        self.last_checked: Dict[str, datetime] = {}
        self.testing = testing
        for user_name in self.user_data.keys():
            self.last_checked[user_name] = datetime.now(timezone.utc) - timedelta(hours=6)
        self.last_checked[self.last_tweet] = datetime.now(timezone.utc) - timedelta(hours=6)

    def load_users_data_from_csv(self) -> None:
        """Load user data from CSV and cache Twitter IDs."""
        try:
            df = pd.read_csv(self.csv_file, dtype={'id': str})

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
                user_name = twitter_handle if twitter_handle else None

                if not user_name:
                    logger.warning(f"Could not determine username for row: {row['Person']}")
                    continue

                if pd.isna(row["id"]) or row["id"] in [None, ""]:
                    try:
                        logger.info(f"Fetching new user data for {user_name}")
                        user_response = self.tweepy_client.get_user(username=user_name)

                        if user_response:
                            user_id = str(user_response.data.id)
                            df.at[idx, "id"] = str(user_id)
                            updated = True
                            self.user_data[user_name] = {
                                "id": str(user_id),
                                "person_name": row.get("Person", "").strip(),
                                "twitter_url": row.get("Twitter URL", "").strip()
                            }
                            logger.info(f"Fetched and cached ID for {user_name}: {str(user_id)}")
                        else:
                            logger.error(f"User {user_name} not found.")

                    except Exception as e:
                        logger.error(f"Error fetching data for {user_name}: {str(e)}")

                else:
                    self.user_data[user_name] = {
                        "id": str(row["id"]),
                        "person_name": row.get("Person", "").strip(),
                        "twitter_url": row.get("Twitter URL", "").strip()
                    }
                    logger.info(f"Using cached ID for {user_name}: {str(row['id'])}")

            if updated:
                df.to_csv(self.csv_file, index=False)
                logger.info(f"CSV '{self.csv_file}' updated with new user IDs.")

            logger.info(f"Successfully loaded {len(self.user_data)} users")

        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise

    def select_next_post(self):
        """
        Asks the agent to decide which account should post next.
        """
        prompt_data = {
            "tweets_left": self.tweet_counts
        }
        response = self.post_selector_agent.run(message=f"this is current schedule {json.dumps(prompt_data)}"
                                                        f"and the last tweet was for {self.last_tweet} ")
        try:
            raw_response = response.content
            lines = raw_response.strip().splitlines()
            if lines[0].startswith("```") and lines[-1].startswith("```"):
                json_str = "\n".join(lines[1:-1]).strip()
            else:
                json_str = raw_response

            response_json = json.loads(json_str)

            user_name = response_json.get("username","")
            category = response_json.get("category","")

            if category:
                logger.info(f"Next post should be from: {user_name} (Category: {category})")
                return user_name, category
            else:
                logger.error("Invalid response from agent: Missing username or category.")
                return None, None
        except json.JSONDecodeError:
            logger.error("Failed to parse agent response.")
            return None, None

    def run(self):
        user_name,category = self.select_next_post()
        logger.info(f"selected {user_name}, {category}")
        if category and self.tweet_counts[user_name]:
            if category == "crypto_only" :
                if user_name == "WuBlockchain" or user_name == "aixbt_agent":
                        self.post_tweet(user_name=user_name)

                elif user_name == "lookonchain":
                    self.post_tweet_with_media(user_name=user_name)

            elif category == "meme":
                self.post_meme_tweet()

            elif category == "company":
                self.post_company_tweet()

            elif category == "crypto_url":
                self.post_reward_tweet()
            self.tweet_counts[user_name] -= 1
        else:
            logger.error(f"No post posted for {category} : {user_name}")
        self.update_last_checked(user_name=user_name)
        self.print_counts_left()


    def post_tweet(self, user_name:str,url:bool=True):
        user_id = self.user_data[user_name].get("id")
        start_time = self.get_start_time(user_name=user_name)
        tweets = self.tweepy_client.get_users_tweets(
            id=user_id,
            max_results=5,
            tweet_fields=["text", "created_at", "entities"],
            start_time=start_time,
        )
        if not tweets.data:
            logger.error(f"No tweets found for {user_name} after {start_time}. Skipping post.")
            return
        logger.info(f"found {len(tweets.data)} for {user_name}, {tweets}")
        if url:
            post_content = self.post_w_url_agent.run(f"{tweets}")
        else :
            post_content = self.post_generator_agent.run(f"{tweets}")
        if not post_content:
            logger.error(f"Generated tweet is empty for {user_name}. Skipping post.")
            return
        if self.testing:
            logger.info(f"Testing mode: Tweet content would be: {post_content.content}")
            return
        else :
            tweet_content = self.clean_text(post_content.content)
            response = self.tweepy_client.create_tweet(text=tweet_content)
            tweet_id = response.data.get("id")
            logging.info(f"Successfully posted tweet with ID {tweet_id}")
            self.tweet_tracker.add_post(tweet_id)
        if response:
            logger.info(f"Tweet posted successfully for {user_name}: {post_content.content}")
        else:
            logger.error(f"Failed to post tweet for {user_name}")

    def post_tweet_with_media(self, user_name):
        user_id = self.user_data[user_name].get("id")
        start_time = self.get_start_time(user_name=user_name)

        tweets = self.tweepy_client.get_users_tweets(
            id=user_id,
            max_results=5,
            tweet_fields=["entities"],
            expansions=["article.media_entities"],
            media_fields=["url"],
            start_time=start_time
        )
        if not tweets.data:
            logger.error(f"No tweets found for {user_name} after {start_time}. Skipping post.")
            return
        tweet_with_media = None
        if tweets.data:
            for tweet in tweets.data:
                if tweet.entities and "media" in tweet.entities:
                    tweet_with_media = tweet
                    break
        if not tweet_with_media:
            logger.info(f"No media found for {user_name}, posting normal tweet instead.")
            self.post_tweet(user_name)
            return

        media_entities = tweet_with_media.entities.get("media", [])
        if not media_entities:
            logger.info(f"Media entities list is empty for {user_name}, posting normal tweet instead.")
            self.post_tweet(user_name)
            return

        media_object = media_entities[0]
        image_url = media_object.get("media_url_https") or media_object.get("media_url")
        if not image_url:
            logger.info(f"No media URL found in tweet for {user_name}, posting normal tweet instead.")
            self.post_tweet(user_name)
            return

        try:
            response = requests.get(image_url)
            if response.status_code != 200:
                logger.error(f"Failed to download image from {image_url}")
                self.post_tweet(user_name)
                return

            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp_file.write(response.content)
            tmp_file.close()
            image_path = tmp_file.name
            logger.info(f"Downloaded image to {image_path}")
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            self.post_tweet(user_name)
            return

        media_ids = []
        try:
            media = self.twitter_api_v1.media_upload(filename=image_path)
            media_id = media.media_id_string
            logger.info(f"Uploaded image with media ID: {media_id}")
            media_ids.append(media_id)
            os.remove(image_path)
        except Exception as e:
            logger.error(f"Error uploading image: {e}. Falling back to normal tweet.")
            if os.path.exists(image_path):
                os.remove(image_path)
            self.post_tweet(user_name)
            return


        post_content = self.post_generator_agent.run(message=f"Generate a post for this {tweet_with_media.text}")
        if not post_content:
            logger.error(f"Generated tweet content is empty for {user_name}, posting normal tweet instead.")
            self.post_tweet(user_name)
            return

        try:
            if self.testing:
                logger.info(f"Testing mode: Tweet content would be: {post_content.content}")
                return
            else:
                tweet_content = self.clean_text(post_content.content)
                tweet_response = self.tweepy_client.create_tweet(
                    text = tweet_content,
                    media_ids = media_ids
                )
                tweet_id = tweet_response.data.get("id")
                logging.info(f"Successfully posted tweet with ID {tweet_id}")
                self.tweet_tracker.add_post(tweet_id)
            if tweet_response:
                logger.info(f"Tweet with media posted successfully for {user_name}: {post_content}")
            else:
                logger.error(f"Failed to post tweet with media for {user_name}")
        except Exception as e:
            logger.error(f"Exception while posting tweet with media: {e}")

    def post_meme_tweet(self):
        user_id = self.user_data["naiivememe"].get("id")
        start_time = self.get_start_time(user_name="naiivememe")
        tweets = self.tweepy_client.get_users_tweets(
            id=user_id,
            max_results=5,
            tweet_fields=["text", "created_at"],
            start_time=start_time
        )

        if not tweets.data:
            logger.error(f"No tweets found for naiivememe after {start_time}. Skipping post.")
            return

        post_content = self.post_w_url_agent.run(f"{tweets.data}")
        if not post_content:
            logger.error(f"Generated tweet is empty for  naiivememe. Skipping post.")
            return

        if self.testing:
            logger.info(f"Testing mode: Tweet content would be: {post_content.content}")
            return
        else:
            tweet_content = self.clean_text(post_content.content)
            response = self.tweepy_client.create_tweet(text=tweet_content)
            tweet_id = response.data.get("id")
            logging.info(f"Successfully posted tweet with ID {tweet_id}")
            self.tweet_tracker.add_post(tweet_id)
        if response:
            logger.info(f"Tweet posted successfully for  naiivememe: {post_content}")
        else:
            logger.error(f"Failed to post tweet for  naiivememe")

    def post_company_tweet(self):
        tweet = self.company_agent.run("Give me one tweet to post on twitter")
        post_content = self.post_generator_agent.run(tweet.content)
        if not post_content:
            logger.error(f"Generated tweet is empty for 365x.ai. Skipping post.")
            return
        if self.testing:
            logger.info(f"Testing mode: Tweet content would be: {post_content.content}")
            return
        else:
            tweet_content = self.clean_text(post_content.content)
            response = self.tweepy_client.create_tweet(text=tweet_content)
            tweet_id = response.data.get("id")
            logging.info(f"Successfully posted tweet with ID {tweet_id}")
            self.tweet_tracker.add_post(tweet_id)
        if response:
            logger.info(f"Tweet posted successfully for 365x.ai: {post_content}")
        else:
            logger.error(f"Failed to post tweet for 365x.ai")

    def post_reward_tweet(self):
        pass

    @staticmethod
    def clean_twitter_handle(handle: str) -> str:
        """Clean Twitter handle by removing '@' if present."""
        return handle.strip().lstrip("@")

    def update_last_checked(self, user_name: str):
        """Updates the last checked time for a user to the current time."""
        self.last_tweet = user_name
        self.last_checked[user_name] = datetime.now(timezone.utc)

    def get_start_time(self,user_name = None) -> datetime:
        """Get timezone-aware start time based on lookback hours."""
        if self.last_checked[user_name]:
            return self.last_checked[user_name]
        else :
            return datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)

    def decrease_count(self,tweet:str):
        self.tweet_counts[tweet] -= 1

    def reset_daily_counts(self):
        """
        Resets the daily tweet limits.
        """
        logger.info("Resetting daily tweet counts.")
        if self.testing:
            self.tweet_counts = {
                "WuBlockchain": 8,
                # "aixbt_agent": 2,
                "lookonchain": 2,
                "naiivememe": 1,
                "365X.ai": 1,
                # "crypto_url": 1
            }
        else:
            self.tweet_counts = {
                "WuBlockchain": 8,
                # "aixbt_agent": 2,
                "lookonchain": 2,
                "naiivememe": 1,
                "365X.ai": 1,
                # "crypto_url": 1
            }

    def print_counts_left(self):
        logger.info(f"{self.tweet_counts}")

    @staticmethod
    def clean_text(text: str) -> str:
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
        return text