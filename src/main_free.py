import os
import logging
import tweepy
import asyncio
import time
from dotenv import load_dotenv
from tweet_tracker import TweetTracker
from post_maker import PostHandler

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()


class TwitterAgent:
    def __init__(self):
        logging.info("Initializing Twitter Agent...")

        self.tweepy_client = self.create_tweepy_client()
        logging.info("Tweepy client created.")

        self.tweet_tracker = TweetTracker()
        logging.info("TweetTracker initialized")

        self.post_handler = PostHandler(
            tweepy_client=self.tweepy_client,
            tweet_tracker=self.tweet_tracker,
            lookback_hours=2,
            #testing=True,
        )
        logging.info("Post handler initialized.")

    @staticmethod
    def create_tweepy_client():
        """Create and return a Tweepy client."""
        user_client = tweepy.Client(
            bearer_token="AAAAAAAAAAAAAAAAAAAAAArG0QEAAAAA1FIqu3JYTgqO5j3a1l%2F%2BVeSLlu0%3DCdjXjRebhsbb5Ip7XNKYfAFrGyr2gWYMccydSqEDzMdPkF4A64",
            consumer_key="eKNbgQI3PsIg5nsRZ98TU7TiM",
            consumer_secret="wnWFCF4pwV9ljlG2lFW4bE8ckQmRubJsSwlw011Mh5soRqx4r8",
            access_token="1908585636254109696-yekpS00uoZZFzd68chTw6lmTpm2jPR",
            access_token_secret="ooOTFf9p0fzN2B0tF5XwZfF43QnPIFazbTPOP4dRFgAzZ",
            wait_on_rate_limit=True
        )
        return user_client
    async def reset_daily_posts(self):
        """Run cycles with specific intervals."""
        while True:
            try:
                self.post_handler.reset_daily_counts()
            except Exception as error:
                logger.error(f"Error in reset post cycle: {error}")
            await asyncio.sleep(86400)

    async def scheduled_posts(self):
        """Post content every 2 hours."""
        logger.info("Starting scheduled posting (every 2 hours)...")
        while True:
            try:
                self.post_handler.run()
                logger.info("Post completed")
            except Exception as error:
                logger.error(f"Error posting content: {error}")

            await asyncio.sleep(7200)  # 7200 seconds = 2 hours

    async def run(self):
        """Run the Twitter agent with scheduled posting."""
        logger.info("Starting Twitter Agent (posting every 2 hours)...")
        try:

            tasks = [
                asyncio.create_task(self.reset_daily_posts()),
                asyncio.create_task(self.scheduled_posts()),
            ]
            # Wait for all tasks to complete (they should run indefinitely)
            await asyncio.gather(*tasks)
        except Exception as error:
            logger.error(f"Critical error: {error}")
        finally:
            logger.info("Twitter Agent shutting down.")


if __name__ == '__main__':
    try:
        agent = TwitterAgent()
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logger.info("Twitter Agent stopped by user.")
    except Exception as e:
        logger.error(f"Twitter Agent stopped due to error: {e}")