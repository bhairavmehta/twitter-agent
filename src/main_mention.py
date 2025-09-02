import os
import time
import asyncio
import logging
from dotenv import load_dotenv
import tweepy
from datetime import datetime, timezone, timedelta
from personality import PersonalityConfig, Personality
from content_generator import ContentGenerator
from mention_handler import MentionResponder
from tweet_tracker import TweetTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class MentionHandlerApp:
    def __init__(self):
        logging.info("Initializing Mention Handler App...")

        # Initialize Tweepy client
        self.tweepy_client = self.create_tweepy_client()
        logging.info("Tweepy client created.")

        # Create personality
        self.personality = self.create_personality()
        logging.info("Personality created.")

        # Create tweet tracker
        self.tweet_tracker = TweetTracker()
        logging.info("Tweet tracker initialized.")

        # Initialize content generator
        self.generator = ContentGenerator(
            personality=self.personality,
            api_key=os.getenv('OPENROUTER_API_KEY'),
            model_name=os.getenv('model_name', 'openai/gpt-4o'),
            post_model_name=os.getenv('post_model_name'),
            validation_model_name=os.getenv('validation_model_name'),
            retriever=None
        )
        logging.info("Content generator initialized.")

        # Initialize mention responder
        self.mention_responder = MentionResponder(
            self.personality,
            self.generator,
            self.tweepy_client,
            [],
            tweet_tracker=self.tweet_tracker
        )
        logging.info("Mention responder initialized.")

        # Set last checked time to 30 minutes ago
        self.last_checked_mention = int((datetime.now(timezone.utc) - timedelta(minutes=30)).timestamp() // 60)
        logging.info("Initialization complete.")

    @staticmethod
    def create_tweepy_client():
        """Create and return a Tweepy client with user authentication."""
        client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
            wait_on_rate_limit=True
        )
        return client

    @staticmethod
    def create_personality():
        """Create a personality for the bot."""
        config = PersonalityConfig(
            tone="professional",
            engagement_style="enthusiastic",
            brand="365x.ai",
            response_temperature=0.7,
            formality_level="semi-formal",
            emoji_usage="moderate",
            content_length_preference="medium",
            sentiment_bias="positive",
            humor_level="subtle",
            slang_usage=False,
            buzzwords=[
                "Crypto",
                "blockchain",
                "stocks",
                "NFT"
            ]
        )
        return Personality(config)

    async def check_mentions(self):
        """Check for mentions and respond to them."""
        try:
            current_time = int(datetime.now(timezone.utc).timestamp() // 60)
            lookback_minutes = current_time - self.last_checked_mention

            if lookback_minutes > 0:
                logging.info(f"Checking for new mentions (lookback: {lookback_minutes} minutes)...")
                mention_stats = await self.mention_responder.process_mentions_and_respond(
                    lookback_minutes=lookback_minutes
                )
                self.last_checked_mention = int(datetime.now(timezone.utc).timestamp() // 60)
                logger.info(f"Mention processing stats: {mention_stats}")
                logging.info("Mention check complete.")
            else:
                logging.info("No time has passed since last check, skipping...")
        except Exception as e:
            logging.error(f"Error checking mentions: {e}", exc_info=True)

    async def run(self):
        """Run the mention handler application."""
        logging.info("Starting Mention Handler App...")

        while True:
            start_time = time.time()

            # Check for mentions
            await self.check_mentions()

            # Calculate how long to sleep to maintain a 90-second interval
            elapsed = time.time() - start_time
            sleep_time = max(0, 90 - elapsed)
            logger.info(f"Mention check took {elapsed:.2f} seconds, sleeping for {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)


if __name__ == "__main__":
    try:
        app = MentionHandlerApp()
        asyncio.run(app.run())
    except KeyboardInterrupt:
        logger.info("Mention Handler App stopped by user.")
    except Exception as e:
        logger.error(f"Mention Handler App stopped due to error: {e}", exc_info=True)