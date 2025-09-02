import os
import json
import asyncio
from dotenv import load_dotenv
import logging
import random
import tweepy
from datetime import datetime, timezone, timedelta
from personality import PersonalityConfig, Personality
from content_generator import ContentGenerator
from mention_handler import MentionResponder
from post_maker import PostHandler
from comment_engager import CommentEngager
from comment_replier import CommentReplier
from rapid_tweepy import RapidTweepy
from scheduler import (
    ScheduleManager, Schedule, CommentManager,
    CompetitorCommentManager, RetweetManager,
    PollScheduleManager
)
from poll_handler import PollHandler
from crypto_scraper import CryptoNewsWorkFlow
from retrieval_agent import RetrievalAgent
from competitor_twitter_pipeline import CompetitorTweetCollector
from retweet_pipeline import RetweetPipeline
from tweet_tracker import TweetTracker
from tweet_pipeline import TweetPipeline
from agents.media_post_agent import create_media_schedule_agent
from tools.schedule_tool import ScheduleTool
import nest_asyncio

nest_asyncio.apply()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()


class TwitterAgent:
    def __init__(self):
        logging.info("Initializing class...")

        # Initialize lock for task synchronization
        self.task_lock = asyncio.Lock()
        logging.info("Task lock initialized.")

        self.docs_path = os.getenv("DOCS_PATH", "docs")
        logging.info(f"Docs path set to: {self.docs_path}")

        self.tweepy_client, self.app_client = self.create_tweepy_client()
        self.rapid_tweepy = RapidTweepy(api_key=os.getenv("X_RAPID_API_KEY"))
        logging.info("Tweepy clients created.")

        self.personality = self.create_dummy_brand_personality()

        logging.info("Brand personality created.")

        self.schedule_manager = ScheduleManager()

        logging.info("Schedule manager initialized.")

        self.poll_manager = PollScheduleManager()

        logging.info("Poll Schedule manager initialized.")

        # self.load_schedule_entries()

        logging.info("Schedule entries loaded.")

        self.tweet_tracker = TweetTracker()

        logging.info("TweetTracker initialized")

        # When initializing TweetPipeline
        self.tweet_pipeline = TweetPipeline(
            docs_folder_automated=self.docs_path + "/automated",
            docs_folder_influencers=self.docs_path + "/influencer",
            lookback=24,
            tweepy_client=self.tweepy_client,
            rapid_client=self.rapid_tweepy,
            influencer_csv=os.getenv("INFLUENCER_CSV"),
            automated_csv=os.getenv("AUTOMATED_CSV"),
            max_docs=1
        )

        logging.info("Tweet pipeline initialized.")

        self.retrieval_agent = RetrievalAgent(
            model_name="openai/gpt-4o-mini",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            docs_path=self.docs_path,
            chunk_size=1000,
            chunk_overlap=200,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.0,
        )

        logging.info("Retrieval agent initialized.")

        self.generator = ContentGenerator(
            retriever=self.retrieval_agent,
            personality=self.personality,
            api_key=os.getenv('OPENROUTER_API_KEY'),
            model_name=os.getenv('model_name', 'meta-llama/llama-3.3-70b-instruct'),
            post_model_name=os.getenv('post_model_name'),
            validation_model_name=os.getenv('validation_model_name'),
        )

        logging.info("Content generator initialized.")

        self.crypto_workflow = CryptoNewsWorkFlow(
            model=os.getenv('model_name', 'meta-llama/llama-3.3-70b-instruct'),
            scheduler=self.schedule_manager,
            number_of_posts=3,
            retriever=self.retrieval_agent,
            poll_scheduler=self.poll_manager,
            analyzer_model=os.getenv("analyzer_model")
        )

        logging.info("Crypto news workflow initialized.")

        self.retweet_manager = RetweetManager()
        logging.info("Retweet manager initialized.")

        self.retweet_pipeline = RetweetPipeline(
            rapid_client=self.rapid_tweepy,
            csv_file=os.getenv('RETWEET_INFLUENCER_CSV'),
            tweepy_client=self.tweepy_client,
            retweet_manager=self.retweet_manager,
            lookback_hours=24
        )

        logging.info("Retweet pipeline initialized.")

        self.competitor_comment_manager = CompetitorCommentManager()
        logging.info("Competitor comment manager initialized.")

        self.comment_scheduler_manager = CommentManager()
        logging.info("Comment scheduler manager initialized.")

        self.comment_collector = CompetitorTweetCollector(
            csv_file=os.getenv('COMPETITOR_CSV'),
            tweepy_client=self.tweepy_client,
            rapid_client=self.rapid_tweepy,
            comment_manager=self.comment_scheduler_manager,
            competitor_comment_manager=self.competitor_comment_manager,
            lookback_hours=24
        )

        logging.info("Competitor tweet collector initialized.")

        self.post_handler = PostHandler(tweepy_client=self.tweepy_client,tweet_tracker=self.tweet_tracker,
                                        lookback_hours=2)
        logging.info("post handler initialized.")
        self.mention_responder = MentionResponder(self.personality, self.generator, self.tweepy_client,
                                                  [], tweet_tracker=self.tweet_tracker)
        logging.info("mention handler initialized.")
        self.tweet_engager = CommentEngager(self.generator, self.tweepy_client,
                                            self.comment_scheduler_manager,
                                            tweet_tracker=self.tweet_tracker)
        logging.info("tweet engager initialized.")
        self.poll_handler = PollHandler(
            tweepy_client=self.tweepy_client,
            poll_scheduler=self.poll_manager,
            tweet_tracker=self.tweet_tracker
        )
        logging.info("Poll handler initialized.")
        self.media_post_agent = create_media_schedule_agent(
            schedule_tool=ScheduleTool(schedulemanager=self.schedule_manager, with_media=True)
        )
        logging.info("Media agent initialized.")
        self.comment_replier = CommentReplier(
            personality=self.personality,
            generator=self.generator,
            tweepy_client=self.tweepy_client,
            rapid_client=self.rapid_tweepy,
            competitor_csv=os.getenv('COMPETITOR_CSV'),
            key_people_csv=os.getenv('KEY_PEOPLE_CSV', 'docs/csv_files/Key_People.csv'),
            engaged_history=[],
            min_comment_likes=0,
            max_comments_per_tweet=1
        )
        logging.info("Comment replier initialized.")
        self.last_checked_mention = int((datetime.now(timezone.utc) - timedelta(minutes=30)).timestamp() // 60)
        logging.info("Class initialization complete.")
        self.final_content_post = None

        # Flag to control the mention thread
        self.mention_thread_running = False
        self.mention_thread = None

    @staticmethod
    def create_tweepy_client() -> tuple:
        """Create and return both user authentication and app-only authentication Tweepy clients."""
        # User authentication client (for actions that require user context like posting tweets)
        user_client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
            wait_on_rate_limit=True
        )

        app_client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            wait_on_rate_limit=True
        )

        return user_client, app_client

    @staticmethod
    def create_dummy_brand_personality() -> Personality:
        """Create a personality for a dummy tech brand."""
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

    @staticmethod
    def load_scheduled_posts() -> list:
        """Load scheduled posts from a JSON file."""
        json_path = os.getenv('SCHEDULED_POSTS_JSON')
        try:
            with open(json_path, 'r') as file:
                posts = json.load(file)
            logger.info(f"Successfully loaded {len(posts)} scheduled posts")
            return posts
        except FileNotFoundError:
            logger.error(f"Scheduled posts file not found at {json_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {json_path}")
            return []

    def load_schedule_entries(self) -> None:
        """
        Create schedule entries from the loaded JSON file and add them to the schedule manager.
        """

        scheduled_posts = self.load_scheduled_posts()
        if isinstance(scheduled_posts, dict):
            scheduled_posts = scheduled_posts.get("scheduled_posts", [])

        for post in scheduled_posts:
            try:
                schedule_time = datetime.fromisoformat(post['scheduled_time'])
                schedule_entry = Schedule(
                    scheduled_time=schedule_time.replace(tzinfo=timezone.utc),
                    current_events=post['current_events'],
                    content=post['content']
                )
                self.schedule_manager.add_schedule(schedule_entry)
                logger.info(f"Added schedule for {schedule_time}")
            except (KeyError, ValueError) as error:
                logger.error(f"Error processing scheduled post: {error}")
                continue

    async def update_crypto_content(self):
        """Run the crypto news workflow and update scheduled posts."""
        while True:
            async with self.task_lock:
                logger.info("Running crypto news workflow...")
                try:
                    output = self.crypto_workflow.run()
                    logger.info(f"Crypto content updated successfully: {output}")
                except Exception as error:
                    logger.error(f"Error updating crypto content: {error}")
            await asyncio.sleep(3600)  # Run every 1 hour

    async def run_tweet_pipeline_daily(self):
        """Run the TweetPipeline update once a day."""
        while True:
            async with self.task_lock:
                logger.info("Running TweetPipeline daily update...")
                try:
                    self.tweet_pipeline.update_docs()
                    self.retrieval_agent.refresh()
                    logger.info("TweetPipeline update completed.")

                    self.comment_collector.schedule_all_competitor_comments(max_results_per_user=5)
                    self.comment_collector.transfer_top_competitor_comments(num_posts=4)
                    logger.info("Competitor comments transferred.")

                    logger.info("Starting daily retweet collection...")
                    collection_result = self.retweet_pipeline.collect_candidate_tweets(max_results=5)
                    logger.info(f"Retweet collection completed: {collection_result}")

                    self.retweet_pipeline.schedule_retweets(num_retweets=10)
                    self.post_handler.reset_daily_counts()
                    logger.info("Retweet scheduling completed.")

                    logger.info("Daily pipeline update completed successfully.")
                except Exception as e:
                    logger.error(f"Error in TweetPipeline daily update: {e}")
            # Sleep outside the lock
            await asyncio.sleep(86400)  # 24 hours

    async def process_retweets(self):
        while True:
            async with self.task_lock:
                try:
                    # Process retweets
                    self.retweet_pipeline.process_retweets(num_retweets=1)
                    logger.info("Retweets processed.")

                except Exception as e:
                    logging.error(f"Error checking mentions: {e}")
            await asyncio.sleep(21600)  # Every 6 hours

    async def run_comment_replier_cycle(self):
        """Run the comment replier every 40 minutes as per client requirements."""
        while True:
            async with self.task_lock:
                try:
                    current_time = datetime.now(timezone.utc)
                    logger.info(f"{current_time.isoformat()} - Starting comment replier cycle...")

                    if random.choice([True, False]):
                        # Process comments
                        tweets_processed, comments_engaged = self.comment_replier.process_comments(
                            lookback_hours=4,
                            max_comments=1,
                            max_comments_per_tweet=1
                        )
                        logger.info(
                            f"Comment replier processed {tweets_processed} tweets and engaged with {comments_engaged} comments")
                    else:
                        # Process tweets
                        tweets_processed, tweets_engaged = self.comment_replier.process_tweets(
                            lookback_hours=4,
                            max_tweets=1,
                            min_likes_threshold=1
                        )
                        logger.info(
                            f"Comment replier processed {tweets_processed} tweets and engaged with {tweets_engaged} tweets")

                except Exception as error:
                    logger.error(f"Error in comment replier cycle: {error}")
            await asyncio.sleep(2400)  # Every 40 minutes (2400 seconds)

    async def scheduled_posts(self):
        """Post one tweet every hour as per client requirements."""
        while True:
            async with self.task_lock:
                try:
                    current_time = datetime.now(timezone.utc)
                    logger.info(f"{current_time.isoformat()} - Starting hourly post cycle...")

                    # Post scheduled content
                    self.post_handler.run()
                    logger.info("Hourly post completed.")

                except Exception as error:
                    logger.error(f"Error in hourly post cycle: {error}")
            await asyncio.sleep(7200)  # 2 hour (7200 seconds)

    async def process_mentions(self):
        while True:
            async with self.task_lock:
                try:
                    current_time = int(datetime.now(timezone.utc).timestamp() // 60)
                    lookback_minutes = current_time - self.last_checked_mention
                    if lookback_minutes:
                        logging.info(f"Checking for new mentions (lookback: {lookback_minutes} minutes)...")
                        mention_stats = await self.mention_responder.process_mentions_and_respond(
                            lookback_minutes=lookback_minutes,
                            max_mentions=2,
                        )
                        self.last_checked_mention = int(datetime.now(timezone.utc).timestamp() // 60)
                        logger.info(f"Mention processing stats: {mention_stats}")
                        logging.info("Mention check complete.")
                except tweepy.TweepyException as e:
                    logger.error(f"Tweepy error while checking mentions: {e}")
                except Exception as e:
                    logger.error(f"Error checking mentions: {e}")

            await asyncio.sleep(7200)  # Check mentions every 2 hours

    async def reset_daily_posts(self):
        """Reset daily post counts."""
        while True:
            try:
                self.post_handler.reset_daily_counts()
                logger.info("Daily post counts reset.")
            except Exception as error:
                logger.error(f"Error in reset post cycle: {error}")
            await asyncio.sleep(86400)  # Once per day

    async def run(self):
        """Main async method to start the Twitter agent."""
        logger.info("Starting Twitter Agent with client requirements: 1 tweet/hour, comment cycle/40min...")
        try:
            # Create tasks for operations with client-specified frequencies
            tasks = [
                asyncio.create_task(self.run_tweet_pipeline_daily()),
                asyncio.create_task(self.scheduled_posts()),
                asyncio.create_task(self.run_comment_replier_cycle()),
                asyncio.create_task(self.process_retweets()),
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