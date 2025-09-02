import logging
from datetime import datetime, timezone
import tweepy
import asyncio
from content_generator import ContentGenerator
from scheduler import ScheduleManager
from media_generator import MediaGenerator
from tweet_tracker import TweetTracker

logging.basicConfig(level=logging.INFO)


class PostHandler:
    def __init__(
            self,
            generator: ContentGenerator,
            tweepy_client:tweepy.Client,
            schedule_manager: ScheduleManager,
            tweet_tracker: TweetTracker,
            media_lock=None,
            schedule_lock=None
    ):
        """
        This class will be used to make posts.

        Args:
            generator: The content generator to use
            tweepy_client: The tweepy client to use
            schedule_manager: The schedule manager to use
            tweet_tracker: The tweet tracker to use
            media_lock: Optional asyncio.Lock for media operations
            schedule_lock: Optional asyncio.Lock for schedule operations
        """
        self.generator = generator
        self.tweepy_client = tweepy_client
        self.schedule_manager = schedule_manager
        self.tweet_tracker = tweet_tracker
        self.media_generator = MediaGenerator()

        # Use provided locks or create new ones if not provided
        self.media_lock = media_lock if media_lock else asyncio.Lock()
        self.schedule_lock = schedule_lock if schedule_lock else asyncio.Lock()

    @staticmethod
    def _is_due(schedule_entry) -> bool:
        """
        Check whether the scheduled time is due (i.e., scheduled time is in the past or now).
        """
        now = datetime.now(timezone.utc)
        return schedule_entry.scheduled_time <= now

    def _generate_post_content(self, scheduled_entry) -> str:
        """
        Build a prompt using the scheduled post's current events and personality's buzzwords,
        then use the generator to produce the tweet content.
        """
        post_content = self.generator.generate_post(scheduled_entry.current_events, scheduled_entry.content)
        return post_content

    async def _generate_and_upload_media(self, scheduled_entry) -> str | None:
        """
        Generate media based on the scheduled entry and upload it to Twitter.
        Uses a lock to ensure exclusive access to media generation.

        Args:
            scheduled_entry: The schedule entry to generate media for.

        Returns:
            The media ID if successful, None otherwise.
        """
        # Use the media_lock to ensure exclusive access
        async with self.media_lock:
            try:
                logging.info(f"Media lock acquired for {scheduled_entry.media_type} generation")
                media_prompt = scheduled_entry.media_prompt
                if scheduled_entry.media_type == "image":
                    media_id = await self.media_generator.generate_and_upload_image(media_prompt)
                    logging.info(f"Generated and uploaded image with ID: {media_id}")
                    return media_id

                elif scheduled_entry.media_type == "video":
                    logging.info(f"Generating video with prompt: {media_prompt}")
                    # Make sure the video generation is properly awaited
                    # If generate_and_upload_video is not async, you'll need to adapt this
                    media_id = await self.media_generator.generate_and_upload_video(media_prompt)
                    logging.info(f"Generated and uploaded video with ID: {media_id}")
                    return media_id

            except Exception as e:
                logging.error(f"Error generating media for scheduled post: {e}")
                return None
            finally:
                logging.info("Media lock released")

    async def post_due_posts(self) -> None:
        """
        Check all pending schedules. For each entry that is due, generate the post content,
        post the tweet using the tweepy client, and mark the entry as completed.
        Uses locks to ensure exclusive access to schedule and media operations.
        """
        # Get all pending posts with schedule lock
        async with self.schedule_lock:
            pending = self.schedule_manager.get_all_pending()
            due_entries = [entry for entry in pending if self._is_due(entry)]

        if not due_entries:
            logging.info("No scheduled posts are due at this time.")
            return

        logging.info(f"Found {len(due_entries)} due posts to process")

        # Process media posts first, then regular posts
        media_posts = [entry for entry in due_entries if getattr(entry, 'include_media', False)]
        regular_posts = [entry for entry in due_entries if not getattr(entry, 'include_media', False)]

        logging.info(f"Processing {len(media_posts)} media posts and {len(regular_posts)} regular posts")

        # First process all media posts
        for entry in media_posts:
            await self.process_single_post(entry)

        # Then process all regular posts
        for entry in regular_posts:
            await self.process_single_post(entry)

    async def process_single_post(self, entry):
        """
        Process a single scheduled post, generating media if needed and posting to Twitter.
        Uses locks to ensure exclusive access to media and schedule operations.

        Args:
            entry: The schedule entry to process

        Returns:
            The tweet ID if successfully posted, None otherwise
        """
        try:
            logging.info(f"Processing post: {entry}")

            # Generate content
            post_content = self._generate_post_content(entry)
            logging.info(f"Generated content: {post_content[:50]}...")

            # Handle media if needed
            media_ids = []
            if getattr(entry, 'include_media', False):
                media_type = getattr(entry, 'media_type', 'image')
                logging.info(f"Processing {media_type} for post...")

                # Wait for media generation to complete
                media_id = await self._generate_and_upload_media(entry)

                if media_id:
                    media_ids.append(media_id)
                    logging.info(f"Media generated successfully with ID: {media_id}")
                else:
                    logging.warning("Media generation failed. Continuing with text-only post.")

            # Post the tweet
            try:
                if media_ids:
                    logging.info(f"Creating tweet with media ID(s): {media_ids}")
                    response = self.tweepy_client.create_tweet(
                        text=post_content,
                        media_ids=media_ids
                    )
                else:
                    logging.info("Creating text-only tweet")
                    response = self.tweepy_client.create_tweet(
                        text=post_content
                    )

                tweet_id = response.data.get("id")
                logging.info(f"Successfully posted tweet with ID {tweet_id}")
                self.tweet_tracker.add_post(tweet_id)

                # Remove post from schedule with lock
                async with self.schedule_lock:
                    self.schedule_manager.remove_scheduled_post(entry)
                    logging.info(f"Removed post from schedule queue")

                return tweet_id

            except tweepy.TweepyException as e:
                logging.error(f"Error posting tweet: {e}")
                return None

        except Exception as e:
            logging.error(f"Unexpected error processing post: {e}")
            return None