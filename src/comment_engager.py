import time
from datetime import datetime, timezone, timedelta
from typing import List
import logging
import tweepy
from scheduler import CommentManager
from tweet_tracker import TweetTracker
from content_generator import ContentGenerator


class CommentEngager:
    def __init__(self,
                 generator:ContentGenerator,
                 tweepy_client:tweepy.client,
                 comment_manager: CommentManager,
                 tweet_tracker: TweetTracker,
                 engaged_history: List[str] = None):
        """
        Initialize the CommentEngager class.

        Args:
            generator: An agent or function that generates a reply based on tweet text.
            tweepy_client (tweepy.Client): A configured Tweepy client.
            comment_manager: A manager that holds competitor comment data.
            tweet_tracker (TweetTracker): An instance of TweetTracker to track bot's tweets.
            engaged_history (List[str], optional): List of tweet IDs that have already been engaged.
        """
        self.generator = generator
        self.tweepy_client = tweepy_client
        self.comment_manager = comment_manager
        self.tweet_tracker = tweet_tracker  # Store tweet tracker instance
        self.engaged_history = engaged_history or []

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def has_already_engaged(self, tweet_id: str) -> bool:
        return tweet_id in self.engaged_history

    @staticmethod
    def ensure_timezone_aware(dt: datetime) -> datetime:
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def is_tweet_valid(self, tweet_time: datetime) -> bool:
        """
        Check if a tweet's timestamp is valid for engagement.

        Args:
            tweet_time (datetime): The tweet's timestamp

        Returns:
            bool: True if the tweet is valid for engagement
        """
        try:
            if tweet_time is None:
                return False

            tweet_time = self.ensure_timezone_aware(tweet_time)
            now = datetime.now(timezone.utc)

            max_age = timedelta(hours=36)
            age = now - tweet_time

            return age <= max_age
        except Exception as e:
            logging.error(f"Error in is_tweet_valid: {str(e)}")
            return False

    def engage_comments(self, num_of_comments: int = 5) -> None:
        """
        Iterate over competitor comment tweets from the comment manager, generate a reply for each,
        and post it as a reply using the Tweepy client. If a reply is successfully posted, record the tweet ID
        in the engaged history.

        Args:
            num_of_comments (int): Maximum number of comments to engage with
        """
        try:
            comments = self.comment_manager.get_future_comments()

            processed = 0

            logging.info(f"Processing up to {num_of_comments} comments from {len(comments)} available")

            for comment in comments:
                if processed >= num_of_comments:
                    break

                if self.has_already_engaged(comment.tweet_id):
                    logging.info(f"Already engaged with tweet {comment.tweet_id}; skipping.")
                    continue

                if self.tweet_tracker.is_our_tweet(comment.tweet_id):
                    logging.info(f"Skipping bot's own tweet {comment.tweet_id}.")
                    continue

                try:
                    comment.time_posted = self.ensure_timezone_aware(comment.time_posted)
                    if not self.is_tweet_valid(comment.time_posted):
                        logging.info(f"Tweet {comment.tweet_id} is too old; skipping.")
                        continue

                    if not self.generator.filter_comment(comment.comment_text):
                        logging.info(f"Tweet {comment.tweet_id} is not crypto related")
                        continue

                    response_text = self.generator.generate_comment(comment.comment_text,self_tweet=False)
                    if not response_text or len(response_text) > 280:
                        logging.warning(f"Invalid response for tweet {comment.tweet_id}; skipping.")
                        continue

                    reply = self.tweepy_client.create_tweet(
                        text=response_text,
                        in_reply_to_tweet_id=comment.tweet_id
                    )

                    if reply and reply.data:
                        reply_id = reply.data.get('id')
                        logging.info(f"Commented on tweet {comment.tweet_id} with reply ID {reply_id}")
                        self.engaged_history.append(comment.tweet_id)
                        self.tweet_tracker.add_reply(reply_id)  # Track the reply
                        processed += 1
                        time.sleep(30)
                    else:
                        logging.error(f"Failed to get reply data for tweet {comment.tweet_id}")

                except tweepy.TweepyException as e:
                    logging.error(f"Tweepy error engaging tweet {comment.tweet_id}: {str(e)}")
                except Exception as e:
                    logging.error(f"Unexpected error engaging tweet {comment.tweet_id}: {str(e)}")
                    continue

            logging.info(f"Engagement complete. Processed {processed} comments.")
        except Exception as e:
            logging.error(f"Error in engage_comments: {str(e)}")
            raise

