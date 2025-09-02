import logging
from datetime import datetime, timezone
from scheduler import PollScheduleManager
import tweepy
from tweet_tracker import TweetTracker

logging.basicConfig(level=logging.INFO)


class PollHandler:
    def __init__(self, tweepy_client: tweepy.Client, poll_scheduler:PollScheduleManager
                 , max_polls_per_run: int = 1, tweet_tracker:TweetTracker = None):
        """
        Initializes the PollHandler with a Tweepy client, a PollScheduler,
        and a limit on the number of polls to post per run.
        Args:
            tweepy_client (tweepy.Client): Authenticated Tweepy client for posting tweets.
            poll_scheduler: Manages the scheduling of polls.
            max_polls_per_run (int): Maximum number of polls to post in a single run. Defaults to 5.
        """
        self.tweepy_client = tweepy_client
        self.poll_scheduler = poll_scheduler
        self.max_polls_per_run = max_polls_per_run
        self.tweet_tracker = tweet_tracker

    @staticmethod
    def _is_due(scheduled_poll) -> bool:
        """
        Checks if the scheduled time is due (i.e., scheduled time is in the past or now).

        Args:
            scheduled_poll: The scheduled poll entry to check.

        Returns:
            bool: True if the scheduled time is due, False otherwise.
        """
        now = datetime.now(timezone.utc)
        return scheduled_poll.scheduled_time <= now

    def post_n_due_polls(self):
        """
        Posts due polls to Twitter, up to the specified maximum per run,
        and marks them as completed in the poll scheduler.
        """
        due_polls = self.poll_scheduler.get_all_pending()

        if not due_polls:
            logging.info("No scheduled polls are due at this time.")
            return

        polls_to_post = due_polls[:self.max_polls_per_run]

        for poll in polls_to_post:
            try:
                response = self.tweepy_client.create_tweet(
                    text=poll.poll_question,
                    poll_options=poll.poll_options,
                    poll_duration_minutes=poll.poll_duration_minutes
                )
                tweet_id = response.data.get("id")
                self.tweet_tracker.add_poll(tweet_id)
                logging.info(f"Successfully posted poll with tweet ID {tweet_id} for schedule: {poll}")
            except tweepy.TweepyException as e:
                logging.error(f"Error posting poll for schedule {poll}: {e}")
                continue

            self.poll_scheduler.mark_poll_completed(poll)
    def post_due_poll(self):
        poll = self.poll_scheduler.get_next_poll()
        if not poll:
            logging.info("No scheduled polls are due at this time.")
            return
        try:
            response = self.tweepy_client.create_tweet(
                text=poll.poll_question,
                poll_options=poll.poll_options,
                poll_duration_minutes=poll.poll_duration_minutes
            )
            tweet_id = response.data.get("id")
            self.tweet_tracker.add_poll(tweet_id)
            logging.info(f"Successfully posted poll with tweet ID {tweet_id} for schedule: {poll}")
        except tweepy.TweepyException as e:
            logging.error(f"Error posting poll for schedule {poll}: {e}")

        self.poll_scheduler.mark_poll_completed(poll)