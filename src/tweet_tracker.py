import logging
from typing import List, Set, Dict


class TweetTracker:
    """
    A class to track tweets, replies, and comments made by the bot.
    Helps prevent the bot from replying to its own posts and limits comments per post.
    """

    def __init__(self, default_comment_limit: int = 5):
        """
        Initialize the tweet tracker.

        Args:
            default_comment_limit: The default maximum number of comments allowed per post.
        """
        self.our_tweet_ids = set()
        self.comment_counts = {}  # Maps tweet_id to number of comments made
        self.comment_limits = {}  # Maps tweet_id to its comment limit
        self.default_comment_limit = default_comment_limit

    def add_tweet(self, tweet_id: str) -> None:
        """
        Record a tweet posted by the bot.

        Args:
            tweet_id: The ID of the tweet posted by the bot.
        """
        if tweet_id:
            tweet_id = str(tweet_id)
            self.our_tweet_ids.add(tweet_id)
            # Initialize comment count for new tweets
            if tweet_id not in self.comment_counts:
                self.comment_counts[tweet_id] = 0
            # Set default comment limit if not already set
            if tweet_id not in self.comment_limits:
                self.comment_limits[tweet_id] = self.default_comment_limit
            logging.debug(f"Added tweet ID {tweet_id} to tracker")

    def add_reply(self, reply_id: str) -> None:
        """
        Record a reply posted by the bot.

        Args:
            reply_id: The ID of the reply tweet posted by the bot.
        """
        self.add_tweet(reply_id)

    def add_comment(self, comment_id: str, parent_tweet_id: str) -> None:
        """
        Record a comment posted by the bot and increment the comment count for the parent tweet.
        and then return whether the comment limit has been reached or not.
        Args:
            comment_id: The ID of the comment posted by the bot.
            parent_tweet_id: The ID of the tweet being commented on.
        """
        self.add_tweet(comment_id)

        parent_tweet_id = str(parent_tweet_id)

        if parent_tweet_id not in self.comment_counts:
            self.comment_counts[parent_tweet_id] = 0

        if parent_tweet_id not in self.comment_limits:
            self.comment_limits[parent_tweet_id] = self.default_comment_limit

        self.comment_counts[parent_tweet_id] += 1
        logging.debug(
            f"Added comment to tweet {parent_tweet_id}, current count: {self.comment_counts[parent_tweet_id]}")

    def add_retweet(self, retweet_id: str) -> None:
        """
        Record a retweet made by the bot.

        Args:
            retweet_id: The ID of the retweet made by the bot.
        """
        self.add_tweet(retweet_id)

    def add_poll(self, tweet_id: str) -> None:
        """
        Record a poll made by the bot.

        Args:
            tweet_id: The ID of the poll made by the bot.
        """
        self.add_tweet(tweet_id)

    def add_post(self, tweet_id: str) -> None:
        """
        Record a post made by the bot.

        Args:
            tweet_id: The ID of the post made by the bot.
        """
        self.add_tweet(tweet_id)

    def is_our_tweet(self, tweet_id: str) -> bool:
        """
        Check if a tweet was posted by the bot.

        Args:
            tweet_id: The ID of the tweet to check.

        Returns:
            True if the tweet was posted by the bot, False otherwise.
        """
        return str(tweet_id) in self.our_tweet_ids

    def set_comment_limit(self, tweet_id: str, limit: int) -> None:
        """
        Set a custom comment limit for a specific tweet.

        Args:
            tweet_id: The ID of the tweet to set the limit for.
            limit: The maximum number of comments allowed for this tweet.
        """
        tweet_id = str(tweet_id)
        self.comment_limits[tweet_id] = limit
        logging.debug(f"Set comment limit for tweet {tweet_id} to {limit}")

    def get_comment_count(self, tweet_id: str) -> int:
        """
        Get the current number of comments made on a tweet.

        Args:
            tweet_id: The ID of the tweet to check.

        Returns:
            The number of comments made on the tweet.
        """
        tweet_id = str(tweet_id)
        return self.comment_counts.get(tweet_id, 0)

    def can_comment(self, tweet_id: str) -> bool:
        """
        Check if the bot can still comment on a specific tweet.

        Args:
            tweet_id: The ID of the tweet to check.

        Returns:
            True if the comment limit has not been reached, False otherwise.
        """
        tweet_id = str(tweet_id)
        current_count = self.get_comment_count(tweet_id)
        limit = self.comment_limits.get(tweet_id, self.default_comment_limit)
        can_comment = current_count < limit

        if not can_comment:
            logging.info(f"Comment limit ({limit}) reached for tweet {tweet_id}")

        return can_comment

    def get_tweet_count(self) -> int:
        """
        Get the number of tweets tracked.

        Returns:
            The number of tweets in the tracker.
        """
        return len(self.our_tweet_ids)

    def get_all_comment_stats(self) -> Dict[str, Dict]:
        """
        Get statistics about comments on all tracked tweets.

        Returns:
            A dictionary mapping tweet IDs to their comment counts and limits.
        """
        stats = {}
        for tweet_id in self.comment_counts:
            stats[tweet_id] = {
                "count": self.comment_counts[tweet_id],
                "limit": self.comment_limits.get(tweet_id, self.default_comment_limit),
                "remaining": self.comment_limits.get(tweet_id, self.default_comment_limit) - self.comment_counts[
                    tweet_id]
            }
        return stats