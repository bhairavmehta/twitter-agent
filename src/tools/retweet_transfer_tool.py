from datetime import datetime, timezone, timedelta
from typing import List, Optional
from phi.tools import Toolkit
from phi.utils.log import logger
from src.scheduler import RetweetSchedule, RetweetManager, RetweetCandidate


class RetweetTransferTool(Toolkit):
    """
    RetweetTransferTool transfers candidate tweets to scheduled retweets.
    It handles the process of selecting high-value tweets and scheduling them for retweeting.
    """

    def __init__(self, retweet_manager: RetweetManager):
        super().__init__(name="retweet_transfer_tool")
        self.retweet_manager = retweet_manager
        self.register(self.transfer_retweet)
        self.register(self.list_all_candidates)

    def list_all_candidates(self) -> str:
        """
        Return a formatted string listing all candidate tweets available for retweeting.

        Returns:
            str: A newline-separated string with details of each candidate tweet or a message if none exist.
        """
        candidates = self.retweet_manager.get_all_candidates()
        if not candidates:
            return "No candidate tweets available for retweeting."

        lines = []
        for candidate in candidates:
            engagement = candidate.like_count + candidate.retweet_count
            lines.append(
                f"Tweet ID: {candidate.tweet_id} | "
                f"Posted: {candidate.time_posted.isoformat()} | "
                f"From: {candidate.source_acc} | "
                f"Engagement: {engagement} | "
                f"Text: {candidate.tweet_text[:200]}..."
            )
        all_candidates = "\n".join(lines)
        return all_candidates

    def transfer_retweet(self, tweet_id: str) -> str:
        """
        Transfer a candidate tweet to the scheduled retweets.

        Args:
            tweet_id (str): Unique identifier of the tweet to be scheduled for retweeting.

        Returns:
            str: A confirmation message with the scheduled time.
        """
        # Calculate scheduling time (e.g., 1 minute from now)
        scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=1)

        # Try to schedule the retweet
        result = self.retweet_manager.schedule_from_candidate(tweet_id, scheduled_time)

        if result is None:
            return f"No available candidate found with tweet_id: {tweet_id}"

        logger.info(f"Scheduled retweet of tweet {tweet_id} for {scheduled_time.isoformat()}")
        return f"Scheduled retweet of tweet {tweet_id} for {scheduled_time.isoformat()}"