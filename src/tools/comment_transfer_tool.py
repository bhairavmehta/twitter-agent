from datetime import datetime, timezone, timedelta
from typing import List, Optional
from pydantic import BaseModel
from phi.tools import Toolkit
from phi.utils.log import logger
from src.scheduler import CommentSchedule, CommentManager, CompetitorCommentManager


class CommentTransferTool(Toolkit):
    """
    CommentTransferTool transfers a competitor comment (identified by tweet_id)
    from the CompetitorCommentManager to the CommentSchedulerManager.

    It looks up the competitor comment in the competitor manager,
    removes it from that list, creates a scheduled comment entry (e.g. scheduled 1 minute in the future),
    and adds it to the scheduler manager.
    """

    def __init__(self, competitor_manager: CompetitorCommentManager, scheduler_manager: CommentManager):
        super().__init__(name="comment_transfer_tool")
        self.competitor_manager = competitor_manager
        self.scheduler_manager = scheduler_manager
        self.register(self.transfer_comment)
        self.register(self.list_all_competitor_comments)

    def list_all_competitor_comments(self) -> str:
        """
        Return a formatted string listing all competitor comments available in the competitor manager.

        Iterates through all competitor comments stored in the competitor comment manager,
        and returns a string with key details (tweet ID, posted time, and a snippet of the comment text)
        for each comment. If no comments are available, it returns a message indicating so.

        Returns:
            str: A newline-separated string with details of each competitor comment or a message if none exist.
        """
        comments = self.competitor_manager.get_all()
        if not comments:
            return "No competitor comments available."
        lines = []
        for comment in comments:
            # Ensure time_posted has timezone info when formatting
            posted_time = comment.time_posted
            if posted_time.tzinfo is None:
                posted_time = posted_time.replace(tzinfo=timezone.utc)

            lines.append(
                f"Tweet ID: {comment.tweet_id} | Posted: {posted_time.isoformat()} | Text: {comment.comment_text}..."
            )
        all_comments = "\n".join(lines)
        return all_comments

    def transfer_comment(self, tweet_id: str) -> str:
        """
        Transfer a competitor comment to the scheduling system.

        Retrieves the competitor comment data corresponding to the given tweet_id from the raw competitor comment manager,
        removes it from that manager, creates a new scheduled comment entry with a scheduled time (e.g. 1 minute from now),
        and adds it to the CommentSchedulerManager.

        Args:
            tweet_id (str): A unique identifier of the competitor tweet to be transferred as a scheduled comment.

        Returns:
            str: A confirmation message with the ISO-formatted scheduled time of the scheduled comment.
        """
        candidate = None
        for comment in self.competitor_manager.get_all():
            if comment.tweet_id == tweet_id:
                candidate = comment
                break

        if candidate is None:
            return f"No competitor comment found with tweet_id: {tweet_id}"
        self.competitor_manager.remove_comment(candidate)

        # Create an offset-aware datetime for scheduling
        scheduled_time = datetime.now(timezone.utc)

        # Ensure time_posted is also offset-aware
        time_posted = candidate.time_posted
        if time_posted.tzinfo is None:
            time_posted = time_posted.replace(tzinfo=timezone.utc)

        new_schedule = CommentSchedule(
            scheduled_time=scheduled_time,
            tweet_id=candidate.tweet_id,
            comment_text=candidate.comment_text,
            company_link=candidate.company_link,
            time_posted=time_posted
        )

        self.scheduler_manager.add_comment(new_schedule)
        logger.info(f"Transferred tweet {tweet_id} to comment scheduler scheduled for {scheduled_time.isoformat()}")
        return f"Added tweet {tweet_id} to comment scheduler."