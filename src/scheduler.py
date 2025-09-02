from datetime import datetime, timezone, timedelta
from typing import List, Optional, Literal
from pydantic import BaseModel, Field



class ActionMeta(BaseModel):
    scheduled_time: datetime  # When the post should be posted
    completed: bool = False  # Flag indicating if the post has been completed

class PollSchedule(ActionMeta):
    poll_question: str  # The question to ask in the poll
    poll_options: List[str] = Literal[2,3,4]  # Twitter allows 2 to 4 options
    poll_duration_minutes: int = 1440  # Duration in minutes (default is one day)

class Schedule(ActionMeta):
    current_events: str  # What the post is about
    content: str = ""  # this is the content that will be used in making the posts
    # Media-related fields
    include_media: bool = False
    media_type: Optional[Literal["image", "video"]] = None
    media_prompt: str = ""


class RetweetSchedule(ActionMeta):
    time_posted: datetime  # The time the tweet was posted
    source_acc: str  # The source account from which the tweet originates
    tweet_id: str  # The tweet ID of the tweet to be retweeted
    like_count: int = 0  # The number of likes the tweet has
    retweet_count: int = 0  # The number of retweets the tweet has


class CommentSchedule(ActionMeta):
    time_posted: datetime  # The time the tweet was posted
    tweet_id: str  # The competitor's tweet ID to comment on
    comment_text: str  # The text of the comment
    company_link: Optional[
        str] = None  # link to the company's competitor.This will be used to provide additional context


class CompetitorCommentData(BaseModel):
    time_posted: datetime  # When the tweet was posted.
    tweet_id: str  # The unique tweet ID.
    comment_text: str  # The comment text to use.
    company_link: Optional[str] = None  # Optional link to the competitor's company.


class RetweetCandidate(BaseModel):
    time_posted: datetime  # When the original tweet was posted
    source_acc: str  # Source account of the tweet
    tweet_id: str  # Tweet ID
    tweet_text: str  # The actual content of the tweet
    like_count: int = 0  # Number of likes
    retweet_count: int = 0  # Number of retweets
    selected: bool = False  # Flag to mark if this tweet has been selected for scheduling


# Helper function for timezone handling - add this to all classes
def ensure_timezone_aware(dt):
    """Ensure a datetime object is timezone-aware."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class ScheduleManager:
    def __init__(self, ):
        # List of schedule entries yet to be posted.
        self.pending_schedules: List[Schedule] = []
        # List of schedule entries that have already been posted.
        self.completed_schedules: List[Schedule] = []

    def add_schedule(self, schedule_entry: Schedule) -> None:
        """
        Add a new schedule entry. The entry can include multiple details,
        and the manager will sort all pending entries by time it is important as later in post handler class
        we need to post the posts that are almost there
        """
        # Ensure datetime is timezone-aware
        schedule_entry.scheduled_time = ensure_timezone_aware(schedule_entry.scheduled_time)
        self.pending_schedules.append(schedule_entry)
        self.sort_schedules()

    def add_schedule_with_media(self, schedule_entry: Schedule) -> None:
        schedule_entry.scheduled_time = ensure_timezone_aware(schedule_entry.scheduled_time)
        self.pending_schedules.append(schedule_entry)
        self.sort_schedules()

    def sort_schedules(self) -> None:
        """Sort the pending schedules in ascending order based on their scheduled_time."""
        self.pending_schedules.sort(key=lambda entry: entry.scheduled_time)

    def get_next_schedule(self) -> Optional[Schedule]:
        """
        Return the next scheduled post (the one with the earliest scheduled_time).
        If no pending schedule exists, return None.
        """
        if self.pending_schedules:
            return self.pending_schedules[0]
        return None

    def remove_scheduled_post(self, schedule_entry: Schedule) -> None:
        """
        Remove a schedule entry from the pending schedules, mark it as completed,
        and add it to the completed schedules.
        """
        if schedule_entry in self.pending_schedules:
            self.pending_schedules.remove(schedule_entry)
            schedule_entry.completed = True
            self.completed_schedules.append(schedule_entry)

    def get_all_pending(self) -> List[Schedule]:
        """Return all pending schedule entries."""
        return self.pending_schedules

    def get_all_completed(self) -> List[Schedule]:
        """Return all completed schedule entries."""
        return self.completed_schedules

    def get_all_events(self) -> List[Schedule]:
        """
        Return all schedule entries (both pending and completed),
        sorted in ascending order based on scheduled_time.
        """
        all_events = self.pending_schedules + self.completed_schedules
        return sorted(all_events, key=lambda entry: entry.scheduled_time)

    def get_overdue_events(self) -> List[Schedule]:
        """
        Return all pending schedule entries that are overdue (i.e., scheduled_time is in the past).
        """
        now = datetime.now(timezone.utc)  # Create timezone-aware now
        return [entry for entry in self.pending_schedules if ensure_timezone_aware(entry.scheduled_time) < now]

    def get_future_events(self) -> List[Schedule]:
        """
        Return all pending schedule entries with scheduled_time in the future.
        """
        now = datetime.now(timezone.utc)  # Create timezone-aware now
        return [entry for entry in self.pending_schedules if ensure_timezone_aware(entry.scheduled_time) >= now]


class RetweetManager:
    def __init__(self,max_age: int = 24):
        self.candidate_tweets: List[RetweetCandidate] = []
        self.pending_retweets: List[RetweetSchedule] = []
        self.completed_retweets: List[RetweetSchedule] = []
        self.max_age = max_age

    def already_retweeted(self, tweet_id: str) -> bool:
        """Return True if a retweet for the given tweet_id is pending or completed."""
        for entry in self.pending_retweets + self.completed_retweets:
            if entry.tweet_id == tweet_id:
                return True
        return False

    def add_candidate(self, candidate: RetweetCandidate) -> None:
        """Add a new tweet as a candidate for retweeting"""
        # Ensure datetime is timezone-aware
        candidate.time_posted = ensure_timezone_aware(candidate.time_posted)
        if not self._is_duplicate_candidate(candidate.tweet_id):
            self.candidate_tweets.append(candidate)
            self.sort_pending()

    def _is_duplicate_candidate(self, tweet_id: str) -> bool:
        """Check if tweet is already in candidates or scheduled"""
        # Check in candidates
        if any(c.tweet_id == tweet_id for c in self.candidate_tweets):
            return True
        # Check in pending retweets
        if any(r.tweet_id == tweet_id for r in self.pending_retweets):
            return True
        # Check in completed retweets
        if any(r.tweet_id == tweet_id for r in self.completed_retweets):
            return True
        return False

    def schedule_from_candidate(self, tweet_id: str, scheduled_time: datetime) -> Optional[RetweetSchedule]:
        """
        Create a retweet schedule from a candidate tweet
        Returns the created schedule if successful, None if candidate not found
        """
        # Ensure datetime is timezone-aware
        scheduled_time = ensure_timezone_aware(scheduled_time)

        candidate = next(
            (tweet for tweet in self.candidate_tweets if tweet.tweet_id == tweet_id and not tweet.selected), None)
        if candidate is None:
            return None

        # Ensure time_posted is timezone-aware
        time_posted = ensure_timezone_aware(candidate.time_posted)

        retweet = RetweetSchedule(
            scheduled_time=scheduled_time,
            time_posted=time_posted,
            source_acc=candidate.source_acc,
            tweet_id=candidate.tweet_id,
            like_count=candidate.like_count,
            retweet_count=candidate.retweet_count
        )

        candidate.selected = True
        self.remove_candidate(tweet_id)
        self.pending_retweets.append(retweet)
        self.sort_pending()
        return retweet

    def mark_completed(self, retweet: RetweetSchedule) -> None:
        """Mark a retweet as completed and move it to completed list"""
        if retweet in self.pending_retweets:
            self.pending_retweets.remove(retweet)
            retweet.completed = True
            self.completed_retweets.append(retweet)

    def remove_candidate(self, tweet_id: str) -> None:
        """Remove a tweet from candidates list"""
        self.candidate_tweets = [c for c in self.candidate_tweets if c.tweet_id != tweet_id]

    def sort_pending(self) -> None:
        """Sort pending retweets by scheduled time"""
        now = ensure_timezone_aware(datetime.utcnow())
        cutoff_time = now - timedelta(hours=self.max_age)

        before_removal = len(self.candidate_tweets)
        self.candidate_tweets = [
            candidate for candidate in self.candidate_tweets
            if ensure_timezone_aware(candidate.time_posted) > cutoff_time
        ]
        removed_count = before_removal - len(self.candidate_tweets)

        if removed_count > 0:
            print(f"Removed {removed_count} expired candidates (older than {self.max_age} hours).")
        self.candidate_tweets.sort(key=lambda c: (c.retweet_count, c.like_count), reverse=True)


    def add_retweet(self, retweet: RetweetSchedule) -> None:
        if self.already_retweeted(retweet.tweet_id):
            print(f"Retweet for tweet {retweet.tweet_id} already scheduled or completed.")
            return

        # Ensure datetimes are timezone-aware
        retweet.scheduled_time = ensure_timezone_aware(retweet.scheduled_time)
        retweet.time_posted = ensure_timezone_aware(retweet.time_posted)

        self.pending_retweets.append(retweet)
        self.sort_retweets()

    def get_all_candidates(self) -> List[RetweetCandidate]:
        """
        Returns a filtered list of retweet candidates that are within the max_age limit.
        """
        now = ensure_timezone_aware(datetime.now(timezone.utc))
        cutoff_time = now - timedelta(hours=self.max_age)

        valid_candidates = [
            candidate for candidate in self.candidate_tweets
            if ensure_timezone_aware(candidate.time_posted) > cutoff_time
        ]

        return valid_candidates

    def sort_retweets(self) -> None:
        self.pending_retweets.sort(key=lambda r: r.scheduled_time)

    def get_next_retweet(self) -> Optional[RetweetSchedule]:
        return self.pending_retweets[0] if self.pending_retweets else None

    def get_all_pending(self) -> List[RetweetSchedule]:
        """Get all pending retweets"""
        return self.pending_retweets

    def mark_retweet_completed(self, retweet: RetweetSchedule) -> None:
        if retweet in self.pending_retweets:
            self.pending_retweets.remove(retweet)
            retweet.completed = True
            self.completed_retweets.append(retweet)

    def get_overdue_retweets(self) -> List[RetweetSchedule]:
        now = datetime.now(timezone.utc)  # Create timezone-aware now
        return [r for r in self.pending_retweets if ensure_timezone_aware(r.scheduled_time) < now]

    def get_future_retweets(self) -> List[RetweetSchedule]:
        now = datetime.now(timezone.utc)  # Create timezone-aware now
        return [r for r in self.pending_retweets if ensure_timezone_aware(r.scheduled_time) >= now]


class CommentManager:
    def __init__(self):
        self.pending_comments: List[CommentSchedule] = []
        self.completed_comments: List[CommentSchedule] = []

    def add_comment(self, comment: CommentSchedule) -> None:
        # Ensure datetimes are timezone-aware
        comment.scheduled_time = ensure_timezone_aware(comment.scheduled_time)
        comment.time_posted = ensure_timezone_aware(comment.time_posted)

        self.pending_comments.append(comment)
        self.sort_comments()

    def sort_comments(self) -> None:
        self.pending_comments.sort(key=lambda c: c.scheduled_time)

    def get_next_comment(self) -> Optional[CommentSchedule]:
        return self.pending_comments[0] if self.pending_comments else None

    def mark_comment_completed(self, comment: CommentSchedule) -> None:
        if comment in self.pending_comments:
            self.pending_comments.remove(comment)
            comment.completed = True
            self.completed_comments.append(comment)

    def get_overdue_comments(self) -> List[CommentSchedule]:
        now = datetime.now(timezone.utc)  # Create timezone-aware now
        return [c for c in self.pending_comments if ensure_timezone_aware(c.scheduled_time) < now]

    def get_future_comments(self) -> List[CommentSchedule]:
        now = datetime.now(timezone.utc)  # Create timezone-aware now

        # Add debug logging to help identify the issue
        future_comments = []
        for c in self.pending_comments:
            # Ensure the datetime is timezone-aware before comparison
            c_time = ensure_timezone_aware(c.scheduled_time)
            if c_time <= now:  # Changed from >= to <= for "future" comments that are ready to be processed
                # Also ensure time_posted is timezone-aware
                c.time_posted = ensure_timezone_aware(c.time_posted)
                future_comments.append(c)

        return future_comments

    def get_all_events(self) -> List[Schedule]:
        """
        Return all schedule entries (both pending and completed),
        sorted in ascending order based on scheduled_time.
        """
        all_events = self.pending_comments + self.completed_comments
        return sorted(all_events, key=lambda entry: entry.scheduled_time)


class CompetitorCommentManager:
    def __init__(self):
        self.pending_comments: List[CompetitorCommentData] = []

    def add_comment(self, comment: CompetitorCommentData) -> None:
        # Ensure time_posted is timezone-aware
        comment.time_posted = ensure_timezone_aware(comment.time_posted)

        self.pending_comments.append(comment)
        self.pending_comments.sort(key=lambda x: x.time_posted)

    def remove_comment(self, comment: CompetitorCommentData) -> None:
        """
        Remove a competitor comment from the pending list without marking it as completed.
        """
        if comment in self.pending_comments:
            self.pending_comments.remove(comment)

    def get_all(self) -> List[CompetitorCommentData]:
        # Ensure all time_posted values are timezone-aware
        for comment in self.pending_comments:
            comment.time_posted = ensure_timezone_aware(comment.time_posted)

        return self.pending_comments

class PollScheduleManager:
    def __init__(self):
        # List of poll schedule entries yet to be posted.
        self.pending_polls: List[PollSchedule] = []
        # List of poll schedule entries that have already been posted.
        self.completed_polls: List[PollSchedule] = []

    def add_poll_schedule(self, poll_schedule: PollSchedule) -> None:
        """
        Add a new poll schedule entry.
        """
        poll_schedule.scheduled_time = ensure_timezone_aware(poll_schedule.scheduled_time)
        self.pending_polls.append(poll_schedule)
        self.sort_polls()

    def sort_polls(self) -> None:
        """
        Sort pending poll schedules in ascending order based on scheduled_time.
        """
        self.pending_polls.sort(key=lambda poll: poll.scheduled_time)

    def get_next_poll(self) -> Optional[PollSchedule]:
        """
        Return the next scheduled poll (the one with the earliest scheduled_time).
        """
        if self.pending_polls:
            return self.pending_polls[0]
        return None

    def mark_poll_completed(self, poll_schedule: PollSchedule) -> None:
        """
        Mark a poll schedule as completed and move it to the completed list.
        """
        if poll_schedule in self.pending_polls:
            self.pending_polls.remove(poll_schedule)
            poll_schedule.completed = True
            self.completed_polls.append(poll_schedule)

    def get_all_pending(self) -> List[PollSchedule]:
        """
        Return all pending poll schedules.
        """
        return self.pending_polls

    def get_all_completed(self) -> List[PollSchedule]:
        """
        Return all completed poll schedules.
        """
        return self.completed_polls
    def get_all_polls(self)  -> List[PollSchedule]:
        """
        Return all poll schedules.
        """
        return self.completed_polls+self.pending_polls

    def get_overdue_polls(self) -> List[PollSchedule]:
        """
        Return all pending poll schedules that are overdue.
        """
        now = datetime.now(timezone.utc)
        return [poll for poll in self.pending_polls if ensure_timezone_aware(poll.scheduled_time) < now]

    def get_future_polls(self) -> List[PollSchedule]:
        """
        Return all pending poll schedules with scheduled_time in the future.
        """
        now = datetime.now(timezone.utc)
        return [poll for poll in self.pending_polls if ensure_timezone_aware(poll.scheduled_time) >= now]




