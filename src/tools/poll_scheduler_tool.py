from datetime import datetime,timezone
from typing import List
from phi.tools import Toolkit
from phi.utils.log import logger
from scheduler import PollSchedule, PollScheduleManager

class PollSchedulerTool(Toolkit):
    """
    Tool for scheduling polls.
    """

    def __init__(self, schedule_manager: PollScheduleManager):
        """
        Initialize the PollSchedulerTool with a PollScheduleManager instance.

        Args:
            schedule_manager (PollScheduleManager): Manages the scheduling of polls.
        """
        super().__init__(name="poll_scheduler_tool")
        self.manager = schedule_manager

        self.register(self.add_poll)
        self.register(self.get_all_polls_str)

    def add_poll(self, poll_question: str, poll_options: List[str], poll_duration_minutes: int) -> str:
        """
        Add a new poll to the schedule.

        Args:
            poll_question (str): The question to ask in the poll.
            poll_options (List[str]): Options for the poll (2 to 4 options) in 20 characters or less.
            poll_duration_minutes (int): Duration of the poll in minutes.

        Returns:
            str: Confirmation message with the ISO-formatted scheduled time.
        """
        # Validate the number of poll options.
        if not (2 <= len(poll_options) <= 4):
            return "Poll must have between 2 and 4 options."
        scheduled_time = datetime.now(timezone.utc)
        new_poll = PollSchedule(
            poll_question=poll_question,
            poll_options=poll_options,
            poll_duration_minutes=poll_duration_minutes,
            scheduled_time=scheduled_time
        )

        self.manager.add_poll_schedule(new_poll)

        # Log the addition for debugging and audit purposes.
        logger.info(f"Added poll: {new_poll}")

        # Return a formatted confirmation message.
        return f"Poll scheduled for {scheduled_time.isoformat()}."

    def get_all_polls(self) -> List[dict]:
        """
        Retrieve all scheduled polls.

        Returns:
            List[dict]: A list of dictionaries representing each scheduled poll.
        """
        # Retrieve all polls from the manager.
        polls = self.manager.get_all_polls()

        # Convert each poll to a dictionary.
        return [poll.model_dump() for poll in polls]

    def get_all_polls_str(self) -> str:
        """
        Retrieve scheduled polls formatted as a readable string.

        Returns:
            str: A newline-separated string containing details of all scheduled polls,
                 or a message indicating that no polls are scheduled.
        """
        # Fetch scheduled polls from the manager.
        polls = self.manager.get_all_polls()

        # If no polls are available, return an appropriate message.
        if not polls:
            return "No polls scheduled."

        # Initialize an empty list to hold formatted poll details.
        poll_lines = []

        # Loop through each poll and format its details.
        for poll in polls:
            poll_lines.append(
                f"Poll: {poll.poll_question} | Scheduled: {poll.scheduled_time.isoformat()} | "
                f"Options: {', '.join(poll.poll_options)} | Duration: {poll.poll_duration_minutes} minutes"
            )

        # Join the formatted lines with newline characters and return the result.
        return "\n".join(poll_lines)
