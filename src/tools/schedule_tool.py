from datetime import datetime, timedelta, timezone
from typing import List
from phi.tools import Toolkit
from phi.utils.log import logger
from scheduler import Schedule, ScheduleManager


class ScheduleTool(Toolkit):
    """
    ScheduleTool is a Phidata Toolkit for managing scheduling events.

    This tool allows agents to add new schedules and retrieve events (all, overdue,
    or future) in various formats. The methods in this class are registered as tools,
    so they can be called by a Phidata agent to integrate scheduling functionality into workflows.
    """

    def __init__(self, schedulemanager: ScheduleManager, with_media: bool = False):
        """
        Initialize the ScheduleTool with a ScheduleManager instance.

        Args:
            schedulemanager (ScheduleManager): An instance responsible for managing schedule events.
        """
        super().__init__(name="schedule_tool")
        self.manager = schedulemanager

        # Register tool functions to make them accessible for agent use.
        self.register(self.get_all_events_str)
        if with_media :
            self.register(self.add_schedule_with_media)
        else:
            self.register(self.add_schedule)


    def add_schedule(self, event: str, post_content: str = "") -> str:
        """
        Add a new schedule to the system.

        Creates a new Schedule object with the given scheduled time, event description,
        and additional content. The new schedule is then added to the ScheduleManager.

        Args:
            event (str): A brief description or title of the event.
            post_content (str, optional): Additional details or content about the event must be given.

        Returns:
            str: A confirmation message with the ISO-formatted scheduled time.
        """
        scheduled_time = datetime.now(timezone.utc)
        new_schedule = Schedule(scheduled_time=scheduled_time, current_events=event, content=post_content)

        self.manager.add_schedule(new_schedule)

        # Log the addition for debugging and audit purposes.
        logger.info(f"Added schedule: {new_schedule}")

        # Return a formatted confirmation message.
        return f"Schedule added for {scheduled_time}."

    @staticmethod
    def get_all_events(self) -> List[dict]:
        """
        Retrieve all scheduled events.

        Fetches all events stored in the ScheduleManager and converts each event
        into a dictionary for standardized processing.

        Returns:
            List[dict]: A list of dictionaries representing each scheduled event.
        """
        # Retrieve all events from the manager.
        events = self.manager.get_all_events()

        # Convert each event to a dictionary using model_dump() for consistency.
        return [event.model_dump() for event in events]

    def get_all_events_str(self) -> str:
        """
        Retrieve future events formatted as a readable string.

        This method fetches all future events, formats each event's key details (description,
        scheduled time, and content) into a string, and then concatenates them into a single string.

        Returns:
            str: A newline-separated string containing details of all future events, or a message
                 indicating that no future events are scheduled.
        """
        # Fetch future events from the manager.
        events = self.manager.get_all_events()

        # If no future events are available, return an appropriate message.
        if not events:
            return "No future events scheduled."

        # Initialize an empty list to hold formatted event details.
        event_lines = []

        # Loop through each event and format its details.
        for event in events:
            event_lines.append(
                f"Event: {event.current_events} | Scheduled: {event.scheduled_time.isoformat()} | Content:"
                f" {event.content}"
            )

        # Join the formatted lines with newline characters and return the result.
        return "\n".join(event_lines)

    def add_schedule_with_media(self, event: str, post_content: str = "",
                                media_type: str = "image",
                                media_prompt: str = "Bitcoin changing with time"
                                ) -> str:
        """
        Add a new schedule that includes media details.

        This method creates a new Schedule object with media-related fields and adds it to the ScheduleManager.
        You wont be giving a time to it as the time will be set automatically

        Args:
            event (str): A brief description or title of the event.
            post_content (str, optional): Additional details or content about the event.
            media_type (str) : The type of media ("image" or "video").
            media_prompt (str): A creative prompt for generating media.

        Returns:
            str: A confirmation message with the ISO-formatted scheduled time.
        """
        scheduled_time = datetime.now(timezone.utc)
        new_schedule = Schedule(
            scheduled_time=scheduled_time,
            current_events=event,
            content=post_content,
            include_media=True,
            media_type=media_type,
            media_prompt=media_prompt
        )

        self.manager.add_schedule_with_media(new_schedule)
        logger.info(f"Added schedule: {new_schedule}")
        return f"Media schedule added for {scheduled_time}."
