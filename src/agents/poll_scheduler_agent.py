import logging
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
import os
from dotenv import load_dotenv
from tools.poll_scheduler_tool import PollSchedulerTool
from datetime import datetime, timezone
load_dotenv()

def create_poll_scheduler_agent(
    model: str = "openai/gpt-4o-mini",
    poll_scheduler_tool: PollSchedulerTool = None,
    markdown: bool = False,
    show_tool_calls: bool = False
) -> Agent:
    """
    Creates an agent responsible for scheduling polls.

    Args:
        model (str): The language model to use.
        poll_scheduler_tool (PollSchedulerTool): Instance of the poll scheduling tool.
        markdown (bool): Whether to format responses in Markdown.
        show_tool_calls (bool): Whether to display tool calls.

    Returns:
        Agent: Configured agent for scheduling polls.
    """
    if poll_scheduler_tool is None:
        logging.error("No poll scheduling tool provided; please supply a valid PollSchedulerTool instance.")
        return None
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return Agent(
        model=OpenRouter(id=model, api_key=os.getenv("OPENROUTER_API_KEY"),temperature=0.6),
        tools=[poll_scheduler_tool],
        instructions=[
            (
                "You are a poll scheduling agent responsible for generating new poll entries based on current trends and crypto market "
                "First, retrieve existing polls using the provided poll scheduling tool (`get_all_polls_str`). "
                "Then, analyze recent data to create new, unique poll that encourage user interaction without duplicating existing polls. "
                "The poll question should be interactive and based on the latest info provided to you"
                "Ensure that each poll has between 2 and 4 options and is scheduled at optimal times for maximum engagement. "
                "Make sure the options are not more than 20 characters long "
                "Finally, return a summary of the new poll added. Make sure you only add one poll "
                "For example : one instance can be 31x in 7 hours for $PVS - what happens next 1.dump to .5m 2.stable at 5m+"
                f"Also for your reference current time is {current_date_time}"
            )
        ],
        description=(
            "Agent that manages the scheduling of polls, ensuring they are engaging, unique, and scheduled at times that maximize user interaction. "
            "The agent considers current trends and avoids duplicating existing polls."
        ),
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("""\
            Poll summary: {poll_summary}
        """)
    )
