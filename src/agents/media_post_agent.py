import logging
from textwrap import dedent
from phi.agent import Agent
from tools.schedule_tool import ScheduleTool
from phi.tools.exa import ExaTools
from phi.model.openrouter import OpenRouter
import os
from datetime import datetime,timezone
from dotenv import load_dotenv
load_dotenv()

def create_media_schedule_agent(model: str = "openai/gpt-4o-mini",
                          schedule_tool: ScheduleTool = None,
                          markdown: bool = False,
                          show_tool_calls: bool = False) -> Agent:
    exa_tool = ExaTools(api_key=os.getenv("EXA_API_KEY"))
    if schedule_tool is None:
        logging.error("No scheduling tool provided; please supply a valid scheduling tool instance.")
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return Agent(
        model=OpenRouter(id=model, api_key=os.getenv("OPENROUTER_API_KEY"),temperature=0.4),
        tools=[schedule_tool,exa_tool],
        instructions=[
            (
                "You are a scheduling agent responsible for curating high-quality, factually accurate crypto posts with creative media prompts. "
                "Start by retrieving all existing scheduled posts using the method `get_all_events_str`. "
                "Carefully analyze these past posts to extract key insights, recent crypto market trends, and ongoing narratives. "
                "Use only the extracted data and avoid generating generic or repetitive content. "
                "Ensure the new post introduces a fresh perspective or an important update that complements past posts while maintaining factual correctness. "
                "Create a compelling media prompt that visually represents the core theme of the post, ensuring relevance to the content. "
                "Never generate more than one post at a time.Use the tool given to you to schedule the post with the media prompt and other things you have decided on "
                "If no meaningful insights can be derived, avoid speculation and instead refine an existing topic from previous posts with deeper analysis. "
                "Strictly **avoid vague, generic posts**—always ensure specificity and accuracy.Make sure that the post is not generic "
                "And the post should be bringing maximum engagement too. Make sure that the post_content is no more than 200 characters"
                "Also make sure that the pictures are creative not just any stats or charts And you have to schedule at least one post "
                "You can also use the exa tool given to you to get more info about anything before scheduling the post too"
                f"All the information should be factually correct and today's time is{current_date_time}"
            )
        ],
        description="Agent that generates a new crypto post with a creative image prompt by analyzing existing scheduled posts.",
        show_tool_calls=show_tool_calls,
        structured_outputs=True,
        markdown=markdown,
        expected_output=dedent("""\
        {
            summary of the post you have scheduled 
        }
        """)
    )
