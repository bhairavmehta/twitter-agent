import logging
import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv
from datetime import datetime, timezone
load_dotenv()

def create_post_generator_agent(
    model: str = "deepseek/deepseek-r1:free",
    api_key: str = "",
    markdown: bool = False,
    show_tool_calls: bool = False
) -> Agent:
    """
    Creates an agent that generates tweet content for social media posts based on the provided input (e.g., current events).

    Instructions:
      - You are a tweet post generator agent.
      - Your task is to craft an engaging tweet post based on the input content that reflects current events and any relevant details.
      - Ensure your output is in plain text (no markdown formatting) and does not exceed Twitter's 280-character limit.
      - The tweet should be concise, clear, and align with the brand's personality.
      - Incorporate any key details from the input seamlessly.
      - Return the tweet post labeled as "Tweet Post".

    The agent uses the provided model and API key via the OpenRouter interface.
    """
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return Agent(
        model=OpenRouter(id=model, api_key=api_key),
        instructions=[
            (
                "You are a tweet post composer agent. Your task is to generate an engaging and edgy tweet post using two inputs: "
                "the event details and the provided content for the post. Incorporate all the information that is "
                "given, but do not mention or speculate about any details that are missing from the context. "
                "Your final tweet must be in plain text (no markdown), not exceed 280 characters, and should adopt a "
                "snarky, edgy tone with a bit of slang. Output only the tweet text that will be sent."
                "generate an edgy, snarky, and very short tweet post. adopt that edgy tone and use a bit of slang too"
                "make sure you are giving a post that is engaging yet not too long and dont use more than 1 emoji and also "
                "Dont use any hashtags either."
                f"Also for your reference current time is {current_date_time}"
            )
        ],
        description="Agent that generates tweet content for social media posts based on current events.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,

    )
