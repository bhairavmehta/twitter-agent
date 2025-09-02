import logging
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from phi.tools.exa import ExaTools
import os
from dotenv import load_dotenv
from src.tools.retweet_transfer_tool import RetweetTransferTool
from datetime import datetime, timezone
load_dotenv()


def create_retweet_agent(
        model: str = "google/gemini-2.0-flash-001",
        retweet_transfer_tool: RetweetTransferTool = None,
        exa_api_key: str = "",
        markdown: bool = False,
        show_tool_calls: bool = False
) -> Agent:
    """
    Create an agent that evaluates and schedules retweets based on engagement metrics
    and content relevance.
    """
    if retweet_transfer_tool is None:
        logging.error("No retweet transfer tool provided; please supply a valid tool instance.")

    if exa_api_key:
        exa_tool = ExaTools(api_key=exa_api_key)
    else:
        logging.error("No ExaTools API key provided; attempting to load from .env")
        if os.getenv("EXA_API_KEY"):
            exa_tool = ExaTools(api_key=os.getenv("EXA_API_KEY"))
        else:
            logging.error("No EXA_API_KEY found in .env; using default ExaTools instance")
            exa_tool = ExaTools(api_key="")
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return Agent(
        model=OpenRouter(id=model, api_key=os.getenv("OPENROUTER_API_KEY")),
        tools=[retweet_transfer_tool, exa_tool],
        instructions=[
            dedent("""\
            You are a retweet scheduling agent for 365x.ai.
            Your job is to select the most impactful tweets to retweet based on:
            1. Engagement metrics (likes and retweets)
            2. Content relevance to cryptocurrency and finance

            Process:
            1. First, use list_all_candidates to get available tweets
            2. Analyze each tweet's metrics and content
            3. Select the specified number of best tweets
            4. Use transfer_retweet for each selected tweet
            5. Provide a summary of scheduled retweets

            Important considerations:
            - Prioritize tweets with high engagement ratios
            - Favor tweets from authoritative sources
            - Avoid controversial or negative content
            
            make sure to at least select one retweet
            """)
        ],
        description="Agent that manages retweet scheduling by selecting high-value tweets based on metrics and content.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("""\
            Number of retweets scheduled: {num_retweets}
            Selected tweets summary: {tweet_summary}
        """)
    )
