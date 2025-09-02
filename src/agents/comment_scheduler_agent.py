import logging
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from phi.tools.exa import ExaTools
import os
from dotenv import load_dotenv
from src.tools.comment_transfer_tool import CommentTransferTool
from datetime import datetime, timezone
load_dotenv()

def create_competitor_comment_agent(
    model: str = "google/gemini-2.0-flash-001",
    comment_transfer_tool: CommentTransferTool = None,
    exa_api_key: str = "",
    markdown: bool = False,
    show_tool_calls: bool = False
) -> Agent:
    if comment_transfer_tool is None:
        logging.error("No comment transfer tool provided; please supply a valid tool instance.")
    if exa_api_key:
        exa_tool = ExaTools(api_key=exa_api_key)
    else:
        logging.error("No ExaTools API key provided; attempting to load from .env")
        if os.getenv("EXA_API_KEY"):
            exa_tool = ExaTools(api_key=os.getenv("EXA_API_KEY"))
        else:
            logging.error("No EXA_API_KEY found in .env; using default ExaTools instance")
            exa_tool = ExaTools(api_key="")
    return Agent(
        model=OpenRouter(id=model, api_key=os.getenv("OPENROUTER_API_KEY")),
        tools=[comment_transfer_tool,exa_tool],
        instructions=[
            dedent("""\
            You are a competitor comment scheduling agent for 365x.ai.
            First, invoke the tool's list_all_competitor_comments function to retrieve all available competitor comment tweets.
            Then, based on the number of tweets provided in your prompt, select exactly that many competitor tweet IDs 
            and invoke the transfer_comment function for each tweet ID to schedule them.
            Finally, return a summary indicating the number of comments scheduled and a brief summary of each.
            The comments should be related to crypto ,finance or something similar.
            """)
        ],
        description="Agent that manages competitor comment scheduling: selecting competitor tweets and scheduling comments for 365x.ai",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("""\
            Number of comments scheduled: {num_comments}
            Comment summary: {comment_summary}
        """)
    )
