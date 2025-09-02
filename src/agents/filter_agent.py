import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()


def create_crypto_filter_agent(
        model: str = "openai/o3-mini-high",
        api_key: str = "",
        markdown: bool = False,
        show_tool_calls: bool = False
) -> Agent:
    """
    Creates a crypto reply filter agent that evaluates a comment to determine whether it is related
    to cryptocurrencies or blockchain technology.

    The agent analyzes the comment for crypto-related keywords (e.g., 'crypto', 'Bitcoin', 'Ethereum',
    'altcoin', 'blockchain') and outputs a JSON object with:
      - "should_reply": a boolean that is true if the comment is crypto related and merits a reply,
                        or false if it is not.

    Example outputs:
      {"should_reply": true}
      {"should_reply": false}
    """
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    instructions = dedent(f"""
        You are a crypto reply filter agent. Your task is to analyze the provided comment text and determine whether it is related
        to cryptocurrencies or blockchain technology or anything related to finance or something useful for our company 365x.ai which
        is an AI automation solution provider company. Look for references to keywords such as 'crypto', 'Bitcoin', 'Ethereum',
        'altcoin', 'blockchain', or similar terms. If the comment is related to crypto, output a JSON object with 
        "should_reply" set to true; otherwise, output a JSON object with "should_reply" set to false.

        Ensure that your output is plain text JSON without any markdown, quotes, or extra formatting, and do not include any additional text.

        Also, for your reference, the current time is {current_date_time}.
    """).strip()

    return Agent(
        model=OpenRouter(id=model, api_key=api_key if api_key else os.getenv("OPENROUTER_API_KEY")),
        instructions=[instructions],
        description="Crypto reply filter agent that outputs a JSON decision on whether a comment is related to cryptocurrencies and worth replying to.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("{\"should_reply\": <boolean>}")
    )
