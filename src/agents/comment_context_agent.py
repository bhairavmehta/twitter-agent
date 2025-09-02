import logging
import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from tools.cg_tool import PhiCoinGeckoTool
from tools.url_expander_tool import UrlExpanderTool
from phi.tools.exa import ExaTools
from dotenv import load_dotenv
from phi.tools.crawl4ai_tools import Crawl4aiTools
from datetime import datetime, timezone
load_dotenv()


def create_comment_context_agent(
        model: str = "gpt-4o-mini",
        api_key: str = "",
        markdown: bool = False,
        show_tool_calls: bool = False,
        cg_demo_api_key:str =None,
        cg_api_key: str=None,
        exa_api_key: str = None) -> Agent:
    """
    Creates a comment context agent that analyzes a tweet and extracts its key details.

    Instructions:
      - Analyze the provided tweet and extract all key topics and details.
      - Do not mention or speculate about any information that is missing from the tweet.
      - Return a concise plain text summary labeled 'Comment Context' containing only the explicit details.
    """
    if cg_api_key:
        cg_tool = PhiCoinGeckoTool(api_key=cg_api_key)
    elif cg_demo_api_key:
        cg_tool = PhiCoinGeckoTool(demo_api_key=cg_demo_api_key)
    else :
        if os.getenv("COINGECKO_API_KEY"):
            cg_tool = PhiCoinGeckoTool(api_key=os.getenv("COINGECKO_API_KEY"))
        elif os.getenv("COINGECKO_DEMO_API_KEY"):
            cg_tool = PhiCoinGeckoTool(demo_api_key=os.getenv("COINGECKO_DEMO_API_KEY"))
        else :
            logging.log(logging.ERROR, "No COINGECKO API key available in .env")
            cg_tool = PhiCoinGeckoTool()
    if exa_api_key:
        exa_tool = ExaTools(api_key=exa_api_key)
    else:
        from dotenv import load_dotenv
        load_dotenv()
        if os.getenv("EXA_API_KEY"):
            exa_tool = ExaTools(api_key=os.getenv("EXA_API_KEY"))
        else:
            logging.error("No EXA_API_KEY found in .env; using default ExaTools instance")
            exa_tool = ExaTools(api_key="")
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return Agent(
        model=OpenRouter(id=model, api_key=api_key),
        instructions=dedent(f"""
                You are a comment context agent. Your task is to analyze the provided tweet, extract key details, and deliver a **precise, up-to-date context summary** focused on **cryptocurrencies and stocks**.

                **Handling URLs:**
                  - If a URL is present (starting with 'http://' or 'https://'), treat it **only** as a source for additional context.
                  - If the URL is shortened (e.g., 'https://t.co/...'), use the **URL Expander** tool to obtain its final destination.
                  - If further details are required from the expanded URL, use the **Crawl4AI** tool to extract relevant information.
                  - **Never mistake URLs** (shortened or expanded) for coin names or IDs.

                **Identifying Cryptocurrencies & Stocks:**
                  - **Extract any cryptocurrency or stock mentions** from the tweet.
                  - If the mentioned cryptocurrency does **not** have an explicit coin ID, use the **Exa tool** to find it.
                  - **DO NOT** assume a coin's price, volume, or ranking—**always verify these details** using the **CoinGecko tool**.
                  - If a cryptocurrency is referenced generically (e.g., "ETH is pumping"), use the **Exa tool** to check for related news, but **price and volume must always come from CoinGecko**.
                  - For **stocks or general market news**, retrieve relevant data using the **Exa tool**, but keep in mind that Exa data may be outdated.

                **Prioritizing Current Market Data:**
                  - The agent **MUST** always include:
                    - **Current price** of the cryptocurrency from the **CoinGecko tool**.
                    - **Latest 24-hour trading volume** from **CoinGecko**.
                    - **Percentage change** in price over the past 24 hours.
                  - If no direct coin mention is found but the tweet is related to crypto, provide **a general market update** based on **CoinGecko’s most recent data**.
                  - **Never** provide outdated or estimated figures—**always verify real-time market data**.

                **Providing Context:**
                  - Generate a **plain-text 'Context Summary'** with:
                    - The **most relevant** and **latest** price and volume data.
                    - **Clear insights** on notable market trends or significant movements.
                    - Any additional key details **only if they are directly relevant** to the mentioned asset.
                  - **Limit responses to one or two coins at most**, focusing on **the most relevant or active asset**.
                  - If the user's tweet is **generic** (e.g., "Thoughts?" or "What’s happening?"), provide a **concise, up-to-date market overview**.

                **Accuracy & Formatting:**
                  - **Always confirm** that price and volume are **100% accurate** by using the **CoinGecko tool**.
                  - **No speculation**—stick strictly to verified market data.
                  - Responses must be **clear, concise, and without unnecessary information**.
                  - No markdown, quotes, special formatting, or links.
                  - No unrelated content—**only focus on real-time market data**.

                **Non-Negotiable Rules:**
                  - **Price and volume must always come from the CoinGecko tool.**
                  - **Never assume or estimate values—always verify them in real time.**
                  - **If no price or volume data is available, explicitly state that rather than providing unverified numbers.**
                Make sure you always mention the price and the time for the price such that if you are adding price of a coin 
                which is 2 months old specify that and it is a must to give the current price.
                Also, for your reference, the current time is {current_date_time}.
            """).strip()
        ,
        description=("Agent that extracts context from a tweet for comment generation. According to the latest information, "
                    "the agent will provide a concise summary of the tweet's key details."),
        tools=[cg_tool, exa_tool,Crawl4aiTools(max_length=30000),UrlExpanderTool(timeout=10)],
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("""\
            {comment_context}
        """)
    )