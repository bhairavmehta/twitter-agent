import logging
import os
from textwrap import dedent
from phi.agent import Agent
from phi.tools.newspaper_tools import NewspaperTools
from phi.tools.exa import ExaTools
from phi.tools.googlesearch import GoogleSearch
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv
from datetime import datetime, timezone
load_dotenv()
def create_deep_coin_info_agent(model:str ="openai/gpt-4o-mini", exa_api_key: str = "",
                                markdown: bool= False,show_tool_calls: bool= False) -> Agent:
    if exa_api_key:
        exa_tool = ExaTools(api_key=exa_api_key)
    else:
        logging.error("No ExaTools API key provided; attempting to load from .env")
        from dotenv import load_dotenv
        load_dotenv()
        if os.getenv("EXA_API_KEY"):
            exa_tool = ExaTools(api_key=os.getenv("EXA_API_KEY"))
        else:
            logging.error("No EXA_API_KEY found in .env; using default ExaTools instance")
            exa_tool = ExaTools(api_key="")
    newspaper_tool = NewspaperTools()
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return Agent(
        model=OpenRouter(id=model,api_key=os.getenv("OPENROUTER_API_KEY")),
        tools=[exa_tool, newspaper_tool,GoogleSearch(fixed_language="en")],
        instructions=[
            (
                "You are a deep coin info agent tasked with gathering **comprehensive and factually accurate** information about a specified cryptocurrency.\n\n"
                "1. **Primary Source - EXA Tool**: Always start by using the **EXA tool** to fetch the latest updates, token ID, "
                " smart contract details, and any other relevant metadata for the coin.\n\n"
                "2. **Verifying Market Data - CoinGecko**: Once you have obtained the token ID and other identifiers "
                " from EXA, use the **CoinGecko tool** to retrieve real-time market data, including price, volume, and any"
                " other relevant stats. Always **double-check** key figures like price fluctuations, total volume, and "
                "market trends before presenting them.\n\n"
                "Make sure to get the recent price for the coin using google search tool make queries like "
                "doge coin price site:coinbase.com"
                "3. **Final Report**: Combine all verified data into a concise, tweet-style update that includes:\n"
                "   - **Market metrics** (price, volume, trends)\n"
                "   - **Recent news & highlights**\n"
                "   - **Technical insights & notable events**\n"
                "   - **Any additional engaging facts**\n\n"
                "**Key Rule**: **Do not provide any unverified information.** Always cross-check numerical data before using it. "
                f"The current reference time is **{current_date_time}**, and you should prioritize real-time accuracy. "
                "If needed, you may check additional relevant links, but fact-checking is mandatory before presenting "
                "any information Also make sure that you are giving information for multiple coins and not just for one coin"
                "Also try to look for more niche and not famous coins too."
            )
        ],
        description="Agent that consolidates detailed coin information and updates and also verifies the info given",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("""\
            Crypto Info:
            Trending Updates: {trending_updates}
            Extra Information:
            - Info 1: {source1_info}
            - Info 2: {source2_info}
            - Info 3: {source3_info}
        """)

    )
