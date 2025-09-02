import logging
import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv
from tools.cg_tool import PhiCoinGeckoTool
from phi.tools.exa import ExaTools
from phi.tools.crawl4ai_tools import Crawl4aiTools
from tools.url_expander_tool import UrlExpanderTool
from datetime import datetime, timezone
load_dotenv()

def create_reply_context_agent(model: str = "openai/gpt-4o",cg_demo_api_key:str =None,cg_api_key: str=None,
                                exa_api_key: str = None, markdown: bool = False, show_tool_calls: bool = False
                               ,api_key:str = None) -> Agent:
    """
    Creates an agent that takes the text of a social media post and extracts context and relevant details
    to help in composing a reply. It should:
      - Read the content of the post.
      - Identify topics, keywords, or any data (such as currency trends, market data, etc.)
        that are relevant to the post.
      - Use any available tools to fetch additional details if necessary.
      - Return a concise, plain text summary of the extracted context.
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
        # You can add additional research tools here if needed in the tools list.
        instructions=dedent(f"""
            You are a reply context agent. Your task is to analyze the provided tweet, extract key details,
            and provide accurate, concise context. Your primary focus is on cryptocurrencies and stocks.
            If the Tweet is just a normal tweet like whats up or hello or hi then give a market update
            Market data should always be real time and should be verified from the coingecko tool.
            **Handling URLs:**
              - If a URL is present (starting with 'http://' or 'https://'), treat it as a source of additional context.
              - If the URL is shortened (e.g., 'https://t.co/...'), use the **URL Expander** tool to get the final destination.
              - If further details are required from the expanded URL, use the **Crawl4AI** tool to extract relevant information.
              - Never confuse URLs (shortened or expanded) with coin names or IDs.

            **Identifying Cryptocurrencies & Stocks:**
              - Extract any cryptocurrency or stock mentions from the tweet.
              - For **cryptocurrencies**, use the **Exa tool** to retrieve their coin ID if it's not explicitly mentioned.
              - If a cryptocurrency is referenced in a generic way (e.g., "Bitcoin is pumping"), 
              use the **Exa tool** to check for related news.
              - To get **real-time stats** (price, volume, market cap, etc.), first obtain the coin ID from Exa, 
              then query the **CoinGecko tool** for detailed data.

            **Providing Context:**
              - After gathering all relevant data, generate a **plain-text 'Context Summary'** with:
                - The latest market stats (accurate prices, percentage changes, etc.).
                - Notable trends, movements, and any critical news about the mentioned assets.
                - Financial terms clearly explained and unrelated mentions ignored.
                - If the user’s tweet is **generic** (e.g., "Hey" or "Hello"), provide a **market update** with the 
                    latest significant movements and factually correct information.

            **Accuracy & Formatting:**
              - Ensure **all figures** (prices, changes, volume, etc.) are **precise and up to date**.
              - The final response must be clear, concise, and **without unnecessary speculation**.
              - Do not include markdown, quotes, or special formatting.
              - No extraneous or unrelated information.
              
            Make sure you always mention the price and the time for the price such that if you are adding price of a coin 
            which is 2 months old specify that and it is a must to give the current price.
            Also, for your reference, the current time is {current_date_time} ensure that the data given is upto date.
        """).strip(),
        description="Agent that extracts context and relevant data from a post dont give any negative comments.",
        tools=[cg_tool,exa_tool,Crawl4aiTools(max_length=None),UrlExpanderTool(timeout=10)],
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("""{context_summary}""")
    )
