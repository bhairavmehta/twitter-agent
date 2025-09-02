import logging
import os
from textwrap import dedent
from phi.agent import Agent
from tools.cg_tool import PhiCoinGeckoTool
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv
from datetime import datetime, timezone
load_dotenv()

def create_trending_crypto_agent(model:str = "openai/gpt-4o-mini", demo_api_key:str = None,api_key: str= None,
                                 markdown: bool= False,show_tool_calls: bool= False) -> Agent:
    # Initialize the CoinGecko tool with its API key (hardcoded demo key in this example)
    if api_key:
        cg_tool = PhiCoinGeckoTool(api_key=api_key)
    elif demo_api_key:
        cg_tool = PhiCoinGeckoTool(demo_api_key=demo_api_key)
    else :
        logging.log(logging.ERROR, "No API key provided taking api key from .env")
        if os.getenv("COINGECKO_API_KEY"):
            cg_tool = PhiCoinGeckoTool(api_key=os.getenv("COINGECKO_API_KEY"))
        elif os.getenv("COINGECKO_DEMO_API_KEY"):
            cg_tool = PhiCoinGeckoTool(demo_api_key=os.getenv("COINGECKO_DEMO_API_KEY"))
        else :
            logging.log(logging.ERROR, "No API key available in .env")
            cg_tool = PhiCoinGeckoTool()
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return Agent(
        model= OpenRouter(id=model,api_key=os.getenv("OPENROUTER_API_KEY")),
        tools=[cg_tool],
        instructions=[
            (
                "You are a crypto market data aggregator specialized in leveraging the CoinGecko API to retrieve and analyze trending cryptocurrency data. Your task is as follows:\n\n"
                "1. Use the CoinGecko tool's get_trending_coins function to retrieve a comprehensive dataset of trending cryptocurrencies. Ensure that the data includes each coin’s name, market cap, current price, trading volume, and percentage changes over various timeframes.\n\n"
                "2. Analyze the retrieved dataset to determine the top trending coin based on market capitalization. Carefully evaluate the metrics to ensure that the coin with the highest market cap is correctly identified.\n\n"
                "3. Once the top trending coin is identified, invoke the CoinGecko tool’s additional functions (such as get_price) to fetch detailed information for that coin. This detailed information must include the current price, market cap, 24-hour trading volume, 24-hour percentage change, and 7-day percentage change.\n\n"
                "4. Additionally, perform a technical review of the coin's performance by extracting key technical insights and identifying any notable trends or anomalies in the data. Provide actionable key takeaways regarding the coin's performance.\n\n"
                "5. Finally, compile all of the gathered information and insights into a concise, plain text summary labeled 'Trending Coin Summary'. This summary should clearly present the detailed metrics along with the technical insights and key takeaways, ensuring that no required information is missing."
                "Also you will be given some information you will to get the latest data for the same and give that as well"
                "Try not to always take btc and eth as trending coins, try to find other coins that are trending"
                f"Also for your reference current time is {current_date_time}"
            )
        ],
        description="Agent that gathers trending crypto market information and adheres to the expected output format.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("""\
            Top Trending Coin: {trending_coin}
            Price: ${price} | Market Cap: ${market_cap} | 24h Vol: ${volume_24h}
            24h Change: {change_24h}% | 7d Change: {change_7d}%
            Insights: {trending_insights}
            Key Takeaways:
            - {takeaway_1}
            - {takeaway_2}
            - {takeaway_3}
        """),
    )
