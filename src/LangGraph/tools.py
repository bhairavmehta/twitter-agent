from pycoingecko import CoinGeckoAPI
from crawl4ai import AsyncWebCrawler
from exa_py import Exa
import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any, List, Annotated, Union
from langchain_core.tools import tool
from src import Schedule, ScheduleManager
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class ScheduleTool:
    """
    ScheduleTool is a LangChain-style tool for managing scheduling events.

    This tool allows agents to add new schedule events and retrieve events (all, overdue,
    or future) from an injected ScheduleManager.
    """

    def __init__(self, schedulemanager: ScheduleManager):
        """
        Initialize the ScheduleTool with a ScheduleManager instance.

        Args:
            schedulemanager (ScheduleManager): An instance responsible for managing schedule events.
        """
        self.manager = schedulemanager

    @tool
    def add_schedule(
            self,
            event: Annotated[str, "A brief description or title of the event"],
            post_content: Annotated[str, "Additional details or content for the event"] = ""
    ) -> Annotated[str, "Confirmation message with the scheduled time in ISO format"]:
        """
        Add a new schedule event to the system.

        This method creates a new Schedule object with the provided event description
        and post content. The new schedule is added to the ScheduleManager.

        Usage:
          Provide an event description and Post content as strings.

        Returns:
          A confirmation message
        """
        scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=30)
        new_schedule = Schedule(
            scheduled_time=scheduled_time,
            current_events=event,
            content=post_content
        )
        self.manager.add_schedule(new_schedule)
        logger.info(f"Added schedule: {new_schedule}")
        return f"Schedule added."

    @tool
    def get_all_events(self) -> Annotated[List[dict], "A list of all schedule entries as dictionaries"]:
        """
        Retrieve all schedule events from the ScheduleManager.

        Usage:
          No input arguments.

        Returns:
          A list of dictionaries, each representing a scheduled event.
        """
        events = self.manager.get_all_events()
        return [event.model_dump() for event in events]

    @tool
    def get_overdue_events(self) -> Annotated[List[dict], "A list of overdue schedule entries as dictionaries"]:
        """
        Retrieve all overdue schedule events.

        Overdue events are those whose scheduled time is in the past.

        Usage:
          No input arguments.

        Returns:
          A list of dictionaries, each representing an overdue schedule event.
        """
        events = self.manager.get_overdue_events()
        return [event.model_dump() for event in events]

    @tool
    def get_future_events(self) -> Annotated[List[dict], "A list of future schedule entries as dictionaries"]:
        """
        Retrieve all future schedule events.

        Future events are those with a scheduled time in the future.

        Usage:
          No input arguments.

        Returns:
          A list of dictionaries, each representing a future schedule event.
        """
        events = self.manager.get_future_events()
        return [event.model_dump() for event in events]

    @tool
    def get_all_events_str(self) -> Annotated[str, "A human-readable string of all schedule events"]:
        """
        Retrieve all schedule events as a formatted human-readable string.

        This method fetches all schedule events and formats each event's key details,
        including the event description, scheduled time, and content.

        Usage:
          No input arguments.

        Returns:
          A newline-separated string of all schedule events, or a message indicating that no events are scheduled.
        """
        events = self.manager.get_all_events()
        if not events:
            return "No future events scheduled."
        lines = []
        for event in events:
            lines.append(
                f"Event: {event.current_events} | Scheduled: {event.scheduled_time.isoformat()} | Content:"
                f" {event.content}"
            )
        return "\n".join(lines)


@tool
def exa_search_tweet(query: str) -> str:
    """
    Uses the Exa API to search for content in the "tweet" category.

    Args:
        query (str): The search query.

    Returns:
        str: The search result as a string.
    """
    # Initialize the Exa client with the API key from the environment,
    # or use the provided default demo key.
    exa = Exa(api_key=os.getenv("EXA_API_KEY", "e971cf80-4fdf-4796-a349-c2da53a8ffa9"))
    try:
        result = exa.search(query, category="tweet")
        logger.info(f"exa_search_tweet result: {result}")
        return str(result)
    except Exception as e:
        logger.error(f"Error in exa_search_tweet: {e}")
        return f"Error: {e}"


@tool
def exa_search_news(query: str) -> str:
    """
    Uses the Exa API to search for content in the "news articles" category.

    Args:
        query (str): The search query.

    Returns:
        str: The search result as a string.
    """
    # Initialize the Exa client with the API key from the environment,
    # or use the provided default demo key.
    exa = Exa(api_key=os.getenv("EXA_API_KEY", "e971cf80-4fdf-4796-a349-c2da53a8ffa9"))
    try:
        result = exa.search(query, category="news")
        logger.info(f"exa_search_tweet result: {result}")
        return str(result)
    except Exception as e:
        logger.error(f"Error in exa_search_tweet: {e}")
        return f"Error: {e}"


@tool
def coingecko_get_price(
        ids: Annotated[Union[str, List[str]], "Comma-separated coin IDs or a list of coin IDs"],
        **kwargs: Any
) -> Annotated[str, "The API response with price data as a string"]:
    """
    Retrieves the current price for specified coin IDs in target currencies.
    """
    api_key = os.getenv("COINGECKO_API_KEY")
    demo_api_key = os.getenv("COINGECKO_DEMO_API_KEY")
    if api_key:
        cg = CoinGeckoAPI(api_key=api_key)
    elif demo_api_key:
        cg = CoinGeckoAPI(demo_api_key=demo_api_key)
    else:
        cg = CoinGeckoAPI()
    if isinstance(ids, str):
        ids = ids.replace(" ", "")
    try:
        result = cg.get_price(
            ids=ids,
            **kwargs
        )
        logger.info(f"coingecko_get_price result: {result}")
        return str(result)
    except Exception as e:
        logger.error(f"Error in coingecko_get_price: {e}")
        return f"Error: {e}"


@tool
def coingecko_get_coins_markets(
        vs_currency: Annotated[str, "Target currency (e.g., 'usd')"],
        **kwargs: Any
) -> Annotated[str, "The market data for coins as a string"]:
    """
    Retrieves market data for coins for the specified target currency.
    """
    api_key = os.getenv("COINGECKO_API_KEY")
    demo_api_key = os.getenv("COINGECKO_DEMO_API_KEY")
    if api_key:
        cg = CoinGeckoAPI(api_key=api_key)
    elif demo_api_key:
        cg = CoinGeckoAPI(demo_api_key=demo_api_key)
    else:
        cg = CoinGeckoAPI()
    try:
        result = cg.get_coins_markets(vs_currency=vs_currency, **kwargs)
        logger.info(f"coingecko_get_coins_markets result: {result}")
        return str(result)
    except Exception as e:
        logger.error(f"Error in coingecko_get_coins_markets: {e}")
        return f"Error: {e}"


@tool
def coingecko_get_search_trending(
        **kwargs: Any
) -> Annotated[str, "The trending search results from CoinGecko as a string"]:
    """
    Retrieves trending search results from CoinGecko.
    """
    api_key = os.getenv("COINGECKO_API_KEY")
    demo_api_key = os.getenv("COINGECKO_DEMO_API_KEY")
    if api_key:
        cg = CoinGeckoAPI(api_key=api_key)
    elif demo_api_key:
        cg = CoinGeckoAPI(demo_api_key=demo_api_key)
    else:
        cg = CoinGeckoAPI()
    try:
        result = cg.get_search_trending(**kwargs)
        logger.info(f"coingecko_get_search_trending result: {result}")
        return str(result)
    except Exception as e:
        logger.error(f"Error in coingecko_get_search_trending: {e}")
        return f"Error: {e}"


@tool
def coingecko_get_global_data(
        **kwargs: Any
) -> Annotated[Any, "The global cryptocurrency data including active cryptocurrencies, markets, and total market cap"]:
    """
    Retrieves global cryptocurrency data including active cryptocurrencies, markets, and total market cap.
    """
    api_key = os.getenv("COINGECKO_API_KEY")
    demo_api_key = os.getenv("COINGECKO_DEMO_API_KEY")
    if api_key:
        cg = CoinGeckoAPI(api_key=api_key)
    elif demo_api_key:
        cg = CoinGeckoAPI(demo_api_key=demo_api_key)
    else:
        cg = CoinGeckoAPI()
    try:
        result = cg.get_global(**kwargs)
        logger.info(f"coingecko_get_global_data result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in coingecko_get_global_data: {e}")
        return {"error": str(e)}


@tool
def crawl4ai_scraper(
    url: Annotated[str, "The URL to crawl and extract content from"]
) -> Annotated[str, "The scraped content from the URL in markdown format"]:
    """
    Uses Crawl4AI's AsyncWebCrawler to scrape the provided URL and return the extracted markdown content.
    """
    async def async_scrape(urls: str) -> str:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=urls)
            return result.markdown

    try:
        scraped_content = asyncio.run(async_scrape(url))
        logger.info(f"crawl4ai_scraper scraped content length: {len(scraped_content)}")
        return scraped_content
    except Exception as e:
        logger.error(f"Error in crawl4ai_scraper: {e}")
        return f"Error: {e}"
