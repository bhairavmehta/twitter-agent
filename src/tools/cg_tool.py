import os
from typing import Optional, Union, List
import json
from pycoingecko import CoinGeckoAPI
from phi.tools import Toolkit
from phi.utils.log import logger
from dotenv import load_dotenv

load_dotenv()


class PhiCoinGeckoTool(Toolkit):
    def __init__(self, api_key: Optional[str] = None, demo_api_key: Optional[str] = None):
        """
        Initializes the CoinGecko tool for Phi Data.

        Args:
            api_key (Optional[str]): Your CoinGecko Pro API key.
            demo_api_key (Optional[str]): Your CoinGecko demo API key.
            If neither is provided, the free public API is used.
        """
        super().__init__(name="coingecko_tool")
        if api_key:
            self.cg = CoinGeckoAPI(api_key=api_key)
        elif demo_api_key:
            self.cg = CoinGeckoAPI(demo_api_key=demo_api_key)
        else:
            self.cg = CoinGeckoAPI()  # Use free public API

        # Register tool functions for Phi Data
        self.register(self.get_price)
        self.register(self.get_supported_vs_currencies)
        #self.register(self.get_coins_markets)
        self.register(self.get_trending_coins)
        self.register(self.get_global_data)

    def get_price(
        self,
        ids: Union[str, List[str]],
        vs_currencies: Union[str, List[str]],
        include_market_cap: Union[bool, str] = False,
        include_24hr_vol: Union[bool, str] = False,
        include_24hr_change: Union[bool, str] = False,
        include_last_updated_at: Union[bool, str] = False,
        precision: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Get the current price for specified coin IDs in target currencies.
        Returns:
            str: The API response as a string.
        """
        if isinstance(ids, str):
            ids = ids.replace(" ", "")
        if isinstance(vs_currencies, str):
            vs_currencies = vs_currencies.replace(" ", "")
        result = self.cg.get_price(
            ids=ids,
            vs_currencies=vs_currencies,
            include_market_cap=include_market_cap,
            include_24hr_vol=include_24hr_vol,
            include_24hr_change=include_24hr_change,
            include_last_updated_at=include_last_updated_at,
            precision=precision,
            **kwargs
        )
        logger.info(f"get_price result: {result}")
        # Return as string
        return json.dumps(result, indent=2)

    def get_supported_vs_currencies(self, **kwargs) -> str:
        """
        Get the list of supported target currencies.
        Returns:
            str: The API response as a string.
        """
        result = self.cg.get_supported_vs_currencies(**kwargs)
        logger.info(f"get_supported_vs_currencies result: {result}")
        return json.dumps(result, indent=2)

    def get_coins_markets(self, vs_currency: str, **kwargs) -> str:
        """
        Retrieve market data for coins.
        Returns:
            str: The API response with market data as a string.
        """
        result = self.cg.get_coins_markets(vs_currency=vs_currency, **kwargs)
        logger.info(f"get_coins_markets result: {result}")
        return json.dumps(result, indent=2)

    def get_trending_coins(self, **kwargs) -> str:
        """
        Retrieve trending search results for cryptocurrency.
        Returns:
            str: The API response with trending search data as a string.
        """
        result = self.cg.get_search_trending(**kwargs)
        return self.format_trending_coins_response(result)

    def get_global_data(self, **kwargs) -> str:
        """
        Retrieve global cryptocurrency data including active cryptocurrencies, markets,
        total crypto market cap, etc.
        Returns:
            str: The API response from the /global endpoint as a JSON-formatted string.
        """
        try:
            result = self.cg.get_global(**kwargs)
            logger.info(f"get_global_data result: {result}")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error in get_global_data: {e}")
            return json.dumps({"error": str(e)}, indent=2)


    @staticmethod
    def format_trending_coins_response(trending_json: dict) -> str:
        """
        Process the trending coins JSON response and returns a formatted string
        with selected fields for each coin.

        Expected fields:
          - id, name, symbol, price (BTC & USD), score,
          - 24h change (USD & BTC), market cap, and market cap rank.

        Args:
            trending_json (dict): The JSON response from get_search_trending.

        Returns:
            str: A formatted string with coin details.
        """
        coins = trending_json.get("coins", [])
        formatted_coins = []

        for coin_obj in coins:
            item = coin_obj.get("item", {})
            coin_id = item.get("id", "N/A")
            name = item.get("name", "N/A")
            symbol = item.get("symbol", "N/A")
            price_btc = item.get("price_btc", "N/A")
            score = item.get("score", "N/A")
            market_cap_rank = item.get("market_cap_rank", "N/A")

            data = item.get("data", {})
            price_usd = data.get("price", "N/A")
            market_cap = data.get("market_cap", "N/A")
            change_data = data.get("price_change_percentage_24h", {})
            change_usd = change_data.get("usd", "N/A")
            change_btc = change_data.get("btc", "N/A")

            coin_str = (
                f"Coin: {name} ({symbol})\n"
                f"  ID: {coin_id} | Market Cap Rank: {market_cap_rank}\n"
                f"  Price: ${price_usd} (BTC: {price_btc}) | Score: {score}\n"
                f"  24h Change: USD: {change_usd}% | BTC: {change_btc}%\n"
                f"  Market Cap: {market_cap}\n"
            )
            formatted_coins.append(coin_str)
        response = "\n".join(formatted_coins)
        logger.info(f"get_trending_coins : {response}")
        return response

