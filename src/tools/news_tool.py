import os
from typing import Any, Dict, Union, List
from newsdataapi import NewsDataApiClient
from phi.tools import Toolkit
from phi.utils.log import logger
from dotenv import load_dotenv
import json

load_dotenv()


class NewsDataApiTool(Toolkit):
    def __init__(self,api_key:str=""):
        super().__init__(name="news_data_api_tool")
        self.api = NewsDataApiClient(apikey=api_key)
        self.register(self.get_latest_news)

    @staticmethod
    def clean_article(article: Dict[str, Any]) -> Dict[str, Any]:
        keys = ["title", "description", "country", "category"]
        cleaned = {}
        for key in keys:
            if key in article and article[key] is not None:
                if isinstance(article[key], str):
                    value = article[key].strip()
                    if value.upper().startswith("ONLY AVAILABLE"):
                        continue
                    cleaned[key] = value
                elif isinstance(article[key], list):
                    filtered = [
                        item.strip() for item in article[key]
                        if isinstance(item, str) and not item.strip().upper().startswith("ONLY AVAILABLE")
                    ]
                    if filtered:
                        cleaned[key] = filtered
                else:
                    cleaned[key] = article[key]
        return cleaned

    def get_latest_news(
        self,
        q: str = "",
        language: str = "en",
        country: str = "",
        category: str = ""
    ) -> str:
        """
        Retrieve the latest news articles based on a query.
        """
        try:
            result = self.api.latest_api(q=q, language=language, country=country, category=category)
            if isinstance(result, dict) and "results" in result:
                cleaned_results = [self.clean_article(article) for article in result["results"]]
                return json.dumps({"results": cleaned_results}, indent=2)
            else:
                return ""
        except Exception as e:
            logger.error(f"Error in get_latest_news: {e}")
            return json.dumps({"error": str(e)}, indent=2)