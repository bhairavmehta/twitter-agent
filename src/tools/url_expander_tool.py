from typing import Optional
import requests
from phi.tools import Toolkit
from phi.utils.log import logger
from dotenv import load_dotenv

load_dotenv()

class UrlExpanderTool(Toolkit):
    def __init__(self, timeout: Optional[int] = 5):
        """
        Initializes the URL Expander Tool

        Args:
            timeout (Optional[int]): The timeout in seconds for the HTTP HEAD request. Defaults to 5.
        """
        super().__init__(name="url_expander_tool")
        self.timeout = timeout
        self.register(self.expand_url)

    def expand_url(self, url: str) -> str:
        """
        Expands a shortened URL to its final destination by performing an HTTP HEAD request
        that follows redirects.

        Args:
            url (str): The URL to expand.

        Returns:
            str: The final destination URL if successful; otherwise, returns the original URL.
        """
        try:
            response = requests.head(url, allow_redirects=True, timeout=self.timeout)
            final_url = response.url
            logger.info(f"expand_url: {url} expanded to {final_url}")
            return final_url
        except Exception as e:
            logger.error(f"Error expanding URL {url}: {e}")
            return url
