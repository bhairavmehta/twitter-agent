import http.client
import json
from dataclasses import dataclass
from datetime import datetime,timezone


@dataclass
class Rapid_User:
    id: str
    name: str
    followers_count: int

@dataclass
class Rapid_Tweet:
    id: str
    text: str
    likes: int
    username: str
    conversation_id: str
    created_at: datetime
    retweet: int

@dataclass
class Rapid_Comment:
    parent_id:str
    parent_text:str
    comment_id: str
    user_id: str
    username: str
    text: str
    created_at: datetime
    likes: int
    replies: int


class RapidTweepy:
    def __init__(self, api_key: str):
        self.api_host = "twitter241.p.rapidapi.com"
        self.api_key = api_key

    @staticmethod
    def _format_tweet_timestamp(stamp_str: str) -> datetime:
        """Convert Twitter's timestamp string to a datetime object."""
        try:
            return datetime.strptime(stamp_str, "%a %b %d %H:%M:%S %z %Y")
        except ValueError:
            return datetime.now(timezone.utc)

    def get_user_tweets(self, user_id: str, count: int = 5):
        """Fetch recent tweets from a specific user."""
        conn = http.client.HTTPSConnection(self.api_host)
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.api_host
        }
        endpoint = f"/user-tweets?user={user_id}&count={count}"
        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()
        return self._parse_tweets(json.loads(data.decode("utf-8")))

    def get_post_comments(self, tweet_id: str,tweet_text:str, count: int = 5, ranking_mode: str = "Relevance"):
        """Fetch comments on a specific tweet."""
        conn = http.client.HTTPSConnection(self.api_host)
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.api_host
        }
        endpoint = f"/comments?pid={tweet_id}&count={count}&rankingMode={ranking_mode}"
        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()
        return self._parse_comments(json.loads(data.decode("utf-8")),tweet_id,tweet_text)

    def get_user_info(self, username: str):
        """Fetch user info based on username."""
        conn = http.client.HTTPSConnection(self.api_host)
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.api_host
        }
        endpoint = f"/user?username={username}"
        conn.request("GET", endpoint, headers=headers)
        res = conn.getresponse()
        data = res.read()
        return self._parse_user_info(json.loads(data.decode("utf-8")))

    def _parse_tweets(self, response_data):
        """Extracts tweet details from API response."""
        tweets = []

        if "result" in response_data:
            instructions = response_data["result"].get("timeline", {}).get("instructions", [])

            for instruction in instructions:
                if isinstance(instruction, dict):
                    entries = instruction.get("entries", [])
                    for entry in entries:
                        if entry.get("entryId", "").startswith("tweet-"):
                            tweet_data = entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get(
                                "result", {})

                            tweet_info = Rapid_Tweet(
                                id=tweet_data.get("rest_id", ""),
                                text=tweet_data.get("legacy", {}).get("full_text", ""),
                                likes=tweet_data.get("legacy", {}).get("favorite_count", 0),
                                username=tweet_data.get("core", {}).get("user_results", {}).get("result", {}).get(
                                    "legacy", {}).get("screen_name", ""),
                                conversation_id=tweet_data.get("legacy", {}).get("conversation_id_str", ""),
                                created_at=self._format_tweet_timestamp(tweet_data.get("legacy", {})
                                                                        .get("created_at", "")),
                                retweet=tweet_data.get("legacy", {}).get("retweet_count", 0)
                            )
                            tweets.append(tweet_info)
        return tweets

    @staticmethod
    def _parse_user_info(response_data):
        """Extracts user ID, name, and follower count from API response."""
        if "result" in response_data:
            user_data = response_data["result"].get("data", {}).get("user", {}).get("result", {})
            legacy_data = user_data.get("legacy", {})
            return Rapid_User(
                id=user_data.get("rest_id", ""),
                name=legacy_data.get("name", ""),
                followers_count=legacy_data.get("followers_count", 0)
            )
        return Rapid_User(id="", name="", followers_count=0)

    def _parse_comments(self, response_data,parent_id:str,parent_text):
        """Extracts structured comment details from API response."""
        parsed_comments = []
        if "result" in response_data:
            instructions = response_data["result"].get("instructions", [])
            for instruction in instructions:
                if isinstance(instruction, dict):
                    entries = instruction.get("entries", [])
                    for entry in entries:
                        if entry.get("content",{}).get("__typename") == 'TimelineTimelineModule':
                            #print(entry.get("content",{}).keys())
                            #print(entry.get("content",{}).get('items')[0].get('item').get('itemContent').get('tweet_results').keys())
                            comment_data = (entry.get("content",{}).get('items')[0].get('item')
                                            .get('itemContent').get('tweet_results').get('result'))
                            parsed_comments.append(Rapid_Comment(
                                parent_id=parent_id,
                                parent_text=parent_text,
                                comment_id=comment_data.get("rest_id", ""),
                                user_id=comment_data.get("core", {}).get("user_results", {}).get("result", {}).
                                get("rest_id", ""),
                                username=comment_data.get("core", {}).get("user_results", {}).get("result", {})
                                .get("legacy", {}).get("screen_name", ""),
                                text=comment_data.get("legacy", {}).get("full_text", ""),
                                created_at=self._format_tweet_timestamp(comment_data.get("legacy", {})
                                                                        .get("created_at", "")),
                                likes=comment_data.get("legacy", {}).get("favorite_count", 0),
                                replies=comment_data.get("legacy", {}).get("reply_count", 0),
                            ))

        return parsed_comments


