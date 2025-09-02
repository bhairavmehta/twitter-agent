from datetime import datetime, timedelta, timezone
import logging
from typing import List
import tweepy
import time
import json
from agents.filter_agent import create_crypto_filter_agent
from content_generator import ContentGenerator
import tweepy
logging.basicConfig(level=logging.INFO)


class TweetEngager:
    def __init__(self, personality,
                 generator: ContentGenerator,
                 tweepy_client: tweepy.client,
                 engaged_history: List = None,
                 min_engagement_count: int = 100,
                 optimal_followers: int = 1000
                 ):
        """
        Initialize the TweetEngager class. This class will search for tweets related to
        buzzwords and engage (reply) on those tweets if they meet the engagement criteria.

        Parameters:
          personality: An object containing configuration including buzzwords.
          generator: An agent for generating tweet comments.
          tweepy_client: A configured tweepy client.
          engaged_history: A list of tweet IDs that have already been engaged with.
          min_engagement_count: The minimum total engagement required for a tweet to be considered.
          optimal_followers: A target followers count for the tweet's author to be considered influential.
        """
        self.personality = personality
        self.generator = generator
        self.tweepy_client = tweepy_client
        self.engaged_history = engaged_history or []
        self.buzzwords = [bw.lower().strip() for bw in personality.config.buzzwords]

        self.min_engagement_count = min_engagement_count
        self.optimal_followers = optimal_followers
        self.reply_filter_agent = create_crypto_filter_agent()

    def _construct_query(self) -> str:
        """
        Construct a search query based on buzzwords.
        Combines buzzwords with an OR operator and excludes retweets.
        query will be improved in the later versions
        """
        query = " OR ".join(self.buzzwords) + " -is:retweet lang:en"
        return query

    def search_tweets(self, lookback_minutes: int = 15, max_results: int = 10) -> List[tweepy.Tweet]:
        """
        Search for recent tweets that match the buzzwords.
        """
        start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        query = self._construct_query()
        try:
            response = self.tweepy_client.search_recent_tweets(
                query=query,
                start_time=start_time.isoformat(),
                tweet_fields=['created_at', 'text', 'author_id'],
                max_results=max_results
            )
            tweets = response.data or []
            logging.info(f"Found {len(tweets)} tweets matching buzzwords.")
            return tweets
        except tweepy.TweepyException as e:
            logging.error(f"Error during tweet search: {e}")
            return []

    def has_already_engaged(self, tweet: tweepy.Tweet) -> bool:
        """
        Check if a tweet has already been engaged with, based on its ID.
        """
        return tweet.id in self.engaged_history


    def is_engagement_sufficient(self, tweet: tweepy.Tweet) -> bool:
        """
        Determine if a tweet has enough engagement to justify a reply.
        The check is based on:
         - Total engagement count (likes, retweets, replies, quotes)
         - Engagement ratio relative to the author's followers
        """
        try:
            total_engagement = tweet.likes

            user_response = self.tweepy_client.get_user(
                id=tweet.author_id,
                user_fields=['public_metrics']
            )
            if user_response.data and hasattr(user_response.data, "public_metrics"):
                followers_count = user_response.data.public_metrics.get("followers_count", 0)
            else:
                followers_count = 0

            logging.info(
                f"Tweet {tweet.id} metrics: total_engagement={total_engagement}, "
            )
            prompt_input = (
                f"Tweet Text: {tweet.text}\n"
                f"Optimal Total Engagement: {self.min_engagement_count}\n"
                f"Current Total Engagement: {total_engagement}\n"
                f"Optimal Followers: {self.optimal_followers}\n"
                f"Author Followers: {followers_count}\n"
                "Context: This tweet is related to cryptocurrencies, 365x.ai, and stocks. "
                "Evaluate if the tweet is worth replying to based on its substance, engagement metrics, "
                "and the influence of the author."
            )


            return self.generator.filter_comment(comment_context=prompt_input)

        except Exception as e:
            logging.error(f"Error checking engagement for tweet {tweet.id}: {e}")
            return False


    def engage_with_tweets(self, lookback_minutes: int = 15, max_results: int = 10) -> None:
        """
        Search for tweets that match buzzwords, then generate and post a reply for each
        tweet that hasn't already been engaged with.
        """
        tweets = self.search_tweets(lookback_minutes, max_results)
        for tweet in tweets:
            logging.info(f"checking tweet : {tweet}")
            if self.has_already_engaged(tweet):
                logging.info(f"Already engaged with tweet {tweet.id}; skipping.")
                continue

            if not self.is_engagement_sufficient(tweet):
                logging.info(f"Tweet {tweet.id} does not meet engagement criteria; skipping.")
                continue

            time.sleep(30)
            try:
                # generating a contextual response using the tweet's text as input.
                logging.info(f"checking tweet : {tweet}")
                response_text = self.generator.generate_comment(tweet.text,self_tweet=False)

                if len(response_text) < 280:
                    reply = self.tweepy_client.create_tweet(
                        text=response_text,
                        in_reply_to_tweet_id=tweet.id
                    )
                    logging.info(f"Replied to tweet {tweet.id} with reply {reply.data.get('id')}.")

                else:
                    logging.info(f"Reply to tweet {tweet.id} was too long response_text.")

                #This will record that this tweet has been engaged with.
                self.engaged_history.append(tweet.id)
            except tweepy.TweepyException as e:
                logging.error(f"Error engaging with tweet {tweet.id}: {e}")
