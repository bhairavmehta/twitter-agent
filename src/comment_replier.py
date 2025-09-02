from datetime import datetime, timedelta, timezone
import logging
import tweepy
import time
import pandas as pd
import os
from typing import List, Dict, Optional, Tuple
from rapid_tweepy import RapidTweepy, Rapid_Comment
from content_generator import ContentGenerator


class CommentReplier:
    """
    CommentReplier identifies and engages with high-engagement comments on competitor
    and key people's tweets.

    This class monitors tweets from designated accounts, identifies comments with
    significant engagement, and replies to those comments if they meet criteria.
    """

    def __init__(
            self,
            personality,
            generator: ContentGenerator,
            tweepy_client: tweepy.Client,
            rapid_client: RapidTweepy,
            competitor_csv: str = None,
            key_people_csv: str = None,
            engaged_history: List[str] = None,
            min_comment_likes: int = 5,
            max_comments_per_tweet: int = 3
    ):
        """
        Initialize the CommentReplier.

        Args:
            personality: Personality configuration with buzzwords.
            generator: Content generator for producing replies.
            tweepy_client: Authenticated Tweepy client.
            competitor_csv: Path to CSV file with competitor Twitter handles.
            key_people_csv: Path to CSV file with key people Twitter handles.
            engaged_history: List of comment IDs already engaged with.
            min_comment_likes: Minimum likes for a comment to be considered significant.
            max_comments_per_tweet: Maximum number of comments to engage with per tweet.
        """
        self.personality = personality
        self.generator = generator
        self.tweepy_client = tweepy_client
        self.competitor_csv = competitor_csv
        self.rapid_client = rapid_client
        self.key_people_csv = key_people_csv
        self.engaged_history = engaged_history or []
        self.min_comment_likes = min_comment_likes
        self.max_comments_per_tweet = max_comments_per_tweet

        # Cache for user IDs
        self.user_cache = {}

        # Cache for recent tweets
        self.tweets_cache = {
            'last_fetched': datetime.now(timezone.utc) - timedelta(days=1),
            'tweets': [],
            'params': {}
        }

        # Load competitor and key people accounts
        self.target_accounts = self._load_target_accounts()

        logging.info(f"CommentReplier initialized with {len(self.target_accounts)} target accounts")

    def _load_target_accounts(self) -> Dict[str, Dict]:
        """
        Load target accounts from the competitor and key people CSV files.

        Returns:
            Dict[str, Dict]: Dictionary mapping Twitter handle to account info.
        """

        accounts = {}

        # Load competitor accounts
        if self.competitor_csv and os.path.exists(self.competitor_csv):
            try:
                df = pd.read_csv(self.competitor_csv, dtype={'id': str})
                for _, row in df.iterrows():
                    if "Twitter handle" in df.columns and not pd.isna(row["Twitter handle"]):
                        handle = row["Twitter handle"].strip().lstrip('@')
                        accounts[handle] = {
                            "id": row.get("id", None),
                            "type": "competitor",
                            "name": row.get("Name Company", handle)
                        }
            except Exception as e:
                logging.error(f"Error loading competitor CSV: {e}")

        # Load key people accounts
        if self.key_people_csv and os.path.exists(self.key_people_csv):
            try:
                df = pd.read_csv(self.key_people_csv, dtype={'id': str})
                for _, row in df.iterrows():
                    if "Twitter handle" in df.columns and not pd.isna(row["Twitter handle"]):
                        handle = row["Twitter handle"].strip().lstrip('@')
                        accounts[handle] = {
                            "id": row.get("id", None),
                            "type": "key_person",
                            "name": row.get("Person", handle)
                        }
            except Exception as e:
                logging.error(f"Error loading key people CSV: {e}")

        return accounts

    def _get_user_id(self, username: str) -> Optional[str]:
        """
        Get user ID for a Twitter handle, using cache if available.

        Args:
            username: Twitter handle without @ symbol.

        Returns:
            Optional[str]: User ID if found, None otherwise.
        """
        if username in self.user_cache:
            return self.user_cache[username]

        if username in self.target_accounts and self.target_accounts[username]["id"]:
            self.user_cache[username] = self.target_accounts[username]["id"]
            return self.user_cache[username]

        try:
            user_info = self.rapid_client.get_user_info(username)
            if user_info and user_info.id:
                self.user_cache[username] = user_info.id
                return user_info.id
        except Exception as e:
            logging.error(f"Error fetching user ID for {username}: {e}")

        return None

    def get_recent_tweets(self, lookback_hours: int = 24, max_tweets_per_account: int = 3,
                          min_char = 10) -> List[Dict]:
        """
        Get recent tweets from all target accounts within the lookback period.
        Uses cache if available and requested within 5 hours of the last fetch with the same parameters.

        Args:
            lookback_hours: Hours to look back for tweets.
            max_tweets_per_account: Maximum tweets to fetch per account.

        Returns:
            List[Dict]: List of dictionaries with tweet information.
        """
        current_time = datetime.now(timezone.utc)

        # Check if we have a valid cache that matches the requested parameters
        if len(self.tweets_cache['tweets']):

            # Calculate how old the cache is
            cache_age = current_time - self.tweets_cache['last_fetched']

            if cache_age < timedelta(hours=5):
                logging.info(f"Using cached tweets from {self.tweets_cache['last_fetched']} (age: {cache_age})")
                return self.tweets_cache['tweets']
            else:
                logging.info(f"Cache expired (age: {cache_age}), fetching fresh tweets")
        else:
            logging.info("Cache parameters mismatch or no cache available, fetching fresh tweets")

        # If we reach here, we need to fetch fresh tweets
        start_time = current_time - timedelta(hours=lookback_hours)
        all_tweets = []

        for username, account_info in self.target_accounts.items():
            try:
                user_id = str(account_info.get("id") or self._get_user_id(username))
                if not user_id:
                    logging.warning(f"Could not get user ID for {username}, skipping")
                    continue

                logging.info(f"Fetching tweets for {username} (ID: {str(user_id)})")

                tweets = self.rapid_client.get_user_tweets(user_id, count=max_tweets_per_account)
                if tweets:
                    for tweet in tweets:
                        tweet_created_at = tweet.created_at if hasattr(tweet, "created_at") else start_time
                        if tweet_created_at >= start_time:
                            if len(tweet.text.strip()) < min_char:
                                logging.info(
                                    f"Skipping tweet {tweet.id} by @{tweet.username} because it's under {min_char} characters")
                                continue

                            all_tweets.append({
                                "id": tweet.id,
                                "text": tweet.text,
                                "created_at": tweet_created_at,
                                "username": tweet.username,
                                "conversation_id": tweet.conversation_id,
                                "metrics": {'likes': tweet.likes},
                                "account_type": account_info["type"],
                                "account_name": account_info["name"]
                            })
                    logging.info(f"Found {len(tweets)} tweets for {username}")
                else:
                    logging.info(f"No recent tweets found for {username}")

            except Exception as e:
                logging.error(f"Error fetching tweets for {username}: {e}")

        logging.info(f"Total tweets collected: {len(all_tweets)}")

        # Update the cache
        self.tweets_cache = {
            'last_fetched': datetime.now(timezone.utc),
            'tweets': all_tweets,
            'params': {
                'lookback_hours': lookback_hours,
                'max_tweets_per_account': max_tweets_per_account
            }
        }

        return all_tweets

    def get_tweet_comments(self, tweet_id: str, tweet_text: str,
                           min_likes: Optional[int] = None, min_char: int = 10) -> List[Rapid_Comment]:
        """
        Get direct reply comments for a specific tweet, filtered by minimum likes if specified.

        This method checks the 'referenced_tweets' field for a reference of type "replied_to"
        that matches the original tweet's ID.

        Args:
            tweet_id: ID of the tweet to get comments for.
            tweet_text: text of the parent tweet.
            min_likes: Minimum number of likes for a comment to be included.
            min_char: Minimum number of characters for a comment to be included.

        Returns:
            List[Dict]: List of dictionaries with comment information.
        """
        comments = []
        try:

            response_data = self.rapid_client.get_post_comments(tweet_id, tweet_text, count=5,
                                                                ranking_mode="Relevance")
            logging.info(f"Search response for tweet {tweet_id}: {response_data}")
            if not response_data:
                logging.info(f"No comments found for tweet {tweet_id}")
                return comments

            for comment in response_data:
                like_count = comment.likes

                if min_likes is not None and like_count < min_likes:
                    continue
                if len(comment.text.strip()) < min_char:
                    logging.info(f"Skipping tweet {comment.comment_id} by @{comment.username} because it's under {min_char} characters")
                    continue
                comments.append(comment)

            comments.sort(key=lambda x: x.likes, reverse=True)
            logging.info(f"Found {len(comments)} direct comments for tweet {tweet_id} with min likes {min_likes}")

        except Exception as e:
            logging.error(f"Error fetching comments for tweet {tweet_id}: {e}")

        return comments

    def has_already_engaged(self, comment_id: str) -> bool:
        """
        Check if we have already engaged with this comment.

        Args:
            comment_id: ID of the comment.

        Returns:
            bool: True if already engaged, False otherwise.
        """
        return comment_id in self.engaged_history

    def reply_to_tweet(self, tweet: Dict) -> bool:
        """
        Generate and post a direct reply to a tweet from a target account.

        Args:
            tweet: Dictionary with tweet information.

        Returns:
            bool: True if reply was successful, False otherwise.
        """
        try:
            reply_text = self.generator.generate_response(
                input_text=f"""Reply to this tweet on Twitter: {tweet['text']}\n
                This tweet was posted by {tweet['username']} who is a {tweet['account_type']} - {tweet['account_name']}.\n
                We want to engage professionally with this content to increase our visibility.""",
                self_tweet=False
            )

            response = self.tweepy_client.create_tweet(
                text=reply_text,
                in_reply_to_tweet_id=tweet['id']
            )

            if response and response.data:
                logging.info(f"Successfully replied to tweet {tweet['id']} by @{tweet['username']}")
                # Add the tweet ID to engaged history to avoid duplicate replies
                self.engaged_history.append(tweet['id'])
                return True
            else:
                logging.warning(f"Failed to reply to tweet {tweet['id']} by @{tweet['username']}")
                return False

        except Exception as e:
            logging.error(f"Error replying to tweet {tweet['id']}: {e}")
            return False

    def reply_to_comment(self, comment: Rapid_Comment) -> bool:
        """
        Generate and post a reply to the comment.

        Args:
            comment: Dictionary with comment information.

        Returns:
            bool: True if reply was successful, False otherwise.
        """
        try:
            reply_text = self.generator.generate_response(
                input_text=f"""Reply to this comment on Twitter: {comment.text}\n
                The parent tweet is: {comment.parent_text}\n
                Note: This comment is on someone else's post, not on our own. 
                Please avoid replying as if it were our tweet.""", self_tweet=False
            )
            response = self.tweepy_client.create_tweet(
                text=reply_text,
                in_reply_to_tweet_id=comment.comment_id
            )
            if response and response.data:
                logging.info(f"Successfully replied to comment {comment.comment_id}")
                self.engaged_history.append(comment.comment_id)
                return True
            else:
                logging.warning(f"Failed to reply to comment {comment.comment_id}")
                return False

        except Exception as e:
            logging.error(f"Error replying to comment {comment.comment_id}: {e}")
            return False

    def process_comments(
            self,
            lookback_hours: int = 24,
            max_comments: int = 10,
            max_comments_per_tweet: Optional[int] = 2
    ) -> Tuple[int, int]:
        """
        Process comments on tweets from target accounts.

        Args:
            lookback_hours: Hours to look back for tweets.
            max_comments: Maximum tweets to process.
            max_comments_per_tweet: Maximum comments to engage with per tweet (defaults to self.max_comments_per_tweet).

        Returns:
            Tuple[int, int]: (tweets_processed, comments_engaged)
        """
        if max_comments_per_tweet is None:
            max_comments_per_tweet = self.max_comments_per_tweet

        tweets = self.get_recent_tweets(lookback_hours=lookback_hours)
        tweets.sort(
            key=lambda x: sum(x["metrics"].values()) if "metrics" in x and x["metrics"] else 0,
            reverse=True
        )
        tweets = tweets

        tweets_processed = 0
        comments_engaged = 0

        for tweet in tweets:
            try:
                if self.generator.filter_comment(tweet["text"]):
                    logging.info(f"Skipping tweet {tweet['id']} by @{tweet['username']}")
                    continue
                logging.info(f"Processing tweet {tweet['id']} by @{tweet['username']}")
                comments = self.get_tweet_comments(tweet['conversation_id'], tweet["text"],
                                                   min_likes=self.min_comment_likes)
                comments_processed = 0
                for comment in comments:
                    if self.has_already_engaged(comment.comment_id):
                        logging.info(f"Already engaged with comment {comment.comment_id}, skipping")
                        continue

                    if comments_processed >= max_comments_per_tweet:
                        logging.info(
                            f"Reached max comments ({max_comments_per_tweet}) for tweet {tweet['id']}, moving to next tweet")
                        break
                    success = None
                    if self.generator.filter_comment(
                            comment_context=f"parent post:{comment.parent_text} and comment text"
                                            f"{comment.text}"):
                        success = self.reply_to_comment(comment)
                    if success:
                        comments_processed += 1
                        comments_engaged += 1
                        if comments_processed == max_comments_per_tweet:
                            break

                    time.sleep(30)

                tweets_processed += 1
                if comments_engaged == max_comments:
                    break

            except Exception as e:
                logging.error(f"Error processing tweet {tweet['id']}: {e}")

        logging.info(
            f"Comment replier processed {tweets_processed} tweets and engaged with {comments_engaged} comments")
        return tweets_processed, comments_engaged

    def process_tweets(
            self,
            lookback_hours: int = 24,
            max_tweets: int = 5,
            min_likes_threshold: int = 10
    ) -> Tuple[int, int]:
        """
        Process tweets from target accounts and reply directly to them.

        Args:
            lookback_hours: Hours to look back for tweets.
            max_tweets: Maximum tweets to reply to.
            min_likes_threshold: Minimum likes a tweet should have to be considered for reply.

        Returns:
            Tuple[int, int]: (tweets_processed, tweets_replied)
        """
        tweets = self.get_recent_tweets(lookback_hours=lookback_hours)

        tweets.sort(
            key=lambda x: x["metrics"].get('likes', 0) if "metrics" in x else 0,
            reverse=True
        )

        tweets_processed = 0
        tweets_replied = 0

        for tweet in tweets:
            try:
                if self.has_already_engaged(tweet['id']):
                    logging.info(f"Already engaged with tweet {tweet['id']}, skipping")
                    continue

                if self.generator.filter_comment(tweet["text"]):
                    logging.info(f"Skipping tweet {tweet['id']} by @{tweet['username']} based on content filter")
                    continue

                likes = tweet["metrics"].get('likes', 0) if "metrics" in tweet else 0
                if likes < min_likes_threshold:
                    logging.info(
                        f"Skipping tweet {tweet['id']} with only {likes} likes (below threshold of {min_likes_threshold})")
                    continue

                logging.info(f"Processing tweet {tweet['id']} by @{tweet['username']}")

                success = self.reply_to_tweet(tweet)

                if success:
                    tweets_replied += 1

                tweets_processed += 1

                # Break if we've reached the maximum number of tweets to reply to
                if tweets_replied >= max_tweets:
                    break

                # Add delay between replies to avoid rate limiting
                time.sleep(30)

            except Exception as e:
                logging.error(f"Error processing tweet {tweet['id']}: {e}")

        logging.info(f"Direct tweet replier processed {tweets_processed} tweets and replied to {tweets_replied} tweets")
        return tweets_processed, tweets_replied

    def clear_tweet_cache(self):
        """
        Clear the tweets cache to force a fresh fetch on the next call to get_recent_tweets.
        """
        self.tweets_cache = {
            'last_fetched': None,
            'tweets': [],
            'params': {}
        }
        logging.info("Tweet cache cleared")