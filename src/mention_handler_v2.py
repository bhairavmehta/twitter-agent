import logging
import re
import asyncio
import json
from typing import List, Dict, Optional, Any, Tuple # Added Tuple
from datetime import datetime, timedelta, timezone
import tweepy
from personality import Personality
from content_generator import ContentGenerator
from media_generator import MediaGenerator
from tweet_tracker import TweetTracker # Assuming TweetTracker is correct
from agents.structured_tweet_response_agent import create_structured_response_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Assume TweetTracker class is defined correctly elsewhere

class MentionResponder:
    def __init__(self,
                 personality: Personality,
                 generator: ContentGenerator,
                 tweepy_client: tweepy.Client,
                 response_history: List = None,
                 tweet_tracker: TweetTracker = None,
                 mention_lock=None,
                 default_comment_limit: int = 5):
        """
        Initialize the MentionResponder.
        """
        self.personality = personality
        self.generator = generator
        self.tweepy_client = tweepy_client
        self.response_history = response_history or []
        self.media_generator = MediaGenerator()
        # Use a combined method to get ID and username, store consistently
        self.me_id, self.me_username = self._get_me_info()
        self.tweet_tracker = tweet_tracker if tweet_tracker else TweetTracker(
            default_comment_limit=default_comment_limit)
        # Ensure lock is an asyncio.Lock
        self.mention_lock = mention_lock if mention_lock is not None else asyncio.Lock()
        self.structured_response_agent = create_structured_response_agent(
            model="openai/gpt-4o",
            markdown=False,
            show_tool_calls=False
        )
        if not self.me_id:
             raise ValueError("Could not retrieve bot's user ID. Cannot initialize MentionResponder.")


    def _get_me_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Gets the bot's user ID (as string) and username."""
        try:
            user_response = self.tweepy_client.get_me(user_fields=["username"])
            if user_response.data:
                user_id = str(user_response.data.id) # Store as string
                username = user_response.data.username
                logging.info(f"Initialized with bot user ID: {user_id}, Username: @{username}")
                return user_id, username
            else:
                logging.error("No user data returned from get_me().")
                return None, None
        except tweepy.TweepyException as e:
            logging.error(f"Failed to get bot's user info: {e}")
            return None, None
        except Exception as e:
            logging.error(f"An unexpected error occurred in _get_me_info: {e}")
            return None, None

    @staticmethod
    def _format_datetime(dt: datetime) -> str:
        """Format datetime to RFC3339."""
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    def get_mentions(self, lookback_minutes: int = 60) -> List[tweepy.Tweet]:
        """Retrieve recent mentions."""
        if not self.me_id:
            logging.error("Cannot fetch mentions, bot user ID is not set.")
            return []
        start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        try:
            response = self.tweepy_client.get_users_mentions(
                id=self.me_id,
                start_time=self._format_datetime(start_time),
                expansions=['referenced_tweets.id', 'author_id'],
                tweet_fields=['created_at', 'conversation_id', 'text', 'author_id'] # Ensure author_id is fetched
            )
            mentions = response.data or []
            # Ensure mentions are actual Tweet objects and filter out None if any crept in
            mentions = [m for m in mentions if isinstance(m, tweepy.Tweet)]
            logging.info(f"Fetched {len(mentions)} mentions from the last {lookback_minutes} minutes.")
            return mentions
        except tweepy.errors.NotFound:
             logging.warning(f"Mention endpoint returned 404 for user {self.me_id}. Maybe no mentions yet?")
             return []
        except tweepy.TweepyException as e:
            logging.error(f"Failed to fetch mentions: {e}")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred fetching mentions: {e}")
            return []

    # Keep _get_parent_tweet as it is used by _should_process and _generate_and_send_response
    # BUT acknowledge it causes redundant API calls - this is a *minimal change* fix
    def _get_parent_tweet(self, conversation_id: int) -> Optional[tweepy.Tweet]:
        """
        Fetch the parent tweet (original tweet of the conversation).
        Returns None if not found. Also returns the mention itself if it's the root.
        """
        if not conversation_id:
            logging.warning("Cannot get parent tweet for invalid conversation_id: None")
            return None
        try:
            tweet_response = self.tweepy_client.get_tweet(
                conversation_id, # Fetch the conversation's root tweet
                tweet_fields=["text", "author_id", "created_at"] # Add fields as needed
            )
            return tweet_response.data
        except tweepy.errors.NotFound:
             logging.warning(f"Parent tweet with conversation_id {conversation_id} not found.")
             return None
        except tweepy.TweepyException as e:
            # Log specific error types if needed (e.g., rate limits)
            logging.error(f"Failed to fetch parent tweet for conversation {conversation_id}: {e}")
            return None
        except Exception as e:
             logging.error(f"Unexpected error fetching parent tweet {conversation_id}: {e}")
             return None


    # Critical check method
    def _should_process(self, mention: tweepy.Tweet) -> bool:
        """
        Decide if a mention should be processed based on:
        1. Not our own tweet/comment
        2. Author is not self
        3. Comment limit not reached for the conversation thread.
        """
        mention_id_str = str(mention.id)

        # 1. Skip if it's one of our own tweets/comments tracked by TweetTracker
        if self.tweet_tracker.is_our_tweet(mention_id_str):
            logging.debug(f"Skipping mention {mention_id_str} (our own tweet/comment in tracker)")
            return False

        # 2. Skip if the author is the bot itself (important check)
        # Ensure author_id is available on the mention object
        if not hasattr(mention, 'author_id') or mention.author_id is None:
             logging.warning(f"Skipping mention {mention_id_str} due to missing author_id.")
             # Decide if this should be treated as processable or not. Skipping is safer.
             return False
        if str(mention.author_id) == self.me_id:
            logging.debug(f"Skipping mention {mention_id_str} (author is self)")
            return False

        # 3. Check comment limit for the conversation thread
        if not hasattr(mention, 'conversation_id') or mention.conversation_id is None:
             logging.warning(f"Skipping mention {mention_id_str} due to missing conversation_id.")
             return False # Cannot check limit without conversation_id

        # Determine the ID for comment tracking (always the conversation_id)
        parent_id_str = str(mention.conversation_id)

        # Perform the check using TweetTracker
        if not self.tweet_tracker.can_comment(parent_id_str):
            # Logging is done within can_comment, but we add context here
            logging.info(f"Skipping mention {mention_id_str} in conversation {parent_id_str}: comment limit reached.")
            return False

        # If all checks pass
        return True

    # _get_context can remain, acknowledging it fetches parent again
    def _get_context(self, mention: tweepy.Tweet) -> str:
        """Get context from parent tweet or mention text."""
        context_parts = []
        # This fetches parent again - inefficiency accepted for minimal change
        parent_tweet = self._get_parent_tweet(mention.conversation_id)

        mention_text = mention.text if hasattr(mention, 'text') else ''
        cleaned_mention_text = mention_text
        # Clean bot's username from mention text for context
        if self.me_username:
            cleaned_mention_text = re.sub(rf'@{self.me_username}\b', '', mention_text, flags=re.IGNORECASE).strip()

        if parent_tweet and str(parent_tweet.id) != str(mention.id):
             parent_text = parent_tweet.text if hasattr(parent_tweet, 'text') else '[Parent tweet text not available]'
             parent_author_id = parent_tweet.author_id if hasattr(parent_tweet, 'author_id') else 'unknown'
             context_parts.append(f"Parent tweet (@{parent_author_id}): {parent_text}")

        mention_author_id = mention.author_id if hasattr(mention, 'author_id') else 'unknown'
        context_parts.append(f"Mention (@{mention_author_id}): {cleaned_mention_text}")

        # Adding previous response history might still be useful for the LLM context
        # previous_responses = [entry for entry in self.response_history
        #                       if str(entry.get('conversation_id')) == str(mention.conversation_id)]
        # if previous_responses:
        #     last_reply = previous_responses[-1].get('response_text', '')
        #     if last_reply:
        #         context_parts.append(f"Our previous reply in thread: {last_reply}")

        return " | ".join(context_parts)


    # Media handlers need to correctly identify parent_id and call tracker
    async def _handle_media_request(self,
                                    mention: tweepy.Tweet,
                                    parent_id_str: str, # Pass the correct ID
                                    response_text: str,
                                    media_prompt: str,
                                    media_type: str) -> Optional[Dict]:
        """Unified media handler."""
        media_id = None
        try:
            if media_type == "image":
                media_id = await self.media_generator.generate_and_upload_image(media_prompt)
            elif media_type == "video":
                media_id = await self.media_generator.generate_and_upload_video(media_prompt)
            else:
                logging.error(f"Unsupported media type '{media_type}' requested for mention {mention.id}")
                return None

            if not media_id:
                logging.error(f"Failed to generate/upload {media_type} for mention {mention.id}. Prompt: {media_prompt}")
                # Fallback: Post text only?
                # return await self._post_text_reply(mention, parent_id_str, response_text)
                return None # Fail for now if media fails

            # Clean self-mentions from the final text before posting
            if self.me_username:
                response_text = re.sub(rf'@{self.me_username}\b', '', response_text, flags=re.IGNORECASE).strip()

            if not response_text:
                 logging.warning(f"Generated empty text message for media reply to mention {mention.id}, skipping.")
                 return None

            # Post the tweet with media, replying to the *mention*
            response = self.tweepy_client.create_tweet(
                text=response_text,
                in_reply_to_tweet_id=mention.id,
                media_ids=[media_id]
            )

            if response and response.data:
                response_id = str(response.data['id'])
                logging.info(f"Posted {media_type} reply {response_id} to mention {mention.id} in thread {parent_id_str}")
                # *** Track the successful comment ***
                self.tweet_tracker.add_comment(response_id, parent_id_str)
                return response.data
            else:
                logging.error(f"Tweet creation with media for mention {mention.id} returned no data.")
                return None

        except Exception as e:
            logging.error(f"Error handling {media_type} request for mention {mention.id}: {e}", exc_info=True)
            return None

    async def _post_text_reply(self, mention: tweepy.Tweet, parent_id_str: str, text: str) -> Optional[Dict]:
         """Posts a text-only reply tweet and tracks it."""
         try:
            # Clean self-mentions from the final text before posting
            if self.me_username:
                text = re.sub(rf'@{self.me_username}\b', '', text, flags=re.IGNORECASE).strip()

            if not text:
                 logging.warning(f"Generated empty text for mention {mention.id}, skipping reply.")
                 return None

            # Post reply to the *mention*
            response = self.tweepy_client.create_tweet(
                text=text,
                in_reply_to_tweet_id=mention.id
            )

            if response and response.data:
                response_id = str(response.data['id'])
                logging.info(f"Posted text reply {response_id} to mention {mention.id} in thread {parent_id_str}")
                # *** Track the successful comment ***
                self.tweet_tracker.add_comment(response_id, parent_id_str)
                return response.data
            else:
                logging.error(f"Text tweet creation for mention {mention.id} returned no data.")
                return None
         except tweepy.TweepyException as e:
            # Handle specific errors like duplicate content
            if isinstance(e, tweepy.errors.Forbidden) and 'duplicate content' in str(e).lower():
                logging.warning(f"Skipping reply to {mention.id} due to duplicate content detection.")
            else:
                logging.error(f"Error sending text reply for mention {mention.id}: {e}")
            return None
         except Exception as e:
            logging.error(f"Unexpected error posting text reply for mention {mention.id}: {e}", exc_info=True)
            return None

    # Removed the redundant can_comment check inside this method
    async def _generate_and_send_response(self, mention: tweepy.Tweet) -> Optional[Dict]:
        """
        Generate and post a reply. Assumes _should_process check has passed.
        """
        # Determine parent_id (conversation_id) ONCE for this function's scope
        if not hasattr(mention, 'conversation_id') or mention.conversation_id is None:
            logging.error(f"Cannot generate response for mention {mention.id}: missing conversation_id.")
            return None
        parent_id_str = str(mention.conversation_id)

        llm_context = self._get_context(mention)
        mention_text = mention.text if hasattr(mention, 'text') else ''

        # Prepare input for the structured response agent (example structure)
        agent_input = f"""
        Analyze the following Twitter mention and context to determine the appropriate response type and content.

        Context:
        {llm_context}

        Mention Text We Are Replying To:
        {mention_text}

        Output a JSON object: {{ "type": "normal|image|video|no_reply", "prompt": "...", "message": "..." }}
        Do NOT include @{self.me_username} in the message.
        Use "no_reply" for spam/unclear/hostile mentions.
        """

        # Get Structured Response from Agent
        try:
            structured_response_obj = self.structured_response_agent.run(agent_input)
            structured_response_text = str(structured_response_obj.content)
            structured_response = json.loads(structured_response_text)
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON from agent for mention {mention.id}: {e}. Falling back.")
            # Fallback to simple text generation
            try:
                 fallback_text = self.generator.generate_response(llm_context, self_tweet=False)
                 structured_response = {"type": "normal", "prompt": "", "message": fallback_text}
            except Exception as gen_e:
                 logging.error(f"Fallback text generation failed for mention {mention.id}: {gen_e}")
                 return None
        except Exception as e:
            logging.error(f"Error running structured response agent for mention {mention.id}: {e}", exc_info=True)
            return None

        # Process Agent Response
        response_type = structured_response.get("type", "normal").lower()
        message = structured_response.get("message", "").strip()
        prompt = structured_response.get("prompt", "").strip()

        # Final cleaning of self-mention just in case
        if self.me_username:
            message = re.sub(rf'@{self.me_username}\b', '', message, flags=re.IGNORECASE).strip()

        # Handle Different Response Types
        response_data = None
        if response_type == "no_reply":
            logging.info(f"Agent determined no reply needed for mention {mention.id}.")
            return None
        elif response_type == "video":
            if not prompt or not message:
                 logging.warning(f"Video response for {mention.id} missing prompt or message. Skipping.")
                 return None # Or fallback to text? Decide behavior.
            response_data = await self._handle_media_request(mention, parent_id_str, message, prompt, "video")
        elif response_type == "image":
            if not prompt or not message:
                 logging.warning(f"Image response for {mention.id} missing prompt or message. Skipping.")
                 return None # Or fallback to text?
            response_data = await self._handle_media_request(mention, parent_id_str, message, prompt, "image")
        elif response_type == "normal":
             if not message:
                 logging.warning(f"Normal response for {mention.id} has empty message. Skipping.")
                 return None
             response_data = await self._post_text_reply(mention, parent_id_str, message)
        else:
             logging.warning(f"Unknown response type '{response_type}' for {mention.id}. Defaulting to text.")
             if not message:
                 logging.warning(f"Defaulting response for {mention.id} has empty message. Skipping.")
                 return None
             response_data = await self._post_text_reply(mention, parent_id_str, message)

        # Note: The call to self.tweet_tracker.add_comment happens INSIDE
        # _handle_media_request and _post_text_reply upon SUCCESSFUL posting.

        return response_data # Return the data of the created tweet, or None if failed/skipped

    def log_response(self, mention: tweepy.Tweet, response: Dict) -> None:
        """Log the response details."""
        try:
            # Ensure IDs are stored consistently (e.g., as strings)
            entry = {
                'conversation_id': str(mention.conversation_id),
                'mention_id': str(mention.id),
                'response_id': str(response.get('id')),
                'response_text': response.get('text'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.response_history.append(entry)
            logging.debug(f"Logged response {entry['response_id']} for mention {entry['mention_id']}.")
        except Exception as e:
            logging.error(f"Failed to log response for mention {mention.id}: {e}")


    # *** Apply proper locking here ***
    async def process_mentions_and_respond(self, lookback_minutes: int = 60, max_mentions=5) -> Dict:
        """
        Fetch mentions, process them using a lock, generate responses, and reply.
        """
        # Use async context manager for the lock
        async with self.mention_lock:
            # The lock ensures that only one invocation of _process_mentions runs at a time,
            # preventing race conditions between checking the limit and adding a comment.
            logging.info(f"Acquired mention processing lock. Processing up to {max_mentions} mentions from last {lookback_minutes} mins.")
            return await self._process_mentions(lookback_minutes, max_mentions)
        # Lock is automatically released here, even if errors occur inside _process_mentions

    # Keep _process_mentions mostly the same structure, but it now runs under the lock
    async def _process_mentions(self, lookback_minutes: int = 20, max_mentions_to_process=10) -> Dict: # Use internal name consistency
        """
        Internal logic to fetch and process mentions. Assumes lock is held.
        """
        mentions = self.get_mentions(lookback_minutes=lookback_minutes)
        processed_count = 0
        # Use compatible stat keys if needed by calling code
        stats = {'total': len(mentions), 'attempted': 0, 'responded': 0, 'skipped': 0, 'errors': 0, 'comment_stats': {}}

        # Optional: Pre-fetch comment stats for reporting (can be kept or removed)
        if self.tweet_tracker:
             parent_ids_in_batch = set(str(m.conversation_id) for m in mentions if hasattr(m, 'conversation_id') and m.conversation_id)
             comment_stats_snapshot = {}
             for p_id in parent_ids_in_batch:
                 limit = self.tweet_tracker.comment_limits.get(p_id, self.tweet_tracker.default_comment_limit)
                 count = self.tweet_tracker.get_comment_count(p_id)
                 comment_stats_snapshot[p_id] = {'count': count, 'limit': limit, 'remaining': max(0, limit - count)}
             stats['comment_stats'] = comment_stats_snapshot

        for mention in mentions:
             if processed_count >= max_mentions_to_process:
                 logging.info(f"Reached processing limit ({max_mentions_to_process}). Skipping remaining {len(mentions) - processed_count} mentions.")
                 stats['skipped'] += (len(mentions) - processed_count)
                 break

             processed_count += 1
             stats['attempted'] += 1
             logging.debug(f"Processing mention {mention.id} (Attempt {processed_count}/{max_mentions_to_process})")

             try:
                 # *** CRITICAL: Check should_process *within the lock* ***
                 if self._should_process(mention):
                      # If check passes, proceed to generate and send
                      response_data = await self._generate_and_send_response(mention)

                      # If response was successful (tweet created and tracked)
                      if response_data and isinstance(response_data, dict) and response_data.get('id'):
                          self.log_response(mention, response_data)
                          stats['responded'] += 1
                      else:
                          # _generate_and_send_response handles skipping/failure logging
                          stats['skipped'] += 1 # Count as skipped if no response sent
                 else:
                      # _should_process logs the reason for skipping (e.g., limit reached)
                      stats['skipped'] += 1

             except Exception as e:
                 # Catch errors during the processing of a single mention
                 logging.exception(f"Critical error processing mention {mention.id}: {e}")
                 stats['errors'] += 1
                 # Continue to the next mention

        logging.info(f"Mention processing finished. Stats: {stats}")
        # Use 'total' key if needed for compatibility
        final_stats = {
            'total': stats['total'],
            'attempted': stats['attempted'],
            'responded': stats['responded'],
            'skipped': stats['skipped'],
            'errors': stats['errors'],
            'comment_stats': stats['comment_stats'] # Include the snapshot
        }
        return final_stats