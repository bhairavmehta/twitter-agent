import logging
import os
import re
import asyncio
import json
from textwrap import dedent
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, timezone
import tweepy
from dotenv import load_dotenv
from personality import Personality
from content_generator import ContentGenerator
from media_generator import MediaGenerator
from tweet_tracker import TweetTracker
from agents.structured_tweet_response_agent import create_structured_response_agent
from agents.mention_desision_agent import create_mention_responder_decision_agent

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MentionResponder:
    def __init__(self,
                 personality: Personality,
                 generator: ContentGenerator,
                 tweepy_client: tweepy.Client,
                 response_history: List,
                 tweet_tracker: TweetTracker,
                 mention_lock=None,
                 default_comment_limit: int = 3,
                 persona_notes: Optional[str] = None
                 ):
        """
        Initialize the MentionResponder.

        Args:
            personality: Contains configuration (e.g., buzzwords, persona).
            generator: Provides a generate_response() method.
            tweepy_client: An authenticated Tweepy client.
            response_history: List tracking processed conversation IDs (legacy?).
            tweet_tracker: Tracks tweets and comment limits for conversations. REQUIRED.
            mention_lock: Asyncio lock for thread safety.
            default_comment_limit: Default maximum comments per parent tweet (used if tweet_tracker needs it).
            persona_notes: Specific notes about the bot's persona for the decision agent.
                           If None, attempts to get from personality object.
        """
        if tweet_tracker is None:
             raise ValueError("TweetTracker instance must be provided.")

        self.personality = personality
        self.generator = generator
        self.tweepy_client = tweepy_client
        self.response_history = response_history
        self.media_generator = MediaGenerator()
        self.tweet_tracker = tweet_tracker
        self.mention_lock = mention_lock if mention_lock else asyncio.Lock()
        self.me_id = None
        self.me_username = None
        self._get_me_id()
        self.structured_response_agent = create_structured_response_agent(
            model="openai/gpt-4o-mini",
            markdown=False,
            show_tool_calls=False
        )

        self.persona_notes = persona_notes or getattr(personality, 'persona_notes', "Default persona: Be helpful and relevant.")
        logging.info(f"Using persona notes for decision agent: '{self.persona_notes}'")
        self.decision_agent = create_mention_responder_decision_agent()
        if self.me_id:
            logging.info(f"MentionResponder initialized for user ID: {self.me_id}, username: {self.me_username}")
        else:
            logging.error("MentionResponder initialized WITHOUT bot user ID/username.")


    def _get_me_id(self) -> None:
        """Gets and sets the bot's user ID and username."""
        try:
            user_response = self.tweepy_client.get_me(user_fields=["username"])
            if user_response.data:
                self.me_username = user_response.data.username
                self.me_id = user_response.data.id
                logging.info(f"Successfully retrieved bot ID: {self.me_id}, Username: {self.me_username}")
            else:
                logging.error("No user data returned from get_me(). Cannot determine bot ID.")
        except tweepy.TweepyException as e:
            logging.error(f"Failed to get bot's user ID: {e}")
        except Exception as e:
             logging.error(f"An unexpected error occurred in _get_me_id: {e}")

    @staticmethod
    def _format_datetime(dt: datetime) -> str:
        """Format datetime to RFC3339."""
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    def get_mentions(self, lookback_minutes: int = 60) -> List:
        """Retrieve recent mentions of the bot, excluding self-mentions."""
        if not self.me_id:
            logging.error("Cannot fetch mentions, bot user ID is not set.")
            return []

        start_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        try:
            response = self.tweepy_client.get_users_mentions(
                id=self.me_id,
                start_time=self._format_datetime(start_time),
                expansions=['referenced_tweets.id', 'author_id'],
                tweet_fields=['created_at', 'conversation_id', 'text', 'author_id']
            )
            mentions = response.data or []
            logging.info(f"Fetched {len(mentions)} mentions from the last {lookback_minutes} minutes.")

            # Filter out mentions where the author is the bot itself
            filtered_mentions = [m for m in mentions if m.author_id != self.me_id]
            if len(filtered_mentions) < len(mentions):
                logging.info(f"Filtered out {len(mentions) - len(filtered_mentions)} mentions authored by the bot.")

            return filtered_mentions
        except tweepy.TweepyException as e:
            logging.error(f"Failed to fetch mentions: {e}")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred fetching mentions: {e}")
            return []

    def _should_process_basic(self, mention, parent) -> bool:
        """
        Perform basic checks: not our own tweet and comment limit not reached for the conversation.
        """
        # Skip if it's one of our own tweets (should be filtered by get_mentions, but double-check)
        if mention.author_id == self.me_id:
             logging.info(f"Skipping mention {mention.id} (basic check: authored by self).")
             return False
        if self.tweet_tracker.is_our_tweet(str(mention.id)):
            logging.info(f"Skipping mention {mention.id} (basic check: recognized as our previous tweet/reply).")
            return False
        logging.info(f"mention {mention} passed basic checks. Parent tweet: {parent}")
        conversation_id = str(mention.conversation_id)
        if not self.tweet_tracker.can_comment(conversation_id) or not self.tweet_tracker.can_comment(str(parent.id)):
            logging.info(f"Skipping mention {mention.id} (basic check: comment limit reached for conversation {conversation_id}).")
            return False

        return True

    async def _should_reply_based_on_content(self, mention, parent_tweet) -> bool:
        """
        Uses the decision agent to determine if the bot should reply based on content and persona.
        Returns True if the decision is "reply", False otherwise.
        """
        mention_text = getattr(mention, 'text', '')
        parent_text = getattr(parent_tweet, 'text', 'N/A') # Use N/A if parent_tweet is None

        # Clean the mention text for the agent input
        cleaned_mention = mention_text
        if self.me_username:
             # Remove the bot's handle anywhere in the text
             cleaned_mention = re.sub(rf'@{self.me_username}\b', '', cleaned_mention, flags=re.IGNORECASE).strip()

        agent_input = f"""
        Mention Text: "{cleaned_mention}"
        Conversation Context: "{parent_text}"
        Bot Persona Notes: "{self.persona_notes}"
        """

        logging.info(f"Running decision agent for mention {mention.id} with input:\n{agent_input}")
        try:
            decision_response_str = await asyncio.to_thread(self.decision_agent.run, agent_input.strip())

            if not decision_response_str:
                logging.warning(f"Decision agent returned empty response for mention {mention.id}. Defaulting to 'ignore'.")
                return False

            raw_agent_response_content = decision_response_str.content
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_agent_response_content, re.DOTALL)

            json_string_to_parse = None
            if match:
                json_string_to_parse = match.group(1).strip()

            else:
                potential_json = raw_agent_response_content.strip()
                if potential_json.startswith('{') and potential_json.endswith('}'):
                    json_string_to_parse = potential_json

            if not json_string_to_parse:
                logging.error(
                    f"Could not extract JSON from agent response for mention {mention.id}. Raw: '{raw_agent_response_content}'")
                return False

            decision_data = json.loads(json_string_to_parse)
            decision = decision_data.get("decision", "").lower() # Normalize to lowercase
            reason = decision_data.get("reason", "N/A")
            logging.info(f"Decision agent for mention {mention.id}: {decision.upper()} - Reason: {reason}")
            return decision == "reply"

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse decision agent JSON response for mention {mention.id}: {e}. Response: '{decision_response_str}'. Defaulting to 'ignore'.")
            return False # Don't reply if parsing fails
        except Exception as e:
            # Catch any other exceptions during agent execution
            logging.exception(f"Error running decision agent for mention {mention.id}: {e}. Defaulting to 'ignore'.") # Use exception for stack trace
            return False # Don't reply on unexpected error


    def _get_parent_tweet(self, conversation_id: int) -> Optional[Any]:
        """
        Fetch the parent tweet (original tweet of the conversation).
        Returns None if not found or on error.
        """
        if not conversation_id: return None
        try:
            # Fetch necessary fields for context and potential replies
            tweet_response = self.tweepy_client.get_tweet(
                conversation_id,
                tweet_fields=["text", "author_id", "created_at"],
                expansions=["author_id"] # Expand author for potential future use
            )
            return tweet_response.data
        except tweepy.TweepyException as e:
            # Handle not found specifically
            if hasattr(e, 'api_codes') and 404 in e.api_codes:
                 logging.warning(f"Parent tweet for conversation {conversation_id} not found (404).")
            # Log other Tweepy errors
            else:
                 logging.error(f"Failed to fetch parent tweet for conversation {conversation_id}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error fetching parent tweet for conversation {conversation_id}: {e}")
            return None

    async def _handle_media_request(self, mention, conversation_id_str: str, response_text: str, prompt: str, media_type: str):
        """Internal helper to generate and post media replies."""
        media_id = None
        reply_to_id = mention.id # Reply directly to the mention

        try:
            logging.info(f"Generating {media_type} for mention {mention.id} with prompt: '{prompt}'")
            if media_type == "image":
                media_id = await self.media_generator.generate_and_upload_image(prompt)
            elif media_type == "video":
                media_id = await self.media_generator.generate_and_upload_video(prompt)

            if not media_id:
                logging.error(f"Failed to generate/upload {media_type} for mention {mention.id}")
                # Optional: Fallback to text-only reply?
                # response = self.tweepy_client.create_tweet(text=response_text, in_reply_to_tweet_id=reply_to_id)
                return None # Indicate failure

            logging.info(f"Posting {media_type} reply to mention {mention.id}")
            response = self.tweepy_client.create_tweet(
                text=response_text,
                in_reply_to_tweet_id=reply_to_id,
                media_ids=[media_id]
            )

            # Track successful comment in TweetTracker using conversation_id
            if response and hasattr(response, 'data') and response.data.get('id'):
                response_id_str = str(response.data['id'])
                self.tweet_tracker.add_comment(response_id_str, conversation_id_str) # Use conversation ID for tracking
                logging.info(f"Posted {media_type} reply {response_id_str} to mention {mention.id} in conversation {conversation_id_str}")
                return response.data
            else:
                 logging.error(f"Tweet creation with {media_type} media_id {media_id} seemed successful but response data invalid for mention {mention.id}.")
                 return None

        except tweepy.TweepyException as e:
            logging.error(f"Tweepy error posting {media_type} reply for mention {mention.id}: {e}")
            # Specifically check for duplicate status code (187)
            if hasattr(e, 'api_codes') and 187 in e.api_codes:
                logging.warning(f"Skipping {media_type} reply to mention {mention.id} - Twitter API reported duplicate status.")
            # Return None on error so it's not counted as success
            return None
        except Exception as e:
            logging.error(f"Unexpected error handling {media_type} request for mention {mention.id}: {e}")
            # Return None on error
            return None

    async def _generate_and_send_response(self, mention, parent_tweet) -> Optional[Dict]:
        """
        Determines reply type (normal, image, video) using structured_response_agent.
        If type is 'normal', generates text using self.generator, adjusting 'self_tweet'
        flag based on whether the parent tweet was authored by the bot.
        If type is media, uses prompt from structured agent and its message (or default).
        """
        mention_text = getattr(mention, 'text', '')
        conversation_id_str = str(mention.conversation_id)
        # Get parent text from the pre-fetched parent_tweet object
        parent_text = getattr(parent_tweet, 'text', 'N/A')
        reply_to_id = mention.id  # Reply directly to the mention tweet

        logging.info(
            f"[_generate_and_send_response:{mention.id}] Starting response determination. Parent Text available: {parent_text != 'N/A'}")

        # --- Prepare Input for Structured Response Agent ---
        cleaned_mention_for_struct_agent = mention_text
        if self.me_username:
            cleaned_mention_for_struct_agent = re.sub(rf'@{self.me_username}\b', '',
                                                      cleaned_mention_for_struct_agent, flags=re.IGNORECASE).strip()

        # Combine mention and parent context
        agent_input_context = f"Mention: {cleaned_mention_for_struct_agent}"
        if parent_text != "N/A":
            agent_input_context += f" | Parent Context: {parent_text}"

        logging.info(
            f"[_generate_and_send_response:{mention.id}] Input for structured_response_agent (type/prompt determination): '{agent_input_context}'")

        structured_response_obj = None
        structured_response_text = None
        try:
            # Assuming .run is synchronous; wrap if needed: await asyncio.to_thread(...)
            structured_response_obj = self.structured_response_agent.run(agent_input_context)
            structured_response_text = structured_response_obj.content if structured_response_obj else None
            logging.debug(
                f"[_generate_and_send_response:{mention.id}] Raw structured_response_agent output: '{structured_response_text}'")
        except Exception as e:
            logging.error(
                f"[_generate_and_send_response:{mention.id}] Error running structured_response_agent: {e}. Assuming 'normal' response type.")
            structured_response_text = None

        response_type = "normal"  # Default type
        structured_message = ""
        prompt = ""  # Prompt for media generation

        if structured_response_text:
            try:
                structured_response = json.loads(structured_response_text)
                response_type = structured_response.get("type", "normal").lower()
                # Get potential message and prompt
                structured_message = structured_response.get("message", "").strip()
                prompt = structured_response.get("prompt", "").strip()
                logging.info(
                    f"[_generate_and_send_response:{mention.id}] Parsed structured response: type={response_type}, potential_media_message='{structured_message[:50]}...', prompt='{prompt[:50]}...'")
            except json.JSONDecodeError as e:
                logging.error(
                    f"[_generate_and_send_response:{mention.id}] Error parsing structured response JSON: {e}. Response: '{structured_response_text}'. Defaulting type to 'normal'.")
                response_type = "normal"
            except Exception as e:
                logging.error(
                    f"[_generate_and_send_response:{mention.id}] Unexpected error processing structured response: {e}. Defaulting type to 'normal'.")
                response_type = "normal"
        else:
            logging.warning(
                f"[_generate_and_send_response:{mention.id}] Structured response agent returned no content. Defaulting type to 'normal'.")
            response_type = "normal"

        final_message = ""

        if response_type == "normal":
            logging.info(
                f"[_generate_and_send_response:{mention.id}] Response type is 'normal'. Generating text using self.generator.")

            is_reply_to_our_post = False
            if parent_tweet and hasattr(parent_tweet, 'author_id') and parent_tweet.author_id == self.me_id:
                is_reply_to_our_post = True
                logging.info(
                    f"[_generate_and_send_response:{mention.id}] This is a reply within a thread started by the bot. Setting self_tweet=True for generator.")
            else:
                logging.info(
                    f"[_generate_and_send_response:{mention.id}] This is a reply within a thread started by someone else. Setting self_tweet=False for generator.")

            generator_input = agent_input_context
            logging.info(f"[_generate_and_send_response:{mention.id}] Input for self.generator: '{generator_input}'")

            try:
                # Call the generator with the correct self_tweet flag
                final_message = self.generator.generate_response(
                    generator_input,
                    self_tweet=is_reply_to_our_post  # Pass the determined boolean flag
                )
                logging.info(
                    f"[_generate_and_send_response:{mention.id}] Generated message using self.generator (self_tweet={is_reply_to_our_post}): '{final_message[:100]}...'")
            except Exception as e:
                logging.error(
                    f"[_generate_and_send_response:{mention.id}] Text generation failed using self.generator: {e}")
                return None  # Cannot proceed if text generation fails

        elif response_type == "image" or response_type == "video":
            if structured_message:
                final_message = structured_message
                logging.info(
                    f"[_generate_and_send_response:{mention.id}] Using message from structured agent for {response_type} post: '{final_message[:50]}...'")
            else:
                final_message = "Check this out!"
                logging.info(
                    f"[_generate_and_send_response:{mention.id}] Using default text for {response_type} post: '{final_message}'")
        else:
            logging.error(
                f"[_generate_and_send_response:{mention.id}] Unknown response type '{response_type}'. Cannot generate reply.")
            return None

        original_message = final_message
        if self.me_username:
            final_message = re.sub(rf'@{self.me_username}\b', '', final_message, flags=re.IGNORECASE).strip()
            if final_message != original_message:
                logging.debug(f"[_generate_and_send_response:{mention.id}] Cleaned bot handle from final message.")

        if not final_message:
            logging.error(
                f"[_generate_and_send_response:{mention.id}] Cannot reply: Final message is empty after processing.")
            return None

        response_data = None
        try:
            if response_type == "video" and prompt:
                logging.info(
                    f"[_generate_and_send_response:{mention.id}] Handling VIDEO request. Prompt: '{prompt}', Message: '{final_message}'")
                response_data = await self._handle_media_request(mention, conversation_id_str, final_message, prompt,
                                                                 "video")
            elif response_type == "image" and prompt:
                logging.info(
                    f"[_generate_and_send_response:{mention.id}] Handling IMAGE request. Prompt: '{prompt}', Message: '{final_message}'")
                response_data = await self._handle_media_request(mention, conversation_id_str, final_message, prompt,
                                                                 "image")
            elif response_type == "normal":
                logging.info(
                    f"[_generate_and_send_response:{mention.id}] Handling TEXT request. Text: '{final_message}'")
                response_tweet = self.tweepy_client.create_tweet(
                    text=final_message,
                    in_reply_to_tweet_id=reply_to_id
                )

                # Process the response from Twitter API
                if response_tweet and hasattr(response_tweet, 'data') and response_tweet.data.get('id'):
                    response_id_str = str(response_tweet.data['id'])
                    self.tweet_tracker.add_comment(response_id_str, conversation_id_str)  # Track successful comment
                    logging.info(
                        f"[_generate_and_send_response:{mention.id}] Successfully posted TEXT reply {response_id_str} in conversation {conversation_id_str}")
                    response_data = response_tweet.data
                else:
                    logging.error(
                        f"[_generate_and_send_response:{mention.id}] Text tweet creation successful but response data invalid. Response: {response_tweet}")
            else:
                # This case might be hit if media type decided but prompt is missing
                logging.warning(
                    f"[_generate_and_send_response:{mention.id}] Cannot send reply for type '{response_type}' - missing prompt or other issue.")

        # --- Error Handling During Send ---
        except tweepy.TweepyException as e:
            logging.error(f"[_generate_and_send_response:{mention.id}] Tweepy error sending final reply: {e}")
            if hasattr(e, 'api_codes') and 187 in e.api_codes:
                logging.warning(
                    f"[_generate_and_send_response:{mention.id}] Skipping final reply - Twitter API reported duplicate status.")
            response_data = None
        except Exception as e:
            logging.error(f"[_generate_and_send_response:{mention.id}] Unexpected error sending final reply: {e}")
            response_data = None

        logging.info(
            f"[_generate_and_send_response:{mention.id}] Finished processing. Response successful: {response_data is not None}")
        return response_data



    def log_response(self, mention, response: Dict) -> None:
        """Logs response details to the legacy response_history list."""
        try:
            entry = {
                'conversation_id': mention.conversation_id,
                'mention_id': mention.id,
                'response_id': response.get('id'),
                'response_text': response.get('text'),
                'timestamp': datetime.now(timezone.utc)
            }
            self.response_history.append(entry)
            logging.debug(f"Logged response for mention {mention.id} in response_history.")
        except Exception as e:
            logging.error(f"Failed to log response for mention {mention.id} to history: {e}")


    async def process_mentions_and_respond(self, lookback_minutes: int = 60, max_mentions_to_process=5):
        """
        Fetches recent mentions and responds if conditions (limits, content appropriateness) are met.

        Args:
            lookback_minutes: How far back to fetch mentions.
            max_mentions_to_process: Max mentions to attempt processing in one run.

        Returns:
            A dictionary containing stats about the mentions processed.
        """
        if not self.me_id:
             logging.error("Cannot process mentions, bot user ID not available.")
             return {"status": "error", "reason": "no_bot_id"}

        # Acquire lock to prevent concurrent processing runs
        if self.mention_lock.locked():
            logging.warning("Mention processing skipped: Lock already acquired.")
            return {"status": "skipped", "reason": "lock_acquired"}

        async with self.mention_lock:
             logging.info(f"Acquired mention lock. Processing mentions (lookback={lookback_minutes}m, max={max_mentions_to_process}).")
             # Call the internal processing logic now that the lock is held
             return await self._process_mentions_internal(lookback_minutes, max_mentions_to_process)


    async def _process_mentions_internal(self, lookback_minutes: int, max_mentions_to_process: int) -> Dict:
        """Internal mention processing logic, called when lock is held."""
        mentions = self.get_mentions(lookback_minutes=lookback_minutes)

        if not mentions:
            logging.info("No new mentions found to process.")
            return {'total_fetched': 0, 'processed': 0, 'responded': 0, 'skipped_limit': 0, 'skipped_agent': 0, 'skipped_other': 0, 'errors': 0, 'comment_stats': {}}

        # Sort mentions by creation time (oldest first) to process chronologically
        mentions.sort(key=lambda m: m.created_at)

        # Initialize stats with new categories for skips
        stats = {
            'total_fetched': len(mentions),
            'processed': 0, # Mentions attempted (passed basic checks)
            'responded': 0, # Mentions successfully replied to
            'skipped_limit': 0, # Skipped due to comment limit or self-mention
            'skipped_agent': 0, # Skipped by decision agent (content inappropriate)
            'skipped_other': 0, # Skipped for other reasons (e.g., processing limit, send failure)
            'errors': 0, # Unhandled errors during processing loop
            'comment_stats': {} # Populated later
            }

        processed_conversations = set() # Track conversations touched for stats reporting

        mentions_to_attempt = mentions[:max_mentions_to_process]
        logging.info(f"Attempting to process up to {len(mentions_to_attempt)} mentions.")

        for mention in mentions_to_attempt:
            stats['processed'] += 1 # Increment attempt counter for each mention considered
            conversation_id_str = str(mention.conversation_id)
            processed_conversations.add(conversation_id_str) # Track for stats

            try:



                parent_tweet = self._get_parent_tweet(mention.conversation_id)
                if not self._should_process_basic(mention,parent_tweet):
                    stats['skipped_limit'] += 1
                    continue
                should_reply_content = await self._should_reply_based_on_content(mention, parent_tweet)
                if not should_reply_content:
                    # Reason logged within _should_reply_based_on_content
                    stats['skipped_agent'] += 1
                    continue # Move to next mention


                logging.info(f"Checks passed for mention {mention.id}. Proceeding to generate/send response.")
                response = await self._generate_and_send_response(mention, parent_tweet) # Pass parent_tweet
                if response:
                    self.log_response(mention, response) # Log to legacy history if needed
                    stats['responded'] += 1
                else:
                    # _generate_and_send_response returned None (e.g., duplicate, generation/send failure)
                    logging.warning(f"Response generation/sending failed or skipped for mention {mention.id} after checks passed.")
                    stats['skipped_other'] += 1

            except Exception as e:
                # Catch unexpected errors during the processing loop for a single mention
                logging.exception(f"Unhandled error processing mention {mention.id}: {e}")
                stats['errors'] += 1

        # Calculate skips due to hitting the max_mentions_to_process limit
        skipped_due_to_limit = len(mentions) - len(mentions_to_attempt)
        if skipped_due_to_limit > 0:
             logging.info(f"Skipped {skipped_due_to_limit} mentions due to max_mentions_to_process limit ({max_mentions_to_process}).")
             stats['skipped_other'] += skipped_due_to_limit # Add these to 'other' skips

        # Add current comment stats for relevant conversations
        if self.tweet_tracker:
            current_comment_stats = self.tweet_tracker.get_all_comment_stats()
            # Filter stats to only include conversations encountered in this run
            relevant_stats = {conv_id: data for conv_id, data in current_comment_stats.items() if conv_id in processed_conversations}
            stats['comment_stats'] = relevant_stats

        logging.info(f"Mention processing finished. Stats: {stats}")
        return stats