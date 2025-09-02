import logging
import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv
from datetime import datetime, timezone
load_dotenv()


def load_tweets_from_doc(doc_path: str):
    """Load tweets from the specified document."""
    try:
        with open(doc_path, "r", encoding="utf-8") as file:
            tweets = file.readlines()
        return [tweet.strip() for tweet in tweets if tweet.strip()]
    except Exception as e:
        logging.error(f"Error loading tweets: {e}")
        return []


def create_company_info_agent(model: str = "openai/gpt-4o-mini", markdown: bool = False, show_tool_calls: bool = False) -> Agent:
    # Load tweets from the document specified in the .env file
    tweet_doc_path = os.getenv("COMPANY_TWEET_DOCS")
    if not tweet_doc_path:
        logging.error("COMPANY_TWEET_DOCS environment variable not set.")
        return None

    tweets = load_tweets_from_doc(tweet_doc_path)
    if not tweets:
        logging.error("No tweets found in the document.")
        return None
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return Agent(
        model=OpenRouter(id=model, api_key=os.getenv("OPENROUTER_API_KEY"),temperature=0.7),
        tools=[],
        instructions=[
            dedent(
                f"""
                You are a tweet-generation AI specializing in corporate messaging. Your task is to:
                1. Choose a tweet from the provided list.
                2. Generate a new tweet inspired by it, maintaining a professional and engaging tone.
                3. Ensure the new tweet aligns with the company's branding and public image.
                
                The goal is to create fresh, engaging tweets that resonate with the audience while staying 
                true to the company's values.these are tweets you can refer {tweets}
                Also for your reference current time is {current_date_time}
                """
            )
        ],
        description="Agent that selects and generates company tweets.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("""
            Selected Tweet: {selected_tweet}
            Generated Tweet: {new_tweet}
        """
        )
    )

