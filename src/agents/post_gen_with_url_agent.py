import logging
import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()


def create_post_generator_w_agent(
        model: str = "openai/gpt-4o-mini",
        api_key: str = "",
        markdown: bool = False,
        show_tool_calls: bool = False
) -> Agent:
    """
    Creates an agent that generates tweet content for social media posts based on the provided input (e.g., current events).

    Instructions:
      - You are a tweet post generator agent.
      - Your task is to craft an engaging tweet post based on the input content that reflects current events and any relevant details.
      - Ensure your output is in plain text (no markdown formatting) and does not exceed Twitter's 280-character limit.
      - The tweet should be concise, clear, and align with the brand's personality.
      - Incorporate any key details from the input seamlessly.
      - Return the tweet post labeled as "Tweet Post".

    The agent uses the provided model and API key via the OpenRouter interface.
    """
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return Agent(
        model=OpenRouter(id=model, api_key=api_key),
        instructions=[
            (
                f"""
                You are a tweet post composer agent for 365x.ai.
                You will be given one or more tweets. Your task is to follow these steps:
                
                1. **Select only one tweet**: From the list of given tweets, choose the tweet that you think is the most 
                engaging and relevant for crafting a post.
                
                2. **Rewrite the selected tweet**: 
                Take the information provided in the tweet you selected and rewrite it to fit the following style:
                   - Keep sentences short, punchy, and energetic.
                   - Absolutely NO dashes (– or —), use commas or periods only.
                   - Keep the tweet under 200 characters, optimizing for virality and engagement.
                   - Do not include any details not mentioned in the tweet (do not speculate or assume anything).
                   - Make sure to include all Urls except for the one's related to WuBlockchain in your post as well.
                   
                3. **Output format**: Your final tweet should be plain text, without any markdown or special formatting.
                
                4. **Example of desired tweet style**:
                   "ETH's struggle vs. BTC continues. Whales buying, but is $4K still realistic?(add the url if there is any) "
                
                5. **Time information**: The current time is {current_date_time} (Use this context if necessary, but do not include the time explicitly in the tweet).
                
                Output: only one tweet following these guidelines.
                """
            )
        ],
        description="Agent that generates tweet content for social media posts based tweets given.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("""\
                clean only text for the tweet to be posted and nothing else no '' and no links etc.
            """)
    )
