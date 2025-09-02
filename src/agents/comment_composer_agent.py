import logging
import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv
from datetime import datetime, timezone


load_dotenv()

def create_comment_composer_agent(
        model: str = "openai/gpt-4o-mini",
        api_key: str = None,
        markdown: bool = False,
        show_tool_calls: bool = False,
        self_tweet:bool = True,
        reply_examples_file: str = "docs/reply_examples.txt",
) -> Agent:
    """
    Creates a comment composer agent that generates a tweet comment.

    Instructions:
      - Accept as input the original tweet along with a context summary labeled 'Comment Context' that contains the extracted details.
      - Compose an engaging, edgy, and snarky tweet comment that incorporates the provided context.
      - The final comment must be in plain text (without markdown) and must not exceed 280 characters.
      - Output only the tweet comment text labeled as 'Tweet Comment'.
    """
    example_content = ""
    if os.path.exists(reply_examples_file):
        with open(reply_examples_file, "r", encoding="utf-8") as f:
            example_content += f.read()

    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if self_tweet:
        instructions = dedent(f"""
        You are a tweet reply composer agent. Your task is to generate a concise and engaging tweet reply based on
         the provided mention and context.
        Make sure that the reply is within 200 character and no more than that 
        Inputs:
          1. The original mention comment.
          2. A comprehensive context summary that may include cryptocurrency details and other relevant information.

        Guidelines:
         The reply must be in plain text—without any markdown, quotes, or extraneous formatting.
                - The reply must be plain text—no markdown, quotes, or extraneous formatting.
                - The response must be within 200 characters. When a terse answer is sufficient, respond in 4-5 words.
                - Use short, punchy, and energetic sentences.
                  - Absolutely NO dashes (– or —), under any circumstances.
                  - And no (" or ') as well
                  - Do not use emojis in the response.
                  - Optimize for virality and engagement on Twitter
                  - Examples provided later are for reference only to mimic style and tone; do not use their content verbatim.
                    Example of Desired Tweet Style:
                   ETH's struggle vs. BTC continues. Whales buying, but is $4K still realistic?
         Also for your reference current time is {current_date_time}
    """).strip()
    else :
        instructions = dedent(f""" You are a tweet reply composer agent. Your task is to generate a concise and engaging 
             tweet reply on our competitors or a famous personality. Note: This is not our own comment but a comment made by
             either our competitors or a notable figure, so your tone and content must reflect that perspective. 
             Inputs:
              1. The original mention comment. 
              2. A comprehensive context summary that may include cryptocurrency details and 
              other relevant information.
            Guidelines:
                - Short, punchy, energetic sentences.
                - Absolutely NO dashes (– or —), under any circumstances.
                - If you include punctuation, stick only to commas or periods.
                - Optimize for virality and engagement on Twitter.
                - If the original tweet contains a reference URL, include it.
                - Absolutely NO dashes (– or —), quotes (" or ').
                - Do not use emojis in the response.
                - Examples provided later are for reference only to mimic style and tone; do not use their content verbatim.
                    Example of Desired Tweet Style:
                    ETH's struggle vs. BTC continues. Whales buying, but is $4K still realistic?
            Also for your reference current time is {current_date_time}
                """).strip()
    return Agent(
        model=OpenRouter(id=model, api_key=api_key,temp=0.3),
        instructions=[
            instructions
        ],
        description="Agent that composes tweet comments based on extracted context from the original tweet.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("""{tweet_comment}""")
    )
