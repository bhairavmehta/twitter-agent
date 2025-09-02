import datetime
import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv
from datetime import datetime, timezone
from phi.tools.googlesearch import GoogleSearch
load_dotenv()


def create_validator_agent(
        model: str = "gpt-4o",
        api_key: str = "",
        markdown: bool = False,
        show_tool_calls: bool = False,
        text_type: str = "comment",  # Allowed values: "post", "comment", "reply"
        post_examples_file: str = "docs/post_examples.txt",
        reply_examples_file: str = "docs/reply_examples.txt"
) -> Agent:
    """
    Creates a tweet generation agent that generates tweets in the style of provided examples.

    Additional rule: If the post mentions or is related to 365x.ai, the agent must adopt an official
    social media communication tone representing 365x.ai. Do not state that you are the social media manager;
    simply present the update as coming directly from 365x.ai, for example:

    """
    text_type = text_type.lower()
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    example_content = ""
    if text_type == "post":
        if os.path.exists(post_examples_file):
            with open(post_examples_file, "r", encoding="utf-8") as f:
                example_content = f.read()
        dynamic_instructions = (
                "For a tweet post: the text should be plain text without markdown, quotes, or extra formatting. "
                "It must be informative, professional, and polished—steering away from an overly edgy or snarky tone. "
                "Use the provided examples only to guide the tone, but do not reproduce any part of them. "
                "Examples: " + example_content
        )
    elif text_type == "comment":
        if os.path.exists(reply_examples_file):
            with open(reply_examples_file, "r", encoding="utf-8") as f:
                example_content = f.read()
        dynamic_instructions = (
                "For a tweet comment: the text should be plain text with no extra formatting. "
                "It must be witty yet professional, ensuring clarity without being overly edgy or snarky. "
                "Use the external examples solely as style guidance without copying them directly. "
                "Examples: " + example_content
        )
    elif text_type == "reply":
        if os.path.exists(reply_examples_file):
            with open(reply_examples_file, "r", encoding="utf-8") as f:
                example_content = f.read()
        dynamic_instructions = (
                "For a tweet reply: the text must be plain text without markdown, quotes, or extra symbols. "
                "It should directly address the mention in a clear and professional manner, avoiding overly edgy or snarky language. "
                "Rely on the external examples for style only, and do not include any of their exact content. "
                "Examples: " + example_content
        )
    else:
        raise ValueError("Invalid text_type. Must be one of 'post', 'comment', or 'reply'.")

    examples_note = (
        "Note: The external examples are provided strictly as stylistic references. "
        "Do not incorporate or output any direct phrases or content from these examples."
        "For example if the text is "
    )

    # New instruction for 365x.ai-related content
    company_instruction = (
        "If the post mentions or relates to 365x.ai, adopt an official social media communication tone as if representing Coinbase. "
        "Present the update in a professional manner without stating that you are the social media manager. "
        "For instance: \"We’re excited to share that 365x.ai has posted a 130% revenue increase and earned the title of 'Best Prime Broker'! "
        "With our new price target raised to $475, we remain committed to driving innovation and serving our community. "
        "Thank you for believing in us, the future looks bright!\""
    )

    instructions = dedent(f""" You are a tweet generation agent. Your task is to generate a tweet of type '{text_type}' 
    based on the provided context. Note: This tweet is not our own comment but comes from our competitors or a famous
    personality, so your tone and content must reflect that external perspective.
    Guidelines:
      1. The tweet must be in plain text—without any markdown, quotes, or extraneous formatting.
      2. The tweet must not exceed 200 characters.
      3. The reply must be in plain text—without any markdown, quotes, or extraneous formatting.
      4. Short, punchy, energetic sentences.
         - Funny, sarcastic tone with clear stakes.
         - Include playful metaphors, especially comparisons to reality TV or competitions.
         - Clearly emphasize winners vs. losers ("going home salty," "golden tickets," "underdogs eliminated," "winner takes all," "crypto idol," "the tribe has spoken" etc.).
         - Casual slang encouraged (e.g., "bulls hyped," "hopium," "vibing," "salty,").
         - Humorous skepticism encouraged ("minimum due diligence, maximum drama").
         - Absolutely NO dashes (– or —), under any circumstances.
         - Do not use emojis in the response.
         - Optimize for virality and engagement on Twitter.
      5. If the original tweet contains a reference URL, include it clearly and naturally at the end of the tweet.

      Example of Desired Tweet Style:

      Binance drops a token popularity contest to spice up listings. Your fave crypto fighting to survive. 
      Two get listed, the rest head home salty. Minimum diligence, max drama. Let's go.

      6. {company_instruction}
      7. Maintain a professional tone that aligns with 365x.ai's brand voice when representing the company.
      8. Keep the post, comment, or reply short and concise—aim for 100 characters or fewer whenever possible. If the reply or comment can be answered with one word or a simple sentence, make sure you use that only. For example, if someone says hello you can simply say gm or gn based on the time.
      9. Ensure that any quoted statements, such as "sure you should buy that crypto," are rewritten without quotation marks unless absolutely necessary for meaning or clarity.
      10. Make sure that your responses are short, concise, and engaging.
      11. Responses should not always focus on the company and should remove all hashtags.

      Also for your reference current time is {current_date_time}

      {dynamic_instructions} 

      {examples_note}
    """).strip()

    return Agent(
        model=OpenRouter(id=model,temperature=0.4, api_key=api_key if api_key else os.getenv("OPENROUTER_API_KEY")),
        tools=[],
        instructions=[instructions],
        description=f"Tweet generation agent for {text_type} texts. Generates tweets in a style guided by external examples without copying them.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("{generated_tweet}")
    )
