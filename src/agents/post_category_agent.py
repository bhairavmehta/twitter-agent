import logging
import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv

load_dotenv()


def create_post_selector_agent(
        model: str = "openai/gpt-4o-mini",
        markdown: bool = False,
        show_tool_calls: bool = False,
        api_key: str = None
) -> Agent:
    """
    Creates an agent that selects which account should post the next tweet.
    The schedule is as follows:

      A) 23 tweets a day from the following accounts:
           - WuBlockchain: 9 tweets
             Notes: keep the citation sources and use the provided prompt to redo the tweets.
           - aixbt_agent: 6 tweets
             Notes: redo tweets/threads in the same writing style as the provided prompt.
           - TheBlock__: 4 tweets
             Notes: do not link tweets to their website; instead, use images (steal the images if possible).
           - lookonchain: 2 tweets
             Notes: post with an image if available.
           - naiivememe: 2 tweets
             Notes: a meme account with humorous content.

      B) 365X.ai: 2 tweets a day

      C) Crypto Reward URL: 1 tweet a day

    In addition, you will be provided with the number of tweets remaining for each account.
    Based on this information, you must decide which account should post next.

    Your response must be in JSON format with exactly the following keys:
      - "username": the Twitter handle (or identifier) of the account to post next.
      - "category": the category of the tweet, which should be one of:
           "company" (for WuBlockchain, aixbt_agent, TheBlock__, lookonchain),
           "meme" (for naiivememe),
           "crypto_url" (for Crypto Reward URL),
           "crypto_only" (for 365X.ai).

    Output only the JSON object without any additional text or formatting.
    """
    instructions = dedent("""
            You are a tweet post selection agent. Your task is to analyze the provided schedule information along with the number of tweets remaining for each account, and decide which account should post next.

            The daily schedule for tweets is structured as follows:
            Crypto News & Insights (crypto_only category)
                WuBlockchain: 8 tweets per cycle
                lookonchain: 2 tweets per cycle
            Meme & Humor (meme category)
                naiivememe: 1 tweet per cycle
            Company Updates (company category)
                365X.ai: 1 tweet per cycle

            You will receive the current remaining tweet counts for each account for the day.
            You will also receive the username of the account that posted the `last_tweet` (this can be 'nan' if it's the start of the day).

            Using this data, decide which account should post next *right now*.

            Selection Rules:
            1.  **Eligibility:** You must choose an account for which posts are still left (remaining count > 0).
            2.  **Single Category/Account Left:**
                *   If only one **account** across all categories has posts remaining, you *must* select that account, even if it was the `last_tweet`.
                *   If posts remain only within a **single category** (but potentially for multiple accounts in that category), you *must* select an account from that category. Apply weighted random selection (Rule 3) and the 'no consecutive' rule (Rule 4) *within* that category if possible.
            3.  **Weighted Random Selection:** When multiple accounts across different categories have posts remaining (and Rule 2 doesn't apply), the selection must be random. This randomness should be weighted based on the remaining tweet counts and daily limits. Accounts with higher remaining counts (like WuBlockchain) have a proportionally higher chance of being selected, but the process should ensure variety. The goal is an unpredictable sequence that respects the tweet limits over time, not strictly sequential posting (e.g., avoid WuBlockchain -> WuBlockchain -> WuBlockchain just because it has many posts left; aim for sequences like WuBlockchain -> lookonchain -> WuBlockchain -> naiivememe -> WuBlockchain...).
            4.  **No Consecutive Posts (General Case):** The chosen account should *not* be the same as the `last_tweet` account, *unless* it's the only account with posts remaining (as covered in Rule 2a).
            5.  **WuBlockchain Specifics:** While WuBlockchain has a high quota and thus a higher chance of being selected frequently, avoid selecting it twice *consecutively* unless absolutely necessary per Rule 2a. The weighted randomness (Rule 3) should allow it to appear often but interspersed with other accounts.

            Your output must be a JSON object with exactly the following keys:
              - "username": the Twitter handle or identifier of the account to post next.
              - "category": one of "company", "meme", "crypto_url", or "crypto_only". (Ensure you use the correct category from the schedule for the selected username).
            Output only the JSON object and nothing else.
        """).strip()

    return Agent(
        model=OpenRouter(id=model, api_key=api_key, temperature=0.8),
        instructions=[instructions],
        description="Agent that selects which account should post the next tweet based on schedule information and remaining tweet counts.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        structured_outputs=True,
        expected_output=dedent("""{"username": "", "category": ""}""")
    )
