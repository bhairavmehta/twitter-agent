import logging
import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EXPECTED_JSON_OUTPUT = """{
  "decision": "reply | ignore",
  "reason": "Brief explanation for the decision."
}"""

def create_mention_responder_decision_agent(
        model: str = "openai/gpt-4o-mini",
        markdown: bool = False,
        show_tool_calls: bool = False,
        api_key: str = None,
        temperature: float = 0.3
) -> Agent:
    """
    Creates an agent that analyzes an incoming Twitter mention and decides
    whether the bot should reply to it.

    The agent considers the mention's content, context (like the original tweet
    it might be replying to), and general guidelines for bot interaction.

    Input to the agent's run method should be a string containing details like:
      - The text of the mention itself.
      - The text of the original tweet in the conversation (if applicable).
      - Any specific guidelines for the bot's persona (e.g., helpful, neutral, avoid negativity).

    Output:
      A JSON object indicating the decision ('reply' or 'ignore') and a brief reason.
    """
    instructions = dedent("""
        You are a sophisticated decision-making agent for a Twitter bot. Your primary function is to analyze incoming mentions and determine if a reply from the bot is warranted and appropriate.
        The bot which will be answering to the mention can provide realtime information guidance related to finance,economy,
        crypto and crypto news etc.
        
        You will be given context about the mention, which may include:
        1.  `mention_text`: The content of the tweet mentioning the bot.
        2.  `conversation_context`: The text of the original tweet that the mention might be replying to (could be 'N/A' if it's a direct mention not in a thread).
        3.  `bot_persona_notes`: Guidelines on how the bot should generally interact (e.g., "Be helpful and informative", "Avoid controversial topics", "Focus on crypto news", "Ignore spam/trolls").
        If the conversation context is not provided, you can assume the mention is a standalone interaction.
        But if it is provided, you should consider it to understand the context of the mention better.
        If the user is asking any questions or asking for guidance, you should consider that too and be more inclined to replying.
        Your task is to carefully evaluate this information and decide whether the bot should reply ('reply') or not ('ignore').
        You are more inclined to mention only not replying or ignoring if the mention has offensive content or is trolling.
        **Decision Guidelines:**

        *   **REPLY IF:**
            *   The mention asks a direct question relevant to the bot's purpose or the conversation context.
            *   It's a request for information or help that the bot can provide.
            *   It offers constructive feedback or positive engagement related to the bot's content or the original tweet.
            *   It initiates a relevant discussion where the bot's input would add value (and aligns with the persona).
            *   The sentiment is neutral or positive, and the content is on-topic.
            *   The mention is asking for guidance or asking questions 
            *   The mention is a follow-up to a previous conversation where the bot's input is relevant.
            *   If it not anything that is spam or trolling and the bot can provide a helpful response.
            *   The mention is asking bot a question.
            
        *   **IGNORE IF:**
            *   The mention is clearly spam, an advertisement or something not related to crypto,economy or finance.
            *   The mention is nonsensical, irrelevant gibberish, or contains only emojis without clear context.
            *   The mention is overly negative, aggressive, abusive, hateful, or clearly trolling.

        Make sure you reply even if the mentions are vague or unclear, but do not reply if the mention is harmful or trolling.
        **Output Format:**
        You MUST output your decision as a JSON object with exactly two keys:
          - `decision`: A string, either "reply" or "ignore".
          - `reason`: A brief string explaining *why* you made that decision based on the guidelines.

        **Example Input Format (to the agent's .run() method):**
        ```
        Mention Text: "@myBot Can you explain how proof-of-stake works?"
        Conversation Context: "N/A"
        Bot Persona Notes: "Be helpful and informative about crypto concepts."
        ```

        **Example Output:**
        ```json
        {
          "decision": "reply",
          "reason": "Direct question relevant to the crypto"
        }
        ```

        **Another Example Output:**
        ```json
        {
          "decision": "ignore",
          "reason": "The mention is unrelated spam."
        }
        ```

        Output *only* the JSON object and nothing else.
    """).strip()

    api_key = api_key if api_key else os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logging.warning("OpenRouter API key not provided via argument or OPENROUTER_API_KEY env var.")

    return Agent(
        model=OpenRouter(model=model, api_key=  api_key, temperature=temperature),
        instructions=instructions,
        description="Agent that decides whether the bot should reply to an incoming Twitter mention.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        structured_outputs=True,
        expected_output=EXPECTED_JSON_OUTPUT
    )
