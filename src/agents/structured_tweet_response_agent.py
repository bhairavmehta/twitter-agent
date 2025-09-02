import logging
import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv

load_dotenv()


def create_structured_response_agent(model: str = "openai/gpt-4o",
                                     markdown: bool = False,
                                     show_tool_calls: bool = False,
                                     api_key: str = None) -> Agent:
    """
    Creates an agent that analyzes a tweet mention and produces a structured JSON response.
    The output JSON contains:
      - type: "video", "image", or "normal"
      - prompt: a creative prompt if the type is "video" or "image" (empty string for "normal")
      - message: the text for the post reply
    """
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    instructions = dedent(f"""
        You are a tweet response composer agent. Your task is to generate a structured JSON output based on the provided tweet mention text.
        You want to increase the hype for the company you are working for which is 365x.ai and you have to make sure that 
        the response you will create will be interactive too.
        The output must be in JSON format and include exactly the following keys:
          - "type": either "video", "image", or "normal" or "no_reply" if the mention is harmfull or rude
          . "video", "image" only if asked for
          - "prompt": a creative prompt to be used if the type is "video" or "image". Leave as an empty string if not applicable.
          - "message": the text that will be used in the reply post.

        Guidelines:
          1. Analyze the mention text carefully.
          2. If the mention text suggests that a video should be created (for example, using keywords like "video", "clip", "watch"), then set "type" to "video" and include a relevant creative prompt in the "prompt" field.
          3. If the mention text implies that an image would be more appropriate (for example, using keywords like "photo", "picture", "image"),then set "type" to "image" and include a creative prompt.
          4. If neither a video nor an image is needed, set "type" to "normal" and leave the "prompt" field empty.
          5. Ensure that the "message" is engaging, concise, and directly relevant to the mention text.
          6. If the mention text is harmful or rude, set "type" to "no_reply"
          7. Output only the JSON object in the exact following format:

             {{
               "type": "<video/image/normal/no_reply>",
               "prompt": "<creative prompt if applicable, else empty>",
               "message": "<post reply text>"
             }}

        Do not include any additional text or formatting outside of the JSON object.
    """).strip()

    return Agent(
        model=OpenRouter(id=model, api_key=api_key, temperature=0.4),
        instructions=[instructions],
        description="Social media manager agent that composes tweet responses in a structured JSON format, including type, prompt, and message.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        structured_outputs=True,
        expected_output=dedent("""{"type": "", "prompt": "", "message": ""}""")
    )

