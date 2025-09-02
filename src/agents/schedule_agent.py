import logging
from textwrap import dedent
from phi.agent import Agent
from tools.schedule_tool import ScheduleTool
from phi.model.openrouter import OpenRouter
import os
from dotenv import load_dotenv
from phi.tools.googlesearch import GoogleSearch
load_dotenv()
def create_schedule_agent(model:str ="openai/gpt-4o", schedule_tool:ScheduleTool = None,
                          markdown: bool= False,show_tool_calls: bool= False,
                          cg_demo_api_key:str=None,cg_api_key: str=None) -> Agent:
    if schedule_tool is None:
        logging.error("No scheduling tool provided; please supply a valid scheduling tool instance.")
    return Agent(
        model=OpenRouter(id=model,api_key=os.getenv("OPENROUTER_API_KEY")),
        tools=[schedule_tool,GoogleSearch()],
        instructions=[
            (
                "You are a scheduling agent responsible for generating new schedule entries for crypto events and news posts. "
                "Your process follows these steps:"
    
                "1. Retrieve all existing posts using the provided scheduling tool (`get_all_events_str`). Ensure "
                "you have a complete list of current posts before proceeding."
                "2. Review the retrieved posts and extract only new and unique updates that are not already scheduled. "
                "Make sure each post is distinct and that no two posts focus on the same coin."
                "3. Before scheduling a post, confirm all critical information, including price and other relevant data, "
                "using the provided tools. If any information is unclear or outdated, do not add the post to the schedule."
                "but if you do get the corrected info you can use that information to schedule the post"
                "4. When creating new posts, prioritize trending coins while ensuring that no more than two coins are "
                "included per post. Keep posts concise and relevant."
                "5. After verification, schedule the posts one by one, ensuring that each is accurate and meets all criteria."
                "6. Finally, return a summary of the newly scheduled posts."
                "You can use queries like this for getting coin price doge coin price site:coinbase.com"
                "Always use the available tools to verify all information before scheduling a post to maintain accuracy and avoid outdated data."
            )
        ],
        description="Agent that manages the scheduling of events and posts and add as many posts as user asks for "
                    "in case there is not much content for scheduling the posts you can decrease the number of posts.",
        show_tool_calls=show_tool_calls,
        markdown=markdown,
        expected_output=dedent("""\
            Number of posts added: {num_posts}
            Post summary: {post_summary}
        """)
    )
