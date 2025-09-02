import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv
from glob import glob

load_dotenv()

class BestTweetFinderAgent:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        docs_path: str = "docs",
        markdown: bool = False,
        show_tool_calls: bool = False
    ):
        """
        Initialize the Best Tweet Finder Agent:.

        Args:
            model: The LLM model to use (e.g., "gpt-4o-mini").
            api_key: API key for OpenRouter (if None, loaded from environment).
            docs_path: Folder path where text documents reside.
            markdown: Whether to return markdown formatted output.
            show_tool_calls: Whether to display tool call details.
        """
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
        self.docs_path = docs_path
        self.api_key = api_key

        # Create an agent with clear instructions for trend analysis.
        self.agent = Agent(
            model=OpenRouter(id=model, api_key=self.api_key),
            instructions=[
                dedent("""
                You are a tweet selector agent. Your task is to retrieve the latest trending crypto market data 
                and analyze recent tweets about the top trending coin. Then, based solely on tweet content , 
                select the single best tweet that is most relevant for the current market situation.
                And a tweet that is insightful and engaging for the audience.

                Provide only the selected tweet text as your final output in the following format:
                Do not include any additional commentary or explanation.
                """)
            ],
            description="An agent that analyzes provided documents to extract and summarize crypto market trends.",
            show_tool_calls=show_tool_calls,
            markdown=markdown,
            expected_output=dedent("""\
                {trend_summary}
            """)
        )

        self.context = self.load_documents()

    def load_documents(self) -> str:
        """
        Read all .txt files from the fixed docs_path and concatenate their contents.
        """
        all_text = ""
        if not os.path.exists(self.docs_path):
            print(f"Documents path '{self.docs_path}' does not exist.")
            return all_text

        file_paths = glob(os.path.join(self.docs_path, "**", "*.txt"), recursive=True)
        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    all_text += f.read() + "\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        return all_text

    def update_context(self):
        """
        Reload documents from the docs_path and update the internal context.
        """
        print("Updating context from documents...")
        self.context = self.load_documents()
        print("Context updated.")

    def get_best_tweet(self) -> str:
        """
        Build a prompt using the current context and invoke the agent to produce a trend analysis summary.
        """
        prompt_template = dedent("""
        Read the following context and extract the key trends and insights regarding the crypto market:
        dont mention any @ or # just give the statistical information present in it
        {context}
        """)
        prompt = prompt_template.format(context=self.context)
        response = self.agent.run(prompt)
        return response.content