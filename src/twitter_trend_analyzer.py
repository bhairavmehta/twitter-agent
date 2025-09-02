import os
from textwrap import dedent
from phi.agent import Agent
from phi.model.openrouter import OpenRouter
from dotenv import load_dotenv
from glob import glob

load_dotenv()

class TrendAnalyzerAgent:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        docs_path: str = "docs",
        markdown: bool = False,
        show_tool_calls: bool = False
    ):
        """
        Initialize the Trend Analyzer Agent.

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
                You are a trend analyzer agent. Your task is to read all the provided documents in your knowledge base 
                and extract key trends, patterns, and significant insights regarding the crypto market. Analyze the content carefully and rely solely on the data provided.

                Provide a clear, concise summary labeled 'Trend Analysis Summary' that includes:
                  • An overview of the key trends observed.
                  • Notable events, statistics, or market shifts.
                  • Actionable insights if available.

                If the documents do not provide sufficient information, state that clearly.
                """)
            ],
            description="An agent that analyzes provided documents to extract and summarize crypto market trends.",
            show_tool_calls=show_tool_calls,
            markdown=markdown,
            expected_output=dedent("""\
                {trend_summary}
            """)
        )

        # Load documents as context.
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

    def analyze_trends(self) -> str:
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
