import os
from phi.utils.log import logger
from twitter_trend_analyzer import TrendAnalyzerAgent
from retrieval_agent import RetrievalAgent
from agents.trending_crypto_agent import create_trending_crypto_agent
from agents.deep_coin_info_agent import create_deep_coin_info_agent
from agents.company_info_agent import create_company_info_agent
from agents.schedule_agent import create_schedule_agent
from scheduler import ScheduleManager,PollScheduleManager
from tools.schedule_tool import ScheduleTool
from dotenv import load_dotenv
from agents.poll_scheduler_agent import create_poll_scheduler_agent
from tools.poll_scheduler_tool import PollSchedulerTool
from best_tweet_finder import BestTweetFinderAgent
load_dotenv()

class CryptoNewsWorkFlow:
    def __init__(
            self,
            poll_scheduler: PollScheduleManager,
            scheduler: ScheduleManager,
            model: str = "openai/gpt-4o",
            number_of_posts: int = 2,
            retriever: RetrievalAgent = None,
            docs_path: str = "docs",
            scheduler_model: str = "openai/gpt-4o-mini",
            analyzer_model: str = "google/gemini-2.0-flash-001"
    ) -> None:
        """
        Initializes the CryptoNewsWorkflow.

        :param model: The model identifier (e.g. "openai/gpt-4o-mini").
        :param scheduler: Instance of ScheduleManager.
        :param number_of_posts: Number of posts to schedule (default is 2).
        :param retriever: this agent will be used to retrieve relevant information
        :param scheduler_model: this model will be used to schedule the posts should be a better model
        """
        self.model = model
        self.scheduler = scheduler
        self.scheduler_model = scheduler_model
        self.number_of_posts = number_of_posts
        self.poll_scheduler = poll_scheduler
        self.retrieval_agent = retriever
        self.analyzer_model = analyzer_model
        self.schedule_tool = ScheduleTool(schedulemanager=self.scheduler)
        self.poll_scheduler_tool = PollSchedulerTool(schedule_manager=self.poll_scheduler)
        self.poll_agent = create_poll_scheduler_agent(poll_scheduler_tool=self.poll_scheduler_tool)
        self.trend_analyzer_agent = TrendAnalyzerAgent(
            model=self.analyzer_model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            docs_path=docs_path,
            markdown=False,
            show_tool_calls=False
        )
        self.tweet_finder_agent = BestTweetFinderAgent(
            model=self.analyzer_model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            docs_path=docs_path,
            markdown=False,
            show_tool_calls=False
        )
        self.trending_crypto_agent = create_trending_crypto_agent(model=self.model,
                                                                  demo_api_key=os.getenv("COINGECKO_DEMO_API_KEY"))
        self.deep_coin_info_agent = create_deep_coin_info_agent(model=self.model,exa_api_key=os.getenv("EXA_API_KEY"))
        self.company_info_agent = create_company_info_agent(model=self.model)
        self.schedule_agent = create_schedule_agent(
            model=self.scheduler_model,
            schedule_tool=self.schedule_tool,
            show_tool_calls=True,
            cg_demo_api_key=os.getenv("COINGECKO_DEMO_API_KEY")
        )

    def run(self) -> dict:
        """
        Executes the workflow:
          1. Analyzes overall trends from documents.
          2. Retrieves trending crypto information based on the trend analysis.
          3. Retrieves detailed coin info using the trend insights.
          4. Retrieves positive company info.
          5. Uses the trend analysis to fetch additional context via the retrieval agent.
          6. Schedules posts using the aggregated context.

        :return: A dictionary containing the final aggregated output.
        """

        # Step 1: Analyze overall trends from documents.
        """logger.info("Step 1: Analyzing overall trends from documents...")
        trend_analysis = self.trend_analyzer_agent.analyze_trends()
        logger.info(f"Trend Analysis: {trend_analysis}")"""

        # Step 2: Retrieve trending crypto info using the trend analysis as input  and a tweet that we will take from
        # them.
        logger.info("Step 2: Retrieving trending crypto information based on trend analysis...")
        trending_response = self.trending_crypto_agent.run(
            message=f"provide current trending crypto market data."
        )
        logger.info(f"Trending Response: {trending_response.content}")
        trending_data = trending_response.content



        # Step 3: Retrieve detailed coin information using the trend analysis.
        logger.info(f"Step 3: Retrieving detailed info for coins mentioned in trends: {trending_response.content}")
        deep_info_response = self.deep_coin_info_agent.run(
            message=f"Get detailed info and trending data for coin(s) mentioned in: {trending_response.content}"
        )
        logger.info(f"Deep Coin Info: {deep_info_response.content}")

        # Step 4: Retrieve positive company information (kept as is).
        logger.info("Step 4: Retrieving positive company info for '365x.ai'...")
        company_response = self.company_info_agent.run(message="Get positive info about 365x.ai")
        logger.info(f"Company Info: {company_response.content}")

        # Step 5: Retrieve additional context based on the trend analysis.
        """logger.info("Step 5: Retrieving additional context based on trend analysis...")
        retrieval_info = self.retrieval_agent.query(trending_response.content)
        logger.info(f"Retrieval Info: {retrieval_info}")"""

        tweet_to_add = self.tweet_finder_agent.get_best_tweet()
        logger.info(f"Tweet to add: {tweet_to_add}")
        # Step 6: Schedule posts and polls using the aggregated context.
        aggregated_context = {
            #"trend_analysis": trend_analysis,
            "trending": trending_data,
            "deep_coin_info": deep_info_response.content,
            "company_info": company_response.content,
            #"retrieval_info": retrieval_info
        }
        logger.info(f"Step 6: Scheduling {self.number_of_posts} posts with aggregated context: {aggregated_context}")
        schedule_response = self.schedule_agent.run(
            message=f"Schedule posts using the following aggregated context: {aggregated_context}"
                    f" for {self.number_of_posts} posts."
                    f"Also add this tweet to the scheduled posts {tweet_to_add} but after verifying its prices and rates etc"
                    f"Make sure you are also making the post for 365x.ai using "
                    f"{company_response.content} you may use the given info as it is to make the post"
        )
        logger.info(f"Schedule Response: {schedule_response.content}")
        poll_response = self.poll_agent.run(
            message =f"Schedule a poll using the following aggregated context: "
                     f"trend_analysis: {trending_response.content} trending_data: {trending_data}"
        )
        final_output = {
            #"trend_analysis": trend_analysis,
            "trending": trending_data,
            "deep_coin_info": deep_info_response.content,
            "company_info": company_response.content,
            #"retrieval_info": retrieval_info,
            "scheduled": schedule_response.content,
            "polls" :poll_response.content
        }
        return final_output


    def update_context(self):
        self.trend_analyzer_agent.update_context()
        self.tweet_finder_agent.update_context()


if __name__ == "__main__":
    # Initialize required components
    poll_scheduler = PollScheduleManager()
    scheduler = ScheduleManager()
    retriever = RetrievalAgent()

    # Instantiate the workflow
    crypto_workflow = CryptoNewsWorkFlow(
        poll_scheduler=poll_scheduler,
        scheduler=scheduler,
        retriever=retriever,
        number_of_posts=4
    )

    # Run the workflow
    output = crypto_workflow.run()

    # Print the output
    print("Final Output:")
    print(scheduler.get_all_pending())
