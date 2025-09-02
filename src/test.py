import logging
import os
from textwrap import dedent
from phi.agent import Agent
from phi.tools.newspaper_tools import NewspaperTools
from phi.tools.exa import ExaTools
from phi.tools.googlesearch import GoogleSearch
from phi.model.openrouter import OpenRouter
from tools.cg_tool import PhiCoinGeckoTool
from dotenv import load_dotenv
from datetime import datetime, timezone
load_dotenv()
def create_deep_coin_info_agent(model:str ="openai/gpt-4o-mini", exa_api_key: str = "",
                                markdown: bool= False,show_tool_calls: bool= False,
                                cg_api_key:str=None,cg_demo_api_key:str=None) -> Agent:
    if cg_api_key:
        cg_tool = PhiCoinGeckoTool(api_key=cg_api_key)
    elif cg_demo_api_key:
        cg_tool = PhiCoinGeckoTool(demo_api_key=cg_demo_api_key)
    else :
        if os.getenv("COINGECKO_API_KEY"):
            cg_tool = PhiCoinGeckoTool(api_key=os.getenv("COINGECKO_API_KEY"))
        elif os.getenv("COINGECKO_DEMO_API_KEY"):
            cg_tool = PhiCoinGeckoTool(demo_api_key=os.getenv("COINGECKO_DEMO_API_KEY"))
        else :
            logging.log(logging.ERROR, "No COINGECKO API key available in .env")
            cg_tool = PhiCoinGeckoTool()
    if exa_api_key:
        exa_tool = ExaTools(api_key=exa_api_key)
    else:
        logging.error("No ExaTools API key provided; attempting to load from .env")
        from dotenv import load_dotenv
        load_dotenv()
        if os.getenv("EXA_API_KEY"):
            exa_tool = ExaTools(api_key=os.getenv("EXA_API_KEY"))
        else:
            logging.error("No EXA_API_KEY found in .env; using default ExaTools instance")
            exa_tool = ExaTools(api_key="")
    newspaper_tool = NewspaperTools()
    current_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return Agent(
        model=OpenRouter(id=model,api_key=os.getenv("OPENROUTER_API_KEY")),
        tools=[exa_tool,GoogleSearch(),cg_tool],
        instructions=[
            (
                "You are a coin price getting agent you have to get us price for the latest price you can get  "
                "Always use the google search tool and search coin id which then you will use to get information about"
                "the coin to get its information with the coingecko tool "
                "should always be for usd price and usd only"
                f"For you reference today's date and time is {current_date_time}"
                f"Make sure you dont give conflicting prices rather just one price which and backed with information "
            )
        ],
        description="Agent that consolidates detailed coin information and updates and also verifies the info given",
        show_tool_calls=True,
        markdown=True,
        expected_output=dedent("""\
            Crypto Info:
            Trending Updates: {trending_updates}
            Extra Information:
            - Info 1: {source1_info}
            - Info 2: {source2_info}
            - Info 3: {source3_info}
        """)

    )

agent  = create_deep_coin_info_agent(exa_api_key=os.getenv("EXA_API_KEY"))
ans = agent.run(message="Get me price for dawg ai coin")
print(ans.content)