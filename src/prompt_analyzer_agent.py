import os
from typing import Dict, List, Literal, TypedDict, Annotated, Union, Optional
from enum import Enum
import json
import logging

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph
# Removed the problematic import
# from langgraph.prebuilt import ToolExecutor
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the state for our graph
class GraphState(TypedDict):
    messages: List[BaseMessage]
    analysis: Optional[Dict]
    final_response: Optional[Dict]

# Define possible request types
class RequestType(str, Enum):
    CRYPTO_IMAGE = "crypto_image"
    CRYPTO_VIDEO = "crypto_video"
    NON_CRYPTO_MEDIA = "non_crypto_media"
    CRYPTO_INFO = "crypto_info"
    OTHER = "other"

# Define response types
class ResponseStrategy(str, Enum):
    GENERATE_CRYPTO_IMAGE = "generate_crypto_image"
    GENERATE_CRYPTO_VIDEO = "generate_crypto_video"
    SARCASTIC_RESPONSE = "sarcastic_response"
    SUGGEST_MEMECOINS = "suggest_memecoins"
    PROVIDE_CRYPTO_INFO = "provide_crypto_info"

# Functions for analyzing the request
analyze_request_schema = {
    "name": "analyze_request",
    "description": "Analyze a user's request to determine if it's about crypto and if it asks for image/video generation",
    "parameters": {
        "type": "object",
        "properties": {
            "request_type": {
                "type": "string",
                "enum": [rt.value for rt in RequestType],
                "description": "The type of request being made"
            },
            "is_crypto_related": {
                "type": "boolean",
                "description": "Whether the request is related to cryptocurrency, blockchain, or finance"
            },
            "wants_media": {
                "type": "boolean",
                "description": "Whether the user is asking for an image or video to be generated"
            },
            "media_type": {
                "type": "string",
                "enum": ["image", "video", "none"],
                "description": "The type of media being requested, if any"
            },
            "crypto_topic": {
                "type": "string",
                "description": "The specific cryptocurrency topic being discussed, if any"
            },
            "suggested_prompt": {
                "type": "string",
                "description": "A suggested prompt for generating media if applicable"
            }
        },
        "required": ["request_type", "is_crypto_related", "wants_media", "media_type"]
    }
}

determine_response_schema = {
    "name": "determine_response",
    "description": "Determine how to respond to the user's request based on analysis",
    "parameters": {
        "type": "object",
        "properties": {
            "response_strategy": {
                "type": "string",
                "enum": [rs.value for rs in ResponseStrategy],
                "description": "How to respond to the user's request"
            },
            "media_prompt": {
                "type": "string",
                "description": "If generating media, the prompt to use"
            },
            "text_response": {
                "type": "string",
                "description": "The text to include in the response"
            },
            "memecoin_suggestions": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of memecoin suggestions if applicable"
            }
        },
        "required": ["response_strategy"]
    }
}

class PromptAnalyzerAgent:
    """
    An agent that analyzes user prompts to determine if they're asking for crypto-related images/videos
    and generates appropriate responses using LangGraph.
    """
    
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the PromptAnalyzerAgent.
        
        Args:
            model_name: The model to use for analysis
            api_key: OpenAI API key (will use env var if None)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=self.api_key,
            temperature=0,
        )
        
        # Create the analysis model with function calling
        self.analysis_model = self.llm.bind_functions([analyze_request_schema])
        
        # Create the response model with function calling
        self.response_model = self.llm.bind_functions([determine_response_schema])
        
        # Build the LangGraph
        self.graph = self._build_graph()
        logging.info("Graph built")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for the prompt analyzer."""
        
        # Node for analyzing the request
        def analyze_request(state: GraphState) -> GraphState:
            """Analyze the user's request to determine intent and content."""
            messages = state["messages"]
            
            # Create a system prompt for analysis
            system_prompt = """
            You are an expert cryptocurrency analyst with knowledge of all cryptocurrencies, tokens, blockchain projects, and financial markets.
            Your job is to analyze user requests to determine:
            1. If they are asking about cryptocurrency, blockchain, or financial topics
            2. If they are requesting image or video generation
            3. What specific crypto topics they're interested in
            
            Focus on detecting mentions of specific cryptocurrencies (Bitcoin, Ethereum, etc.), 
            blockchain projects, tokens, market trends, trading, and financial metrics.
            
            Also determine if they're asking for visual representations through images or videos.
            """

            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                ("user", "Analyze this request and determine if it's crypto-related and if it asks for media generation.")
            ])
            
            response = self.analysis_model.invoke(analysis_prompt.format(messages=messages))
            analysis = response.additional_kwargs.get("function_call", {})
            logging.info(response)
            if analysis and "arguments" in analysis:
                try:
                    parsed_analysis = json.loads(analysis["arguments"])
                    logging.info(f"Request analysis: {parsed_analysis}")
                    return {"messages": state["messages"], "analysis": parsed_analysis}
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse analysis: {analysis['arguments']}")
            
            # Default analysis if parsing fails
            return {
                "messages": state["messages"], 
                "analysis": {
                    "request_type": RequestType.OTHER.value,
                    "is_crypto_related": False,
                    "wants_media": False,
                    "media_type": "none",
                    "crypto_topic": "",
                    "suggested_prompt": ""
                }
            }
        
        # Node for determining response strategy
        def determine_response(state: GraphState) -> GraphState:
            """Determine how to respond based on the analysis."""
            messages = state["messages"]
            analysis = state["analysis"]
            
            # Create a system prompt for response determination
            system_prompt = """
            You are an expert crypto agent that specializes in engaging with users about cryptocurrency topics.
            Based on the analysis of the user's request, determine the best way to respond:
            
            - If they're asking about crypto AND requesting an image → Generate a crypto-related image
            - If they're asking about crypto AND requesting a video → Generate a crypto-related video
            - If they're asking about crypto but NOT requesting media → Provide crypto information
            - If they're asking for media but NOT about crypto → Respond sarcastically and suggest they ask about crypto instead
            - If their request is neither about crypto nor media → Suggest some trending memecoins they might be interested in
            
            For sarcastic responses, be witty but not rude. For memecoin suggestions, include a mix of established and newer meme tokens.
            """
            
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                ("user", f"Based on this analysis: {json.dumps(analysis)}, determine how to respond.")
            ])
            
            response = self.response_model.invoke(response_prompt.format(messages=messages))
            response_strategy = response.additional_kwargs.get("function_call", {})
            logging.info(response)
            if response_strategy and "arguments" in response_strategy:
                try:
                    parsed_strategy = json.loads(response_strategy["arguments"])
                    logging.info(f"Response strategy: {parsed_strategy}")
                    return {
                        "messages": state["messages"],
                        "analysis": state["analysis"],
                        "final_response": parsed_strategy
                    }
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse response strategy: {response_strategy['arguments']}")
            
            # Default strategy if parsing fails
            return {
                "messages": state["messages"],
                "analysis": state["analysis"],
                "final_response": {
                    "response_strategy": ResponseStrategy.PROVIDE_CRYPTO_INFO.value,
                    "text_response": "I can help you with cryptocurrency information. What would you like to know?"
                }
            }
        
        # Define the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("analyze_request", analyze_request)
        workflow.add_node("determine_response", determine_response)
        
        # Define edges
        workflow.add_edge("analyze_request", "determine_response")
        workflow.set_entry_point("analyze_request")
        workflow.set_finish_point("determine_response")
        
        return workflow.compile()
    
    def analyze_prompt(self, user_prompt: str) -> Dict:
        """
        Analyze a user prompt to determine the intent and how to respond.
        
        Args:
            user_prompt: The user's message
            
        Returns:
            A dictionary with the analysis and response strategy
        """
        # Initialize the state with the user message
        initial_state = {
            "messages": [HumanMessage(content=user_prompt)],
            "analysis": None,
            "final_response": None
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "analysis": result["analysis"],
            "response": result["final_response"]
        }