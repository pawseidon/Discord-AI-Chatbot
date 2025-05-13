import os
import asyncio
import json
import aiohttp
from typing import Dict, List, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.tavily_search import TavilySearchResults
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from bot_utilities.config_loader import config
from bot_utilities.ai_utils import get_crypto_price, search_internet

# Initialize the GROQ model
def get_groq_llm():
    api_key = os.environ.get("API_KEY")
    model_name = config.get('MODEL_ID', 'meta-llama/llama-4-scout-17b-16e-instruct')
    
    return ChatGroq(
        api_key=api_key,
        model_name=model_name
    )

# Define our tools

# Web search tool 
def create_search_tool():
    """Create a web search tool using either Tavily (if API key exists) or DuckDuckGo"""
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    
    if tavily_api_key:
        # Use Tavily if API key is available
        return TavilySearchResults(max_results=5)
    else:
        # Fall back to DuckDuckGo which doesn't require an API key
        return DuckDuckGoSearchRun()

# Use your existing search function from ai_utils.py
async def search_internet_sync(query: str) -> str:
    """Perform internet search using DuckDuckGo."""
    result = await search_internet(query)
    return result

# Function to run async functions synchronously (for LangChain tools)
def run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

# Wrap the async function to be used in a LangChain tool
def search_internet_wrapper(query: str) -> str:
    return run_async(search_internet_sync(query))

# Define more specialized tools
class WeatherInput(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")

# Weather tool (this could be expanded with a real API)
def get_weather(location: str) -> str:
    """Get the current weather in a given location"""
    # This is a placeholder - you would integrate with a real weather API
    return f"It's currently sunny and 72 degrees in {location}."

# Crypto price tool that uses your existing get_crypto_price function
async def get_crypto_price_sync(crypto_name: str) -> str:
    """Get real-time cryptocurrency price information."""
    # Use your existing crypto price function
    loop = asyncio.get_event_loop()
    price_info = await get_crypto_price(crypto_name)
    if price_info:
        return price_info
    return f"Could not find price data for {crypto_name}"

# Wrap the async function to be used in a LangChain tool
def get_crypto_price_wrapper(crypto_name: str) -> str:
    return run_async(get_crypto_price_sync(crypto_name))

def create_tools():
    # Use search tool factory that handles both options
    search_tool = create_search_tool()
    
    # Also add our own search tool based on the bot's existing functionality
    custom_search_tool = Tool(
        name="InternetSearch",
        description="Search the internet for current information. Useful for questions about recent events, current prices, or factual information.",
        func=search_internet_wrapper,
    )
    
    # Define other tools
    weather_tool = Tool(
        name="Weather",
        description="Get the current weather in a location",
        func=get_weather,
    )
    
    # Add crypto price tool
    crypto_tool = Tool(
        name="CryptoPrice",
        description="Get current cryptocurrency prices. Input should be a cryptocurrency name or symbol like 'bitcoin', 'btc', 'ethereum', etc.",
        func=get_crypto_price_wrapper,
    )
    
    return [search_tool, custom_search_tool, weather_tool, crypto_tool]

# Create the agent
def create_agent():
    llm = get_groq_llm()
    tools = create_tools()
    
    # Define the agent prompt
    prompt = PromptTemplate.from_template(
        """You are an intelligent Discord bot assistant that can help with various tasks.
        
        You have access to the following tools:
        
        {tools}
        
        Use the following format:
        
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        
        Begin!
        
        Question: {input}
        Thought: """
    )
    
    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

# Function to run the agent (async for Discord)
async def run_agent(query: str) -> str:
    """Run the agent with the given query"""
    agent_executor = create_agent()
    
    # Run the agent in a thread to avoid blocking Discord
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: agent_executor.invoke({"input": query}))
    
    # Extract the response
    if "output" in result:
        return result["output"]
    return "I couldn't process that request. Please try again." 