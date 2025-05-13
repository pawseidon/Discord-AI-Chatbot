import os
import asyncio
import json
import aiohttp
from typing import Dict, List, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from bot_utilities.config_loader import config
from bot_utilities.ai_utils import get_crypto_price, search_internet
from bot_utilities.memory_utils import ConversationMemory
from bot_utilities.rag_utils import RAGSystem
from bot_utilities.token_utils import token_optimizer

# Initialize memory system
conversation_memory = ConversationMemory()
server_knowledge_bases = {}

def get_server_rag(server_id: str) -> RAGSystem:
    """Get or create a RAG system for the server"""
    if server_id not in server_knowledge_bases:
        server_knowledge_bases[server_id] = RAGSystem(server_id)
    return server_knowledge_bases[server_id]

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
    # Optimize the search result to reduce tokens
    if result:
        result = token_optimizer.clean_text(result)
        result = token_optimizer.truncate_text(result, max_tokens=1000)
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

# Knowledge base query tool
async def query_knowledge_base_sync(server_id: str, query: str) -> str:
    """Query the server's knowledge base"""
    # Get the RAG system for this server
    rag_system = get_server_rag(server_id)
    
    # Query the knowledge base
    results = await rag_system.query(query, k=3)
    
    if not results:
        return "No relevant information found in the knowledge base."
    
    # Format results
    response = "Knowledge base results:\n\n"
    for i, doc in enumerate(results):
        # Optimize content to reduce tokens
        content = token_optimizer.clean_text(doc.page_content)
        content = token_optimizer.truncate_text(content, max_tokens=500)
        
        response += f"[Document {i+1}]:\n{content}\n\n"
        if "source" in doc.metadata:
            response += f"Source: {doc.metadata['source']}\n"
    
    return response

# Wrapper for knowledge base query tool
def query_knowledge_base_wrapper(server_id: str, query: str) -> str:
    return run_async(query_knowledge_base_sync(server_id, query))

def create_tools(server_id=None):
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
    
    tools = [search_tool, custom_search_tool, weather_tool, crypto_tool]
    
    # Add knowledge base tool if server_id is provided
    if server_id:
        kb_tool = Tool(
            name="KnowledgeBase",
            description="Query the server's knowledge base for information that has been stored by users. Use this for server-specific information before searching the internet.",
            func=lambda query: query_knowledge_base_wrapper(server_id, query),
        )
        tools.append(kb_tool)
    
    return tools

# Create the agent
def create_agent(user_memory="", kb_context="", server_id=None):
    llm = get_groq_llm()
    tools = create_tools(server_id)
    
    # Optimize memory and KB context to reduce tokens
    if user_memory:
        user_memory = token_optimizer.optimize_memory(user_memory, max_tokens=800)
    
    if kb_context:
        kb_context = token_optimizer.truncate_text(kb_context, max_tokens=1000)
    
    # Define the agent prompt with memory context and knowledge base context
    prompt = PromptTemplate.from_template(
        """You are an intelligent Discord bot assistant that can help with various tasks.
        
        {memory}
        
        {kb_context}
        
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
        handle_parsing_errors=True,
        max_iterations=5  # Limit iterations to save tokens
    )
    
    return agent_executor

# Function to run the agent (async for Discord)
async def run_agent(query: str, user_id: str = None, channel_id: str = None, server_id: str = None) -> str:
    """Run the agent with the given query"""
    user_memory = ""
    kb_context = ""
    
    # If user_id is provided, get their conversation history
    if user_id:
        user_memory = await conversation_memory.format_memory_for_context(user_id)
    
    # If server_id is provided, query the knowledge base for relevant context
    if server_id:
        rag_system = get_server_rag(server_id)
        results = await rag_system.query(query, k=2)
        if results:
            kb_context = await rag_system.format_results_as_context(results)
    
    # Create agent with memory and knowledge base context
    agent_executor = create_agent(user_memory=user_memory, kb_context=kb_context, server_id=server_id)
    
    # Run the agent in a thread to avoid blocking Discord
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: agent_executor.invoke({
        "input": query,
        "memory": user_memory,
        "kb_context": kb_context
    }))
    
    # Extract the response
    if "output" in result:
        response = result["output"]
        
        # Store the interaction in memory if user_id and channel_id are provided
        if user_id and channel_id:
            await conversation_memory.store_interaction(user_id, channel_id, query, response)
            
        return response
    
    return "I couldn't process that request. Please try again." 