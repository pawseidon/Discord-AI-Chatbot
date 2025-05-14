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
from bot_utilities.rag_utils import RAGSystem, get_server_rag
from bot_utilities.token_utils import TokenOptimizer
from dotenv import load_dotenv
import time
import traceback

# Initialize memory system
conversation_memory = ConversationMemory()
server_knowledge_bases = {}

# Initialize token optimizer for reducing token usage
token_optimizer = TokenOptimizer()

# Environment variables
load_dotenv()

class AgentTools:
    """Collection of agent tools with async interfaces for seamless integration with reasoning systems"""
    
    def __init__(self):
        """Initialize the agent tools"""
        self.token_optimizer = TokenOptimizer()
        self.tool_capabilities = {
            'search_web': 'Search the internet for current information',
            'retrieve_knowledge': 'Access stored knowledge from the knowledge base',
            'analyze_sentiment': 'Analyze sentiment and emotions in text',
            'get_crypto_price': 'Retrieve cryptocurrency price information',
            'analyze_image': 'Analyze and describe image content',
            'generate_image': 'Create images from text descriptions',
            'transcribe_audio': 'Convert speech audio to text',
            'summarize_text': 'Create concise summaries of longer texts'
        }
        
    async def search_web(self, query: str, max_results: int = 5) -> str:
        """
        Search the web for information related to the query.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            str: Formatted search results
        """
        try:
            # Record start time for performance tracking
            start_time = time.time()
            
            # Call the search_internet function with fallback mechanisms
            raw_results = await search_internet(query)
            
            # If no results, try a different query formulation
            if not raw_results or len(raw_results.strip()) < 50:
                # Try a reformulated query
                reformulated_query = f"detailed information about {query}"
                raw_results = await search_internet(reformulated_query)
            
            # Clean and optimize results
            if raw_results:
                # Clean text formatting
                clean_results = self.token_optimizer.clean_text(raw_results)
                
                # Truncate to reasonable size
                truncated_results = self.token_optimizer.truncate_text(
                    clean_results, 
                    max_tokens=1000
                )
                
                # Format results for readability
                formatted_results = self._format_search_results(truncated_results, query)
                
                # Log performance
                duration = time.time() - start_time
                print(f"Web search for '{query}' completed in {duration:.2f} seconds")
                
                return formatted_results
            else:
                return f"I couldn't find specific information about '{query}' on the web."
                
        except Exception as e:
            print(f"Error in search_web: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            return f"I encountered an error while searching for '{query}'. Please try a more specific query or try again later."
    
    def _format_search_results(self, results: str, query: str) -> str:
        """Format search results for readability"""
        if not results or len(results.strip()) < 30:
            return f"No significant information found for: {query}"
            
        # Format for better readability with reasoning indicators
        formatted = f"🔍 Web search results for: '{query}'\n\n"
        
        # Split into paragraphs/sections if available
        paragraphs = results.split('\n\n')
        
        # Limit the number of paragraphs
        limited_paragraphs = paragraphs[:5]  # Limit to 5 paragraphs
        
        # Add each paragraph with formatting
        for i, para in enumerate(limited_paragraphs):
            if para.strip():
                formatted += f"{para.strip()}\n\n"
        
        # Add source information reminder
        formatted += "Note: This information was retrieved from web searches and may not be completely up-to-date."
        
        return formatted
    
    async def retrieve_knowledge(self, 
                               query: str, 
                               server_id: str, 
                               max_results: int = 3,
                               include_reasoning: bool = True) -> str:
        """
        Retrieve information from the server's knowledge base.
        
        Args:
            query: The search query
            server_id: The server ID to retrieve knowledge from
            max_results: Maximum number of results to return
            include_reasoning: Whether to include reasoning indicators
            
        Returns:
            str: Formatted knowledge base results
        """
        try:
            # Get the RAG system for this server
            rag_system = get_server_rag(server_id)
            
            # Query the knowledge base
            results = await rag_system.query(query, k=max_results)
            
            if not results:
                return "No relevant information found in the knowledge base."
            
            # Format results
            prefix = "📚 " if include_reasoning else ""
            response = f"{prefix}Knowledge base results:\n\n"
            for i, doc in enumerate(results):
                # Optimize content to reduce tokens
                content = self.token_optimizer.clean_text(doc.page_content)
                content = self.token_optimizer.truncate_text(content, max_tokens=500)
                
                response += f"[Document {i+1}]:\n{content}\n\n"
                if "source" in doc.metadata:
                    response += f"Source: {doc.metadata['source']}\n"
            
            return response
            
        except Exception as e:
            print(f"Error in retrieve_knowledge: {e}")
            return "I encountered an error while querying the knowledge base. Please try again later."
    
    async def analyze_sentiment(self, text: str, include_reasoning: bool = True) -> str:
        """
        Analyze the sentiment of the provided text.
        
        Args:
            text: The text to analyze
            include_reasoning: Whether to include reasoning indicators
            
        Returns:
            str: Sentiment analysis results
        """
        try:
            # Handle import here to avoid circular dependencies
            from bot_utilities.sentiment_utils import analyze_text_sentiment
            
            # Get sentiment analysis
            sentiment_result = await analyze_text_sentiment(text)
            
            # Add reasoning indicator if requested
            if include_reasoning:
                sentiment_result = f"😊 Sentiment Analysis: {sentiment_result}"
                
            return sentiment_result
            
        except Exception as e:
            print(f"Error in analyze_sentiment: {e}")
            return f"I encountered an error while analyzing sentiment: {str(e)}"
    
    async def get_crypto_price_async(self, crypto_name: str, include_reasoning: bool = True) -> str:
        """
        Get cryptocurrency price information.
        
        Args:
            crypto_name: Name or symbol of the cryptocurrency
            include_reasoning: Whether to include reasoning indicators
            
        Returns:
            str: Formatted price information
        """
        try:
            # Use existing crypto price function
            price_info = await get_crypto_price(crypto_name)
            
            # Add reasoning indicator if requested
            if include_reasoning and price_info:
                price_info = f"🔢 Cryptocurrency Price: {price_info}"
                
            if price_info:
                return price_info
            return f"Could not find price data for {crypto_name}"
            
        except Exception as e:
            print(f"Error in get_crypto_price_async: {e}")
            return f"I encountered an error while fetching the price for {crypto_name}. Please try again later."
            
    def get_tool_capabilities(self) -> Dict[str, str]:
        """Get a mapping of available tools and their capabilities"""
        return self.tool_capabilities
        
    def get_appropriate_tools_for_reasoning(self, reasoning_type: str) -> List[str]:
        """
        Get the appropriate tools for a given reasoning type.
        
        Args:
            reasoning_type: The reasoning type
            
        Returns:
            List[str]: List of appropriate tool names
        """
        # Define tool affinities for different reasoning types
        reasoning_tool_map = {
            'sequential': ['search_web', 'retrieve_knowledge'],
            'rag': ['search_web', 'retrieve_knowledge'],
            'knowledge': ['retrieve_knowledge', 'search_web'],
            'verification': ['search_web', 'retrieve_knowledge'],
            'conversational': ['analyze_sentiment'],
            'creative': ['generate_image'],
            'calculation': ['get_crypto_price'],
            'graph': ['search_web', 'retrieve_knowledge'],
            'multi_agent': ['search_web', 'retrieve_knowledge', 'analyze_sentiment'],
            'react': ['search_web', 'retrieve_knowledge', 'get_crypto_price', 'analyze_sentiment'],
            'step_back': ['search_web', 'retrieve_knowledge']
        }
        
        # Return the appropriate tools or all tools if reasoning type not found
        return reasoning_tool_map.get(reasoning_type, list(self.tool_capabilities.keys()))

def get_server_rag(server_id: str) -> RAGSystem:
    """Get or create a RAG system for the server"""
    if server_id not in server_knowledge_bases:
        server_knowledge_bases[server_id] = RAGSystem(server_id)
    return server_knowledge_bases[server_id]

# Initialize the GROQ model
def get_groq_llm():
    """Create a Groq LLM client for agents"""
    # Use smaller model for cost efficiency in agents
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("API_KEY")
    return ChatGroq(
        temperature=0.7, # Higher temp for more creativity
        model="llama3-70b-8192", # Cheaper model for agents
        groq_api_key=api_key,
        max_tokens=1200, # Limit output tokens
    )

# Define our tools

# Web search tool 
def create_search_tool():
    """Create a web search tool using either Tavily (if API key exists) or DuckDuckGo with fallbacks"""
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    
    if tavily_api_key:
        # Use Tavily if API key is available
        print("Using Tavily for web search")
        return TavilySearchResults(max_results=5)
    else:
        # Create a more robust DDG search with fallbacks
        print("Using DuckDuckGo with fallbacks for web search")
        return Tool(
            name="WebSearch",
            description="Search the web for current information. Use this for questions about recent events or factual information.",
            func=search_internet_wrapper,
        )

# Use your existing search function from ai_utils.py
async def search_internet_sync(query: str) -> str:
    """Perform internet search with robust fallback mechanisms."""
    # Record the start time for performance tracking
    start_time = time.time()
    
    try:
        # Call the enhanced search_internet function with fallback mechanisms
        result = await search_internet(query)
        
        # Optimize the search result to reduce tokens
        if result:
            result = token_optimizer.clean_text(result)
            result = token_optimizer.truncate_text(result, max_tokens=1000)
        
        # Log success and timing
        duration = time.time() - start_time
        print(f"Search completed in {duration:.2f} seconds")
        return result
    except Exception as e:
        # Comprehensive error handling
        print(f"Error in search_internet_sync: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        
        # Return a helpful error message
        return f"I encountered an error while searching for '{query}'. Please try a more specific query or try again later."

# Function to run async functions synchronously (for LangChain tools)
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except Exception as e:
        print(f"Error in run_async: {e}")
        return f"An error occurred while processing your request: {str(e)}"

# Wrap the async function to be used in a LangChain tool
def search_internet_wrapper(query: str) -> str:
    try:
        return run_async(search_internet_sync(query))
    except Exception as e:
        print(f"Error in search_internet_wrapper: {e}")
        # Provide a fallback response rather than exposing the error
        return f"I'm having trouble searching for '{query}'. Please try a different query or try again later."

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
    try:
        # Use your existing crypto price function
        loop = asyncio.get_event_loop()
        price_info = await get_crypto_price(crypto_name)
        if price_info:
            return price_info
        return f"Could not find price data for {crypto_name}"
    except Exception as e:
        print(f"Error in get_crypto_price_sync: {e}")
        return f"I encountered an error while fetching the price for {crypto_name}. Please try again later."

# Wrap the async function to be used in a LangChain tool
def get_crypto_price_wrapper(crypto_name: str) -> str:
    try:
        return run_async(get_crypto_price_sync(crypto_name))
    except Exception as e:
        print(f"Error in get_crypto_price_wrapper: {e}")
        return f"I couldn't retrieve the current price for {crypto_name}. There might be an issue with the data source."

# Knowledge base query tool
async def query_knowledge_base_sync(server_id: str, query: str) -> str:
    """Query the server's knowledge base"""
    try:
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
    except Exception as e:
        print(f"Error in query_knowledge_base_sync: {e}")
        return "I encountered an error while querying the knowledge base. Please try again later."

# Wrapper for knowledge base query tool
def query_knowledge_base_wrapper(server_id: str, query: str) -> str:
    try:
        return run_async(query_knowledge_base_sync(server_id, query))
    except Exception as e:
        print(f"Error in query_knowledge_base_wrapper: {e}")
        return "I'm having trouble accessing the server's knowledge base right now. Please try again later."

def create_tools(server_id=None):
    # Use search tool factory that handles both options
    search_tool = create_search_tool()
    
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
        
    # Start with a base set of tools
    tools = [search_tool, weather_tool, crypto_tool]
    
    # Add knowledge base tool if server_id is provided
    if server_id:
        kb_tool = Tool(
            name="KnowledgeBase",
            description=f"Search the server's knowledge base for information. Use this for questions about server-specific information or topics that have been added to the knowledge base.",
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