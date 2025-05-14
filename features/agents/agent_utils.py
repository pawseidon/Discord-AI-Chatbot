"""
Agent utilities module for Discord AI Chatbot.

This module provides various agent capabilities, tool integrations,
and agent creation utilities.
"""

import os
import asyncio
import json
import aiohttp
import time
import traceback
from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field

# Try to import optional dependencies
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.prompts import PromptTemplate
    from langchain_groq import ChatGroq
    from langchain_core.tools import Tool
    from langchain_community.tools.tavily_search.tool import TavilySearchResults
    from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
    HAVE_LANGCHAIN = True
except ImportError:
    HAVE_LANGCHAIN = False

# Import from new module structure
from utils.token_utils import token_optimizer
from core.ai_provider import get_ai_provider
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AgentFactory:
    """Factory class for creating and running agents with various tools"""
    
    def __init__(self, 
                ai_provider=None, 
                server_id: Optional[str] = None,
                max_thinking_length: int = 1200,
                max_iterations: int = 5):
        """
        Initialize the agent factory
        
        Args:
            ai_provider: Optional AI provider for agent operations
            server_id: Optional server ID for server-specific tools
            max_thinking_length: Maximum token length for thinking steps
            max_iterations: Maximum number of reasoning iterations
        """
        self.ai_provider = ai_provider
        self.server_id = server_id
        self.max_thinking_length = max_thinking_length
        self.max_iterations = max_iterations
        self.tools = {}
        
        # Initialize available tools
        self._initialize_tools()
    
    async def ensure_ai_provider(self):
        """Ensure AI provider is available"""
        if self.ai_provider is None:
            self.ai_provider = await get_ai_provider()
    
    def _initialize_tools(self):
        """Initialize available tools for agents"""
        # Register search tool
        self.tools["search"] = self._create_search_tool()
        
        # Register crypto price tool
        self.tools["crypto"] = Tool(
            name="CryptoPrice",
            description="Get current cryptocurrency prices. Input should be a cryptocurrency name or symbol like 'bitcoin', 'btc', 'ethereum', etc.",
            func=self._get_crypto_price_wrapper,
        )
        
        # Register basic weather tool (placeholder)
        self.tools["weather"] = Tool(
            name="Weather",
            description="Get the current weather in a location",
            func=self._get_weather,
        )
        
        # Add server-specific knowledge base tool if server_id is provided
        if self.server_id:
            self.tools["kb"] = Tool(
                name="KnowledgeBase",
                description="Query the server's knowledge base for information that has been stored by users. Use this for server-specific information before searching the internet.",
                func=lambda query: self._query_knowledge_base_wrapper(self.server_id, query),
            )
    
    def _create_search_tool(self):
        """Create a web search tool using available providers"""
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        
        if tavily_api_key and HAVE_LANGCHAIN:
            # Use Tavily if API key is available
            print("Using Tavily for web search")
            return TavilySearchResults(max_results=5)
        else:
            # Create a more robust search tool with fallbacks
            print("Using fallback search mechanism")
            return Tool(
                name="WebSearch",
                description="Search the web for current information. Use this for questions about recent events or factual information.",
                func=self._search_internet_wrapper,
            )
    
    def _run_async(self, coro):
        """Run async functions synchronously for tool compatibility"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        except Exception as e:
            print(f"Error in run_async: {e}")
            return f"An error occurred while processing your request: {str(e)}"
    
    def _search_internet_wrapper(self, query: str) -> str:
        """Wrapper for internet search tool"""
        try:
            return self._run_async(self._search_internet_async(query))
        except Exception as e:
            print(f"Error in search_internet_wrapper: {e}")
            return f"I'm having trouble searching for '{query}'. Please try a different query or try again later."
    
    async def _search_internet_async(self, query: str) -> str:
        """Perform internet search with fallback mechanisms"""
        start_time = time.time()
        
        try:
            # Ensure AI provider is available
            await self.ensure_ai_provider()
            
            # Use the AI provider's search capability if available
            if hasattr(self.ai_provider, 'search_internet'):
                result = await self.ai_provider.search_internet(query)
            elif hasattr(self.ai_provider, 'search_information'):
                result = await self.ai_provider.search_information(query)
            else:
                return "Web search capability not available."
            
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
            print(f"Error in search_internet_async: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            
            # Return a helpful error message
            return f"I encountered an error while searching for '{query}'. Please try a more specific query or try again later."
    
    def _get_weather(self, location: str) -> str:
        """Get the current weather in a given location (placeholder)"""
        # This is a placeholder - would be integrated with a real weather API
        return f"This is a placeholder. In a complete implementation, I would check the weather in {location}."
    
    def _get_crypto_price_wrapper(self, crypto_name: str) -> str:
        """Wrapper for crypto price check tool"""
        try:
            return self._run_async(self._get_crypto_price_async(crypto_name))
        except Exception as e:
            print(f"Error in get_crypto_price_wrapper: {e}")
            return f"I couldn't retrieve the current price for {crypto_name}. There might be an issue with the data source."
    
    async def _get_crypto_price_async(self, crypto_name: str) -> str:
        """Get real-time cryptocurrency price information"""
        try:
            # Ensure AI provider is available
            await self.ensure_ai_provider()
            
            # Use the AI provider's crypto capability if available
            if hasattr(self.ai_provider, 'get_crypto_price'):
                price_info = await self.ai_provider.get_crypto_price(crypto_name)
                if price_info:
                    return price_info
            
            # Fallback to searching for crypto prices
            search_query = f"current price of {crypto_name} cryptocurrency"
            return await self._search_internet_async(search_query)
        except Exception as e:
            print(f"Error in get_crypto_price_async: {e}")
            return f"I encountered an error while fetching the price for {crypto_name}. Please try again later."
    
    def _query_knowledge_base_wrapper(self, server_id: str, query: str) -> str:
        """Wrapper for knowledge base query tool"""
        try:
            return self._run_async(self._query_knowledge_base_async(server_id, query))
        except Exception as e:
            print(f"Error in query_knowledge_base_wrapper: {e}")
            return "I'm having trouble accessing the server's knowledge base right now. Please try again later."
    
    async def _query_knowledge_base_async(self, server_id: str, query: str) -> str:
        """Query the server's knowledge base"""
        try:
            # Ensure AI provider is available
            await self.ensure_ai_provider()
            
            # Use the AI provider's knowledge base querying if available
            if hasattr(self.ai_provider, 'query_knowledge_base'):
                results = await self.ai_provider.query_knowledge_base(server_id, query)
                
                if not results:
                    return "No relevant information found in the knowledge base."
                
                # Format results
                if isinstance(results, str):
                    return results
                
                # Handle structured results
                if isinstance(results, list):
                    response = "Knowledge base results:\n\n"
                    for i, doc in enumerate(results):
                        if hasattr(doc, 'page_content'):
                            # Handle langchain document format
                            content = token_optimizer.clean_text(doc.page_content)
                            content = token_optimizer.truncate_text(content, max_tokens=500)
                            
                            response += f"[Document {i+1}]:\n{content}\n\n"
                            if hasattr(doc, 'metadata') and "source" in doc.metadata:
                                response += f"Source: {doc.metadata['source']}\n"
                        elif isinstance(doc, dict):
                            # Handle dictionary format
                            content = doc.get('content', 'No content available')
                            content = token_optimizer.clean_text(content)
                            content = token_optimizer.truncate_text(content, max_tokens=500)
                            
                            response += f"[Document {i+1}]:\n{content}\n\n"
                            if 'source' in doc:
                                response += f"Source: {doc['source']}\n"
                    
                    return response
            
            return "Knowledge base querying not available."
        except Exception as e:
            print(f"Error in query_knowledge_base_async: {e}")
            return "I encountered an error while querying the knowledge base. Please try again later."
    
    def get_available_tools(self, tool_names: Optional[List[str]] = None) -> List[Tool]:
        """
        Get a list of available tools based on names
        
        Args:
            tool_names: List of tool names to include, None for all
            
        Returns:
            List of Tool objects
        """
        if tool_names is None:
            return list(self.tools.values())
        
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    async def create_langchain_agent(self, 
                                  user_memory: str = "", 
                                  kb_context: str = "",
                                  tool_names: Optional[List[str]] = None) -> Any:
        """
        Create a LangChain agent if LangChain is available
        
        Args:
            user_memory: User memory context
            kb_context: Knowledge base context
            tool_names: List of tool names to include
            
        Returns:
            AgentExecutor if LangChain is available, None otherwise
        """
        if not HAVE_LANGCHAIN:
            return None
        
        # Get specified tools or all available tools
        tools = self.get_available_tools(tool_names)
        
        # Optimize memory and KB context to reduce tokens
        if user_memory:
            user_memory = token_optimizer.truncate_text(user_memory, max_tokens=800)
        
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
            Thought: {agent_scratchpad}
            """
        )
        
        # Create LLM for agent
        await self.ensure_ai_provider()
        
        # Use Groq if available, otherwise use the main AI provider
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if groq_api_key and HAVE_LANGCHAIN:
            llm = ChatGroq(
                temperature=0.7,
                model="llama3-70b-8192",
                groq_api_key=groq_api_key,
                max_tokens=self.max_thinking_length,
            )
        elif hasattr(self.ai_provider, 'langchain_llm'):
            llm = self.ai_provider.langchain_llm
        else:
            # No suitable LLM available
            return None
        
        # Create the agent
        agent = create_react_agent(llm, tools, prompt)
        
        # Create the agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=self.max_iterations,
            memory_context=user_memory,
            kb_context=kb_context,
            early_stopping_method="generate",
        )
        
        return agent_executor
    
    async def run_custom_agent(self, 
                            query: str, 
                            user_memory: str = "", 
                            kb_context: str = "",
                            tool_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a custom agent implementation using AI provider
        
        Args:
            query: User query
            user_memory: User memory context
            kb_context: Knowledge base context
            tool_names: List of tool names to include
            
        Returns:
            Dict with answer and agent trace
        """
        await self.ensure_ai_provider()
        
        # Get specified tools or all available tools
        tools = self.get_available_tools(tool_names)
        
        # Format tools for prompt
        tools_desc = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        
        # Create the ReAct prompt
        prompt = f"""You are an intelligent Discord bot assistant that can help with various tasks.
        
        {user_memory}
        
        {kb_context}
        
        You have access to the following tools:
        
        {tools_desc}
        
        Think step by step to determine which tool to use, then get the information needed to answer the question.
        You can use up to {self.max_iterations} steps of reasoning.
        
        User query: {query}
        
        Now, think step by step and use tools as needed to answer the query.
        """
        
        try:
            # Use the AI provider to generate reasoning
            response = await self.ai_provider.async_call(prompt)
            
            # Format the response
            return {
                "answer": response,
                "agent_trace": "Custom agent execution (trace not available)",
                "tools_used": [tool.name for tool in tools]
            }
        except Exception as e:
            print(f"Error in run_custom_agent: {e}")
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "agent_trace": f"Error: {str(e)}",
                "tools_used": []
            }
    
    async def run_agent(self, 
                      query: str, 
                      user_memory: str = "", 
                      kb_context: str = "",
                      tool_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run an agent to answer a query using available tools
        
        Args:
            query: User query
            user_memory: User memory context
            kb_context: Knowledge base context
            tool_names: List of tool names to include
            
        Returns:
            Dict with answer and agent trace
        """
        # Try LangChain agent first if available
        if HAVE_LANGCHAIN:
            try:
                agent_executor = await self.create_langchain_agent(
                    user_memory=user_memory,
                    kb_context=kb_context,
                    tool_names=tool_names
                )
                
                if agent_executor:
                    # Run the agent
                    result = await agent_executor.ainvoke({"input": query})
                    
                    # Extract the final answer and agent trace
                    answer = result.get("output", "No answer was generated.")
                    
                    return {
                        "answer": answer,
                        "agent_trace": result.get("intermediate_steps", []),
                        "tools_used": [tool.name for tool in self.get_available_tools(tool_names)]
                    }
            except Exception as e:
                print(f"Error in LangChain agent execution: {e}")
                print(f"Stack trace: {traceback.format_exc()}")
                # Fall back to custom agent
        
        # Fall back to custom agent implementation
        return await self.run_custom_agent(
            query=query,
            user_memory=user_memory,
            kb_context=kb_context,
            tool_names=tool_names
        )

# Global instance for singleton pattern
_agent_factory = None

def create_agent(
    ai_provider=None,
    server_id: Optional[str] = None,
    max_thinking_length: int = 1200,
    max_iterations: int = 5
) -> AgentFactory:
    """
    Create or get the global agent factory instance
    
    Args:
        ai_provider: Optional AI provider for agent operations
        server_id: Optional server ID for server-specific tools
        max_thinking_length: Maximum token length for thinking steps
        max_iterations: Maximum number of reasoning iterations
        
    Returns:
        AgentFactory instance
    """
    global _agent_factory
    
    if _agent_factory is None:
        _agent_factory = AgentFactory(
            ai_provider=ai_provider,
            server_id=server_id,
            max_thinking_length=max_thinking_length,
            max_iterations=max_iterations
        )
        
    return _agent_factory 