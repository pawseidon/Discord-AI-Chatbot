import os
import asyncio
from typing import List, Dict, Any, Callable, Awaitable, Optional
import json

from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

class MCPToolsManager:
    """Manages MCP tools and integrations"""
    
    def __init__(self, api_key: str = None, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        """Initialize the MCP tools manager"""
        self.api_key = api_key or os.environ.get("API_KEY")
        self.model_name = model_name
        self.tools_cache = {}
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name
        )
    
    async def get_mcp_tools(self, session: ClientSession) -> List[BaseTool]:
        """Load MCP tools from the session"""
        return await load_mcp_tools(session)
    
    async def create_mcp_agent(self, 
                              session: ClientSession,
                              system_message: str = None) -> AgentExecutor:
        """Create an agent with MCP tools"""
        # Load tools from the session
        tools = await self.get_mcp_tools(session)
        
        # Set default system message if none provided
        if not system_message:
            system_message = """You are a helpful assistant with access to tools. 
            Use these tools to best help the user with their request.
            Always think step by step to determine the best course of action.
            If you don't know how to do something or the tools available don't support it, 
            be honest and explain what you can and cannot do."""
        
        # Create the agent with React framework
        agent = create_react_agent(
            self.llm,
            tools,
            system_message=system_message
        )
        
        # Return the agent executor
        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    async def run_mcp_agent(self, 
                           read_func: Callable[[], Awaitable[Dict[str, Any]]], 
                           write_func: Callable[[Dict[str, Any]], Awaitable[None]],
                           query: str,
                           system_message: str = None,
                           config: Optional[RunnableConfig] = None) -> str:
        """Run an MCP agent with the given read and write functions"""
        async with ClientSession(read_func, write_func) as session:
            # Create the agent
            agent_executor = await self.create_mcp_agent(session, system_message)
            
            # Run the agent
            response = await agent_executor.ainvoke(
                {"input": query}, 
                config=config
            )
            
            return response.get("output", "I wasn't able to complete the task.")
    
    async def simple_read_func(self) -> Dict[str, Any]:
        """Simple read function for MCP that returns an empty dict"""
        return {}
    
    async def simple_write_func(self, data: Dict[str, Any]) -> None:
        """Simple write function for MCP that prints the data"""
        print(f"MCP Tool Output: {json.dumps(data, indent=2)}")
    
    async def run_simple_mcp_agent(self, 
                                  query: str, 
                                  system_message: str = None,
                                  config: Optional[RunnableConfig] = None) -> str:
        """Run an MCP agent with simple read/write functions for basic usage"""
        return await self.run_mcp_agent(
            self.simple_read_func,
            self.simple_write_func,
            query,
            system_message,
            config
        )

class MCPTool:
    """Base class for custom MCP tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with the given parameters"""
        raise NotImplementedError("Tool execution not implemented")
    
    def to_mcp_manifest(self) -> Dict[str, Any]:
        """Convert to MCP manifest format"""
        raise NotImplementedError("MCP manifest conversion not implemented") 