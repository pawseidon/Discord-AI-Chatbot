import os
import asyncio
import json
import sys
from typing import List, Dict, Any, Callable, Awaitable, Optional

# Improved handling of ExceptionGroup based on Python version
if sys.version_info >= (3, 11):
    # Python 3.11+ has built-in ExceptionGroup
    from builtins import ExceptionGroup
else:
    try:
        # Try to import from exceptiongroup package
        from exceptiongroup import ExceptionGroup
    except ImportError:
        # Fallback implementation for older Python versions
        class ExceptionGroup(Exception):
            """Fallback implementation of ExceptionGroup for older Python versions"""
            def __init__(self, message, exceptions):
                super().__init__(message)
                self.exceptions = exceptions
                
            def __str__(self):
                return f"{self.args[0]} ({len(self.exceptions)} sub-exceptions)"

from mcp import ClientSession
from mcp.types import JSONRPCMessage, JSONRPCRequest
from mcp.shared.message import SessionMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

class ReadStream:
    """A read stream that wraps a read function"""
    
    def __init__(self, read_func):
        self.read_func = read_func
    
    async def receive(self):
        """Receive data from the read function"""
        data = await self.read_func()
        # Important: Return data directly without any transformation
        # to preserve the original message structure expected by MCP
        return data
    
    async def __aenter__(self):
        """Support for async context manager protocol"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support for async context manager protocol"""
        pass
        
    def __aiter__(self):
        """Support for async iteration protocol"""
        return self
        
    async def __anext__(self):
        """Get the next item for async iteration"""
        try:
            result = await self.receive()
            if result is None:
                raise StopAsyncIteration
            return result
        except Exception as e:
            # Convert any exception to StopAsyncIteration to end the iteration
            raise StopAsyncIteration from e

class WriteStream:
    """A write stream that wraps a write function"""
    
    def __init__(self, write_func):
        self.write_func = write_func
    
    async def send(self, data):
        """Send data to the write function"""
        # For logging purposes only, don't modify the actual data
        try:
            # Create a safe representation for logging without modifying the object
            if isinstance(data, SessionMessage):
                log_data = f"MCP SessionMessage object"
            elif hasattr(data, 'to_dict'):
                log_data = f"Object with to_dict: {type(data).__name__}"
            elif hasattr(data, '__dict__'):
                log_data = f"Object with __dict__: {type(data).__name__}"
            else:
                log_data = f"Data of type: {type(data).__name__}"
                
            print(f"MCP Tool Output: {log_data}")
        except Exception as e:
            print(f"MCP Tool Output (logging error): {e}")
            
        # Important: Pass the original data object without any transformation
        await self.write_func(data)
    
    async def __aenter__(self):
        """Support for async context manager protocol"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support for async context manager protocol"""
        pass

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
                           read_func: Callable[[], Awaitable[Any]], 
                           write_func: Callable[[Any], Awaitable[None]],
                           query: str,
                           system_message: str = None,
                           config: Optional[RunnableConfig] = None) -> str:
        """Run an MCP agent with the given read and write functions"""
        # Create the read and write stream objects
        read_stream = ReadStream(read_func)
        write_stream = WriteStream(write_func)
        
        # Use the streams with the session
        try:
            async with ClientSession(read_stream, write_stream) as session:
                try:
                    # Initialize the session
                    await session.initialize()
                    
                    # Create the agent
                    print("Creating MCP agent...")
                    agent_executor = await self.create_mcp_agent(session, system_message)
                    
                    # Run the agent
                    print(f"Running agent with query: {query}")
                    response = await agent_executor.ainvoke(
                        {"input": query}, 
                        config=config
                    )
                    
                    return response.get("output", "I wasn't able to complete the task.")
                except ExceptionGroup as eg:
                    # Handle ExceptionGroup specifically as they can be harder to debug
                    print(f"Exception in MCP agent (ExceptionGroup): {str(eg)}")
                    # Extract and print the first exception for clearer debug information
                    first_exc = next(eg.exceptions).__cause__ if eg.exceptions else None
                    print(f"First exception in group: {str(first_exc)}")
                    if "ValidationError" in str(eg):
                        return "I encountered a validation error with the sequential thinking protocol. Please try a different approach."
                    return "I encountered a protocol error in my thinking process. Please try again."
                except Exception as e:
                    print(f"Error in MCP agent: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return f"I encountered an error: {str(e)}"
        except ExceptionGroup as eg:
            print(f"TaskGroup Exception: {str(eg)}")
            errors = "\n".join([str(e) for e in eg.exceptions]) if hasattr(eg, "exceptions") else str(eg)
            print(f"TaskGroup Exception details: {errors}")
            return "I encountered a protocol initialization error. Sequential thinking is unavailable at the moment."
        except Exception as e:
            print(f"Outer session error: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"I encountered a connection error: {str(e)}"
    
    async def simple_read_func(self) -> Any:
        """Simple read function for MCP that returns a properly structured empty response"""
        # This is a bit of special case handling for MCP - in a real application we would
        # have a properly structured two-way communication channel with the MCP server
        
        # Since we're mocking this connection, we need to return None to indicate
        # that there's no more data to read, which allows the session to move forward
        # without waiting for more input that will never come
        return None
    
    async def simple_write_func(self, data: Any) -> None:
        """Simple write function for MCP that prints the data"""
        # The data is already logged in WriteStream.send, so we don't need to do anything else
        pass
    
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