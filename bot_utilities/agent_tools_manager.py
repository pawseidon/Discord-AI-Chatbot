"""
Agent Tools Manager

This module provides a registry and execution system for tools that
agents can use to interact with external systems and perform specialized tasks.
"""

import logging
import asyncio
import inspect
from typing import Dict, Any, Optional, List, Callable, Union, Awaitable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('agent_tools_manager')

# Type for tool execution functions
ToolCallable = Callable[[Dict[str, Any], Dict[str, Any]], Union[Any, Awaitable[Any]]]

class AgentToolsManager:
    """
    Manages the registration and execution of tools that can be used by agents.
    """
    
    def __init__(self):
        """Initialize the tools manager"""
        self.tools = {}  # Maps tool_name to tool definition
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register the default built-in tools"""
        # Web search tool
        self.register_tool(
            name="search",
            description="Search the web for information",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            func=self._search_tool
        )
        
        # Calculator tool
        self.register_tool(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            func=self._calculator_tool
        )
        
        # Memory tool
        self.register_tool(
            name="memory",
            description="Access conversation memory",
            parameters={
                "action": {
                    "type": "string",
                    "description": "Action to perform (get, store)",
                    "enum": ["get", "store"]
                },
                "key": {
                    "type": "string",
                    "description": "Key for the memory item"
                },
                "value": {
                    "type": "string",
                    "description": "Value to store (only for store action)",
                    "required": False
                }
            },
            func=self._memory_tool
        )
    
    def register_tool(
        self, 
        name: str, 
        description: str, 
        parameters: Dict[str, Any],
        func: ToolCallable
    ) -> None:
        """
        Register a new tool
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            parameters: The parameters the tool accepts
            func: The function to call when executing the tool
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "func": func
        }
        logger.info(f"Registered tool: {name}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools and their descriptions
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for tool in self.tools.values()
        ]
    
    def get_tool_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool definition by name
        
        Args:
            name: The name of the tool
            
        Returns:
            The tool definition or None if not found
        """
        return self.tools.get(name)
    
    async def execute_tool(
        self, 
        tool_name: str, 
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a tool with the given arguments
        
        Args:
            tool_name: The name of the tool to execute
            args: The arguments to pass to the tool
            context: Optional context information
            
        Returns:
            The result from the tool execution
            
        Raises:
            ValueError: If the tool does not exist or arguments are invalid
        """
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # Validate arguments against parameters
        self._validate_arguments(tool, args)
        
        # Execute the tool function
        func = tool["func"]
        context = context or {}
        
        try:
            if asyncio.iscoroutinefunction(func):
                # Async function
                result = await func(args, context)
            else:
                # Sync function
                result = func(args, context)
                
            return result
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {str(e)}")
            raise
    
    def _validate_arguments(self, tool: Dict[str, Any], args: Dict[str, Any]) -> None:
        """
        Validate that the arguments match the tool's parameters
        
        Args:
            tool: The tool definition
            args: The arguments to validate
            
        Raises:
            ValueError: If arguments are invalid
        """
        parameters = tool["parameters"]
        
        # Check required parameters
        for param_name, param_def in parameters.items():
            if param_def.get("required", True) and param_name not in args:
                raise ValueError(f"Missing required parameter: {param_name}")
            
            # Type checking could be added here for more validation
    
    # Built-in tool implementations
    
    async def _search_tool(self, args: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Search tool implementation
        
        Args:
            args: Tool arguments
            context: Execution context
            
        Returns:
            Search results as text
        """
        query = args.get("query", "")
        
        # Import lazily to avoid circular dependencies
        from .services.agent_service import agent_service
        
        try:
            results = await agent_service.search_web(query)
            return results
        except Exception as e:
            logger.error(f"Error in search tool: {str(e)}")
            return f"Error performing search: {str(e)}"
    
    async def _calculator_tool(self, args: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Calculator tool implementation
        
        Args:
            args: Tool arguments
            context: Execution context
            
        Returns:
            Calculation result as text
        """
        expression = args.get("expression", "")
        
        # Import lazily to avoid circular dependencies
        from .services.symbolic_reasoning_service import symbolic_reasoning_service
        
        try:
            result = await symbolic_reasoning_service.evaluate_expression(expression)
            return str(result.get("result", "Error: No result"))
        except Exception as e:
            logger.error(f"Error in calculator tool: {str(e)}")
            return f"Error evaluating expression: {str(e)}"
    
    async def _memory_tool(self, args: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Memory tool implementation
        
        Args:
            args: Tool arguments
            context: Execution context
            
        Returns:
            Memory operation result as text
        """
        action = args.get("action", "")
        key = args.get("key", "")
        
        # Import lazily to avoid circular dependencies
        from .services.memory_service import memory_service
        
        try:
            if action == "get":
                # Get from memory
                conversation_id = context.get("conversation_id")
                user_id = context.get("user_id")
                
                if not conversation_id or not user_id:
                    return "Error: Missing conversation_id or user_id in context"
                
                value = await memory_service.get_memory_item(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    key=key
                )
                
                if value:
                    return f"Retrieved memory for '{key}': {value}"
                else:
                    return f"No memory found for key: {key}"
                    
            elif action == "store":
                # Store in memory
                value = args.get("value", "")
                conversation_id = context.get("conversation_id")
                user_id = context.get("user_id")
                
                if not conversation_id or not user_id:
                    return "Error: Missing conversation_id or user_id in context"
                    
                if not value:
                    return "Error: No value provided for storage"
                    
                await memory_service.store_memory_item(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    key=key,
                    value=value
                )
                
                return f"Stored memory with key: {key}"
                
            else:
                return f"Unknown memory action: {action}"
                
        except Exception as e:
            logger.error(f"Error in memory tool: {str(e)}")
            return f"Error accessing memory: {str(e)}" 