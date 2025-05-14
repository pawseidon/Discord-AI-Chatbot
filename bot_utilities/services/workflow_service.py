"""
Workflow Service

This service manages complex agent workflows using graph-based control flow.
It handles the creation, execution, and state management of multi-agent
workflows using LangGraph.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
import json

# Import LangGraph components
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logging.warning("LangGraph not installed. Run 'pip install langgraph' to enable advanced workflow capabilities.")

# Import from our services
from .agent_service import agent_service, AgentCommand, AgentCommandType
from .memory_service import memory_service

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('workflow_service')

class WorkflowState:
    """
    State container for agent workflows.
    
    This class maintains the shared state between agents in a workflow,
    including conversation history, intermediate results, and metadata.
    """
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        """Initialize workflow state"""
        self.state = initial_state or {}
        self.state.setdefault("messages", [])
        self.state.setdefault("current_agent", None)
        self.state.setdefault("scratchpad", {})
        self.state.setdefault("tool_results", {})
        self.state.setdefault("metadata", {})
        
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        self.state["messages"].append({"role": role, "content": content})
        
    def add_tool_result(self, tool_name: str, result: Any) -> None:
        """Store the result of a tool call"""
        self.state["tool_results"][tool_name] = result
        
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in the conversation history"""
        return self.state["messages"]
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary representation"""
        return self.state
        
    def update(self, updates: Dict[str, Any]) -> None:
        """Update state with new values"""
        self.state.update(updates)
        
    def set_current_agent(self, agent_id: str) -> None:
        """Set the currently active agent"""
        self.state["current_agent"] = agent_id
        
    def add_to_scratchpad(self, agent_id: str, note: str) -> None:
        """Add a note to an agent's scratchpad"""
        if agent_id not in self.state["scratchpad"]:
            self.state["scratchpad"][agent_id] = []
        self.state["scratchpad"][agent_id].append(note)
        
    def get_scratchpad(self, agent_id: str) -> List[str]:
        """Get an agent's scratchpad notes"""
        return self.state["scratchpad"].get(agent_id, [])
        
    def get_combined_scratchpad(self) -> str:
        """Get all scratchpad notes combined"""
        combined = []
        for agent_id, notes in self.state["scratchpad"].items():
            combined.append(f"== {agent_id} Notes ==")
            combined.extend(notes)
        return "\n".join(combined)

class WorkflowService:
    """
    Service for managing complex agent workflows using LangGraph.
    """
    def __init__(self):
        """Initialize the workflow service"""
        self._initialized = False
        self.workflows = {}
        
    async def ensure_initialized(self):
        """Initialize the service if not already initialized"""
        if self._initialized:
            return
            
        # Initialize agent service if needed
        await agent_service.ensure_initialized()
        
        logger.info("Workflow service initialized")
        self._initialized = True
        
    async def create_workflow(self, workflow_id: str, agent_ids: List[str]) -> bool:
        """
        Create a new agent workflow.
        
        Args:
            workflow_id: Unique identifier for this workflow
            agent_ids: List of agent IDs to include in the workflow
            
        Returns:
            Success status
        """
        await self.ensure_initialized()
        
        if not LANGGRAPH_AVAILABLE:
            logger.error("LangGraph is required for workflows but is not installed")
            return False
            
        try:
            # Create state graph
            builder = StateGraph(WorkflowState)
            
            # Add nodes for each agent
            for agent_id in agent_ids:
                builder.add_node(agent_id, self._create_agent_node(agent_id))
                
            # Create a default router that determines which agent to call next
            builder.add_node("router", self._create_router_node(agent_ids))
            
            # Set the router as the entry point
            builder.set_entry_point("router")
            
            # Add conditional edges based on agent command types
            for agent_id in agent_ids:
                # From router to this agent
                builder.add_conditional_edges(
                    "router",
                    self._route_condition,
                    {agent_id: agent_id for agent_id in agent_ids}
                )
                
                # From this agent back to router or to END
                builder.add_conditional_edges(
                    agent_id,
                    self._agent_output_condition,
                    {
                        "continue": "router",
                        "complete": END
                    }
                )
            
            # Compile the graph
            self.workflows[workflow_id] = builder.compile()
            logger.info(f"Created workflow '{workflow_id}' with {len(agent_ids)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating workflow '{workflow_id}': {str(e)}")
            return False
        
    def _create_agent_node(self, agent_id: str) -> Callable:
        """Create a node function for an agent in the workflow"""
        async def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # Extract current query and context from state
            messages = state["messages"]
            query = messages[-1]["content"] if messages else ""
            
            # Build context dictionary
            context = {
                "conversation_history": messages,
                "tool_results": state.get("tool_results", {}),
                "scratchpad": state.get("scratchpad", {}).get(agent_id, [])
            }
            
            # Process with agent service
            command = await agent_service.process_agent(agent_id, query, context)
            
            # Update state based on command
            updates = {}
            
            if command.command_type == AgentCommandType.RESPONSE:
                # Final response
                updates["messages"] = add_messages(state["messages"], [
                    {"role": "assistant", "content": command.content}
                ])
                updates["metadata"] = {"status": "complete"}
                
            elif command.command_type == AgentCommandType.DELEGATE:
                # Delegation to another agent
                updates["current_agent"] = command.target_agent
                updates["metadata"] = {
                    "status": "delegated", 
                    "target": command.target_agent,
                    "delegation_context": command.content
                }
                
                # Add delegation note to scratchpad
                if agent_id not in state["scratchpad"]:
                    state["scratchpad"][agent_id] = []
                state["scratchpad"][agent_id].append(
                    f"Delegated to {command.target_agent}: {command.content}"
                )
                
            elif command.command_type == AgentCommandType.TOOL_CALL:
                # Tool call result
                tool_result = await agent_service.execute_tool(
                    command.tool_name, 
                    command.tool_params
                )
                
                updates["tool_results"] = state.get("tool_results", {})
                updates["tool_results"][command.tool_name] = {
                    "params": command.tool_params,
                    "result": tool_result
                }
                
                # Add tool use note to scratchpad
                if agent_id not in state["scratchpad"]:
                    state["scratchpad"][agent_id] = []
                state["scratchpad"][agent_id].append(
                    f"Used tool {command.tool_name}: {json.dumps(command.tool_params)[:100]}..."
                )
                
            return {**state, **updates}
        
        return agent_node
        
    def _create_router_node(self, agent_ids: List[str]) -> Callable:
        """Create the router node function"""
        async def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # If a specific next agent is set (e.g., from delegation), use that
            if state.get("current_agent") in agent_ids:
                return state
                
            # Otherwise, determine the best agent for the current query
            messages = state["messages"]
            if not messages:
                # Default to the first agent if no messages
                state["current_agent"] = agent_ids[0]
                return state
                
            # Get the latest user message
            latest_user_message = None
            for message in reversed(messages):
                if message["role"] == "user":
                    latest_user_message = message["content"]
                    break
                    
            if not latest_user_message:
                # No user message found, use default agent
                state["current_agent"] = agent_ids[0]
                return state
                
            # Detect the most appropriate agent using agent_service
            best_agent = await agent_service.detect_reasoning_type(
                latest_user_message,
                conversation_id="workflow"
            )
            
            # If the detected agent is in our workflow, use it
            if best_agent in agent_ids:
                state["current_agent"] = best_agent
            else:
                # Otherwise use the first agent as default
                state["current_agent"] = agent_ids[0]
                
            return state
        
        return router_node
        
    def _route_condition(self, state: Dict[str, Any]) -> str:
        """Route to the next agent based on the current_agent field"""
        return state.get("current_agent", "unknown")
        
    def _agent_output_condition(self, state: Dict[str, Any]) -> str:
        """Determine if the workflow should continue or end"""
        if state.get("metadata", {}).get("status") == "complete":
            return "complete"
        return "continue"
        
    async def run_workflow(
        self, 
        workflow_id: str, 
        user_query: str, 
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        max_steps: int = 10,
        update_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> str:
        """
        Run a workflow with a user query.
        
        Args:
            workflow_id: ID of the workflow to run
            user_query: The user's query to process
            conversation_id: Optional conversation ID for context
            user_id: Optional user ID
            max_steps: Maximum number of steps to run
            update_callback: Optional callback function for progress updates
            
        Returns:
            The final response
        """
        await self.ensure_initialized()
        
        if workflow_id not in self.workflows:
            logger.error(f"Workflow '{workflow_id}' not found")
            return f"Error: Workflow '{workflow_id}' not found"
            
        try:
            # Initialize the workflow state
            initial_state = WorkflowState()
            initial_state.add_message("user", user_query)
            
            # Add conversation history if available
            if conversation_id and user_id:
                history = await memory_service.get_conversation_history(
                    user_id=user_id,
                    channel_id=conversation_id,
                    limit=5  # Only use last 5 messages for context
                )
                
                if history:
                    # Convert to the format expected by WorkflowState
                    for msg in history:
                        if msg["role"] == "user":
                            initial_state.add_message("user", msg["content"])
                        else:
                            initial_state.add_message("assistant", msg["content"])
            
            # Execute the workflow
            workflow = self.workflows[workflow_id]
            
            # Set up event stream for step-by-step execution
            config = {"recursion_limit": max_steps}
            steps = workflow.stream(initial_state.to_dict(), config=config)
            
            step_count = 0
            final_state = None
            
            async for step in steps:
                step_count += 1
                current_state = step["state"]
                
                # Call update callback if provided
                if update_callback:
                    status = f"Step {step_count}: "
                    if current_state.get("current_agent"):
                        status += f"Using agent '{current_state['current_agent']}'"
                    else:
                        status += "Planning next step"
                        
                    metadata = {
                        "step": step_count,
                        "current_agent": current_state.get("current_agent"),
                        "status": current_state.get("metadata", {}).get("status", "processing")
                    }
                    
                    await update_callback(status, metadata)
                
                final_state = current_state
                
                # Log progress
                logger.info(f"Workflow '{workflow_id}' - Step {step_count} - "
                           f"Agent: {current_state.get('current_agent')}")
                
            # Extract the final response
            if not final_state:
                return f"Error: Workflow '{workflow_id}' produced no result"
                
            messages = final_state.get("messages", [])
            
            # Get the last assistant message
            for message in reversed(messages):
                if message["role"] == "assistant":
                    return message["content"]
                    
            return f"Error: Workflow '{workflow_id}' produced no assistant response"
            
        except Exception as e:
            logger.error(f"Error running workflow '{workflow_id}': {str(e)}")
            return f"Error running workflow: {str(e)}"
            
    async def create_and_run_default_workflow(
        self, 
        user_query: str, 
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        update_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> str:
        """
        Create a default workflow and run it.
        
        This is a convenience method that creates a workflow with the most common agent types,
        then runs it immediately with the provided query.
        
        Args:
            user_query: The user's query
            conversation_id: Optional conversation ID
            user_id: Optional user ID
            update_callback: Optional callback function for progress updates
            
        Returns:
            The final response
        """
        await self.ensure_initialized()
        
        # Create a default workflow ID based on user/conversation
        workflow_id = f"default_{user_id or 'unknown'}_{conversation_id or 'default'}"
        
        # Default set of agents for most queries
        default_agents = [
            "sequential",
            "rag",
            "creative",
            "verification",
            "conversational"
        ]
        
        # Create the workflow if it doesn't exist
        if workflow_id not in self.workflows:
            success = await self.create_workflow(workflow_id, default_agents)
            if not success:
                return "Sorry, I couldn't create a workflow to process your request."
                
        # Run the workflow
        return await self.run_workflow(
            workflow_id=workflow_id,
            user_query=user_query,
            conversation_id=conversation_id,
            user_id=user_id,
            update_callback=update_callback
        )
        
    def clear_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow from memory"""
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            logger.info(f"Deleted workflow '{workflow_id}'")
            return True
        return False
        
    def clear_all_workflows(self) -> None:
        """Delete all workflows from memory"""
        self.workflows.clear()
        logger.info("Cleared all workflows")

# Create a singleton instance
workflow_service = WorkflowService() 