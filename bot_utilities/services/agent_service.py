"""
Agent Service

This module provides a centralized service for agent orchestration,
allowing both event handlers and command cogs to use the same agent system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Tuple
import discord
import re
from enum import Enum, auto
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('agent_service')

# Define Agent Command types for workflow management
class AgentCommandType(Enum):
    """Types of commands an agent can issue"""
    RESPONSE = auto()  # Final response to the user
    DELEGATE = auto()  # Delegate to another agent
    TOOL_USE = auto()  # Use a tool
    ERROR = auto()     # Report an error
    
class AgentCommand:
    """Command issued by an agent"""
    def __init__(self, 
                 command_type: AgentCommandType, 
                 content: str,
                 target_agent: Optional[str] = None,
                 tool_name: Optional[str] = None,
                 tool_args: Optional[Dict[str, Any]] = None):
        self.command_type = command_type
        self.content = content
        self.target_agent = target_agent
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        
    @classmethod
    def response(cls, content: str) -> 'AgentCommand':
        """Create a response command"""
        return cls(AgentCommandType.RESPONSE, content)
        
    @classmethod
    def delegate(cls, target_agent: str, message: str) -> 'AgentCommand':
        """Create a delegation command"""
        return cls(AgentCommandType.DELEGATE, message, target_agent=target_agent)
        
    @classmethod
    def use_tool(cls, tool_name: str, message: str, **tool_args) -> 'AgentCommand':
        """Create a tool use command"""
        return cls(AgentCommandType.TOOL_USE, message, tool_name=tool_name, tool_args=tool_args)
        
    @classmethod
    def error(cls, error_message: str) -> 'AgentCommand':
        """Create an error command"""
        return cls(AgentCommandType.ERROR, error_message)

class AgentService:
    """Service for interacting with the agent orchestration system"""
    
    def __init__(self):
        """Initialize the agent service"""
        self.orchestrator = None
        self.tools_manager = None
        self.memory_manager = None
        self.reasoning_detector = None
        self._initialized = False
        
        # Track the reasoning combinations that have worked well
        self.successful_combos = {}
        
        # Track in-progress requests to prevent duplicates
        self.processing_requests = set()
    
    async def initialize(self, llm_provider = None):
        """Initialize the service with an LLM provider"""
        if not self._initialized:
            if llm_provider is None:
                # Import required modules lazily to avoid circular dependencies
                from ..ai_utils import get_ai_provider
                llm_provider = await get_ai_provider()
                
            # Import required modules lazily 
            from ..agent_orchestrator import AgentOrchestrator  
            from ..agent_tools_manager import AgentToolsManager
            from ..agent_memory import AgentMemoryManager
            from ..reasoning_utils import ReasoningDetector
            
            # Initialize components
            self.tools_manager = AgentToolsManager()
            self.memory_manager = AgentMemoryManager()
            self.reasoning_detector = ReasoningDetector(enable_cache=True)
            self.orchestrator = AgentOrchestrator(
                llm_provider=llm_provider,
                tools_manager=self.tools_manager
            )
            self._initialized = True
            logger.info("Agent service initialized successfully")
            
    async def ensure_initialized(self):
        """Ensure the service is initialized with required components"""
        if not self._initialized:
            await self.initialize()
    
    async def process_query(
        self,
        query: str,
        user_id: str,
        conversation_id: str,
        reasoning_type: str = None,
        update_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_steps: int = 5,
        context: Optional[Dict[str, Any]] = None,
        channel_id: str = None
    ) -> str:
        """
        Process a user query using the agent orchestration system
        
        Args:
            query: The user's query to process
            user_id: The ID of the user making the query
            conversation_id: The ID of the conversation
            reasoning_type: Optional specific reasoning type to use
            update_callback: Optional callback for status updates during processing
            max_steps: Maximum number of processing steps (delegations, tool calls)
            context: Additional context data to include in the processing
            channel_id: Optional channel ID (if not included in conversation_id)
            
        Returns:
            The response from the agent system
        """
        await self.ensure_initialized()
        
        # Create a unique request ID to prevent duplicate processing
        request_id = f"{conversation_id}:{user_id}:{hash(query)}"
        
        # Check if this request is already being processed
        if request_id in self.processing_requests:
            logger.warning(f"Duplicate request detected: {request_id}")
            return "I'm already processing a similar request. Please wait for the response."
        
        # Mark this request as being processed
        self.processing_requests.add(request_id)
        
        try:
            # Extract channel_id from conversation_id if not provided
            if channel_id is None and ":" in conversation_id:
                parts = conversation_id.split(":")
                if len(parts) >= 2:
                    channel_id = parts[1]
            
            # Merge provided context with default context
            full_context = {
                "user_id": user_id,
                "channel_id": channel_id,
                "conversation_id": conversation_id
            }
            if context:
                full_context.update(context)
            
            # Auto-detect reasoning type if not specified
            if reasoning_type is None:
                reasoning_type = await self.detect_reasoning_type(query, conversation_id)
            
            # Check if we should use multiple reasoning types
            should_combine = await self.should_combine_reasoning(query, conversation_id)
            
            if should_combine:
                # Get multiple applicable reasoning types
                reasoning_types = await self.detect_multiple_reasoning_types(query, conversation_id)
                
                # Keep track of which combinations are being used
                combo_key = ":".join(sorted(reasoning_types[:2]))
                
                # Include reasoning types in context
                full_context["reasoning_types"] = reasoning_types
                full_context["primary_reasoning"] = reasoning_types[0]
                full_context["secondary_reasoning"] = reasoning_types[1] if len(reasoning_types) > 1 else None
                
                # Update callback with reasoning type info if provided
                if update_callback:
                    await update_callback("reasoning_switch", {
                        "reasoning_types": reasoning_types,
                        "is_combined": True
                    })
                
                logger.info(f"Using combined reasoning types: {reasoning_types} for query: {query[:100]}")
            else:
                # Just use the primary reasoning type
                full_context["reasoning_types"] = [reasoning_type]
                full_context["primary_reasoning"] = reasoning_type
                
                # Update callback with reasoning type info if provided
                if update_callback:
                    await update_callback("reasoning_switch", {
                        "reasoning_types": [reasoning_type],
                        "is_combined": False
                    })
                
                logger.info(f"Using reasoning type: {reasoning_type} for query: {query[:100]}")
            
            # Process the query using the orchestrator
            response = await self.orchestrator.process_query(
                query=query,
                conversation_id=conversation_id,
                user_id=user_id,
                reasoning_type=reasoning_type,
                max_steps=max_steps,
                update_callback=update_callback,
                context=full_context
            )
            
            # Update successful combinations if this was a combined approach
            if should_combine:
                self.successful_combos[combo_key] = self.successful_combos.get(combo_key, 0) + 1
            
            return response
        finally:
            # Remove request from processing set, regardless of success or failure
            if request_id in self.processing_requests:
                self.processing_requests.remove(request_id)
    
    async def process_agent(self, agent_id: str, query: str, context: Dict[str, Any]) -> AgentCommand:
        """
        Process a query with a specific agent
        
        Args:
            agent_id: The ID of the agent to use
            query: The query to process
            context: Context information for processing
            
        Returns:
            An AgentCommand with the agent's response or action
        """
        # Implementation will invoke the appropriate agent
        # This is a stub for now - will be implemented as needed
        return AgentCommand.response(f"Agent {agent_id} processed: {query[:50]}...")
    
    async def detect_reasoning_type(self, query: str, conversation_id: str = None) -> str:
        """
        Detect the most appropriate primary reasoning type for a query
        
        Args:
            query: The user's query
            conversation_id: The conversation ID
            
        Returns:
            The detected primary reasoning type
        """
        reasoning_types = await self.detect_multiple_reasoning_types(query, conversation_id)
        return reasoning_types[0] if reasoning_types else "conversational"
    
    async def detect_multiple_reasoning_types(self, query: str, conversation_id: str = None, max_types: int = 3) -> List[str]:
        """
        Detect multiple applicable reasoning types for a query, ranked by relevance
        
        Args:
            query: The user query
            conversation_id: Optional conversation ID for context
            max_types: Maximum number of reasoning types to return
            
        Returns:
            List[str]: List of reasoning types, sorted by relevance
        """
        await self.ensure_initialized()
        
        # Define patterns for different reasoning types
        patterns = {
            "sequential": r"(step[s]?[ -]by[ -]step|logical|explain why|explain how|think through|break down|walkthrough|reasoning|analysis)",
            "rag": r"(information|research|look up|find out|search for|latest|recent|news|article|data)",
            "verification": r"(verify|fact check|is it true|confirm|evidence|proof|reliable|ensure|validate)",
            "calculation": r"(calculate|compute|solve|equation|formula|math|add|multiply|divide|subtract|percentage|formula)",
            "creative": r"(creative|story|poem|imagine|pretend|fiction|narrative|write a|generate a)",
            "graph": r"(relationship|network|connect|graph|diagram|map the|connections between|linked|association)"
        }
        
        # Check each pattern against the query
        matches = {}
        for reasoning_type, pattern in patterns.items():
            # Count the number of matches for the pattern
            match_count = len(re.findall(pattern, query, re.IGNORECASE))
            if match_count > 0:
                matches[reasoning_type] = match_count
        
        # If no matches found, default to conversational
        if not matches:
            return ["conversational"]
        
        # Sort reasoning types by match count (descending)
        sorted_types = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N reasoning types
        result = [reasoning_type for reasoning_type, _ in sorted_types[:max_types]]
        
        # If result is empty, default to conversational
        if not result:
            result = ["conversational"]
        
        return result
    
    def _detect_secondary_reasoning_types(self, query: str, primary_type: str) -> List[str]:
        """
        Detect secondary reasoning types that should be used alongside the primary type
        
        Args:
            query: The user's query
            primary_type: The primary reasoning type already detected
            
        Returns:
            List of secondary reasoning types
        """
        secondary_types = []
        lower_query = query.lower()
        
        # Look for research patterns that would benefit from RAG
        if primary_type != "rag" and any(term in lower_query for term in [
            "research", "information about", "find", "search", "latest", "recent",
            "what is", "who is", "learn about", "tell me about", "data on", 
            "statistics", "facts", "history of", "examples of", "articles about"
        ]):
            secondary_types.append("rag")
            
        # Look for verification needs alongside other reasoning
        if primary_type != "verification" and any(term in lower_query for term in [
            "verify", "fact check", "is it true", "confirm", "validate", "accurate",
            "reliable", "trustworthy", "credible", "evidence", "prove", "disprove",
            "debunk", "authenticate", "cross-check", "check sources"
        ]):
            secondary_types.append("verification")
            
        # Sequential thinking often helps with other reasoning types
        if primary_type != "sequential" and any(term in lower_query for term in [
            "step by step", "analyze", "explain thoroughly", "break down",
            "detailed explanation", "in depth", "comprehensive", "process",
            "methodically", "procedure", "sequence", "one by one", "first", "then"
        ]):
            secondary_types.append("sequential")
            
        # Graph thinking for relationship mapping
        if primary_type != "graph" and any(term in lower_query for term in [
            "relationship", "connect", "map", "network", "graph", "between",
            "how does", "relate to", "linked", "association", "diagram", "structure",
            "conceptual map", "mind map", "flowchart", "connections between"
        ]):
            secondary_types.append("graph")
            
        # Multi-agent for complex problems requiring different perspectives
        if primary_type not in ["multi_agent", "workflow"] and (
            "multiple perspectives" in lower_query or 
            "different viewpoints" in lower_query or
            "various angles" in lower_query or
            "different experts" in lower_query or
            "debate" in lower_query or
            "pros and cons" in lower_query or
            "multiple approaches" in lower_query or
            "team of experts" in lower_query
        ):
            secondary_types.append("multi_agent")
            
        # Chain-of-thought for logical reasoning
        if primary_type != "cot" and any(term in lower_query for term in [
            "logical", "deduce", "infer", "reasoning", "think through",
            "conclusion", "premise", "argument", "logic", "deduction",
            "derive", "follow the logic", "rationale"
        ]):
            secondary_types.append("cot")
            
        # Creative reasoning for generative tasks
        if primary_type != "creative" and any(term in lower_query for term in [
            "creative", "imagine", "innovative", "generate", "design", "story",
            "novel idea", "brainstorm", "artistic", "unique", "original",
            "fantasy", "fiction", "imagine if", "create a", "invent"
        ]):
            secondary_types.append("creative")
            
        return secondary_types
    
    def _detect_reasoning_patterns(self, query: str, history: List[Dict[str, Any]] = None) -> List[str]:
        """
        Detect specific patterns in queries that suggest particular reasoning combinations
        
        Args:
            query: The user's query
            history: Optional conversation history
            
        Returns:
            List of reasoning types based on detected patterns
        """
        patterns = []
        lower_query = query.lower()
        
        # Pattern: RAG + Verification (research with fact-checking)
        if (re.search(r'(research|find|search).+(verify|fact.?check|confirm|accurate)', lower_query) or
            re.search(r'(verify|fact.?check|confirm).+(research|find|search)', lower_query)):
            patterns.extend(["rag", "verification"])
            
        # Pattern: Sequential + RAG (step-by-step with research)
        if (re.search(r'(step by step|analyze|explain).+(research|information|find)', lower_query) or
            re.search(r'(research|information|find).+(step by step|analyze|explain)', lower_query)):
            patterns.extend(["sequential", "rag"])
            
        # Pattern: Graph + RAG (mapping relationships with research)
        if (re.search(r'(map|graph|connect|relationship).+(research|information|find)', lower_query) or
            re.search(r'(research|information|find).+(map|graph|connect|relationship)', lower_query)):
            patterns.extend(["graph", "rag"])
            
        # Pattern: Multi-agent + Verification (multiple perspectives with checking)
        if (re.search(r'(multiple perspectives|different viewpoints|debate).+(verify|confirm|accurate)', lower_query) or
            re.search(r'(verify|confirm|accurate).+(multiple perspectives|different viewpoints|debate)', lower_query)):
            patterns.extend(["multi_agent", "verification"])
            
        # Pattern: Creative + Sequential (creative with structured thinking)
        if (re.search(r'(creative|imagine|story).+(step by step|methodically|process)', lower_query) or
            re.search(r'(step by step|methodically|process).+(creative|imagine|story)', lower_query)):
            patterns.extend(["creative", "sequential"])
            
        # Pattern: CoT + Verification (logical reasoning with fact-checking)
        if (re.search(r'(logical|deduce|reasoning).+(verify|check|confirm)', lower_query) or
            re.search(r'(verify|check|confirm).+(logical|deduce|reasoning)', lower_query)):
            patterns.extend(["cot", "verification"])
            
        # Pattern: Step-back + Graph (higher-level perspective with relationships)
        if (re.search(r'(big picture|overview|broader context).+(connect|relationship|map)', lower_query) or
            re.search(r'(connect|relationship|map).+(big picture|overview|broader context)', lower_query)):
            patterns.extend(["step_back", "graph"])
        
        return patterns
            
    def _fallback_reasoning_detection(self, query: str) -> List[str]:
        """
        Perform fallback reasoning type detection when the detector is unavailable
        
        Args:
            query: The user's query
            
        Returns:
            List of detected reasoning types
        """
        lower_query = query.lower()
        reasoning_types = []
        
        # Detect primary reasoning type
        if any(term in lower_query for term in ["step by step", "analyze", "break down", "thorough analysis"]):
            reasoning_types.append("sequential")
        elif any(term in lower_query for term in ["search", "find", "lookup", "information about", "what is", "latest"]):
            reasoning_types.append("rag")
        elif any(term in lower_query for term in ["verify", "fact check", "confirm", "is it true"]):
            reasoning_types.append("verification")
        elif any(term in lower_query for term in ["create", "generate", "write", "story", "poem", "imagine"]):
            reasoning_types.append("creative")
        elif any(term in lower_query for term in ["calculate", "compute", "solve", "equation", "math"]):
            reasoning_types.append("calculation")
        elif any(term in lower_query for term in ["map", "graph", "network", "connections", "relationship"]):
            reasoning_types.append("graph")
        elif any(term in lower_query for term in ["multi agent", "multiple perspectives", "different viewpoints", "debate"]):
            reasoning_types.append("multi_agent")
        elif any(term in lower_query for term in ["chain of thought", "cot", "logical reasoning", "deduce"]):
            reasoning_types.append("cot")
        elif any(term in lower_query for term in ["step back", "big picture", "broader context", "overall view"]):
            reasoning_types.append("step_back")
        elif any(term in lower_query for term in ["workflow", "process flow", "graph workflow", "langgraph"]):
            reasoning_types.append("workflow")
        else:
            reasoning_types.append("conversational")
        
        # Add secondary reasoning types where appropriate
        primary_type = reasoning_types[0]
        secondary_types = self._detect_secondary_reasoning_types(query, primary_type)
        reasoning_types.extend([t for t in secondary_types if t not in reasoning_types])
        
        # Check for specific patterns
        pattern_types = self._detect_reasoning_patterns(query)
        for pattern_type in pattern_types:
            if pattern_type not in reasoning_types:
                reasoning_types.append(pattern_type)
        
        return reasoning_types
    
    async def should_combine_reasoning(self, query: str, conversation_id: str = None) -> bool:
        """
        Determine if multiple reasoning types should be combined for this query
        
        Args:
            query: The user query
            conversation_id: Optional conversation ID for context
            
        Returns:
            bool: True if reasoning types should be combined
        """
        # Detect multiple reasoning types
        reasoning_types = await self.detect_multiple_reasoning_types(query, conversation_id, max_types=2)
        
        # Only consider combining if we have at least 2 types
        if len(reasoning_types) < 2:
            return False
        
        # Define effective combinations
        effective_combinations = [
            {"sequential", "rag"},
            {"verification", "rag"},
            {"calculation", "sequential"},
            {"creative", "sequential"},
            {"graph", "rag"},
            {"graph", "verification"}
        ]
        
        # Check if detected types form an effective combination
        detected_set = set(reasoning_types[:2])
        for combination in effective_combinations:
            if detected_set.issubset(combination) or combination.issubset(detected_set):
                return True
        
        # Special case: For complex questions with multiple keywords, prefer combination
        complex_patterns = [
            r"(why.*how)",
            r"(calculate.*explain)",
            r"(verify.*explain)",
            r"(creative.*structure)",
            r"(search.*analyze)",
            r"(relationship.*data)"
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # Default to not combining
        return False
    
    async def reset_conversation(self, conversation_id: str) -> None:
        """
        Reset a specific conversation
        
        Args:
            conversation_id: The ID of the conversation to reset
        """
        await self.ensure_initialized()
        
        # Reset orchestrator conversation
        if self.orchestrator and hasattr(self.orchestrator, 'reset_conversation'):
            await self.orchestrator.reset_conversation(conversation_id)
        elif self.orchestrator and hasattr(self.orchestrator, 'conversation_memories') and conversation_id in self.orchestrator.conversation_memories:
            self.orchestrator.conversation_memories[conversation_id] = []
            
        logger.info(f"Reset conversation {conversation_id}")
    
    async def clear_user_data(self, user_id: str) -> None:
        """
        Clear all data for a specific user
        
        Args:
            user_id: The ID of the user whose data to clear
        """
        await self.ensure_initialized()
        
        # Clear orchestrator data
        if self.orchestrator and hasattr(self.orchestrator, 'clear_user_data'):
            await self.orchestrator.clear_user_data(user_id)
        
        # Clear memory data
        if self.memory_manager:
            await self.memory_manager.clear_user_data(user_id)
        
        logger.info(f"Cleared all data for user {user_id}")
    
    async def get_agent_emoji(self, agent_type: str) -> str:
        """
        Get the emoji associated with an agent type
        
        Args:
            agent_type: The type of agent
            
        Returns:
            The emoji for the agent type
        """
        # Define emoji mapping for agent types
        emoji_map = {
            # Core reasoning types
            "conversational": "💬",
            "rag": "📚",
            "sequential": "🔄",
            "knowledge": "🧠",
            "verification": "✅",
            "creative": "🎨",
            "calculation": "🧮",
            "planning": "📝",
            "graph": "📊",
            "multi_agent": "👥",
            
            # Additional reasoning approaches
            "react": "⚡",
            "cot": "🔍",
            "step_back": "🔙",
            "workflow": "🔗",
            "reflection": "🪞",
            "symbolic": "🔣",
            "search": "🔎",
            "analysis": "📈",
            "synthesis": "🧩",
            "evaluation": "⚖️",
            "problem_solving": "🛠️",
            "brainstorming": "💭",
            "critique": "🔬",
            "explanation": "📋",
            "summarization": "📝",
            
            # Default fallback
            "default": "🤖"
        }
        
        # Return the emoji from the mapping, or default if not found
        return emoji_map.get(agent_type.lower(), emoji_map["default"])
    
    async def get_agent_emoji_and_description(self, agent_type: str) -> tuple:
        """
        Get the emoji and description for an agent type
        
        Args:
            agent_type: The type of agent
            
        Returns:
            Tuple of (emoji, description)
        """
        # Define descriptions for agent types
        descriptions = {
            "conversational": "Natural, friendly conversation with context awareness.",
            "rag": "Information retrieval and fact-based responses with sources.",
            "sequential": "Step-by-step logical reasoning for complex problems.",
            "knowledge": "Educational content and detailed explanations.",
            "verification": "Fact-checking and verification of information.",
            "creative": "Creative writing, storytelling, and idea generation.",
            "calculation": "Mathematical calculations and numerical analysis.",
            "planning": "Strategic planning and organization.",
            "graph": "Network and relationship analysis.",
            "multi_agent": "Multiple specialized agents working together.",
            "react": "Reasoning combined with action for problem-solving.",
            "cot": "Chain-of-thought reasoning for logical deductions.",
            "step_back": "Takes a step back to analyze a problem from a broader perspective.",
            "workflow": "Graph-based workflow that adapts based on task needs.",
            "reflection": "Self-reflection and evaluation of reasoning processes.",
            "synthesis": "Combining multiple sources and ideas into a cohesive whole."
        }
        
        # Get the emoji for the agent type
        emoji = await self.get_agent_emoji(agent_type)
        
        # Get the description or a default if not found
        description = descriptions.get(
            agent_type.lower(), 
            "A specialized AI reasoning mode."
        )
        
        return emoji, description
    
    async def format_with_agent_emoji(self, response: str, agent_type: str) -> tuple:
        """
        Format a response with the appropriate agent emoji and header
        
        Args:
            response: The response text to format
            agent_type: The type of agent/reasoning
            
        Returns:
            Tuple of (formatted_response, emoji)
        """
        # Check response cache to prevent duplicate processing
        cache_key = f"{hash(response)}:{agent_type}"
        if hasattr(self, '_response_format_cache') and cache_key in self._response_format_cache:
            return self._response_format_cache[cache_key]
            
        # Initialize cache if it doesn't exist
        if not hasattr(self, '_response_format_cache'):
            self._response_format_cache = {}
        
        # Get emoji and description
        emoji, description = await self.get_agent_emoji_and_description(agent_type)
        
        # Create a distinctive header showing the reasoning type
        header = f"{emoji} **{agent_type.capitalize()} Reasoning**"
        
        # Add a separator line between header and response content
        formatted_response = f"{header}\n\n{response}"
        
        # Cache the result
        result = (formatted_response, emoji)
        self._response_format_cache[cache_key] = result
        
        return result
        
    async def search_web(self, query: str) -> str:
        """
        Search the web for information
        
        Args:
            query: The search query
            
        Returns:
            Formatted search results
        """
        try:
            # Import search function from ai_utils
            from ..ai_utils import search_internet
            
            # Perform web search
            search_results = await search_internet(query)
            
            # Validate and format the results
            if not search_results or not search_results.strip():
                return f"No search results found for '{query}'. Please try a different search term."
                
            # Return search results with a heading
            return f"### Search Results for '{query}':\n\n{search_results}"
            
        except Exception as e:
            # Log the error and return a friendly message
            error_traceback = traceback.format_exc()
            print(f"Error in search_web: {e}\n{error_traceback}")
            return f"An error occurred while searching for '{query}': {str(e)[:100]}..."

    async def get_workflow_type(self, query: str, conversation_id: str = None) -> str:
        """
        Get the appropriate workflow type for the query based on reasoning types
        
        Args:
            query: The user query
            conversation_id: Optional conversation ID for context
            
        Returns:
            str: The workflow type to use
        """
        # Import workflow service (lazy import to avoid circular dependency)
        from . import workflow_service
        
        # Check if workflow service is available
        if workflow_service.workflow_service.is_workflow_available():
            # Use workflow service to detect workflow type
            return await workflow_service.workflow_service.detect_workflow_type(query, conversation_id)
        
        # Fallback: Determine workflow based on reasoning types
        reasoning_types = await self.detect_multiple_reasoning_types(query, conversation_id, max_types=2)
        
        # Map reasoning type combinations to workflow types
        if set(reasoning_types[:2]) == {"sequential", "rag"} or "sequential" in reasoning_types and "rag" in reasoning_types:
            return "sequential_rag"
        elif set(reasoning_types[:2]) == {"verification", "rag"} or "verification" in reasoning_types and "rag" in reasoning_types:
            return "verification_rag"
        elif set(reasoning_types[:2]) == {"calculation", "sequential"} or "calculation" in reasoning_types and "sequential" in reasoning_types:
            return "calculation_sequential"
        elif set(reasoning_types[:2]) == {"creative", "sequential"} or "creative" in reasoning_types and "sequential" in reasoning_types:
            return "creative_sequential"
        elif "graph" in reasoning_types and ("rag" in reasoning_types or "verification" in reasoning_types):
            return "graph_rag_verification"
        else:
            # Default to multi-agent for other combinations
            return "multi_agent"

    async def process_query(self, 
                            query: str, 
                            user_id: str, 
                            conversation_id: str = None,
                            reasoning_type: str = "conversational",
                            update_callback: Callable = None) -> str:
        """
        Process a query using a specific reasoning type
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            reasoning_type: The reasoning type to use (default: conversational)
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The response from the agent
        """
        await self.ensure_initialized()
        
        try:
            # Check if we should use a workflow instead of a single reasoning type
            should_combine = await self.should_combine_reasoning(query, conversation_id)
            
            if should_combine:
                # Import workflow service (lazy import to avoid circular dependency)
                from . import workflow_service
                
                # Check if workflow service is available
                if workflow_service.workflow_service.is_workflow_available():
                    # Determine the appropriate workflow type
                    workflow_type = await self.get_workflow_type(query, conversation_id)
                    
                    # Use workflow service for combined reasoning
                    return await workflow_service.workflow_service.process_with_workflow(
                        query=query,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        workflow_type=workflow_type,
                        update_callback=update_callback
                    )
            
            # If not combining or workflow service not available, use single reasoning type
            # Notify about the reasoning type being used
            if update_callback:
                await update_callback("reasoning_switch", {
                    "reasoning_types": [reasoning_type],
                    "is_combined": False
                })
            
            # Process according to reasoning type
            if reasoning_type == "sequential":
                return await self._process_sequential_reasoning(query, user_id, conversation_id, update_callback)
            elif reasoning_type == "rag":
                return await self._process_rag_reasoning(query, user_id, conversation_id, update_callback)
            elif reasoning_type == "verification":
                return await self._process_verification_reasoning(query, user_id, conversation_id, update_callback)
            elif reasoning_type == "calculation":
                return await self._process_calculation_reasoning(query, user_id, conversation_id, update_callback)
            elif reasoning_type == "creative":
                return await self._process_creative_reasoning(query, user_id, conversation_id, update_callback)
            elif reasoning_type == "graph":
                return await self._process_graph_reasoning(query, user_id, conversation_id, update_callback)
            elif reasoning_type == "multi_agent":
                # Import workflow service (lazy import to avoid circular dependency)
                from . import workflow_service
                
                # Use workflow service for multi-agent reasoning
                if workflow_service.workflow_service.is_workflow_available():
                    return await workflow_service.workflow_service.process_with_workflow(
                        query=query,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        workflow_type="multi_agent",
                        update_callback=update_callback
                    )
            else:
                # Default to conversational reasoning
                return await self._process_conversational_reasoning(query, user_id, conversation_id, update_callback)
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in process_query: {e}\n{error_traceback}")
            return f"I encountered an error while processing your request: {str(e)}"

    async def _process_conversational_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None) -> str:
        """
        Process a query using conversational reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The response from the conversational reasoning
        """
        try:
            await self.ensure_initialized()
            
            # Notify about thinking process if callback provided
            if update_callback:
                await update_callback("thinking", {
                    "thinking": "Thinking conversationally about your query..."
                })
                
                await update_callback("agent_switch", {
                    "agent_type": "conversational"
                })
            
            # Get user preferences
            from . import memory_service
            prefs = await memory_service.memory_service.get_user_preferences(user_id)
            
            # Get conversation history
            channel_id = conversation_id.split(':')[1] if conversation_id else None
            history = await memory_service.memory_service.get_conversation_history(user_id, channel_id)
            
            # Prepare system message with user preferences
            system_msg = self._create_system_message(prefs)
            
            # Prepare conversation context for AI
            context = [{"role": "system", "content": system_msg}]
            context.extend(history)
            context.append({"role": "user", "content": query})
            
            # Call AI provider
            from ..ai_utils import get_ai_provider
            ai_provider = await get_ai_provider()
            
            # Create a prompt that combines all messages into a single string
            combined_prompt = system_msg + "\n\n"
            for msg in history:
                role = msg["role"]
                content = msg["content"]
                combined_prompt += f"{role.capitalize()}: {content}\n\n"
            combined_prompt += f"User: {query}\n\nAssistant:"
            
            # Call the AI with the combined prompt
            response = await ai_provider.async_call(
                prompt=combined_prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            # If streaming is requested but not directly supported, simulate it
            if update_callback:
                # Simulate streaming by sending chunks of the response
                full_response = response
                chunk_size = max(20, len(full_response) // 10)  # Divide into ~10 chunks
                
                current_response = ""
                for i in range(0, len(full_response), chunk_size):
                    chunk = full_response[i:i+chunk_size]
                    current_response += chunk
                    await update_callback("update", {"content": current_response})
                    await asyncio.sleep(0.1)  # Short delay between chunks
                
                return full_response
            else:
                # Standard response processing
                return response
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error in _process_conversational_reasoning: {e}\n{error_traceback}")
            return f"I encountered an error while processing your request: {str(e)}"

    async def _process_sequential_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None) -> str:
        """
        Process a query using sequential reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The response from the sequential reasoning
        """
        try:
            await self.ensure_initialized()
            
            # Notify about thinking process if callback provided
            if update_callback:
                await update_callback("thinking", {
                    "thinking": "Breaking down the problem step by step..."
                })
                
                await update_callback("agent_switch", {
                    "agent_type": "sequential"
                })
            
            # Use the sequential_thinking_service to process the query
            from . import sequential_thinking_service
            
            response = await sequential_thinking_service.sequential_thinking_service.process_sequential_thinking(
                problem=query,
                context={"user_id": user_id, "conversation_id": conversation_id},
                prompt_style="sequential",
                enable_revision=True,
                session_id=conversation_id
            )
            
            # Extract the actual response from the tuple
            if isinstance(response, tuple) and len(response) == 2:
                success, actual_response = response
                return actual_response
            
            return response
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in _process_sequential_reasoning: {error_traceback}")
            return f"I encountered an error while processing your request with sequential reasoning: {str(e)}"

    async def _process_rag_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None) -> str:
        """
        Process a query using RAG (Retrieval-Augmented Generation) reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The response from the RAG reasoning
        """
        try:
            await self.ensure_initialized()
            
            # Notify about thinking process if callback provided
            if update_callback:
                await update_callback("thinking", {
                    "thinking": "Searching for relevant information..."
                })
                
                await update_callback("agent_switch", {
                    "agent_type": "rag"
                })
            
            # For now, we'll use a simplified approach
            # In a full implementation, this would query a vector database
            
            # Call the LLM with a RAG-focused prompt
            from ..ai_utils import get_ai_provider
            llm_provider = await get_ai_provider()
            
            system_message = (
                "You are a helpful assistant that provides factual information backed by reliable sources. "
                "Answer the question based on your knowledge, but clearly state if you're uncertain. "
                "Always provide references or sources when possible."
            )
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            # Call the LLM
            result = await llm_provider.generate_text(messages=messages)
            
            # Update conversation history if needed
            from . import memory_service
            if conversation_id:
                # Add user message to history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id,
                    entry={"role": "user", "content": query}
                )
                
                # Add assistant response to history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id,
                    entry={"role": "assistant", "content": result}
                )
            
            return result
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in _process_rag_reasoning: {error_traceback}")
            return f"I encountered an error while processing your request with RAG reasoning: {str(e)}"

    async def _process_verification_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None) -> str:
        """
        Process a query using verification reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The response from the verification reasoning
        """
        try:
            await self.ensure_initialized()
            
            # Notify about thinking process if callback provided
            if update_callback:
                await update_callback("thinking", {
                    "thinking": "Verifying information and checking facts..."
                })
                
                await update_callback("agent_switch", {
                    "agent_type": "verification"
                })
            
            # Call the LLM with a verification-focused prompt
            from ..ai_utils import get_ai_provider
            llm_provider = await get_ai_provider()
            
            system_message = (
                "You are a fact-checking assistant that verifies information. "
                "Carefully analyze the query, identify factual claims, and verify their accuracy. "
                "Provide evidence for your verification and clearly indicate confidence levels. "
                "If you detect misinformation, correct it politely with accurate information."
            )
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            # Call the LLM
            result = await llm_provider.generate_text(messages=messages)
            
            # Update conversation history if needed
            from . import memory_service
            if conversation_id:
                # Add entries to conversation history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id,
                    entry={"role": "user", "content": query}
                )
                
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id,
                    entry={"role": "assistant", "content": result}
                )
            
            return result
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in _process_verification_reasoning: {error_traceback}")
            return f"I encountered an error while processing your request with verification reasoning: {str(e)}"

    async def _process_calculation_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None) -> str:
        """
        Process a query using calculation reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The response from the calculation reasoning
        """
        try:
            await self.ensure_initialized()
            
            # Notify about thinking process if callback provided
            if update_callback:
                await update_callback("thinking", {
                    "thinking": "Calculating the mathematical solution..."
                })
                
                await update_callback("agent_switch", {
                    "agent_type": "calculation"
                })
            
            # Use the symbolic reasoning service if available
            from . import symbolic_reasoning_service
            
            # Check if the symbolic reasoning service is available
            if hasattr(symbolic_reasoning_service, 'symbolic_reasoning_service'):
                # Process calculation using the symbolic reasoning service
                return await symbolic_reasoning_service.symbolic_reasoning_service.process_calculation(
                    query=query,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    update_callback=update_callback
                )
            
            # Fallback: Call the LLM with a calculation-focused prompt
            from ..ai_utils import get_ai_provider
            llm_provider = await get_ai_provider()
            
            system_message = (
                "You are a mathematical reasoning assistant. "
                "Solve calculations step-by-step, showing your work clearly. "
                "For complex problems, break them down into simpler parts. "
                "Double-check your calculations before providing the final answer."
            )
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            # Call the LLM
            result = await llm_provider.generate_text(messages=messages)
            
            # Update conversation history if needed
            from . import memory_service
            if conversation_id:
                # Add entries to conversation history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id,
                    entry={"role": "user", "content": query}
                )
                
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id,
                    entry={"role": "assistant", "content": result}
                )
            
            return result
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in _process_calculation_reasoning: {error_traceback}")
            return f"I encountered an error while processing your request with calculation reasoning: {str(e)}"

    async def _process_creative_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None) -> str:
        """
        Process a query using creative reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The response from the creative reasoning
        """
        try:
            await self.ensure_initialized()
            
            # Notify about thinking process if callback provided
            if update_callback:
                await update_callback("thinking", {
                    "thinking": "Being creative and generating innovative ideas..."
                })
                
                await update_callback("agent_switch", {
                    "agent_type": "creative"
                })
            
            # Call the LLM with a creative-focused prompt
            from ..ai_utils import get_ai_provider
            llm_provider = await get_ai_provider()
            
            system_message = (
                "You are a creative assistant with a vivid imagination. "
                "Generate engaging, imaginative, and original content. "
                "Think outside the box and provide unique perspectives. "
                "Use rich language, sensory details, and evocative imagery when appropriate."
            )
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            # Call the LLM
            result = await llm_provider.generate_text(messages=messages)
            
            # Update conversation history if needed
            from . import memory_service
            if conversation_id:
                # Add entries to conversation history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id,
                    entry={"role": "user", "content": query}
                )
                
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id,
                    entry={"role": "assistant", "content": result}
                )
            
            return result
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in _process_creative_reasoning: {error_traceback}")
            return f"I encountered an error while processing your request with creative reasoning: {str(e)}"

    async def _process_graph_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None) -> str:
        """
        Process a query using graph reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The response from the graph reasoning
        """
        try:
            await self.ensure_initialized()
            
            # Notify about thinking process if callback provided
            if update_callback:
                await update_callback("thinking", {
                    "thinking": "Analyzing relationships and connections in the data..."
                })
                
                await update_callback("agent_switch", {
                    "agent_type": "graph"
                })
            
            # Call the LLM with a graph reasoning focused prompt
            from ..ai_utils import get_ai_provider
            llm_provider = await get_ai_provider()
            
            system_message = (
                "You are a graph-based reasoning assistant that analyzes connections and relationships. "
                "Identify key entities and their relationships in the query. "
                "Map out connections, dependencies, and influences between elements. "
                "Provide insights based on the network structure of information."
            )
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            # Call the LLM
            result = await llm_provider.generate_text(messages=messages)
            
            # Update conversation history if needed
            from . import memory_service
            if conversation_id:
                # Add entries to conversation history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id,
                    entry={"role": "user", "content": query}
                )
                
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id,
                    entry={"role": "assistant", "content": result}
                )
            
            return result
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in _process_graph_reasoning: {error_traceback}")
            return f"I encountered an error while processing your request with graph reasoning: {str(e)}"

    def _create_system_message(self, prefs: dict) -> str:
        """
        Create a system message based on user preferences
        
        Args:
            prefs: User preferences dictionary
            
        Returns:
            str: Formatted system message for the AI
        """
        # Default system message
        system_message = (
            "You are a helpful, friendly AI assistant in a Discord chat. "
            "Provide thoughtful, accurate responses to users' questions and engage in natural conversation. "
            "Be concise but comprehensive, and always strive to be helpful and informative."
        )
        
        # Apply user preferences if available
        if prefs:
            # Apply tone preference
            tone = prefs.get('tone')
            if tone:
                if tone == 'professional':
                    system_message += " Maintain a professional, formal tone in your responses."
                elif tone == 'casual':
                    system_message += " Keep your tone casual and conversational, like chatting with a friend."
                elif tone == 'friendly':
                    system_message += " Be warm, approachable and friendly in your interactions."
                elif tone == 'technical':
                    system_message += " Use technical language and provide detailed explanations."
            
            # Apply verbosity preference
            verbosity = prefs.get('verbosity')
            if verbosity:
                if verbosity == 'concise':
                    system_message += " Keep your responses brief and to the point."
                elif verbosity == 'detailed':
                    system_message += " Provide detailed explanations and thorough responses."
                    
            # Apply expertise areas if specified
            expertise = prefs.get('expertise')
            if expertise and isinstance(expertise, list) and len(expertise) > 0:
                expertise_str = ", ".join(expertise)
                system_message += f" You have special expertise in: {expertise_str}."
                
            # Apply personality traits if specified
            personality = prefs.get('personality')
            if personality and isinstance(personality, list) and len(personality) > 0:
                personality_str = ", ".join(personality)
                system_message += f" Your personality can be described as: {personality_str}."
        
        return system_message

# Create a singleton instance for global access
agent_service = AgentService() 