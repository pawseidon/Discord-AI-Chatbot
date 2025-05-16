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
import time
import json

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
        channel_id: str = None,
        search_results: str = None,
        display_raw_results: bool = False
    ) -> str:
        """
        Process a user query, adding search results to the history. This is the main entry point for
        processing user queries.
        
        Args:
            query: The query to process
            user_id: The user ID for context and memory
            conversation_id: Conversation ID (typically guildId:channelId)
            reasoning_type: Optional reasoning type to use (otherwise auto-detected)
            update_callback: Optional callback for streaming progress updates
            max_steps: Maximum steps for multi-step reasoning
            context: Optional additional context
            channel_id: Optional channel ID (can be parsed from conversation_id)
            search_results: Optional pre-fetched search results
            display_raw_results: Whether to display raw search results to the user
            
        Returns:
            str: Response to the query
        """
        await self.ensure_initialized()
        
        # Initialize context if not provided
        if context is None:
            context = {}
            
        # Add search_results to context if provided
        if search_results:
            context["search_results"] = search_results
            
        # Add display_raw_results to context if True
        if display_raw_results:
            context["display_raw_results"] = display_raw_results
        
        # Parse channel_id from conversation_id if not provided
        if not channel_id and conversation_id and ":" in conversation_id:
            channel_id = conversation_id.split(":")[-1]
            
        # Auto-detect reasoning type if not specified
        if not reasoning_type:
            reasoning_type = await self.detect_reasoning_type(query, conversation_id)
            
        # Determine if we should use a workflow service (complex reasoning) 
        should_use_workflow = False
        prefer_workflow = False
        
        # Get user preferences
        from . import memory_service
        user_prefs = await memory_service.memory_service.get_user_preferences(user_id)
        
        # Check if user has enabled workflows in preferences
        if user_prefs and "enable_workflows" in user_prefs:
            prefer_workflow = user_prefs.get("enable_workflows") is True
            
        # Check if configuration allows workflows and query is suitable
        from bot_utilities.config import config
        if config.get("ENABLE_WORKFLOWS") and prefer_workflow:
            # Check if query is complex enough for workflow
            if len(query.split()) >= 10 or any(x in query.lower() for x in ["complex", "thorough", "workflow", "detail", "elaborate"]):
                should_use_workflow = True
                
        # Add interleaved_format for clear thought revision visibility in context if using sequential reasoning
        if reasoning_type == "sequential" or "sequential" in reasoning_type:
            context["interleaved_format"] = True
                
        # Process with workflow service if appropriate
        if should_use_workflow:
            try:
                from . import workflow_service
                # Check if workflow service is available
                if await workflow_service.workflow_service.is_workflow_available():
                    # Get workflow type (auto-detected if not specified)
                    workflow_type = None
                    if user_prefs and "workflow_type" in user_prefs:
                        workflow_type = user_prefs.get("workflow_type")
                        if workflow_type == "auto":
                            workflow_type = None  # Let workflow service auto-detect
                            
                    # Process with workflow
                    return await workflow_service.workflow_service.process_with_workflow(
                        query=query,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        workflow_type=workflow_type,
                        update_callback=update_callback,
                        search_results=search_results,
                        display_raw_results=display_raw_results
                    )
            except Exception as e:
                print(f"Error using workflow service: {e}")
                traceback.print_exc()
                # Fall back to regular processing
        
        # Process using specialized reasoning agents
        if reasoning_type == "conversational":
            return await self._process_conversational_reasoning(query, user_id, conversation_id, update_callback, context)
        elif reasoning_type == "sequential":
            return await self._process_sequential_reasoning(query, user_id, conversation_id, update_callback, context)
        elif reasoning_type == "rag":
            return await self._process_rag_reasoning(query, user_id, conversation_id, update_callback, context)
        elif reasoning_type == "verification":
            return await self._process_verification_reasoning(query, user_id, conversation_id, update_callback, context)
        elif reasoning_type == "calculation":
            return await self._process_calculation_reasoning(query, user_id, conversation_id, update_callback, context)
        elif reasoning_type == "creative":
            return await self._process_creative_reasoning(query, user_id, conversation_id, update_callback, context)
        elif reasoning_type == "graph":
            return await self._process_graph_reasoning(query, user_id, conversation_id, update_callback, context)
        else:
            # Default to conversational reasoning
            return await self._process_conversational_reasoning(query, user_id, conversation_id, update_callback, context)
    
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
        
        # Pre-process the query to handle special cases
        lower_query = query.lower()
        
        # Handle "how to" questions as sequential by default
        is_how_to = re.search(r'(how to|how do I|how can I|steps to|guide for|tutorial|instructions for)', lower_query)
        
        # Define patterns for different reasoning types
        patterns = {
            "sequential": r"(step[s]?[ -]by[ -]step|logical|explain why|explain how|think through|break down|walkthrough|reasoning|analysis)",
            "rag": r"(information|research|look up|find out|search for|latest|recent|news|article|data)",
            "verification": r"(verify|fact check|is it true|confirm|evidence|proof|reliable|ensure|validate)",
            "calculation": r"(calculate|compute|math problem|equation|formula|numerical|add|multiply|divide|subtract|percentage|formula|=|\+|\-|\*|\/|\^|sqrt|square root|solve for x|find the value|algebra|calculus|arithmetic|[\d]+[\s]*[\+\-\*\/\^])",
            "creative": r"(creative|story|poem|imagine|pretend|fiction|narrative|write a|generate a)",
            "graph": r"(relationship|network|connect|graph|diagram|map the|connections between|linked|association)"
        }
        
        # Check for calculation patterns that should override others
        is_calculation = False
        
        # Check for mathematical expressions
        if re.search(r'(\d+\s*[\+\-\*\/\^]\s*\d+)', query) or \
            re.search(r'(solve for \w+|find the value of \w+|calculate \w+|compute \w+|evaluate \w+)', lower_query):
            is_calculation = True
        
        # Check for equation patterns
        if "=" in query or re.search(r'x\s*=|y\s*=|f\(x\)\s*=', query):
            is_calculation = True
        
        # Avoid treating "how to solve [non-math problem]" as calculation
        if re.search(r'(how to solve|how do I solve)', lower_query):
            # Only mark as calculation if there are clear math terms
            if not any(term in lower_query for term in [
                "equation", "formula", "math", "calculation", "algebra", "geometric", 
                "arithmetic", "variable", "value of", "function", "derivative", "integral",
                "polynomial", "logarithm", "exponential", "number", "digit", "decimal", "fraction"
            ]):
                is_calculation = False
                # Boost sequential for these types of queries
                if is_how_to:
                    return ["sequential", "rag", "conversational"]
        
        # Check each pattern against the query
        matches = {}
        for reasoning_type, pattern in patterns.items():
            # Count the number of matches for the pattern
            match_count = len(re.findall(pattern, query, re.IGNORECASE))
            
            # Special case for calculation
            if reasoning_type == "calculation" and is_calculation:
                matches[reasoning_type] = match_count + 10  # Give it a boost if special patterns detected
            elif reasoning_type == "sequential" and is_how_to:
                matches[reasoning_type] = match_count + 5   # Give sequential a boost for how-to questions
            elif match_count > 0:
                matches[reasoning_type] = match_count
        
        # If no matches found, default to conversational
        if not matches:
            # For how-to questions with no specific matches, default to sequential
            if is_how_to:
                return ["sequential", "rag", "conversational"]
            return ["conversational"]
        
        # Sort reasoning types by match count (descending)
        sorted_types = sorted(matches.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N reasoning types
        result = [reasoning_type for reasoning_type, _ in sorted_types[:max_types]]
        
        # If result is empty, default to conversational
        if not result:
            result = ["conversational"]
        
        # Log the detection result for debugging
        logger.info(f"Detected reasoning types for query '{query}': {result}")
        
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
            bool: Whether multiple reasoning types should be combined
        """
        # Check for explicit combination requests in the query
        lower_query = query.lower()
        combination_markers = [
            "step by step and search", "search and verify", "verify and explain",
            "combine multiple reasoning", "analyze from different perspectives",
            "both search and explain", "calculate and explain", "creative and structured"
        ]
        
        for marker in combination_markers:
            if marker in lower_query:
                return True
                
        # Check query complexity - longer queries often benefit from combined reasoning
        if len(query.split()) > 25:  # Long complex queries
            return True
            
        # Check query patterns that typically benefit from combined reasoning
        complex_patterns = [
            r"(explain|analyze|break down) .* (with evidence|with sources|from research)",
            r"(verify|fact check) .* (and explain|step by step)",
            r"(calculate|solve) .* (and show steps|and explain)",
            r"(write|create|generate) .* (with structure|structured|organized)"
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, lower_query, re.IGNORECASE):
                return True
                
        # Default to simpler single reasoning for most queries
        return False

    async def execute_reasoning(self,
                         reasoning_type: str,
                         query: str,
                         user_id: str,
                         conversation_id: str = None,
                         update_callback: Callable = None,
                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a specific reasoning strategy
        
        Args:
            reasoning_type: The type of reasoning to use
            query: The query to process
            user_id: User ID for memory
            conversation_id: Optional conversation ID
            update_callback: Optional callback for updates
            context: Optional additional context
            
        Returns:
            Dict[str, Any]: The result of the reasoning process
        """
        try:
            await self.ensure_initialized()
            
            # Add context if not provided
            if context is None:
                context = {}
                
            if "user_id" not in context:
                context["user_id"] = user_id
                
            if "conversation_id" not in context and conversation_id:
                context["conversation_id"] = conversation_id
            
            # Process according to reasoning type
            if reasoning_type == "sequential":
                response = await self._process_sequential_reasoning(query, user_id, conversation_id, update_callback)
                return {"response": response, "reasoning_type": reasoning_type}
                
            elif reasoning_type == "sequential_thinking":
                # Import here to avoid circular imports
                from . import sequential_thinking_service
                
                # Get parameters from context
                enable_revision = context.get("enable_revision", True)
                enable_reflection = context.get("enable_reflection", False)
                num_thoughts = context.get("num_thoughts", 5)
                temperature = context.get("temperature", 0.3)
                
                # Process with sequential thinking
                success, response = await sequential_thinking_service.sequential_thinking_service.process_sequential_thinking(
                    problem=query,
                    context=context,
                    prompt_style="sequential",
                    num_thoughts=num_thoughts,
                    temperature=temperature,
                    enable_revision=enable_revision,
                    enable_reflection=enable_reflection,
                    session_id=f"session_{user_id}_{conversation_id}"
                )
                
                return {"response": response, "success": success, "reasoning_type": reasoning_type}
                
            elif reasoning_type == "rag":
                response = await self._process_rag_reasoning(query, user_id, conversation_id, update_callback)
                return {"response": response, "reasoning_type": reasoning_type}
                
            elif reasoning_type == "verification":
                response = await self._process_verification_reasoning(query, user_id, conversation_id, update_callback)
                return {"response": response, "reasoning_type": reasoning_type}
                
            elif reasoning_type == "calculation":
                response = await self._process_calculation_reasoning(query, user_id, conversation_id, update_callback)
                return {"response": response, "reasoning_type": reasoning_type}
                
            elif reasoning_type == "creative":
                response = await self._process_creative_reasoning(query, user_id, conversation_id, update_callback)
                return {"response": response, "reasoning_type": reasoning_type}
                
            elif reasoning_type == "graph":
                response = await self._process_graph_reasoning(query, user_id, conversation_id, update_callback)
                return {"response": response, "reasoning_type": reasoning_type}
                
            elif reasoning_type == "multi_agent":
                # Use the multi-agent workflow
                from . import workflow_service
                if workflow_service.workflow_service.is_workflow_available():
                    response = await workflow_service.workflow_service.multi_agent_workflow(
                        query=query,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        update_callback=update_callback
                    )
                    return {"response": response, "reasoning_type": reasoning_type}
                else:
                    # Fall back to conversational
                    response = await self._process_conversational_reasoning(query, user_id, conversation_id, update_callback)
                    return {"response": response, "reasoning_type": reasoning_type}
            else:
                # Default to conversational reasoning
                response = await self._process_conversational_reasoning(query, user_id, conversation_id, update_callback)
                return {"response": response, "reasoning_type": "conversational"}
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in execute_reasoning: {error_traceback}")
            return {
                "error": str(e),
                "response": f"Error in {reasoning_type} reasoning: {str(e)}",
                "reasoning_type": reasoning_type
            }

    async def get_workflow_type(self, query: str, conversation_id: str = None) -> str:
        """
        Get the appropriate workflow type for a query
        
        Args:
            query: The user query
            conversation_id: Optional conversation ID for context
            
        Returns:
            str: The detected workflow type
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

    async def _process_conversational_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None, context: Dict[str, Any] = None) -> str:
        """
        Process a query using conversational reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID
            update_callback: Optional callback for streaming updates
            context: Optional additional context
            
        Returns:
            str: The response from the conversational reasoning
        """
        try:
            await self.ensure_initialized()
            
            # Get memory service
            from . import memory_service
            
            # Get user preferences
            from ..memory_utils import get_user_preferences
            user_prefs = await get_user_preferences(user_id)
            
            # Get language setting
            language = user_prefs.get("language", "en")
            
            # Get conversation history if available
            history = []
            if conversation_id:
                history = await memory_service.memory_service.get_conversation_history(
                    user_id=user_id,
                    channel_id=conversation_id.split(':')[-1] if ':' in conversation_id else None,
                    limit=10
                )
            
            # Create system message based on preferences
            system_msg = self._create_system_message(user_prefs)
            
            # Call AI provider
            from ..ai_utils import get_ai_provider
            ai_provider = await get_ai_provider()
            
            # Create context messages for a chat model
            messages = [{"role": "system", "content": system_msg}]
            for msg in history:
                if msg.get("role") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": query})
            
            # Call the AI with the messages list
            response = await ai_provider.generate_text(
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Add to conversation history if available
            if conversation_id:
                try:
                    # Add user message to history
                    await memory_service.memory_service.add_to_history(
                        user_id=user_id,
                        channel_id=conversation_id.split(':')[-1] if ':' in conversation_id else None,
                        entry={"role": "user", "content": query}
                    )
                    
                    # Add assistant response to history
                    await memory_service.memory_service.add_to_history(
                        user_id=user_id,
                        channel_id=conversation_id.split(':')[-1] if ':' in conversation_id else None,
                        entry={"role": "assistant", "content": response}
                    )
                except Exception as e:
                    print(f"Error adding to history: {e}")
            
            return response
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in _process_conversational_reasoning: {error_traceback}")
            return f"I encountered an error while processing your request: {str(e)}"

    async def _process_sequential_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None, context: Dict[str, Any] = None) -> str:
        """
        Process a query using sequential reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            context: Optional additional context
            
        Returns:
            str: The response from the sequential reasoning
        """
        try:
            await self.ensure_initialized()
            
            # Notify about thinking process if callback provided
            if update_callback:
                await update_callback("thinking", {
                    "thinking": "Breaking down the problem step by step with thought revision..."
                })
                
                await update_callback("agent_switch", {
                    "agent_type": "sequential"
                })
            
            # Use the sequential_thinking_service to process the query
            from . import sequential_thinking_service
            
            # Ensure the service is initialized
            await sequential_thinking_service.sequential_thinking_service.ensure_initialized()
            
            # Create context if not provided
            if not context:
                context = {"user_id": user_id, "conversation_id": conversation_id}
            else:
                # Update context with user_id and conversation_id if not already there
                if "user_id" not in context:
                    context["user_id"] = user_id
                if "conversation_id" not in context:
                    context["conversation_id"] = conversation_id
            
            # Add interleaved_format for clear thought revision visibility
            context["interleaved_format"] = True
            
            # Process the query with sequential thinking
            success, response = await sequential_thinking_service.sequential_thinking_service.process_sequential_thinking(
                problem=query,
                context=context,
                prompt_style="sequential",
                enable_revision=True,
                session_id=conversation_id or f"seq_{user_id}_{int(time.time())}"
            )
            
            # Add to conversation history if successful
            if success and conversation_id:
                # Extract channel_id from conversation_id
                channel_id = conversation_id.split(':')[1] if ':' in conversation_id else conversation_id
                
                from . import memory_service
                # Add user message to history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=channel_id,
                    entry={"role": "user", "content": query}
                )
                
                # Add assistant response to history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=channel_id,
                    entry={"role": "assistant", "content": response}
                )
            
            return response
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in _process_sequential_reasoning: {error_traceback}")
            return f"I encountered an error while processing your request with sequential reasoning: {str(e)}"

    async def _process_rag_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None, context: Dict[str, Any] = None) -> str:
        """
        Process a query using RAG reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            context: Optional additional context
            
        Returns:
            str: Response
        """
        try:
            # Get web search results
            from ..web_search import search_web
            search_results = await search_web(query)
            
            # Parse search results - handle both string and list formats
            if isinstance(search_results, str):
                formatted_results = search_results
            else:
                # Assume it's a list of dictionaries with title, url, and snippet
                try:
                    formatted_results = "\n\n".join(
                        f"Source {i+1}: {result['title']}\n{result['url']}\n{result['snippet']}"
                        for i, result in enumerate(search_results[:5])
                    )
                except (TypeError, KeyError):
                    # If parsing fails, just use the raw results
                    formatted_results = str(search_results)
            
            # If there are no search results, fall back to conversational
            if not formatted_results.strip():
                print(f"No search results found for '{query}'. Falling back to conversational reasoning.")
                if update_callback:
                    await update_callback("No search results found. Using conversational reasoning instead.", {})
                return await self._process_conversational_reasoning(query, user_id, conversation_id, update_callback)
            
            # Get user preferences
            from ..memory_utils import get_user_preferences
            user_prefs = await get_user_preferences(user_id)
            
            # Get history if available
            history = []
            if conversation_id:
                from . import memory_service
                history = await memory_service.memory_service.get_conversation_history(
                    user_id=user_id, 
                    channel_id=conversation_id.split(':')[-1] if ':' in conversation_id else None,
                    limit=5
                )
            
            # Create system message
            system_msg = f"""You are an AI assistant that provides helpful, accurate information based on search results.
            
User preferences: {json.dumps(user_prefs, indent=2)}

When answering, follow these guidelines:
1. Use the search results provided to inform your response
2. If the search results don't contain relevant information, say so
3. Prioritize recent and authoritative sources
4. Cite sources when referencing specific information
5. If the search results seem outdated or contradictory, note this in your response
6. Be concise and direct in your response, focusing on answering the user's query
"""
            
            # Create messages for context
            messages = [{"role": "system", "content": system_msg}]
            
            # Add history if available
            for msg in history[-5:]:  # Limit to last 5 messages
                if msg.get("role") and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add search results and query
            search_context = f"Search Results:\n{formatted_results}"
            messages.append({"role": "user", "content": f"{search_context}\n\nUser query: {query}"})
            
            # Call AI provider
            from ..ai_utils import get_ai_provider
            ai_provider = await get_ai_provider()
            
            # Generate response
            response = await ai_provider.generate_text(
                messages=messages,
                temperature=0.5,
                max_tokens=2000
            )
            
            # Add to conversation history if available
            if conversation_id:
                try:
                    from . import memory_service
                    # Add user message first
                    await memory_service.memory_service.add_to_history(
                        user_id=user_id,
                        channel_id=conversation_id.split(':')[-1] if ':' in conversation_id else None,
                        entry={"role": "user", "content": query}
                    )
                    
                    # Then add AI response
                    await memory_service.memory_service.add_to_history(
                        user_id=user_id,
                        channel_id=conversation_id.split(':')[-1] if ':' in conversation_id else None,
                        entry={"role": "assistant", "content": response}
                    )
                except Exception as e:
                    print(f"Error adding to history: {e}")
            
            return response
            
        except Exception as e:
            print(f"Error in _process_rag_reasoning: {e}")
            traceback.print_exc()
            return f"I encountered an error while researching information: {str(e)}"

    async def _process_verification_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None, context: Dict[str, Any] = None) -> str:
        """
        Process a query using verification reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            context: Optional additional context
            
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
                # Extract channel_id from conversation_id
                channel_id = conversation_id.split(':')[1] if ':' in conversation_id else conversation_id
                
                # Add entries to conversation history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=channel_id,
                    entry={"role": "user", "content": query}
                )
                
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=channel_id,
                    entry={"role": "assistant", "content": result}
                )
            
            return result
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in _process_verification_reasoning: {error_traceback}")
            return f"I encountered an error while processing your request with verification reasoning: {str(e)}"

    async def _process_calculation_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None, context: Dict[str, Any] = None) -> str:
        """
        Process a query using calculation reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            context: Optional additional context
            
        Returns:
            str: The response from the calculation reasoning
        """
        try:
            await self.ensure_initialized()
            
            # Notify about thinking process if callback provided
            if update_callback:
                await update_callback("thinking", {
                    "thinking": "Performing calculation and symbolic reasoning..."
                })
                
                await update_callback("agent_switch", {
                    "agent_type": "calculation"
                })
            
            # Use the symbolic reasoning service if available
            from . import symbolic_reasoning_service
            
            # Check if the symbolic reasoning service is available
            if hasattr(symbolic_reasoning_service, 'symbolic_reasoning_service'):
                # Process calculation using the symbolic reasoning service
                result = await symbolic_reasoning_service.symbolic_reasoning_service.process_calculation(
                    query=query,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    update_callback=update_callback
                )
                
                # Check if the result is a dictionary (new format) or string (old format)
                if isinstance(result, dict):
                    # Extract the response field or use default message
                    response = result.get("response", "I couldn't process this calculation.")
                    
                    # Update conversation history if needed
                    from . import memory_service
                    if conversation_id:
                        # Extract channel_id from conversation_id
                        channel_id = conversation_id.split(':')[1] if ':' in conversation_id else conversation_id
                        
                        # Add entries to conversation history
                        await memory_service.memory_service.add_to_history(
                            user_id=user_id,
                            channel_id=channel_id,
                            entry={"role": "user", "content": query}
                        )
                        
                        await memory_service.memory_service.add_to_history(
                            user_id=user_id,
                            channel_id=channel_id,
                            entry={"role": "assistant", "content": response}
                        )
                    
                    return response
                else:
                    # It's already a string, return as is
                    return result
            
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
                # Extract channel_id from conversation_id
                channel_id = conversation_id.split(':')[1] if ':' in conversation_id else conversation_id
                
                # Add entries to conversation history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=channel_id,
                    entry={"role": "user", "content": query}
                )
                
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=channel_id,
                    entry={"role": "assistant", "content": result}
                )
            
            return result
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in _process_calculation_reasoning: {error_traceback}")
            return f"I encountered an error while processing your request with calculation reasoning: {str(e)}"

    async def _process_creative_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None, context: Dict[str, Any] = None) -> str:
        """
        Process a query using creative reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            context: Optional additional context
            
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
                # Extract channel_id from conversation_id
                channel_id = conversation_id.split(':')[1] if ':' in conversation_id else conversation_id
                
                # Add entries to conversation history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=channel_id,
                    entry={"role": "user", "content": query}
                )
                
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=channel_id,
                    entry={"role": "assistant", "content": result}
                )
            
            return result
        
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in _process_creative_reasoning: {error_traceback}")
            return f"I encountered an error while processing your request with creative reasoning: {str(e)}"

    async def _process_graph_reasoning(self, query: str, user_id: str, conversation_id: str = None, update_callback: Callable = None, context: Dict[str, Any] = None) -> str:
        """
        Process a query using graph reasoning
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            update_callback: Optional callback for streaming updates
            context: Optional additional context
            
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
                # Extract channel_id from conversation_id
                channel_id = conversation_id.split(':')[1] if ':' in conversation_id else conversation_id
                
                # Add entries to conversation history
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=channel_id,
                    entry={"role": "user", "content": query}
                )
                
                await memory_service.memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=channel_id,
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