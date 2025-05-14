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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('agent_service')

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
    
    async def detect_multiple_reasoning_types(self, query: str, conversation_id: str = None) -> List[str]:
        """
        Detect multiple reasoning types that may be applicable to a query
        
        Args:
            query: The user's query
            conversation_id: The conversation ID
            
        Returns:
            List of applicable reasoning types in order of relevance
        """
        await self.ensure_initialized()
        
        # Get conversation history
        history = []
        if conversation_id and self.orchestrator and hasattr(self.orchestrator, 'conversation_memories') and conversation_id in self.orchestrator.conversation_memories:
            history = self.orchestrator.conversation_memories[conversation_id]
        
        # Initialize reasoning types list
        reasoning_types = []
        
        # Use reasoning detector if available
        if self.reasoning_detector:
            try:
                primary_type, confidence = self.reasoning_detector.detect_reasoning_type(
                    query=query,
                    conversation_history=history,
                    conversation_id=conversation_id
                )
                reasoning_types.append(primary_type)
                
                # Add secondary reasoning types based on query characteristics
                secondary_types = self._detect_secondary_reasoning_types(query, primary_type)
                reasoning_types.extend([t for t in secondary_types if t not in reasoning_types])
                
                # Check for patterns that might indicate specific combined reasoning approaches
                pattern_types = self._detect_reasoning_patterns(query, history)
                for pattern_type in pattern_types:
                    if pattern_type not in reasoning_types:
                        reasoning_types.append(pattern_type)
                
            except Exception as e:
                logger.warning(f"Error in reasoning detector: {e}, using fallback")
                reasoning_types = self._fallback_reasoning_detection(query)
        else:
            # Use fallback detection
            reasoning_types = self._fallback_reasoning_detection(query)
        
        # Ensure we have at least one reasoning type
        if not reasoning_types:
            reasoning_types = ["conversational"]
            
        return reasoning_types
    
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
        Determine if a query would benefit from combining multiple reasoning approaches
        
        Args:
            query: The user's query
            conversation_id: The conversation ID
            
        Returns:
            Boolean indicating if multiple reasoning should be used
        """
        # Get all applicable reasoning types
        reasoning_types = await self.detect_multiple_reasoning_types(query, conversation_id)
        
        # Determine if we should combine based on number and types
        if len(reasoning_types) <= 1:
            return False
            
        # Some combinations work well together
        good_combinations = [
            {"rag", "verification"},           # Research with fact-checking
            {"sequential", "rag"},             # Step-by-step analysis with research
            {"graph", "rag"},                  # Relationship mapping with research
            {"sequential", "calculation"},     # Step-by-step with calculation
            {"multi_agent", "verification"},   # Multiple perspectives with verification
            {"creative", "sequential"},        # Creative with structured thinking
            {"cot", "verification"},           # Logical reasoning with fact-checking
            {"step_back", "graph"},            # Higher-level perspective with relationships
            {"rag", "cot"},                    # Research with logical reasoning
            {"sequential", "verification"},    # Step-by-step with verification
            {"workflow", "multi_agent"},       # Graph workflow with multiple agents
            {"cot", "sequential"},             # Chain-of-thought with sequential thinking
            {"rag", "creative"},               # Research with creative generation
            {"graph", "multi_agent"}           # Relationship mapping with multiple perspectives
        ]
        
        # Check if specific pattern combinations are detected
        pattern_types = self._detect_reasoning_patterns(query)
        if len(pattern_types) >= 2:
            pattern_set = set(pattern_types[:2])
            for good_combo in good_combinations:
                if pattern_set.issuperset(good_combo):
                    logger.info(f"Detected pattern combination: {pattern_types}")
                    return True
        
        # Check if any good combination exists in our detected types
        detected_set = set(reasoning_types[:2])  # Just check the top two types
        for good_combo in good_combinations:
            if detected_set.issuperset(good_combo):
                logger.info(f"Using good combination: {detected_set}")
                return True
        
        # Complex queries often benefit from combined reasoning
        if len(query.split()) > 25:  # If query is long and complex
            for primary in reasoning_types[:1]:
                for secondary in reasoning_types[1:2]:
                    combo_key = ":".join(sorted([primary, secondary]))
                    # If this combination has been successful before, use it again
                    if combo_key in self.successful_combos and self.successful_combos[combo_key] > 0:
                        logger.info(f"Using previously successful combination: {combo_key}")
                        return True
                
        # By default, only use the primary reasoning if no good combo found
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
        Format a response with the appropriate agent emoji
        
        Args:
            response: The response to format
            agent_type: The type of agent that generated the response
            
        Returns:
            Tuple of (formatted_response, emoji)
        """
        if not response:
            return "", ""
            
        # Get the emoji for the agent type
        emoji = await self.get_agent_emoji(agent_type)
        
        # Check if response already starts with an emoji
        if response.strip() and response.strip()[0] in emoji_map_inverted:
            # Already has an emoji, just return as is
            return response, emoji
            
        # Format the response with the emoji
        formatted_response = f"{emoji} {response}"
        
        return formatted_response, emoji

# Create a singleton instance for global access
agent_service = AgentService() 