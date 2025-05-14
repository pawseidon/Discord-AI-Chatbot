"""
Agent Service

This module provides a centralized service for agent orchestration,
allowing both event handlers and command cogs to use the same agent system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
import discord

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
        channel_id: str,
        update_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_steps: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process a user query using the agent orchestration system
        
        Args:
            query: The user's query to process
            user_id: The ID of the user making the query
            channel_id: The ID of the channel where the query was made
            update_callback: Optional callback for status updates during processing
            max_steps: Maximum number of processing steps (delegations, tool calls)
            context: Additional context data to include in the processing
            
        Returns:
            The response from the agent system
        """
        await self.ensure_initialized()
        
        # Create a conversation ID that's consistent for this user-channel combination
        conversation_id = f"{user_id}-{channel_id}"
        
        # Merge provided context with default context
        full_context = {
            "user_id": user_id,
            "channel_id": channel_id,
            "conversation_id": conversation_id
        }
        if context:
            full_context.update(context)
        
        # Process the query using the orchestrator
        return await self.orchestrator.process_query(
            query=query,
            conversation_id=conversation_id,
            user_id=user_id,
            max_steps=max_steps,
            update_callback=update_callback
        )
    
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
        if conversation_id and self.orchestrator and conversation_id in self.orchestrator.conversation_memories:
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
            "what is", "who is", "learn about", "tell me about"
        ]):
            secondary_types.append("rag")
            
        # Look for verification needs alongside other reasoning
        if primary_type != "verification" and any(term in lower_query for term in [
            "verify", "fact check", "is it true", "confirm", "validate", "accurate",
            "reliable", "trustworthy", "credible", "evidence"
        ]):
            secondary_types.append("verification")
            
        # Sequential thinking often helps with other reasoning types
        if primary_type != "sequential" and any(term in lower_query for term in [
            "step by step", "analyze", "explain thoroughly", "break down",
            "detailed explanation", "in depth", "comprehensive"
        ]):
            secondary_types.append("sequential")
            
        # Graph thinking for relationship mapping
        if primary_type != "graph" and any(term in lower_query for term in [
            "relationship", "connect", "map", "network", "graph", "between",
            "how does", "relate to", "linked", "association", "diagram"
        ]):
            secondary_types.append("graph")
            
        # Multi-agent for complex problems requiring different perspectives
        if primary_type not in ["multi_agent", "workflow"] and (
            "multiple perspectives" in lower_query or 
            "different viewpoints" in lower_query or
            "various angles" in lower_query or
            "different experts" in lower_query or
            "debate" in lower_query
        ):
            secondary_types.append("multi_agent")
            
        return secondary_types
            
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
        else:
            reasoning_types.append("conversational")
        
        # Add secondary reasoning types where appropriate
        primary_type = reasoning_types[0]
        secondary_types = self._detect_secondary_reasoning_types(query, primary_type)
        reasoning_types.extend([t for t in secondary_types if t not in reasoning_types])
        
        return reasoning_types
    
    async def should_combine_reasoning(self, query: str) -> bool:
        """
        Determine if a query would benefit from combining multiple reasoning approaches
        
        Args:
            query: The user's query
            
        Returns:
            Boolean indicating if multiple reasoning should be used
        """
        # Get all applicable reasoning types
        reasoning_types = await self.detect_multiple_reasoning_types(query)
        
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
            {"creative", "sequential"}         # Creative with structured thinking
        ]
        
        # Check if any good combination exists in our detected types
        detected_set = set(reasoning_types[:2])  # Just check the top two types
        for good_combo in good_combinations:
            if detected_set.issuperset(good_combo):
                return True
                
        # By default, only use the primary reasoning if no good combo found
        return False
    
    async def clear_user_data(self, user_id: str) -> None:
        """
        Clear all data for a specific user
        
        Args:
            user_id: The ID of the user whose data to clear
        """
        await self.ensure_initialized()
        
        # Clear orchestrator data
        if self.orchestrator:
            self.orchestrator.clear_user_data(user_id)
        
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
            "workflow": "Graph-based workflow that adapts based on task needs."
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
        Format a response with an agent emoji
        
        Args:
            response: The message content
            agent_type: The type of agent that generated the response
            
        Returns:
            Tuple of (formatted_response, emoji)
        """
        emoji = await self.get_agent_emoji(agent_type)
        
        if emoji:
            return f"{emoji} {response}", emoji
        else:
            return response, None

# Create a singleton instance for global access
agent_service = AgentService() 