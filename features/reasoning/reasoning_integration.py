import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set
from enum import Enum
import time
import traceback

from features.reasoning.reasoning_router import ReasoningRouter, ReasoningMethod
from features.caching import ReasoningCacheProxy

logger = logging.getLogger("reasoning_integration")

class TriggerType(Enum):
    """Types of natural language triggers for reasoning methods"""
    SEQUENTIAL = "sequential_trigger"
    RAG = "rag_trigger"
    CRAG = "crag_trigger"
    REACT = "react_trigger"
    GRAPH = "graph_trigger"
    HYBRID = "hybrid_trigger"
    VERIFICATION = "verification_trigger"

class ReasoningState:
    """Class to track the state of a reasoning process across multiple interactions"""
    
    def __init__(self, user_id: str, channel_id: Optional[str] = None):
        self.user_id = user_id
        self.channel_id = channel_id
        self.active_methods: Dict[ReasoningMethod, Dict[str, Any]] = {}
        self.method_results: Dict[ReasoningMethod, List[Dict[str, Any]]] = {}
        self.current_graph = None  # For graph-based reasoning
        self.verification_needed = False
        self.in_transition = False
        self.context_data: Dict[str, Any] = {}
        self.conversation_complexity = 0.0
        self.created_at = None
        self.last_updated = None
        
    def activate_method(self, method: ReasoningMethod, data: Dict[str, Any] = None):
        """Activate a reasoning method with optional data"""
        self.active_methods[method] = data or {}
        
    def deactivate_method(self, method: ReasoningMethod):
        """Deactivate a reasoning method"""
        if method in self.active_methods:
            del self.active_methods[method]
    
    def add_method_result(self, method: ReasoningMethod, result: Dict[str, Any]):
        """Add a result from a reasoning method"""
        if method not in self.method_results:
            self.method_results[method] = []
        self.method_results[method].append(result)
    
    def is_method_active(self, method: ReasoningMethod) -> bool:
        """Check if a method is active"""
        return method in self.active_methods
    
    def get_active_methods(self) -> List[ReasoningMethod]:
        """Get list of active methods"""
        return list(self.active_methods.keys())
    
    def update_context(self, key: str, value: Any):
        """Update context data"""
        self.context_data[key] = value
        
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context data"""
        return self.context_data.get(key, default)
    
    def get_full_context(self) -> Dict[str, Any]:
        """Get full context data dictionary"""
        return self.context_data.copy()
    
    def increase_complexity(self, amount: float = 0.1):
        """Increase conversation complexity"""
        self.conversation_complexity = min(1.0, self.conversation_complexity + amount)
        
    def decrease_complexity(self, amount: float = 0.1):
        """Decrease conversation complexity"""
        self.conversation_complexity = max(0.0, self.conversation_complexity - amount)

class StateManager:
    """Manager for reasoning states across users and channels"""
    
    def __init__(self, ttl: int = 3600):
        """
        Initialize state manager
        
        Args:
            ttl: Time-to-live for states in seconds
        """
        self.states: Dict[str, ReasoningState] = {}
        self.ttl = ttl
    
    def get_state_key(self, user_id: str, channel_id: Optional[str] = None) -> str:
        """Generate a state key from user and channel IDs"""
        if channel_id:
            return f"{user_id}:{channel_id}"
        return user_id
    
    def get_state(self, user_id: str, channel_id: Optional[str] = None) -> ReasoningState:
        """Get or create a reasoning state"""
        key = self.get_state_key(user_id, channel_id)
        
        if key not in self.states:
            self.states[key] = ReasoningState(user_id, channel_id)
        
        return self.states[key]
    
    def clear_state(self, user_id: str, channel_id: Optional[str] = None):
        """Clear a reasoning state"""
        key = self.get_state_key(user_id, channel_id)
        
        if key in self.states:
            del self.states[key]
    
    async def cleanup_expired_states(self):
        """Clean up expired states (should be called periodically)"""
        import time
        
        now = time.time()
        expired_keys = []
        
        for key, state in self.states.items():
            if state.last_updated and (now - state.last_updated) > self.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.states[key]
        
        return len(expired_keys)

class ReasoningIntegration:
    """
    Enhanced reasoning system with context-aware triggering and method coordination
    """
    
    def __init__(self, 
               reasoning_router: ReasoningRouter,
               ai_provider=None,
               context_cache=None):
        """
        Initialize reasoning integration
        
        Args:
            reasoning_router: Underlying reasoning router
            ai_provider: AI provider for context analysis
            context_cache: Cache for conversation context
        """
        self.router = reasoning_router
        self.ai_provider = ai_provider
        self.context_cache = context_cache
        self.state_manager = StateManager()
        
        # Natural language trigger patterns for different reasoning methods
        self.trigger_patterns = {
            TriggerType.SEQUENTIAL: [
                r"\b(step by step|sequential|one by one|in order|procedure)\b",
                r"\b(how (do|can|would) (i|we|you)|explain how to)\b",
                r"\b(walk me through|guide me|instructions for)\b",
                r"\bexplain\s+(\w+\s+){0,3}process\b",
                r"\b(first|second|third|next|finally|lastly)\b.*\b(then|after that)\b"
            ],
            TriggerType.RAG: [
                r"\b(what is|who is|when did|where is|why did|how does)\b.*\?",
                r"\b(information|data|details|facts) (on|about|regarding)\b",
                r"\btell me about\b",
                r"\b(history|background|origin) of\b",
                r"\bdefine\b"
            ],
            TriggerType.CRAG: [
                r"\b(remember|mentioned|earlier|before|previously)\b",
                r"\bwe (talked|discussed|mentioned)\b",
                r"\bin (relation|regards|reference) to (our|the) (conversation|discussion)\b",
                r"\bbuilding on (that|this|what we discussed)\b",
                r"\bfollowing up (on|from)\b"
            ],
            TriggerType.REACT: [
                r"\b(find|search|look up|get|fetch|retrieve)\b",
                r"\b(calculate|compute|determine)\b",
                r"\b(send|email|message|notify)\b",
                r"\b(create|generate|make)\b.*\b(file|document|spreadsheet|image|picture)\b",
                r"\b(check|verify|confirm|validate)\b"
            ],
            TriggerType.GRAPH: [
                r"\b(relationship|connection|link|associate|relate)\b",
                r"\b(compare|contrast|versus|vs|difference between)\b",
                r"\b(categorize|classify|group|organize)\b",
                r"\bhow.*\binfluence|affect|impact\b",
                r"\bcause and effect\b"
            ],
            TriggerType.HYBRID: [
                r"\b(complex|complicated|difficult|intricate)\b.*\b(problem|issue|question|situation)\b",
                r"\banalyz(e|ing)\b.*\b(system|network|framework|structure)\b",
                r"\b(multi|several|various|different)\b.*\b(factor|aspect|component|dimension)\b",
                r"\binterdependence\b",
                r"\b(systemic|holistic)\b"
            ],
            TriggerType.VERIFICATION: [
                r"\b(verify|fact[- ]check|confirm|is it true)\b",
                r"\b(are you sure|certain|confident)\b",
                r"\b(reliable|trustworthy|credible|accurate)\b",
                r"\bsource\b.*\b(information|data|claim|statement)\b",
                r"\bevidence\b"
            ]
        }
        
        # Compile regex patterns
        self.compiled_patterns = {
            trigger_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for trigger_type, patterns in self.trigger_patterns.items()
        }
        
        # Method combinations for hybrid approaches
        self.method_combinations = {
            (ReasoningMethod.SEQUENTIAL, ReasoningMethod.GRAPH_OF_THOUGHT): self._sequential_graph_hybrid,
            (ReasoningMethod.RAG, ReasoningMethod.SEQUENTIAL): self._rag_sequential_hybrid,
            (ReasoningMethod.REACT, ReasoningMethod.SEQUENTIAL): self._react_sequential_hybrid,
        }
    
    async def process_query(self,
                          query: str,
                          user_id: str,
                          channel_id: Optional[str] = None,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query with context-aware reasoning
        
        Args:
            query: User query
            user_id: User ID
            channel_id: Optional channel ID for context
            context: Additional context
            
        Returns:
            Processed result
        """
        context = context or {}
        
        # Get user's reasoning state
        state = self.state_manager.get_state(user_id, channel_id)
        
        # Update context with conversation history if available
        if self.context_cache:
            conversation_context = await self.context_cache.get_response_with_context(
                query=query,
                user_id=user_id,
                guild_id=context.get("guild_id"),
                channel_id=channel_id,
                context_messages=context.get("conversation_history", [])
            )
            
            if conversation_context:
                context["cached_context"] = conversation_context
        
        # Detect natural language triggers
        triggers = await self._detect_triggers(query, state)
        
        # Update reasoning state based on triggers and context
        await self._update_reasoning_state(state, triggers, query, context)
        
        # Determine which reasoning method(s) to use
        methods = await self._select_methods(state, query, context)
        
        # Check if we need to use a hybrid approach
        if len(methods) > 1 and tuple(sorted(methods)) in self.method_combinations:
            hybrid_method = self.method_combinations[tuple(sorted(methods))]
            result = await hybrid_method(query, user_id, state, context)
        elif len(methods) > 1:
            # Use the highest priority method if no specific hybrid is available
            method = methods[0]
            result = await self.router.route_query(
                query=query,
                user_id=user_id,
                method=method,
                context=context
            )
        elif len(methods) == 1:
            # Use the single selected method
            method = methods[0]
            result = await self.router.route_query(
                query=query,
                user_id=user_id,
                method=method,
                context=context
            )
        else:
            # Let the router automatically select the method
            result = await self.router.route_query(
                query=query,
                user_id=user_id,
                context=context
            )
        
        # Update state with results
        method_used = ReasoningMethod(result.get("method", "sequential"))
        state.add_method_result(method_used, result)
        
        # Store context for future reference if cache available
        if self.context_cache and "answer" in result:
            new_message = {
                "user": query,
                "bot": result["answer"],
                "timestamp": result.get("timestamp")
            }
            
            await self.context_cache.update_conversation_context(
                user_id=user_id,
                guild_id=context.get("guild_id"),
                channel_id=channel_id,
                new_message=new_message
            )
        
        return result
    
    async def _detect_triggers(self, query: str, state: ReasoningState) -> Dict[TriggerType, float]:
        """
        Detect natural language triggers in the query
        
        Args:
            query: User query
            state: Current reasoning state
            
        Returns:
            Dictionary of trigger types and confidence scores
        """
        triggers = {}
        
        # Check for trigger patterns
        for trigger_type, patterns in self.compiled_patterns.items():
            confidence = 0.0
            
            for pattern in patterns:
                if pattern.search(query):
                    confidence += 0.2  # Each match increases confidence
            
            if confidence > 0:
                triggers[trigger_type] = min(1.0, confidence)
        
        # Enhance with semantic analysis if available
        if self.ai_provider and hasattr(self.ai_provider, "analyze_query_intent"):
            try:
                intent_analysis = await self.ai_provider.analyze_query_intent(
                    query, state.get_full_context()
                )
                
                if intent_analysis and "detected_intents" in intent_analysis:
                    for intent, score in intent_analysis["detected_intents"].items():
                        if intent == "sequential" and score > 0.5:
                            triggers[TriggerType.SEQUENTIAL] = max(
                                triggers.get(TriggerType.SEQUENTIAL, 0), score
                            )
                        elif intent == "factual" and score > 0.5:
                            triggers[TriggerType.RAG] = max(
                                triggers.get(TriggerType.RAG, 0), score
                            )
                        elif intent == "contextual" and score > 0.5:
                            triggers[TriggerType.CRAG] = max(
                                triggers.get(TriggerType.CRAG, 0), score
                            )
                        elif intent == "action" and score > 0.5:
                            triggers[TriggerType.REACT] = max(
                                triggers.get(TriggerType.REACT, 0), score
                            )
                        elif intent == "relational" and score > 0.5:
                            triggers[TriggerType.GRAPH] = max(
                                triggers.get(TriggerType.GRAPH, 0), score
                            )
                        elif intent == "verification" and score > 0.5:
                            triggers[TriggerType.VERIFICATION] = max(
                                triggers.get(TriggerType.VERIFICATION, 0), score
                            )
            except Exception as e:
                logger.error(f"Failed to analyze query intent: {e}")
        
        return triggers
    
    async def _update_reasoning_state(self, 
                                    state: ReasoningState, 
                                    triggers: Dict[TriggerType, float],
                                    query: str,
                                    context: Dict[str, Any]):
        """
        Update reasoning state based on triggers and context
        
        Args:
            state: Current reasoning state
            triggers: Detected triggers
            query: User query
            context: Additional context
        """
        # Update state based on triggers
        if TriggerType.SEQUENTIAL in triggers and triggers[TriggerType.SEQUENTIAL] > 0.5:
            state.activate_method(ReasoningMethod.SEQUENTIAL)
            
        if TriggerType.RAG in triggers and triggers[TriggerType.RAG] > 0.5:
            state.activate_method(ReasoningMethod.RAG)
            
        if TriggerType.CRAG in triggers and triggers[TriggerType.CRAG] > 0.5:
            state.activate_method(ReasoningMethod.CRAG)
            
        if TriggerType.REACT in triggers and triggers[TriggerType.REACT] > 0.5:
            state.activate_method(ReasoningMethod.REACT)
            
        if TriggerType.GRAPH in triggers and triggers[TriggerType.GRAPH] > 0.5:
            state.activate_method(ReasoningMethod.GRAPH_OF_THOUGHT)
            
        if TriggerType.VERIFICATION in triggers and triggers[TriggerType.VERIFICATION] > 0.5:
            state.verification_needed = True
        
        # Increase complexity for complex queries
        if TriggerType.HYBRID in triggers and triggers[TriggerType.HYBRID] > 0.3:
            state.increase_complexity(0.2)
            
        # Update context data
        for key, value in context.items():
            state.update_context(key, value)
    
    async def _select_methods(self, 
                           state: ReasoningState, 
                           query: str, 
                           context: Dict[str, Any]) -> List[ReasoningMethod]:
        """
        Select reasoning methods based on state, query, and context
        
        Args:
            state: Current reasoning state
            query: User query
            context: Additional context
            
        Returns:
            List of reasoning methods to use, in priority order
        """
        selected_methods = []
        active_methods = state.get_active_methods()
        
        # Start with active methods
        if active_methods:
            selected_methods.extend(active_methods)
        
        # Add verification method if needed
        if state.verification_needed and ReasoningMethod.RAG not in selected_methods:
            selected_methods.append(ReasoningMethod.RAG)
        
        # Check complexity threshold
        if state.conversation_complexity > 0.6:
            # For high complexity, ensure we have either sequential or graph
            if (ReasoningMethod.SEQUENTIAL not in selected_methods and
                ReasoningMethod.GRAPH_OF_THOUGHT not in selected_methods):
                selected_methods.append(ReasoningMethod.SEQUENTIAL)
        
        # If no methods selected, let the router decide
        if not selected_methods:
            # Analyze query to suggest a method
            complexity = await self.router._analyze_query_complexity(query, context)
            
            if complexity > 0.7:
                selected_methods.append(ReasoningMethod.SEQUENTIAL)
            elif "search" in query.lower() or "find" in query.lower():
                selected_methods.append(ReasoningMethod.RAG)
            elif context.get("conversation_history", []):
                selected_methods.append(ReasoningMethod.CRAG)
        
        return selected_methods
    
    async def _sequential_graph_hybrid(self, 
                                    query: str, 
                                    user_id: str, 
                                    state: ReasoningState,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hybrid approach combining sequential thinking and graph reasoning
        
        Args:
            query: User query
            user_id: User ID
            state: Current reasoning state
            context: Additional context
            
        Returns:
            Processed result
        """
        # First, use sequential thinking to break down the problem
        sequential_result = await self.router.reasoning_methods[ReasoningMethod.SEQUENTIAL](
            query, user_id, context
        )
        
        if "thinking_steps" not in sequential_result:
            # If sequential thinking didn't produce steps, return as is
            return sequential_result
        
        # Extract concepts and relationships from sequential thinking
        concepts = []
        relationships = []
        
        for step in sequential_result.get("thinking_steps", []):
            # Extract key concepts from each step
            step_text = step.get("thought", "")
            if step_text:
                # Simple concept extraction (enhanced version would use NLP)
                words = re.findall(r'\b[A-Za-z]{4,}\b', step_text)
                concepts.extend([w for w in words if w.lower() not in ('this', 'that', 'then', 'when', 'where', 'which')])
        
        # Remove duplicates and limit to top concepts
        concepts = list(set(concepts))[:10]
        
        # Now use graph reasoning with these concepts
        graph_context = context.copy()
        graph_context["extracted_concepts"] = concepts
        graph_context["sequential_steps"] = sequential_result.get("thinking_steps", [])
        
        graph_result = await self.router.reasoning_methods[ReasoningMethod.GRAPH_OF_THOUGHT](
            query, user_id, graph_context
        )
        
        # Combine the results
        combined_result = {
            "answer": graph_result.get("answer", sequential_result.get("answer", "")),
            "method": "sequential_graph_hybrid",
            "method_emoji": "🧠📊",
            "thinking_process": {
                "sequential": sequential_result.get("thinking_steps", []),
                "graph": graph_result.get("graph", {})
            }
        }
        
        return combined_result
    
    async def _rag_sequential_hybrid(self, 
                                  query: str, 
                                  user_id: str, 
                                  state: ReasoningState,
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hybrid approach combining RAG and sequential thinking
        
        Args:
            query: User query
            user_id: User ID
            state: Current reasoning state
            context: Additional context
            
        Returns:
            Processed result
        """
        # First, use RAG to retrieve relevant information
        rag_result = await self.router.reasoning_methods[ReasoningMethod.RAG](
            query, user_id, context
        )
        
        # Now use sequential thinking with the retrieved information
        sequential_context = context.copy()
        sequential_context["retrieved_information"] = rag_result.get("retrieved_documents", [])
        
        sequential_result = await self.router.reasoning_methods[ReasoningMethod.SEQUENTIAL](
            query, user_id, sequential_context
        )
        
        # Combine the results
        combined_result = {
            "answer": sequential_result.get("answer", rag_result.get("answer", "")),
            "method": "rag_sequential_hybrid",
            "method_emoji": "📚🧠",
            "thinking_process": {
                "retrieved_documents": rag_result.get("retrieved_documents", []),
                "sequential": sequential_result.get("thinking_steps", [])
            }
        }
        
        return combined_result
    
    async def _react_sequential_hybrid(self, 
                                    query: str, 
                                    user_id: str, 
                                    state: ReasoningState,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hybrid approach combining ReAct and sequential thinking
        
        Args:
            query: User query
            user_id: User ID
            state: Current reasoning state
            context: Additional context
            
        Returns:
            Processed result
        """
        # First, use sequential thinking to formulate a plan
        sequential_result = await self.router.reasoning_methods[ReasoningMethod.SEQUENTIAL](
            query, user_id, context
        )
        
        # Extract action steps from sequential thinking
        action_steps = []
        
        for step in sequential_result.get("thinking_steps", []):
            step_text = step.get("thought", "")
            if any(action_word in step_text.lower() for action_word in 
                   ["search", "find", "look up", "retrieve", "calculate", "check"]):
                action_steps.append(step_text)
        
        # If we found action steps, use ReAct for them
        if action_steps:
            react_context = context.copy()
            react_context["planned_actions"] = action_steps
            
            react_result = await self.router.reasoning_methods[ReasoningMethod.REACT](
                query, user_id, react_context
            )
            
            # Combine the results
            combined_result = {
                "answer": react_result.get("answer", sequential_result.get("answer", "")),
                "method": "react_sequential_hybrid",
                "method_emoji": "⚙️🧠",
                "thinking_process": {
                    "planning": sequential_result.get("thinking_steps", []),
                    "actions": react_result.get("actions", []),
                    "observations": react_result.get("observations", [])
                }
            }
            
            return combined_result
        else:
            # If no action steps were identified, return sequential result
            return sequential_result

def create_reasoning_integration(
    reasoning_router: ReasoningRouter, 
    ai_provider=None,
    context_cache=None
) -> ReasoningIntegration:
    """
    Factory function to create reasoning integration
    
    Args:
        reasoning_router: Reasoning router instance
        ai_provider: AI provider for context analysis
        context_cache: Cache for conversation context
        
    Returns:
        Configured ReasoningIntegration instance
    """
    return ReasoningIntegration(
        reasoning_router=reasoning_router,
        ai_provider=ai_provider,
        context_cache=context_cache
    )

class IntegratedReasoning:
    """
    Integrated reasoning system that coordinates between different reasoning methods
    and provides context-aware method selection and caching
    """
    
    def __init__(self, 
               cache_proxy: ReasoningCacheProxy,
               reasoning_router=None,
               hallucination_handler=None,
               context_cache=None,
               semantic_cache=None):
        """
        Initialize integrated reasoning system
        
        Args:
            cache_proxy: Cache proxy for reasoning results
            reasoning_router: Router for specific reasoning methods
            hallucination_handler: Handler for hallucination detection
            context_cache: Context-aware cache
            semantic_cache: Semantic cache
        """
        self.cache_proxy = cache_proxy
        self.reasoning_router = reasoning_router
        self.hallucination_handler = hallucination_handler
        self.context_cache = context_cache
        self.semantic_cache = semantic_cache
        
        # Method registry
        self.reasoning_methods = {}
        
        # Metrics
        self.metrics = {
            "total_queries": 0,
            "method_usage": {},
            "avg_response_time": 0,
            "total_response_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "adaptive_method_selected": {}
        }
    
    def register_method(self, method_name: str, method_func: Callable):
        """
        Register a reasoning method
        
        Args:
            method_name: Method name
            method_func: Method function
        """
        self.reasoning_methods[method_name] = method_func
        self.metrics["method_usage"][method_name] = 0
        logger.info(f"Registered reasoning method: {method_name}")
    
    async def process_query(self, 
                         query: str, 
                         user_id: str,
                         channel_id: Optional[str] = None,
                         context: Optional[Dict[str, Any]] = None,
                         method: Optional[Union[str, ReasoningMethod]] = None,
                         guild_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query using the most appropriate reasoning method
        
        Args:
            query: User query
            user_id: User ID
            channel_id: Optional channel ID
            context: Optional conversation context
            method: Optional specific reasoning method to use
            guild_id: Optional guild ID
            
        Returns:
            Response data
        """
        start_time = time.time()
        self.metrics["total_queries"] += 1
        
        # Initialize context if not provided
        if context is None:
            context = {}
        
        # Extract requested method if specified
        requested_method = None
        if method:
            if isinstance(method, ReasoningMethod):
                requested_method = method.value
            else:
                requested_method = method
        
        # Try to get from cache first
        cache_key_params = {
            "user_id": user_id,
            "query": query,
            "reasoning_type": requested_method,
            "channel_id": channel_id,
            "guild_id": guild_id
        }
        
        cached_response = await self.cache_proxy.get_cached_reasoning(**cache_key_params)
        
        if cached_response:
            self.metrics["cache_hits"] += 1
            
            # Mark as cached for the caller
            if "result" in cached_response:
                result = cached_response["result"]
                # Flag as coming from cache
                result["from_cache"] = True
                
                # Add verification info if available
                if "verification" in cached_response:
                    result["verification"] = cached_response["verification"]
                
                # Update metrics
                response_time = time.time() - start_time
                self._update_metrics(cached_response.get("reasoning_type", "unknown"), response_time)
                
                return result
            return cached_response
        
        self.metrics["cache_misses"] += 1
        
        # Get enhanced context for better reasoning
        enhanced_context = await self.cache_proxy.get_enhanced_context(
            user_id=user_id,
            query=query,
            channel_id=channel_id,
            guild_id=guild_id
        )
        
        # Merge with provided context
        for key, value in context.items():
            if key not in enhanced_context or not enhanced_context[key]:
                enhanced_context[key] = value
        
        # Determine the reasoning method to use
        method_to_use = requested_method
        
        if not method_to_use or method_to_use == ReasoningMethod.ADAPTIVE.value:
            # Adaptively select the best method based on query and context
            method_to_use = await self._select_reasoning_method(query, enhanced_context)
            self.metrics["adaptive_method_selected"][method_to_use] = self.metrics["adaptive_method_selected"].get(method_to_use, 0) + 1
        
        # Process using the selected method
        try:
            if self.reasoning_router and hasattr(self.reasoning_router, 'route_query'):
                # Use reasoning router if available
                from features.reasoning.reasoning_router import ReasoningMethod as RouterMethod
                
                # Convert method name to router enum
                try:
                    router_method = RouterMethod(method_to_use)
                except (ValueError, TypeError):
                    router_method = None  # Router will use default
                
                response = await self.reasoning_router.route_query(
                    query=query,
                    user_id=user_id,
                    method=router_method,
                    context=enhanced_context
                )
            elif method_to_use in self.reasoning_methods:
                # Use directly registered method
                method_func = self.reasoning_methods[method_to_use]
                response = await method_func(
                    query=query,
                    user_id=user_id,
                    context=enhanced_context
                )
            else:
                # Default case
                logger.warning(f"No reasoning method found for {method_to_use}, using fallback")
                response = {
                    "answer": f"I processed your query but couldn't find an appropriate reasoning method.",
                    "method": "fallback",
                    "method_emoji": "⚠️",
                    "confidence": 0.3
                }
        except Exception as e:
            logger.error(f"Error in reasoning method {method_to_use}: {e}")
            logger.error(traceback.format_exc())
            
            # Create error response
            response = {
                "answer": "I encountered an error while processing your request.",
                "method": "error",
                "method_emoji": "❌",
                "error": str(e),
                "confidence": 0.0
            }
        
        # Record the processing time
        response_time = time.time() - start_time
        
        # Add processing time and method to response
        response["processing_time"] = response_time
        if "method" not in response:
            response["method"] = method_to_use
        
        # Update metrics
        self._update_metrics(method_to_use, response_time)
        
        # Verify response if hallucination handler available
        verification_result = None
        if self.hallucination_handler and hasattr(self.hallucination_handler, 'verify_response'):
            try:
                verification_result = await self.hallucination_handler.verify_response(
                    query=query,
                    response=response["answer"],
                    context=enhanced_context
                )
                
                # Add verification result to response
                response["verification"] = verification_result
                
                # Adjust answer if confidence is low
                if verification_result.get("confidence", 1.0) < 0.5:
                    original_answer = response["answer"]
                    response["answer"] = self._add_uncertainty_qualifiers(
                        original_answer, 
                        verification_result.get("confidence", 0)
                    )
            except Exception as e:
                logger.error(f"Error in hallucination verification: {e}")
        
        # Cache the result
        await self.cache_proxy.cache_reasoning_result(
            user_id=user_id,
            query=query,
            reasoning_type=method_to_use,
            result=response,
            verification_result=verification_result,
            channel_id=channel_id,
            guild_id=guild_id
        )
        
        return response
    
    async def _select_reasoning_method(self, query: str, context: Dict[str, Any]) -> str:
        """
        Select the most appropriate reasoning method based on query and context
        
        Args:
            query: User query
            context: Enhanced context
            
        Returns:
            Selected reasoning method name
        """
        # Analyze query complexity and characteristics
        is_complex = self._is_complex_query(query)
        needs_retrieval = self._needs_retrieval(query, context)
        is_action_oriented = self._is_action_oriented(query)
        needs_verification = self._needs_factual_verification(query)
        has_conversation_context = bool(context.get("conversation_history"))
        
        # Decision logic for method selection
        if needs_verification:
            return ReasoningMethod.CHAIN_OF_VERIFICATION.value
        
        if is_action_oriented:
            return ReasoningMethod.REACT.value
            
        if needs_retrieval:
            if has_conversation_context:
                return ReasoningMethod.REFLECTIVE_RAG.value
            else:
                return ReasoningMethod.SPECULATIVE_RAG.value
        
        if is_complex:
            if self._needs_nonlinear_reasoning(query):
                return ReasoningMethod.GRAPH_OF_THOUGHT.value
            else:
                return ReasoningMethod.SEQUENTIAL.value
        
        # Default case
        return ReasoningMethod.SEQUENTIAL.value
    
    def _is_complex_query(self, query: str) -> bool:
        """Determine if a query is complex and needs structured reasoning"""
        query = query.lower()
        
        # Keywords indicating complex reasoning
        complex_indicators = [
            "explain", "analyze", "compare", "contrast", "evaluate", 
            "synthesize", "examine", "investigate", "debate", "discuss",
            "why", "how would", "what if", "implications", "consequences",
            "pros and cons", "advantages", "disadvantages", "effects of",
            "impact of", "relationship between", "difference between"
        ]
        
        # Check for complex indicators
        for indicator in complex_indicators:
            if indicator in query:
                return True
        
        # Check query length as a heuristic
        if len(query.split()) > 10:
            return True
            
        return False
    
    def _needs_retrieval(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if a query needs information retrieval"""
        query = query.lower()
        
        # Keywords indicating retrieval need
        retrieval_indicators = [
            "what is", "who is", "where is", "when did", "information about",
            "tell me about", "definition of", "find", "search", "look up",
            "data on", "statistics", "facts about", "history of", "how to",
            "guide", "tutorial", "examples of", "instances of"
        ]
        
        # Check for retrieval indicators
        for indicator in retrieval_indicators:
            if indicator in query:
                return True
                
        return False
    
    def _is_action_oriented(self, query: str) -> bool:
        """Determine if a query is action-oriented and needs ReAct reasoning"""
        query = query.lower()
        
        # Keywords indicating action orientation
        action_indicators = [
            "do", "make", "create", "build", "perform", "solve",
            "implement", "execute", "achieve", "accomplish", "generate",
            "calculate", "compute", "organize", "arrange", "plan"
        ]
        
        # Check for action-oriented words at the beginning
        words = query.split()
        if words and words[0] in action_indicators:
            return True
            
        # Check for action phrases
        action_phrases = [
            "how can i", "how do i", "how should i", "how would you",
            "can you help me", "i need to", "i want to"
        ]
        
        for phrase in action_phrases:
            if phrase in query:
                return True
                
        return False
    
    def _needs_factual_verification(self, query: str) -> bool:
        """Determine if a query needs factual verification"""
        query = query.lower()
        
        # Keywords indicating factual verification need
        factual_indicators = [
            "is it true", "fact check", "verify", "validate", "confirm",
            "is this correct", "accurate", "reliable", "trustworthy",
            "evidence", "proof", "source", "citation", "reference"
        ]
        
        # Check for factual verification indicators
        for indicator in factual_indicators:
            if indicator in query:
                return True
                
        return False
    
    def _needs_nonlinear_reasoning(self, query: str) -> bool:
        """Determine if a query needs non-linear reasoning"""
        query = query.lower()
        
        # Keywords indicating non-linear reasoning need
        nonlinear_indicators = [
            "multiple factors", "interconnected", "network", "graph",
            "complex system", "interrelated", "ecosystem", "multifaceted",
            "diverse perspectives", "various aspects", "different angles",
            "consider all", "big picture", "holistic", "systems thinking"
        ]
        
        # Check for non-linear reasoning indicators
        for indicator in nonlinear_indicators:
            if indicator in query:
                return True
                
        return False
    
    def _add_uncertainty_qualifiers(self, answer: str, confidence: float) -> str:
        """
        Add uncertainty qualifiers to answers with low verification confidence
        
        Args:
            answer: Original answer
            confidence: Confidence score (0-1)
            
        Returns:
            Modified answer with appropriate qualifiers
        """
        if confidence < 0.3:
            qualifiers = [
                "I'm uncertain about this, but: ",
                "This might not be accurate, but: ",
                "Take this with caution: ",
                "I'm not confident in this answer, but: "
            ]
            import random
            prefix = random.choice(qualifiers)
            return f"{prefix}{answer}\n\n(Note: This information has low confidence and may not be accurate.)"
        
        elif confidence < 0.5:
            return f"{answer}\n\n(Note: I'm not entirely confident in this answer. Consider verifying this information.)"
        
        return answer
    
    def _update_metrics(self, method: str, response_time: float):
        """Update metrics after processing"""
        # Update method usage
        if method in self.metrics["method_usage"]:
            self.metrics["method_usage"][method] += 1
        else:
            self.metrics["method_usage"][method] = 1
        
        # Update response time metrics
        prev_total = self.metrics["total_response_time"]
        prev_count = self.metrics["total_queries"]
        
        self.metrics["total_response_time"] += response_time
        self.metrics["avg_response_time"] = self.metrics["total_response_time"] / prev_count
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for the integrated reasoning system"""
        metrics = dict(self.metrics)
        
        # Add cache metrics if available
        if self.cache_proxy and hasattr(self.cache_proxy, 'get_metrics'):
            try:
                cache_metrics = await self.cache_proxy.get_metrics()
                metrics["cache"] = cache_metrics
            except Exception as e:
                logger.error(f"Error getting cache metrics: {e}")
        
        return metrics

def create_integrated_reasoning(
    cache_proxy,
    reasoning_router=None,
    hallucination_handler=None,
    context_cache=None,
    semantic_cache=None) -> IntegratedReasoning:
    """
    Factory function to create an integrated reasoning system
    
    Args:
        cache_proxy: Cache proxy for reasoning results
        reasoning_router: Router for reasoning methods
        hallucination_handler: Handler for hallucination detection
        context_cache: Context-aware cache
        semantic_cache: Semantic cache
        
    Returns:
        Configured IntegratedReasoning instance
    """
    return IntegratedReasoning(
        cache_proxy=cache_proxy,
        reasoning_router=reasoning_router,
        hallucination_handler=hallucination_handler,
        context_cache=context_cache,
        semantic_cache=semantic_cache
    ) 