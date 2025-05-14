import logging
import time
import re
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from enum import Enum

from features.reasoning.methods.sequential_thinking import process_sequential_thinking
from features.reasoning.methods.react_reasoning import process_react_reasoning
from features.reasoning.methods.reflective_rag import process_reflective_rag
from features.reasoning.methods.speculative_rag import process_speculative_rag

logger = logging.getLogger("reasoning_router")

class ReasoningMethod(Enum):
    """Enumeration of supported reasoning methods"""
    SEQUENTIAL = "sequential"
    RAG = "rag"
    CRAG = "crag"
    REACT = "react"
    GRAPH_OF_THOUGHT = "graph_of_thought"
    SPECULATIVE = "speculative"
    REFLEXION = "reflexion"
    REFLECTIVE_RAG = "reflective_rag"
    SPECULATIVE_RAG = "speculative_rag"

class ReasoningRouter:
    """
    Router for selecting and executing different reasoning methods based on query analysis
    """
    
    def __init__(self, 
               ai_provider=None, 
               response_cache=None,
               hallucination_handler=None,
               default_method: ReasoningMethod = ReasoningMethod.SEQUENTIAL,
               complexity_threshold: float = 0.7,
               enable_auto_selection: bool = True):
        """
        Initialize reasoning router
        
        Args:
            ai_provider: AI provider for reasoning and query analysis
            response_cache: Cache for responses
            hallucination_handler: Handler for detecting hallucinations
            default_method: Default reasoning method
            complexity_threshold: Threshold for complex query detection
            enable_auto_selection: Enable automatic method selection
        """
        self.ai_provider = ai_provider
        self.response_cache = response_cache
        self.hallucination_handler = hallucination_handler
        self.default_method = default_method
        self.complexity_threshold = complexity_threshold
        self.enable_auto_selection = enable_auto_selection
        
        # Register reasoning methods using imported modules
        self.reasoning_methods = {
            ReasoningMethod.SEQUENTIAL: self._sequential_thinking,
            ReasoningMethod.RAG: self._rag_generation,
            ReasoningMethod.CRAG: self._crag_generation,
            ReasoningMethod.REACT: self._react_reasoning,
            ReasoningMethod.GRAPH_OF_THOUGHT: self._graph_reasoning,
            ReasoningMethod.SPECULATIVE: self._speculative_reasoning,
            ReasoningMethod.REFLEXION: self._reflexion_reasoning,
            ReasoningMethod.REFLECTIVE_RAG: self._reflective_rag_reasoning,
            ReasoningMethod.SPECULATIVE_RAG: self._speculative_rag_reasoning,
        }
        
        # Metrics for reasoning methods
        self.metrics = {
            method.value: {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "cache_hits": 0,
                "avg_time": 0,
                "total_time": 0
            } for method in ReasoningMethod
        }
        
        # Reasoning method emoji indicators
        self.method_emojis = {
            ReasoningMethod.SEQUENTIAL: "🧠",
            ReasoningMethod.RAG: "📚",
            ReasoningMethod.CRAG: "🔍",
            ReasoningMethod.REACT: "⚙️",
            ReasoningMethod.GRAPH_OF_THOUGHT: "📊",
            ReasoningMethod.SPECULATIVE: "🔮",
            ReasoningMethod.REFLEXION: "🪞",
            ReasoningMethod.REFLECTIVE_RAG: "🪞",
            ReasoningMethod.SPECULATIVE_RAG: "🔮"
        }
        
        # Initialize tool providers for ReAct reasoning
        self.tool_providers = {}
        
        # Keywords for complex topic detection
        self.complex_topic_keywords = [
            r"\b(analyze|analysis|complex|complicated|difficult|elaborate|intricate)\b",
            r"\b(multifaceted|sophisticated|comprehensive|detailed|thorough)\b",
            r"\b(explain|understand|assess|evaluate|examine|investigate)\b",
            r"\b(implications|consequences|results|effects|impacts|influences)\b",
            r"\b(relationship|correlation|connection|link|association|causation)\b",
            r"\b(compare|contrast|differentiate|distinguish|versus|vs\.)\b",
            r"\b(pros|cons|advantages|disadvantages|benefits|drawbacks)\b",
            r"\b(theory|framework|model|concept|principle|methodology)\b",
            r"\b(policy|strategy|approach|technique|method|procedure)\b",
            r"\b(history|development|evolution|progression|transformation)\b",
            r"\b(research|study|experiment|investigation|analysis|survey)\b",
            r"\b(economic|political|social|cultural|environmental|technological)\b",
            r"\b(global|international|national|regional|local|worldwide)\b",
            r"\b(future|trend|forecast|prediction|projection|outlook)\b",
            r"\b(debate|controversy|dispute|disagreement|conflict|tension)\b",
            r"\b(challenge|problem|issue|concern|dilemma|obstacle)\b",
            r"\b(solution|resolution|answer|remedy|fix|mitigation)\b",
            r"\b(perspective|viewpoint|standpoint|position|opinion|belief)\b",
            r"\b(factor|element|aspect|dimension|component|variable)\b",
            r"\b(scenario|situation|circumstance|context|condition|case)\b",
            r"\b(step[s]? by step|sequential|in depth|detailed)\b",
            r"\b(math|mathematical|computation|calculate|equation|formula)\b",
            r"\b(programming|code|algorithm|function|method|class|object)\b",
            r"\b(science|scientific|hypothesis|experiment|observation)\b",
            r"\b(ethical|moral|philosophical|ideological|theological)\b"
        ]
        
        # Compile regex patterns for efficiency
        self.complex_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.complex_topic_keywords]
    
    async def route_query(self, 
                        query: str, 
                        user_id: str, 
                        method: Optional[ReasoningMethod] = None,
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route query to the appropriate reasoning method
        
        Args:
            query: User query
            user_id: User ID
            method: Optional specific reasoning method to use
            context: Additional context for reasoning
            
        Returns:
            Result from the selected reasoning method
        """
        start_time = time.time()
        selected_method = method
        context = context or {}
        
        # Check cache first
        if self.response_cache:
            cache_key = f"reasoning:{query.strip().lower()}"
            cached_result = await self.response_cache.get(
                cache_key,
                cache_type="response",
                user_id=user_id
            )
            
            if cached_result:
                cached_method = ReasoningMethod(cached_result.get("method", self.default_method.value))
                self.metrics[cached_method.value]["cache_hits"] += 1
                logger.info(f"Cache hit for query: {query[:50]}...")
                cached_result["from_cache"] = True
                cached_result["method"] = cached_method.value
                cached_result["method_emoji"] = self.method_emojis.get(cached_method, "🤖")
                return cached_result
        
        # Auto-select method if not specified
        if not selected_method and self.enable_auto_selection:
            # Determine complexity
            complexity_score = await self._analyze_query_complexity(query, context)
            logger.info(f"Query complexity score: {complexity_score}")
            
            # Get intent analysis
            intent_analysis = None
            try:
                if self.ai_provider and hasattr(self.ai_provider, "analyze_query_intent"):
                    intent_analysis = await self.ai_provider.analyze_query_intent(query, context)
                    logger.info(f"Query intent analysis: {intent_analysis}")
            except Exception as e:
                logger.error(f"Failed to analyze query intent: {e}")
            
            # Select method based on complexity and intent
            selected_method = await self._select_reasoning_method(
                query, complexity_score, intent_analysis, context
            )
            
            logger.info(f"Selected reasoning method: {selected_method.value}")
        
        # Use default method if still not specified
        if not selected_method:
            selected_method = self.default_method
            logger.info(f"Using default reasoning method: {selected_method.value}")
        
        # Get the reasoning method function
        reasoning_func = self.reasoning_methods.get(selected_method)
        if not reasoning_func:
            logger.warning(f"Unknown reasoning method: {selected_method}, falling back to default")
            selected_method = self.default_method
            reasoning_func = self.reasoning_methods.get(selected_method)
        
        # Update metrics
        self.metrics[selected_method.value]["calls"] += 1
        
        # Execute reasoning method
        try:
            result = await reasoning_func(query, user_id, context)
            
            # Add method info
            result["method"] = selected_method.value
            result["method_emoji"] = self.method_emojis.get(selected_method, "🤖")
            result["processing_time"] = time.time() - start_time
            
            # Cache result if cache available
            if self.response_cache:
                # Only cache if not a personalized or context-specific query
                if not context.get("skip_cache", False):
                    cache_key = f"reasoning:{query.strip().lower()}"
                    await self.response_cache.set(
                        cache_key,
                        result,
                        cache_type="response",
                        ttl=3600,  # 1 hour
                        user_id=user_id
                    )
            
            # Update metrics
            self.metrics[selected_method.value]["successes"] += 1
            self.metrics[selected_method.value]["total_time"] += result["processing_time"]
            self.metrics[selected_method.value]["avg_time"] = (
                self.metrics[selected_method.value]["total_time"] / 
                self.metrics[selected_method.value]["successes"]
            )
            
            # Verify response if hallucination handler available
            if self.hallucination_handler and "answer" in result:
                try:
                    verification = await self.hallucination_handler.verify_response(
                        user_id=user_id,
                        query=query,
                        response=result["answer"],
                        context_data=context
                    )
                    
                    # Add verification results
                    result["verification"] = {
                        "verified": verification.get("verified", True),
                        "confidence": verification.get("confidence", 1.0)
                    }
                    
                    # If grounding was used, update answer
                    if verification.get("grounding_used", False) and "grounded_response" in verification:
                        result["original_answer"] = result["answer"]
                        result["answer"] = verification["grounded_response"]
                        result["verification"]["grounding_used"] = True
                        
                        if "grounding_sources" in verification:
                            result["verification"]["sources"] = verification["grounding_sources"]
                except Exception as e:
                    logger.error(f"Failed to verify response: {e}")
            
            return result
        except Exception as e:
            logger.error(f"Error in reasoning method {selected_method.value}: {e}")
            self.metrics[selected_method.value]["failures"] += 1
            
            # Fallback to default method if different
            if selected_method != self.default_method:
                logger.info(f"Falling back to default method: {self.default_method.value}")
                try:
                    default_func = self.reasoning_methods.get(self.default_method)
                    result = await default_func(query, user_id, context)
                    
                    # Add method info
                    result["method"] = self.default_method.value
                    result["method_emoji"] = self.method_emojis.get(self.default_method, "🤖")
                    result["processing_time"] = time.time() - start_time
                    result["fallback"] = True
                    
                    # Update metrics
                    self.metrics[self.default_method.value]["calls"] += 1
                    self.metrics[self.default_method.value]["successes"] += 1
                    self.metrics[self.default_method.value]["total_time"] += result["processing_time"]
                    self.metrics[self.default_method.value]["avg_time"] = (
                        self.metrics[self.default_method.value]["total_time"] / 
                        self.metrics[self.default_method.value]["successes"]
                    )
                    
                    return result
                except Exception as fallback_error:
                    logger.error(f"Error in fallback method: {fallback_error}")
                    self.metrics[self.default_method.value]["failures"] += 1
            
            # Return error response if all else fails
            return {
                "method": selected_method.value,
                "method_emoji": "❌",
                "error": str(e),
                "answer": f"I encountered an error while processing your request. Please try again or rephrase your question.",
                "processing_time": time.time() - start_time
            }
    
    async def _analyze_query_complexity(self, query: str, context: Dict[str, Any] = None) -> float:
        """
        Analyze the complexity of a query
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Complexity score (0.0-1.0)
        """
        # Start with base complexity
        complexity = 0.3
        
        # Check for complex topic keywords
        keyword_matches = 0
        for pattern in self.complex_patterns:
            if pattern.search(query):
                keyword_matches += 1
        
        # Adjust complexity based on keyword matches
        if keyword_matches > 0:
            keyword_complexity = min(0.3, keyword_matches * 0.02)
            complexity += keyword_complexity
        
        # Consider query length
        if len(query) > 100:
            complexity += 0.1
        if len(query) > 200:
            complexity += 0.1
        
        # Check for multiple questions
        question_count = len(re.findall(r'\?', query))
        if question_count > 1:
            complexity += min(0.2, question_count * 0.05)
        
        # Check for comparison requests
        if re.search(r'\b(compare|versus|vs\.|differences?|similarities?)\b', query, re.IGNORECASE):
            complexity += 0.1
        
        # Check for analysis requests
        if re.search(r'\b(analyze|analysis|evaluate|examine|assess)\b', query, re.IGNORECASE):
            complexity += 0.1
        
        # Check for programming/technical content
        if re.search(r'\b(code|function|algorithm|programming|javascript|python|java|c\+\+)\b', query, re.IGNORECASE):
            complexity += 0.15
        
        # Check for mathematical content
        if re.search(r'\b(calculate|equation|formula|mathematical|probability|statistics)\b', query, re.IGNORECASE):
            complexity += 0.15
        
        # If AI provider available, use it for deeper analysis
        try:
            if self.ai_provider and hasattr(self.ai_provider, "analyze_query_intent"):
                intent_analysis = await self.ai_provider.analyze_query_intent(query, context)
                if intent_analysis and "complexity" in intent_analysis:
                    # Blend our basic analysis with AI analysis
                    ai_complexity = intent_analysis["complexity"]
                    complexity = 0.4 * complexity + 0.6 * ai_complexity
        except Exception as e:
            logger.error(f"Error in AI complexity analysis: {e}")
        
        # Ensure within bounds
        return min(max(complexity, 0.0), 1.0)
    
    async def _select_reasoning_method(self, 
                                     query: str, 
                                     complexity: float,
                                     intent_analysis: Optional[Dict[str, Any]] = None,
                                     context: Dict[str, Any] = None) -> ReasoningMethod:
        """
        Select the appropriate reasoning method based on query analysis
        
        Args:
            query: User query
            complexity: Query complexity score
            intent_analysis: Intent analysis results
            context: Additional context
            
        Returns:
            Selected reasoning method
        """
        # If intent analysis provided reasoning type, use it
        if intent_analysis and "reasoning_type" in intent_analysis:
            reason_type = intent_analysis["reasoning_type"].lower()
            for method in ReasoningMethod:
                if method.value == reason_type:
                    return method
        
        # For high complexity queries
        if complexity >= self.complexity_threshold:
            # Action-oriented queries benefit from ReAct
            if re.search(r'\b(how to|steps to|guide for|instructions for)\b', query, re.IGNORECASE):
                return ReasoningMethod.REACT
            
            # Comparison or multi-faceted analysis benefit from Graph-of-Thought
            if re.search(r'\b(compare|versus|vs\.|differences|similarities|pros and cons)\b', query, re.IGNORECASE):
                return ReasoningMethod.GRAPH_OF_THOUGHT
            
            # Default for complex queries is Sequential Thinking
            return ReasoningMethod.SEQUENTIAL
        
        # For medium complexity queries
        elif complexity >= 0.4:
            # Factual queries benefit from RAG
            if re.search(r'\b(what is|who is|when did|where is|why did|how does)\b', query, re.IGNORECASE):
                # Use CRAG if context is important
                if context and (context.get("conversation_history") or context.get("relevant_history")):
                    return ReasoningMethod.CRAG
                return ReasoningMethod.RAG
            
            # Analysis queries benefit from Sequential Thinking
            return ReasoningMethod.SEQUENTIAL
        
        # For simpler queries, use RAG for factual and CRAG for contextual
        else:
            # Use CRAG if context is important
            if context and (context.get("conversation_history") or context.get("relevant_history")):
                return ReasoningMethod.CRAG
            return ReasoningMethod.RAG
    
    async def _sequential_thinking(self, 
                            query: str, 
                            user_id: str,
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process query with sequential thinking
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            Dict with answer and reasoning details
        """
        start_time = time.time()
        
        try:
            # Use the imported sequential thinking module
            result = await process_sequential_thinking(
                query=query,
                user_id=user_id,
                ai_provider=self.ai_provider,
                response_cache=self.response_cache,
                context=context
            )
            
            # Update metrics
            method = ReasoningMethod.SEQUENTIAL
            self.metrics[method.value]["successes"] += 1
            execution_time = time.time() - start_time
            self._update_metrics(method.value, execution_time)
            
            return result
        except Exception as e:
            logger.error(f"Error in sequential thinking: {e}")
            self.metrics[ReasoningMethod.SEQUENTIAL.value]["failures"] += 1
            
            # Return error response
            return {
                "answer": f"I encountered an error while processing your request using sequential thinking. Please try again or rephrase your question.",
                "error": str(e),
                "method": ReasoningMethod.SEQUENTIAL.value
            }

    async def _react_reasoning(self, 
                        query: str, 
                        user_id: str,
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process query with ReAct reasoning
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            Dict with answer and reasoning details
        """
        start_time = time.time()
        
        try:
            # Use the imported react reasoning module
            result = await process_react_reasoning(
                query=query,
                user_id=user_id,
                ai_provider=self.ai_provider,
                response_cache=self.response_cache,
                tool_providers=self.tool_providers,
                context=context
            )
            
            # Update metrics
            method = ReasoningMethod.REACT
            self.metrics[method.value]["successes"] += 1
            execution_time = time.time() - start_time
            self._update_metrics(method.value, execution_time)
            
            return result
        except Exception as e:
            logger.error(f"Error in ReAct reasoning: {e}")
            self.metrics[ReasoningMethod.REACT.value]["failures"] += 1
            
            # Return error response
            return {
                "answer": f"I encountered an error while processing your request using ReAct reasoning. Please try again or rephrase your question.",
                "error": str(e),
                "method": ReasoningMethod.REACT.value
            }
            
    def _update_metrics(self, method_name: str, execution_time: float):
        """Update performance metrics for a reasoning method"""
        metrics = self.metrics[method_name]
        total_calls = metrics["successes"] + metrics["failures"]
        
        # Update average execution time
        metrics["total_time"] += execution_time
        metrics["avg_time"] = metrics["total_time"] / total_calls if total_calls > 0 else 0
        
        logger.debug(f"Updated metrics for {method_name}: avg_time={metrics['avg_time']:.2f}s, calls={total_calls}")
    
    async def _rag_generation(self, 
                            query: str, 
                            user_id: str,
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform RAG (Retrieval-Augmented Generation)
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            RAG result
        """
        logger.info(f"Performing RAG generation for query: {query[:50]}...")
        
        if not self.ai_provider or not hasattr(self.ai_provider, "rag_generation"):
            return {
                "answer": "RAG generation is not available with the current AI provider.",
                "sources": []
            }
        
        try:
            # Call AI provider for RAG generation
            rag_result = await self.ai_provider.rag_generation(query, user_id, context)
            
            # Format response
            result = {
                "answer": rag_result.get("answer", "No answer generated."),
                "sources": rag_result.get("sources", []),
                "confidence": rag_result.get("confidence", 0.0)
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in RAG generation: {e}")
            raise
    
    async def _crag_generation(self, 
                             query: str, 
                             user_id: str,
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform CRAG (Contextual Retrieval-Augmented Generation)
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            CRAG result
        """
        logger.info(f"Performing CRAG generation for query: {query[:50]}...")
        
        if not self.ai_provider or not hasattr(self.ai_provider, "crag_generation"):
            return {
                "answer": "CRAG generation is not available with the current AI provider.",
                "sources": []
            }
        
        try:
            # Call AI provider for CRAG generation
            crag_result = await self.ai_provider.crag_generation(query, user_id, context)
            
            # Format response
            result = {
                "answer": crag_result.get("answer", "No answer generated."),
                "sources": crag_result.get("sources", []),
                "confidence": crag_result.get("confidence", 0.0),
                "context_utilization": crag_result.get("context_utilization", {})
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in CRAG generation: {e}")
            raise
    
    async def _graph_reasoning(self, 
                             query: str, 
                             user_id: str,
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform Graph-of-Thought reasoning
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            Graph-of-Thought result
        """
        logger.info(f"Performing Graph-of-Thought reasoning for query: {query[:50]}...")
        
        if not self.ai_provider or not hasattr(self.ai_provider, "graph_reasoning"):
            return {
                "answer": "Graph-of-Thought reasoning is not available with the current AI provider.",
                "nodes": [],
                "edges": []
            }
        
        # Get current state if available
        current_state = None
        if context and "reasoning_state" in context:
            current_state = context["reasoning_state"].get("graph", {})
        
        try:
            # Call AI provider for Graph reasoning
            graph_result = await self.ai_provider.graph_reasoning(query, user_id, context, current_state)
            
            # Format response
            result = {
                "answer": graph_result.get("answer", "No answer generated."),
                "nodes": graph_result.get("nodes", []),
                "edges": graph_result.get("edges", []),
                "reasoning_path": graph_result.get("reasoning_path", [])
            }
            
            # Update reasoning state
            new_state = {
                "nodes": result["nodes"],
                "edges": result["edges"],
                "reasoning_path": result["reasoning_path"]
            }
            
            if context:
                if "reasoning_state" not in context:
                    context["reasoning_state"] = {}
                context["reasoning_state"]["graph"] = new_state
            
            return result
        except Exception as e:
            logger.error(f"Error in Graph-of-Thought reasoning: {e}")
            raise
    
    async def _speculative_reasoning(self, 
                                   query: str, 
                                   user_id: str,
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform Speculative reasoning
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            Speculative reasoning result
        """
        logger.info(f"Performing Speculative reasoning for query: {query[:50]}...")
        
        if not self.ai_provider or not hasattr(self.ai_provider, "speculative_reasoning"):
            return {
                "answer": "Speculative reasoning is not available with the current AI provider.",
                "candidates": []
            }
        
        try:
            # Call AI provider for Speculative reasoning
            spec_result = await self.ai_provider.speculative_reasoning(query, user_id, context)
            
            # Format response
            result = {
                "answer": spec_result.get("answer", "No answer generated."),
                "candidates": spec_result.get("candidates", []),
                "verification": spec_result.get("verification", {})
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in Speculative reasoning: {e}")
            raise
    
    async def _reflexion_reasoning(self, 
                                 query: str, 
                                 user_id: str,
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform Reflexion reasoning
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            Reflexion reasoning result
        """
        logger.info(f"Performing Reflexion reasoning for query: {query[:50]}...")
        
        if not self.ai_provider or not hasattr(self.ai_provider, "reflexion_reasoning"):
            return {
                "answer": "Reflexion reasoning is not available with the current AI provider.",
                "reflections": []
            }
        
        # Get current state if available
        current_state = None
        if context and "reasoning_state" in context:
            current_state = context["reasoning_state"].get("reflexion", {})
        
        try:
            # Call AI provider for Reflexion reasoning
            reflex_result = await self.ai_provider.reflexion_reasoning(query, user_id, context, current_state)
            
            # Format response
            result = {
                "answer": reflex_result.get("answer", "No answer generated."),
                "initial_answer": reflex_result.get("initial_answer", ""),
                "reflections": reflex_result.get("reflections", []),
                "improvement": reflex_result.get("improvement", {})
            }
            
            # Update reasoning state
            new_state = {
                "initial_answer": result["initial_answer"],
                "reflections": result["reflections"],
                "improvement": result["improvement"]
            }
            
            if context:
                if "reasoning_state" not in context:
                    context["reasoning_state"] = {}
                context["reasoning_state"]["reflexion"] = new_state
            
            return result
        except Exception as e:
            logger.error(f"Error in Reflexion reasoning: {e}")
            raise
    
    async def _reflective_rag_reasoning(self, 
                                 query: str, 
                                 user_id: str,
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform Reflective RAG reasoning
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            Reflective RAG result
        """
        logger.info(f"Performing Reflective RAG reasoning for query: {query[:50]}...")
        
        try:
            # Use the imported reflective RAG module
            result = await process_reflective_rag(
                query=query,
                user_id=user_id,
                ai_provider=self.ai_provider,
                response_cache=self.response_cache,
                context=context
            )
            
            # Update metrics
            method = ReasoningMethod.REFLECTIVE_RAG
            self.metrics[method.value]["successes"] += 1
            
            return result
        except Exception as e:
            logger.error(f"Error in Reflective RAG reasoning: {e}")
            self.metrics[ReasoningMethod.REFLECTIVE_RAG.value]["failures"] += 1
            
            # Return error response
            return {
                "answer": f"I encountered an error while processing your request using Reflective RAG reasoning. Please try again or rephrase your question.",
                "error": str(e),
                "method": ReasoningMethod.REFLECTIVE_RAG.value
            }
            
    async def _speculative_rag_reasoning(self, 
                                   query: str, 
                                   user_id: str,
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform Speculative RAG reasoning
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            Speculative RAG result
        """
        logger.info(f"Performing Speculative RAG reasoning for query: {query[:50]}...")
        
        try:
            # Use the imported speculative RAG module
            result = await process_speculative_rag(
                query=query,
                user_id=user_id,
                ai_provider=self.ai_provider,
                response_cache=self.response_cache,
                context=context
            )
            
            # Update metrics
            method = ReasoningMethod.SPECULATIVE_RAG
            self.metrics[method.value]["successes"] += 1
            
            return result
        except Exception as e:
            logger.error(f"Error in Speculative RAG reasoning: {e}")
            self.metrics[ReasoningMethod.SPECULATIVE_RAG.value]["failures"] += 1
            
            # Return error response
            return {
                "answer": f"I encountered an error while processing your request using Speculative RAG reasoning. Please try again or rephrase your question.",
                "error": str(e),
                "method": ReasoningMethod.SPECULATIVE_RAG.value
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get reasoning router metrics"""
        return dict(self.metrics)
    
    def get_supported_methods(self) -> List[Dict[str, str]]:
        """
        Get list of supported reasoning methods
        
        Returns:
            List of supported methods with name, emoji and description
        """
        methods = []
        
        for method in ReasoningMethod:
            methods.append({
                "name": method.value,
                "emoji": self.method_emojis.get(method, "🤖"),
                "description": self._get_method_description(method)
            })
        
        return methods
    
    def _get_method_description(self, method: ReasoningMethod) -> str:
        """Get description for a reasoning method"""
        descriptions = {
            ReasoningMethod.SEQUENTIAL: "Step-by-step reasoning for complex problems",
            ReasoningMethod.RAG: "Retrieval-Augmented Generation for factual information",
            ReasoningMethod.CRAG: "Context-aware retrieval for personalized responses",
            ReasoningMethod.REACT: "Reasoning and Acting for task-oriented responses",
            ReasoningMethod.GRAPH_OF_THOUGHT: "Non-linear reasoning for multi-faceted analysis",
            ReasoningMethod.SPECULATIVE: "Generates and verifies multiple candidate responses",
            ReasoningMethod.REFLEXION: "Self-reflective improvement of initial responses",
            ReasoningMethod.REFLECTIVE_RAG: "Self-reflection for RAG quality improvement",
            ReasoningMethod.SPECULATIVE_RAG: "Multi-query speculative approach for better retrieval"
        }
        
        return descriptions.get(method, "Unknown reasoning method")

def create_reasoning_router(ai_provider=None, response_cache=None, 
                          hallucination_handler=None, **kwargs) -> ReasoningRouter:
    """
    Factory function to create a reasoning router
    
    Args:
        ai_provider: AI provider for reasoning
        response_cache: Cache for responses
        hallucination_handler: Handler for detecting hallucinations
        **kwargs: Additional options for ReasoningRouter
        
    Returns:
        Configured ReasoningRouter instance
    """
    return ReasoningRouter(
        ai_provider=ai_provider,
        response_cache=response_cache,
        hallucination_handler=hallucination_handler,
        **kwargs
    ) 