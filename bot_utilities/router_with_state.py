import logging
import asyncio
import json
import re
import time
import uuid
import os
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

from bot_utilities.state_manager import StateManager, create_state_manager
from bot_utilities.response_cache import ResponseCache, create_response_cache
from bot_utilities.ai_utils import get_ai_provider, generate_response, should_use_sequential_thinking
from bot_utilities.sequential_thinking import create_sequential_thinking
from bot_utilities.react_utils import run_react_agent
from bot_utilities.chain_of_verification import run_verification
from bot_utilities.speculative_rag import SpeculativeRAG
from bot_utilities.reflective_rag import SelfReflectiveRAG

# Set up logging
logger = logging.getLogger("sum2act_router")
logger.setLevel(logging.INFO)

class Sum2ActRouter:
    """
    Sum2Act Router: Dynamically selects reasoning methods based on 
    the query and context, maintaining state throughout conversations.
    
    Inspired by "From Summary to Action: Enhancing Large Language Models 
    for Complex Tasks with Open World APIs"
    
    Features:
    - Progressively analyzes and routes queries to appropriate reasoning methods
    - Maintains conversation context and state awareness
    - Uses semantic fingerprinting to retrieve responses for similar queries
    - Provides visual feedback for reasoning method selection
    - Supports parallel method evaluation for critical tasks
    - Integrates with state management and response caching
    - Supports custom pre and post processing hooks
    """
    
    def __init__(self, 
                llm_provider=None,
                state_manager=None,
                response_cache=None,
                sequential_thinking_enabled=True,
                react_enabled=True,
                verification_enabled=True,
                enable_cache=True,
                parallel_evaluation=False,
                config_path: str = "config/router_config.json"):
        """
        Initialize the Sum2Act router
        
        Args:
            llm_provider: LLM provider for AI operations
            state_manager: State manager instance (or None to create)
            response_cache: Response cache instance (or None to create)
            sequential_thinking_enabled: Enable sequential thinking
            react_enabled: Enable ReAct planning
            verification_enabled: Enable chain-of-verification
            enable_cache: Enable response caching
            parallel_evaluation: Evaluate multiple methods in parallel
            config_path: Path to router configuration file
        """
        self.llm_provider = llm_provider
        self.state_manager = state_manager or create_state_manager()
        self.response_cache = response_cache or create_response_cache() if enable_cache else None
        self.config_path = config_path
        
        # Initialize reasoning modules
        self.sequential_thinking = create_sequential_thinking(llm_provider) if sequential_thinking_enabled else None
        self.react_enabled = react_enabled
        self.verification_enabled = verification_enabled
        self.parallel_evaluation = parallel_evaluation
        
        # Reasoning method metadata
        self.method_info = {
            "sequential": {
                "name": "Sequential Thinking",
                "emoji": "🔄",
                "description": "Step-by-step reasoning process"
            },
            "react": {
                "name": "ReAct Planning",
                "emoji": "📝",
                "description": "Reasoning and acting iteratively"
            },
            "verification": {
                "name": "Chain-of-Verification",
                "emoji": "✅",
                "description": "Verify facts and reasoning"
            },
            "speculative_rag": {
                "name": "Speculative RAG",
                "emoji": "🔍",
                "description": "Speculative retrieval for complex queries"
            },
            "reflective_rag": {
                "name": "Reflective RAG",
                "emoji": "🤔",
                "description": "Self-reflective reasoning with retrieval"
            },
            "default": {
                "name": "Standard Processing",
                "emoji": "💬",
                "description": "Direct response generation"
            }
        }
        
        # Load configuration if available
        self.config = self._load_config()
        
        # Setup routing thresholds from config
        self.complexity_threshold = self.config.get("complexity_threshold", 0.65)
        self.verification_threshold = self.config.get("verification_threshold", 0.70)
        
        # Performance metrics
        self.metrics = {
            "requests": 0,
            "successes": 0,
            "failures": 0, 
            "cache_hits": 0,
            "method_usage": {method: 0 for method in self.method_info},
            "latencies": []
        }
        
        # Customization hooks
        self.pre_request_hooks: List[Callable] = []
        self.post_request_hooks: List[Callable] = []
        
        # Will track the most recently used method
        self.last_method = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        config = {
            "complexity_threshold": 0.65,
            "verification_threshold": 0.70,
            "cache_enabled": True,
            "methods": {
                "sequential": True,
                "react": True,
                "verification": True,
                "speculative_rag": False,
                "reflective_rag": False
            },
            "parallel_evaluation": False,
            "prompt_templates": {
                "analysis": "Analyze the following query to determine how to process it:\n\nQuery: \"{query}\"\n\nDetermine:\n1. Is this a factual query that requires verification?\n2. Is this a complex query that requires step-by-step reasoning?\n3. Does this query need external tools or APIs to answer properly?\n4. Does this query need web search or information retrieval?\n5. On a scale of 0.0-1.0, how complex is this query?\n\nProvide your analysis as a JSON object with the following fields:\n- is_factual: boolean\n- is_complex: boolean\n- needs_verification: boolean\n- needs_tools: boolean\n- needs_search: boolean\n- complexity_score: float (0.0-1.0)\n- recommended_method: string (one of: \"default\", \"sequential\", \"react\", \"verification\")\n- reasoning: string (brief explanation)\n\nJSON:"
            },
            "temperature": {
                "analysis": 0.1,
                "sequential": 0.3,
                "react": 0.5,
                "verification": 0.1,
                "default": 0.7
            }
        }
        
        # Try to load from file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                
                # Update config with file values
                self._deep_merge(config, file_config)
                logger.info(f"Loaded router configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Recursively merge nested dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    async def analyze_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Analyze a query to determine its complexity and required reasoning approach
        
        Args:
            query: The user query
            session_id: Optional session ID for context
            
        Returns:
            Dict: Analysis results
        """
        # Run any pre-request hooks
        for hook in self.pre_request_hooks:
            try:
                query = await hook(query)
            except Exception as e:
                logger.error(f"Error in pre-request hook: {e}")
        
        # Get conversation history if session_id provided
        history = []
        if session_id:
            try:
                history = await self.state_manager.get_recent_messages(session_id, limit=5)
            except Exception as e:
                logger.error(f"Error retrieving conversation history: {e}")
        
        # Construct history context if available
        history_context = ""
        if history:
            history_context = "Previous conversation:\n"
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_context += f"{role.upper()}: {content}\n"
            history_context += "\n"
        
        # If no LLM provider, use simple heuristics
        if not self.llm_provider:
            return await self._simple_query_analysis(query)
        
        # Use LLM for more sophisticated analysis
        try:
            # Get the analysis prompt template
            analysis_template = self.config["prompt_templates"]["analysis"]
            
            # Format the prompt with the query and history
            prompt = analysis_template.format(query=query)
            if history_context:
                prompt = f"{history_context}\n{prompt}"
            
            # Get temperature for analysis
            temperature = self.config["temperature"]["analysis"]

            # Get analysis from LLM
            response = await self.llm_provider.async_call(
                prompt=prompt,
                temperature=temperature,
                max_tokens=500
            )
            
            # Parse JSON response
            try:
                # Find JSON block
                json_match = re.search(r'({[\s\S]*})', response)
                if json_match:
                    json_str = json_match.group(1)
                    analysis = json.loads(json_str)
                    
                    # Ensure all required fields are present
                    required_fields = [
                        "is_factual", "is_complex", "needs_verification", 
                        "needs_tools", "needs_search", "complexity_score", 
                        "recommended_method", "reasoning"
                    ]
                    
                    # Fill in any missing fields with defaults
                    for field in required_fields:
                        if field not in analysis:
                            if field in ["is_factual", "is_complex", "needs_verification", "needs_tools", "needs_search"]:
                                analysis[field] = False
                            elif field == "complexity_score":
                                analysis[field] = 0.5
                            elif field == "recommended_method":
                                analysis[field] = "default"
                            elif field == "reasoning":
                                analysis[field] = "Default analysis"
                    
                    # Validate complexity score
                    analysis["complexity_score"] = max(0.0, min(1.0, float(analysis["complexity_score"])))
                    
                    # Check if the recommended method is enabled
                    if (analysis["recommended_method"] != "default" and 
                        not self.config["methods"].get(analysis["recommended_method"], True)):
                        # Fall back to default if method is disabled
                        logger.info(f"Method {analysis['recommended_method']} is disabled, falling back to default")
                        analysis["recommended_method"] = "default"
                    
                    # Add the original query
                    analysis["query"] = query
                    
                    # Store analysis in session if available
                    if session_id:
                        try:
                            await self.state_manager.set_session_metadata(
                                session_id=session_id,
                                metadata={"last_analysis": analysis}
                            )
                        except Exception as e:
                            logger.error(f"Error storing analysis in session: {e}")
                    
                    return analysis
            except Exception as e:
                logger.error(f"Error parsing analysis JSON: {e}")
        
        except Exception as e:
            logger.error(f"Error in LLM query analysis: {e}")
        
        # Fall back to simple analysis if anything fails
        return await self._simple_query_analysis(query)
    
    async def _simple_query_analysis(self, query: str) -> Dict[str, Any]:
        """Simple heuristic-based query analysis when LLM is unavailable"""
        # Initialize analysis
        analysis = {
            "query": query,
            "is_factual": False,
            "is_complex": False,
            "needs_verification": False,
            "needs_tools": False,
            "needs_search": False,
            "complexity_score": 0.5,
            "recommended_method": "default",
            "reasoning": "Simple heuristic analysis"
        }
        
        # Normalize query for analysis
        normalized_query = query.lower().strip()
        
        # Check complexity based on length and structure
        if len(normalized_query.split()) > 20:
            analysis["is_complex"] = True
            analysis["complexity_score"] = 0.7
        
        # Check for factual indicators
        factual_indicators = [
            "what is", "who is", "when did", "where is", "how many",
            "which", "list", "explain", "define", "describe",
            "tell me about", "history of", "facts about"
        ]
        
        if any(indicator in normalized_query for indicator in factual_indicators):
            analysis["is_factual"] = True
            analysis["needs_verification"] = True
        
        # Check for complex reasoning indicators
        reasoning_indicators = [
            "why", "how", "analyze", "evaluate", "compare", "contrast",
            "pros and cons", "advantages", "disadvantages", "implications",
            "solve", "step by step", "sequentially", "systematically",
            "think through", "reason about"
        ]
        
        if any(indicator in normalized_query for indicator in reasoning_indicators):
            analysis["is_complex"] = True
            analysis["complexity_score"] = 0.8
            analysis["recommended_method"] = "sequential"
        
        # Check for tool needs
        tool_indicators = [
            "calculate", "compute", "convert", "search", "find", "lookup",
            "research", "browse", "get me", "fetch", "code", "program",
            "script", "api", "database"
        ]
        
        if any(indicator in normalized_query for indicator in tool_indicators):
            analysis["needs_tools"] = True
            analysis["recommended_method"] = "react"
        
        # If both complex and factual, prioritize verification
        if analysis["is_complex"] and analysis["is_factual"] and analysis["complexity_score"] > 0.7:
            analysis["recommended_method"] = "verification"
        
        # Check if the recommended method is enabled
        if (analysis["recommended_method"] != "default" and 
            not self.config["methods"].get(analysis["recommended_method"], True)):
            # Fall back to default if method is disabled
            analysis["recommended_method"] = "default"
        
        return analysis
    
    async def _process_with_method(self, 
                                  method: str, 
                                  query: str, 
                                  session_id: str,
                                  username: str = None, 
                                  user_id: str = None,
                                  channel_id: str = None,
                                  guild_id: str = None,
                                  history: List[Dict[str, str]] = None, 
                                  enhanced_instructions: str = None,
                                  force_method: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Process a query using the specified method
        
        Args:
            method: Reasoning method to use
            query: User query
            session_id: Session ID for state tracking
            username: User's name for personalization
            user_id: User's ID for state tracking
            channel_id: Channel ID for context
            guild_id: Guild ID for context
            history: Conversation history
            enhanced_instructions: Additional custom instructions
            force_method: Force a specific reasoning method
            
        Returns:
            (response, metadata) tuple
        """
        # Track this method in metrics
        if method in self.metrics["method_usage"]:
            self.metrics["method_usage"][method] += 1
        
        # Save as last method
        self.last_method = method
        
        # Get method info for metadata
        method_data = self.method_info.get(method, self.method_info["default"])
        
        # Create base metadata
        metadata = {
            "method": method,
            "method_name": method_data["name"],
            "method_emoji": method_data["emoji"]
        }
        
        # Set system instructions based on method and any enhanced instructions
        system_instructions = ""
        if enhanced_instructions:
            system_instructions = enhanced_instructions + "\n\n"
        
        # Method-specific temperatures
        temperature = self.config["temperature"].get(method, 0.7)
        
        # Process based on method
        start_time = time.time()
        
        try:
            if method == "sequential" and self.sequential_thinking:
                # Use sequential thinking module
                response = await self.sequential_thinking.process(
                    query=query,
                    username=username,
                    system_instructions=system_instructions,
                    history=history
                )
                
                # Store thinking steps in metadata
                if hasattr(self.sequential_thinking, "last_thinking_steps"):
                    metadata["thinking_steps"] = self.sequential_thinking.last_thinking_steps
            
            elif method == "react" and self.react_enabled:
                # Use ReAct module
                response, react_trace = await run_react_agent(
                    query=query,
                    llm_provider=self.llm_provider,
                    system_instructions=system_instructions,
                    history=history
                )
                
                # Store reasoning trace in metadata
                metadata["react_trace"] = react_trace
                
                # Store reasoning steps in session if available
                if session_id:
                    try:
                        await self.state_manager.set_session_metadata(
                            session_id=session_id,
                            metadata={"last_reasoning_steps": react_trace}
                        )
                    except Exception as e:
                        logger.error(f"Error storing reasoning steps: {e}")
            
            elif method == "verification" and self.verification_enabled:
                # Use verification module
                response, verification_data = await run_verification(
                    query=query,
                    llm_provider=self.llm_provider,
                    system_instructions=system_instructions,
                    history=history
                )
                
                # Store verification data in metadata
                metadata["verification_data"] = verification_data
                
                # Store verification in session if available
                if session_id:
                    try:
                        await self.state_manager.set_session_metadata(
                            session_id=session_id,
                            metadata={"last_verification": verification_data}
                        )
                    except Exception as e:
                        logger.error(f"Error storing verification data: {e}")
            
            elif method == "speculative_rag":
                # Use speculative RAG
                speculative_rag = SpeculativeRAG(self.llm_provider)
                response = await speculative_rag.process(
                    query=query,
                    history=history,
                    system_instructions=system_instructions
                )
                
                # Store retrieval data if available
                if hasattr(speculative_rag, "last_retrieval_data"):
                    metadata["retrieval_data"] = speculative_rag.last_retrieval_data
            
            elif method == "reflective_rag":
                # Use reflective RAG
                reflective_rag = SelfReflectiveRAG(self.llm_provider)
                response = await reflective_rag.process(
                    query=query,
                    history=history,
                    system_instructions=system_instructions
                )
                
                # Store reflection data if available
                if hasattr(reflective_rag, "last_reflection"):
                    metadata["reflection"] = reflective_rag.last_reflection
                
            else:
                # Default method: direct generation
                user_greeting = f"{username}: " if username else ""
                
                # Format history for context
                context = ""
                if history:
                    for msg in history[-5:]:  # Last 5 messages
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "user" and username:
                            context += f"{username}: {content}\n"
                        else:
                            context += f"{role.capitalize()}: {content}\n"
                
                # Generate response with default method
                response = await generate_response(
                    instructions=system_instructions,
                    history=history or [],
                    stream=False
                )
            
            # Record latency
            latency = time.time() - start_time
            metadata["latency"] = latency
            self.metrics["latencies"].append(latency)
            
            # Apply any post-processing hooks
            for hook in self.post_request_hooks:
                try:
                    response = await hook(response, query)
                except Exception as e:
                    logger.error(f"Error in post-request hook: {e}")
            
            return response, metadata
            
        except Exception as e:
            logger.error(f"Error processing with method {method}: {e}")
            
            # Create error metadata
            latency = time.time() - start_time
            metadata["error"] = str(e)
            metadata["latency"] = latency
            
            # Return graceful error message
            return f"I apologize, but I encountered an error while processing your query. Please try again or rephrase your question.", metadata
    
    async def _check_cache(self, query: str, channel_id: str = None, guild_id: str = None) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Check if a similar query exists in the cache
        
        Args:
            query: The query to check
            channel_id: Optional channel ID for context
            guild_id: Optional guild ID for context
            
        Returns:
            Cached response and metadata if found, None otherwise
        """
        if not self.response_cache or not self.config.get("cache_enabled", True):
            return None
        
        try:
            # Try to retrieve from cache
            cached_response = await self.response_cache.retrieve(
                query=query,
                channel_id=channel_id,
                guild_id=guild_id
            )
            
            # If found, format and return
            if cached_response:
                response = cached_response.get("response")
                metadata = cached_response.get("metadata", {})
                metadata["from_cache"] = True
                metadata["similarity"] = cached_response.get("similarity", 1.0)
                
                # Update cache hit count
                self.metrics["cache_hits"] += 1
                
                return response, metadata
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
        
        return None
    
    async def _store_in_cache(self, 
                            query: str, 
                            response: str, 
                            metadata: Dict[str, Any],
                            channel_id: str = None,
                            guild_id: str = None) -> None:
        """
        Store a response in the cache
        
        Args:
            query: The query
            response: The response
            metadata: Response metadata
            channel_id: Optional channel ID for context
            guild_id: Optional guild ID for context
        """
        if not self.response_cache or not self.config.get("cache_enabled", True):
            return
        
        # Don't cache error responses
        if "error" in metadata:
            return
        
        try:
            await self.response_cache.store(
                query=query,
                response=response,
                metadata=metadata,
                channel_id=channel_id,
                guild_id=guild_id
            )
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
    
    async def process(self, 
                     query: str, 
                     username: str = None, 
                     user_id: str = None,
                     session_id: str = None,
                     channel_id: str = None,
                     guild_id: str = None,
                     history: List[Dict[str, str]] = None, 
                     enhanced_instructions: str = None,
                     force_method: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Process a query and return a response.
        
        This is the main entry point for the router.
        
        Args:
            query: The user query to process
            username: User's name for personalization
            user_id: User's ID for state tracking
            session_id: Session ID for state tracking
            channel_id: Channel ID for context 
            guild_id: Guild ID for context
            history: Conversation history
            enhanced_instructions: Custom instructions to include
            force_method: Force a specific reasoning method
            
        Returns:
            Tuple of (response_text, metadata)
        """
        start_time = time.time()
        self.metrics["requests"] += 1
        
        # Create session if needed
        if user_id and not session_id:
            try:
                session_id = await self.state_manager.get_or_create_session(user_id, channel_id, guild_id)
            except Exception as e:
                logger.error(f"Error creating session: {e}")
        
        # Run any pre-request hooks
        for hook in self.pre_request_hooks:
            try:
                query = await hook(query)
            except Exception as e:
                logger.error(f"Error in pre-request hook: {e}")
        
        # Validate force_method if provided
        if force_method and force_method not in self.method_info:
            logger.warning(f"Invalid force_method '{force_method}', falling back to default")
            force_method = "default"
            
        # Check if the forced method is enabled
        if force_method and force_method != "default" and not self.config["methods"].get(force_method, True):
            logger.warning(f"Forced method '{force_method}' is disabled, falling back to default")
            force_method = "default"
        
        # First check cache if appropriate
        cache_result = await self._check_cache(query, channel_id, guild_id)
        if cache_result:
            response, metadata = cache_result
            
            # Apply any post-processing hooks
            for hook in self.post_request_hooks:
                try:
                    response = await hook(response, query)
                except Exception as e:
                    logger.error(f"Error in post-request hook: {e}")
            
            # Store interaction in state manager if we have a session
            if session_id:
                try:
                    # Add user message
                    await self.state_manager.add_message(
                        session_id=session_id,
                        message=query,
                        role="user"
                    )
                    
                    # Add assistant response
                    await self.state_manager.add_message(
                        session_id=session_id,
                        message=response,
                        role="assistant",
                        metadata=metadata
                    )
                except Exception as e:
                    logger.error(f"Error updating session history: {e}")
            
            return response, metadata
        
        try:
            # Determine which method to use
            method = force_method if force_method else None
            
            # If method not forced, analyze query
            if not method:
                analysis = await self.analyze_query(query, session_id)
                method = analysis.get("recommended_method", "default")
                
                # Log analysis results
                logger.info(f"Query analysis: method={method}, complexity={analysis.get('complexity_score', 0.5)}")
            
            # Process with the selected method
            response, metadata = await self._process_with_method(
                method=method,
                query=query,
                session_id=session_id,
                username=username,
                user_id=user_id, 
                channel_id=channel_id,
                guild_id=guild_id,
                history=history,
                enhanced_instructions=enhanced_instructions,
                force_method=force_method
            )
            
            # Calculate total latency
            total_latency = time.time() - start_time
            metadata["total_latency"] = total_latency
            
            # Update metrics
            self.metrics["successes"] += 1
            
            # Store in cache if appropriate
            await self._store_in_cache(
                query=query, 
                response=response, 
                metadata=metadata,
                channel_id=channel_id,
                guild_id=guild_id
            )
            
            # Store interaction in state manager if we have a session
            if session_id:
                try:
                    # Add user message
                    await self.state_manager.add_message(
                        session_id=session_id,
                        message=query,
                        role="user"
                    )
                    
                    # Add assistant response
                    await self.state_manager.add_message(
                        session_id=session_id,
                        message=response,
                        role="assistant",
                        metadata=metadata
                    )
                    
                    # Update session metadata
                    await self.state_manager.set_session_metadata(
                        session_id=session_id,
                        metadata={
                            "last_interaction": time.time(),
                            "last_method": method
                        }
                    )
                except Exception as e:
                    logger.error(f"Error updating session: {e}")
            
            return response, metadata
            
        except Exception as e:
            logger.error(f"Error in processing query: {e}")
            self.metrics["failures"] += 1
            
            # Return error message and metadata
            metadata = {
                "error": str(e),
                "method": "error",
                "method_name": "Error Handling",
                "method_emoji": "⚠️",
                "total_latency": time.time() - start_time
            }
            
            return f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your query.", metadata
    
    def register_pre_request_hook(self, hook: Callable):
        """Register a hook to run before processing a request"""
        self.pre_request_hooks.append(hook)
        
    def register_post_request_hook(self, hook: Callable):
        """Register a hook to run after processing a request"""
        self.post_request_hooks.append(hook)
    
    async def get_router_stats(self) -> Dict[str, Any]:
        """Get statistics about router performance"""
        # Calculate average latency
        avg_latency = sum(self.metrics["latencies"]) / max(1, len(self.metrics["latencies"]))
        
        # Calculate success rate
        success_rate = self.metrics["successes"] / max(1, self.metrics["requests"])
        
        # Get cache stats if available
        cache_stats = {}
        if self.response_cache:
            try:
                cache_stats = await self.response_cache.get_stats()
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")
        
        return {
            "requests": self.metrics["requests"],
            "successes": self.metrics["successes"],
            "failures": self.metrics["failures"],
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "cache_hits": self.metrics["cache_hits"],
            "cache_stats": cache_stats,
            "method_usage": self.metrics["method_usage"]
        }
    
    async def invalidate_cache(self, pattern: str = None, channel_id: str = None, guild_id: str = None) -> int:
        """Invalidate cache entries matching criteria"""
        if not self.response_cache:
            return 0
            
        try:
            return await self.response_cache.invalidate(
                pattern=pattern,
                channel_id=channel_id,
                guild_id=guild_id
            )
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0

def create_sum2act_router(
        llm_provider=None,
        state_manager=None,
        response_cache=None,
        sequential_thinking_enabled=True,
        react_enabled=True,
        verification_enabled=True,
        enable_cache=True,
        parallel_evaluation=False,
        config_path: str = "config/router_config.json"):
    """
    Create and return a Sum2Act router instance
    
    Args:
        llm_provider: LLM provider for AI operations
        state_manager: State manager instance (or None to create)
        response_cache: Response cache instance (or None to create)
        sequential_thinking_enabled: Enable sequential thinking
        react_enabled: Enable ReAct planning
        verification_enabled: Enable chain-of-verification
        enable_cache: Enable response caching
        parallel_evaluation: Evaluate multiple methods in parallel
        config_path: Path to configuration file
        
    Returns:
        Initialized Sum2ActRouter
    """
    # Initialize with provided parameters
    router = Sum2ActRouter(
        llm_provider=llm_provider,
        state_manager=state_manager,
        response_cache=response_cache,
        sequential_thinking_enabled=sequential_thinking_enabled,
        react_enabled=react_enabled,
        verification_enabled=verification_enabled,
        enable_cache=enable_cache,
        parallel_evaluation=parallel_evaluation,
        config_path=config_path
    )
    
    return router 