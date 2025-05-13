import logging
import asyncio
import json
import time
import os
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import re

from bot_utilities.reasoning_router import ReasoningRouter, create_reasoning_router
from bot_utilities.router_with_state import Sum2ActRouter, create_sum2act_router
from bot_utilities.state_manager import create_state_manager
from bot_utilities.response_cache import create_response_cache
from bot_utilities.ai_utils import get_ai_provider, generate_response
from bot_utilities.sequential_thinking import create_sequential_thinking
from bot_utilities.context_manager import get_context_manager

# Set up logging
logger = logging.getLogger("router_compatibility")
logger.setLevel(logging.INFO)

class RouterAdapter:
    """
    Compatibility layer to bridge multiple routing architectures with unified interface.
    Provides seamless integration between different routing engines, fallback mechanisms,
    and support for dynamic routing configuration.
    
    Features:
    - Unified interface across router implementations
    - Automatic fallback to alternative routers when needed
    - Centralized state and cache management
    - Performance monitoring and metrics tracking
    - Framework-agnostic compatibility
    """
    def __init__(self, 
                use_sum2act: bool = True, 
                fallback_enabled: bool = True,
                auto_retry: bool = True,
                state_manager=None, 
                response_cache=None,
                config_path: str = "config/router_config.json"):
        """
        Initialize the router adapter
        
        Args:
            use_sum2act: Whether to use Sum2Act as primary router
            fallback_enabled: Whether to fall back to alternative router on failure
            auto_retry: Whether to automatically retry failed requests with fallback
            state_manager: Custom StateManager (or None to create a default one)
            response_cache: Custom ResponseCache (or None to create a default one)
            config_path: Path to router configuration file
        """
        self.use_sum2act = use_sum2act
        self.fallback_enabled = fallback_enabled
        self.auto_retry = auto_retry
        self.sum2act_router = None
        self.legacy_router = None
        self.llm_provider = None
        self.state_manager = state_manager
        self.response_cache = response_cache
        self.config_path = config_path
        
        # Performance metrics
        self.metrics = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "fallbacks": 0,
            "primary_latency": [],
            "fallback_latency": [],
            "cache_hits": 0
        }
        
        # Customization hooks
        self.pre_request_hooks: List[Callable] = []
        self.post_request_hooks: List[Callable] = []
        
        # Load configuration if available
        self.config = self._load_config()
        
        logger.info(f"RouterAdapter initialized with Sum2Act as {'primary' if use_sum2act else 'secondary'}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load router configuration from file"""
        config = {
            "retry_count": 2,
            "request_timeout": 30,
            "max_retry_wait": 5,
            "method_weights": {
                "sequential": 1.0,
                "react": 0.8,
                "verification": 0.6,
                "default": 0.4
            },
            "complexity_thresholds": {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.8
            }
        }
        
        # Try to load from file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    # Update with file values
                    for key, value in file_config.items():
                        if key in config:
                            config[key] = value
                logger.info(f"Loaded router configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading router configuration: {e}")
        
        return config
    
    async def init_routers(self, llm_provider=None):
        """Initialize both routers with the same LLM provider"""
        self.llm_provider = llm_provider
        
        # Create state manager if not provided
        if self.state_manager is None:
            self.state_manager = create_state_manager()
            
        # Create response cache if not provided
        if self.response_cache is None:
            self.response_cache = create_response_cache()
        
        # Initialize Sum2Act router
        if self.sum2act_router is None:
            self.sum2act_router = create_sum2act_router(
                llm_provider=llm_provider,
                state_manager=self.state_manager,
                response_cache=self.response_cache
            )
            
        # Initialize legacy router
        if self.legacy_router is None:
            self.legacy_router = create_reasoning_router(
                llm_provider=llm_provider
            )
    
    async def analyze_query(self, query: str, user_id: str = None, channel_id: str = None) -> Dict[str, Any]:
        """
        Analyze query using the primary router's analysis method
        
        Args:
            query: The user query
            user_id: Optional user ID for contextual analysis
            channel_id: Optional channel ID for contextual analysis
            
        Returns:
            Dict: Analysis results
        """
        await self.init_routers()
        
        try:
            # Run pre-request hooks
            for hook in self.pre_request_hooks:
                try:
                    query = await hook(query, user_id, channel_id)
                except Exception as e:
                    logger.error(f"Error in pre-request hook: {e}")
            
            if self.use_sum2act:
                # Get session ID if we have a user ID
                session_id = None
                if user_id and self.state_manager:
                    session_id = await self.state_manager.get_session_for_user(user_id, channel_id)
                    if not session_id:
                        session_id = await self.state_manager.create_session(user_id, channel_id)
                
                # Pass session if available
                if session_id:
                    return await self.sum2act_router.analyze_query(query, session_id=session_id)
                else:
                    return await self.sum2act_router.analyze_query(query)
            else:
                # Convert legacy router analysis to similar format
                legacy_analysis = await self.legacy_router.analyze_query(query)
                
                # Transform to match Sum2Act format
                return {
                    "query": query,
                    "is_factual": getattr(legacy_analysis, "needs_verification", False),
                    "is_complex": getattr(legacy_analysis, "is_complex", False),
                    "needs_verification": getattr(legacy_analysis, "needs_verification", False),
                    "needs_tools": getattr(legacy_analysis, "needs_tools", False),
                    "needs_search": getattr(legacy_analysis, "needs_search", False),
                    "complexity_score": getattr(legacy_analysis, "complexity_score", 0.5),
                    "recommended_method": getattr(legacy_analysis, "recommended_method", "default"),
                    "reasoning": getattr(legacy_analysis, "reasoning", "Analysis from legacy router")
                }
        except Exception as e:
            logger.error(f"Error in analyze_query: {e}")
            # Return default analysis
            return {
                "query": query,
                "is_factual": False,
                "is_complex": False,
                "needs_verification": False,
                "needs_tools": False,
                "needs_search": False,
                "complexity_score": 0.5,
                "recommended_method": "default",
                "reasoning": "Default analysis due to error"
            }
    
    async def process(self, 
                     query: str, 
                     username: str = None, 
                     user_id: str = None, 
                     session_id: str = None,
                     channel_id: str = None,
                     guild_id: str = None,
                     history: List[Dict[str, str]] = None, 
                     enhanced_instructions: str = None,
                     force_method: str = None,
                     enhanced_context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Process a query through the router system
        
        Args:
            query: The user query
            username: Optional username for personalization
            user_id: Optional user ID for personalization 
            session_id: Optional session ID for state management
            channel_id: Optional channel ID for context tracking
            guild_id: Optional guild/server ID
            history: Optional conversation history
            enhanced_instructions: Optional custom system instructions
            force_method: Optional method to force (react, sequential, verification, default)
            enhanced_context: Optional enhanced context from context manager
            
        Returns:
            Tuple[response_text, metadata]
        """
        # Initialize stats tracking
        start_time = time.time()
        self.metrics["requests"] += 1
        
        # Initialize routers (idempotent - won't reinitialize if already done)
        await self.init_routers()
        
        # Get thread_id from channel_id if it's a thread
        thread_id = None
        if channel_id and (hasattr(channel_id, "startswith") and channel_id.startswith("thread-")):
            thread_id = channel_id
        
        # Run pre-request hooks
        for hook in self.pre_request_hooks:
            try:
                query = await hook(query, user_id, channel_id)
            except Exception as e:
                logger.error(f"Error in pre-request hook: {e}")
        
        # Check response cache first 
        cache_key = f"{query}|{user_id or 'anonymous'}|{channel_id or 'global'}"
        
        if self.response_cache:
            cached_response = await self.response_cache.retrieve(query, channel_id=channel_id)
            if cached_response:
                self.metrics["cache_hits"] += 1
                logger.info(f"Cache hit for query: {query[:30]}...")
                
                # Extract the response data from the cache
                response_data = cached_response.get("response", "")
                metadata = cached_response.get("metadata", {})
                
                # Handle cached response based on its format
                if isinstance(response_data, tuple):
                    # Extract only the response text from the tuple, no metadata
                    if len(response_data) >= 1:
                        response_text = response_data[0]
                        # Get metadata from second element if it exists
                        if len(response_data) >= 2 and isinstance(response_data[1], dict):
                            metadata_from_tuple = response_data[1]
                            # Merge the metadata if both exist
                            if metadata_from_tuple:
                                metadata = {**metadata, **metadata_from_tuple}
                    else:
                        # Empty tuple, use fallback
                        response_text = "I don't have a good answer for that right now."
                        metadata = {"method": "cache_error", "error": "empty_tuple"}
                    
                    # Ensure response_text is a string
                    if not isinstance(response_text, str):
                        try:
                            response_text = str(response_text)
                        except:
                            response_text = "I'm having trouble formatting my response."
                    
                    # Clean up the response text to remove any metadata or formatting artifacts
                    response_text = self._clean_cache_response(response_text)
                    
                    # Return the properly formatted tuple
                    return response_text, metadata
                else:
                    # If it's just a string or another format, wrap it properly
                    # Ensure it's a string
                    if not isinstance(response_data, str):
                        try:
                            response_data = str(response_data)
                        except:
                            response_data = "I'm having trouble formatting my response."
                    
                    # Clean up any formatting artifacts
                    cleaned_response = self._clean_cache_response(response_data)
                    
                    default_metadata = {
                        "method": "cache", 
                        "method_name": "Cached Response", 
                        "method_emoji": "🔄", 
                        "enhanced_context_used": True
                    }
                    # Merge with any existing metadata
                    combined_metadata = {**default_metadata, **(metadata or {})}
                    return cleaned_response, combined_metadata
        
        # Get enhanced context if not provided
        context_manager = get_context_manager()
        if not enhanced_context and user_id and channel_id:
            enhanced_context = context_manager.get_conversation_context(
                user_id=user_id,
                channel_id=channel_id,
                query=query,
                thread_id=thread_id
            )
        
        # Add the query to context tracking if we have enough info
        if user_id and channel_id:
            context_manager.add_message(
                user_id=user_id,
                channel_id=channel_id,
                thread_id=thread_id,
                message={
                    "role": "user",
                    "content": query,
                    "timestamp": time.time(),
                    "message_id": session_id or f"session-{time.time()}"
                }
            )
        
        # Try with primary router
        try:
            # For legacy router, use our code directly
            if not self.use_sum2act:
                # Enhance history with context if available
                enriched_history = history or []
                if enhanced_context and enhanced_context.get("conversation_summary"):
                    # Add conversation summary as a system message
                    enriched_history.insert(0, {
                        "role": "system",
                        "content": f"[Conversation context: {enhanced_context['conversation_summary']}]"
                    })
                
                # Add relevant thinking if available
                if enhanced_context and enhanced_context.get("relevant_thinking"):
                    relevant_thinking = enhanced_context["relevant_thinking"]
                    if relevant_thinking:
                        thinking_context = "Previous related thoughts:\n"
                        for i, thinking in enumerate(relevant_thinking):
                            thinking_context += f"{i+1}. Problem: {thinking.get('problem', '')}\n"
                            thinking_context += f"   Solution: {thinking.get('solution', '')[:200]}...\n"
                        
                        enriched_history.insert(0, {
                            "role": "system",
                            "content": thinking_context
                        })
                
                # Pass to legacy router
                response, metadata = await self.legacy_router.process(
                    query=query,
                    username=username,
                    user_id=user_id,
                    history=enriched_history,
                    enhanced_instructions=enhanced_instructions
                )
                
                # Store the response in context manager
                if user_id and channel_id and response:
                    context_manager.add_message(
                        user_id=user_id,
                        channel_id=channel_id,
                        thread_id=thread_id,
                        message={
                            "role": "assistant",
                            "content": response,
                            "timestamp": time.time(),
                            "method": metadata.get("method", "default"),
                            "in_response_to": query
                        }
                    )
                    
                    # Store thinking record if available
                    if "thinking" in metadata or "reasoning" in metadata:
                        thinking_record = {
                            "problem": query,
                            "solution": response,
                            "prompt_style": metadata.get("method", "default"),
                            "is_complex": False,
                            "timestamp": time.time(),
                            "success": True,
                            "is_hidden": True,
                            "reasoning": metadata.get("thinking", metadata.get("reasoning", ""))
                        }
                        
                        # Add thinking record to context manager
                        context_manager.add_thinking_process(
                            user_id=user_id,
                            channel_id=channel_id,
                            thread_id=thread_id,
                            thinking=thinking_record
                        )
                
                # Store the response in cache if successful
                if self.response_cache and response:
                    await self.response_cache.store(
                        query=query, 
                        response=(response, metadata),
                        channel_id=channel_id,
                        ttl=300  # 5 minutes
                    )
                
                # Record success metrics
                self.metrics["successes"] += 1
                self.metrics["primary_latency"].append(time.time() - start_time)
                
                return response, metadata
                
            # For Sum2Act router, update to pass enhanced context
            else:
                # Get/create session if needed
                if not session_id and self.state_manager and user_id:
                    session_id = await self.state_manager.get_session_for_user(user_id, channel_id)
                    if not session_id:
                        session_id = await self.state_manager.create_session(user_id, channel_id)
                
                # Pass to Sum2Act router with session context
                response, metadata = await self.sum2act_router.process(
                    query=query,
                    username=username,
                    session_id=session_id,
                    history=history,
                    enhanced_instructions=enhanced_instructions,
                    force_method=force_method
                )
                
                # Add enhanced context information to metadata if available
                if enhanced_context:
                    if not metadata:
                        metadata = {}
                    metadata["enhanced_context_used"] = True
                
                # Store the response in context manager if successful
                if user_id and channel_id and response:
                    context_manager.add_message(
                        user_id=user_id,
                        channel_id=channel_id,
                        thread_id=thread_id,
                        message={
                            "role": "assistant",
                            "content": response,
                            "timestamp": time.time(),
                            "method": metadata.get("method", "default"),
                            "in_response_to": query
                        }
                    )
                    
                    # Store thinking record if available
                    if "thinking" in metadata or "reasoning" in metadata:
                        thinking_record = {
                            "problem": query,
                            "solution": response,
                            "prompt_style": metadata.get("method", "default"),
                            "is_complex": metadata.get("complexity_score", 0) > 0.5,
                            "timestamp": time.time(),
                            "success": True,
                            "is_hidden": False,
                            "reasoning": metadata.get("thinking", metadata.get("reasoning", ""))
                        }
                        
                        # Add thinking record to context manager
                        context_manager.add_thinking_process(
                            user_id=user_id,
                            channel_id=channel_id,
                            thread_id=thread_id,
                            thinking=thinking_record
                        )
                
                # Store in cache if successful and we have a cache
                if self.response_cache and response:
                    await self.response_cache.store(
                        query=query, 
                        response=(response, metadata),
                        channel_id=channel_id,
                        ttl=300  # 5 minutes
                    )
                
                # Record success metrics
                self.metrics["successes"] += 1
                self.metrics["primary_latency"].append(time.time() - start_time)
                
                return response, metadata
                
        except Exception as e:
            logger.error(f"Error with primary router: {e}")
            primary_error = str(e)
            
            # Record failure metrics
            self.metrics["failures"] += 1
            
            # If we don't have fallback or auto-retry, raise the error
            if not self.fallback_enabled or not self.auto_retry:
                error_message = f"Error in query processing: {primary_error}"
                error_metadata = {
                    "error": primary_error,
                    "method": "error",
                    "method_name": "Error",
                    "method_emoji": "❌"
                }
                return error_message, error_metadata
        
        # --------------- FALLBACK HANDLING SECTION ---------------
        # If we get here, the primary router failed and we're using fallback
        
        fallback_start_time = time.time()
        self.metrics["fallbacks"] += 1
        
        logger.info(f"Using fallback for query: {query[:30]}...")
        
        # Try fallback approaches in sequence
        try:
            # First try the other router approach (sum2act <-> reasoning router)
            if self.use_sum2act and self.legacy_router:
                # Try reasoning router as fallback
                try:
                    # Pass to legacy router
                    response, metadata = await self.legacy_router.process(
                        query=query,
                        username=username,
                        user_id=user_id,
                        history=history,
                        enhanced_instructions=enhanced_instructions
                    )
                    
                    # Record fallback metrics
                    self.metrics["fallback_latency"].append(time.time() - fallback_start_time)
                    
                    # Set fallback indicator in metadata
                    metadata["is_fallback"] = True
                    metadata["original_error"] = primary_error
                    
                    # Store the response in context manager if successful
                    if user_id and channel_id and response:
                        context_manager.add_message(
                            user_id=user_id,
                            channel_id=channel_id,
                            thread_id=thread_id,
                            message={
                                "role": "assistant",
                                "content": response,
                                "timestamp": time.time(),
                                "method": metadata.get("method", "default"),
                                "in_response_to": query,
                                "is_fallback": True
                            }
                        )
                    
                    return response, metadata
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback router also failed: {fallback_error}")
                    # Continue to next fallback option
            
            # Try sequential thinking as another fallback
            seq_thinking = create_sequential_thinking(llm_provider=self.llm_provider)
            try:
                # Build context with any available info
                context = {
                    "username": username,
                    "user_id": user_id,
                    "query": query,
                    "history": history,
                    "enhanced_context": enhanced_context or {}
                }
                
                # Use chain-of-thought approach without explicit formatting
                success, thinking_response = await seq_thinking.run(
                    problem=query,
                    context=context,
                    prompt_style="cot",  # Chain of thought is less structured looking
                    num_thoughts=4,  # Fewer steps 
                    temperature=0.3,
                    max_tokens=1500
                )
                
                if success:
                    # Store thinking record in context manager
                    if user_id and channel_id:
                        thinking_record = {
                            "problem": query,
                            "solution": thinking_response,
                            "prompt_style": "cot",
                            "is_complex": False,
                            "timestamp": time.time(),
                            "success": success,
                            "is_hidden": True,
                            "is_fallback": True
                        }
                        
                        # Add thinking record to context manager
                        context_manager.add_thinking_process(
                            user_id=user_id,
                            channel_id=channel_id,
                            thread_id=thread_id,
                            thinking=thinking_record
                        )
                    
                    # Remove explicit formatting markers for a cleaner response
                    response = re.sub(r'\*\*(?:Step|Thought|Conclusion)\s*\d*\*\*:\s*', '', thinking_response)
                    response = re.sub(r'\*\*(?:Conclusion|Summary|Answer)\*\*:\s*', '', response)
                    response = re.sub(r'\n{3,}', '\n\n', response)
                    
                    metadata = {
                        "method": "sequential",
                        "method_name": "Sequential Thinking",
                        "method_emoji": "🧠",
                        "is_fallback": True,
                        "original_error": primary_error
                    }
                    
                    # Store the response in context manager
                    if user_id and channel_id:
                        context_manager.add_message(
                            user_id=user_id,
                            channel_id=channel_id,
                            thread_id=thread_id,
                            message={
                                "role": "assistant",
                                "content": response,
                                "timestamp": time.time(),
                                "method": "sequential_fallback",
                                "in_response_to": query,
                                "is_fallback": True
                            }
                        )
                    
                    # Record fallback metrics
                    self.metrics["fallback_latency"].append(time.time() - fallback_start_time)
                    
                    return response, metadata
            
            except Exception as sequential_error:
                logger.error(f"Sequential thinking fallback failed: {sequential_error}")
                # Continue to next fallback option
                
            # Final fallback: simple AI response
            try:
                # Create a basic system message
                system_message = enhanced_instructions or "You are a helpful AI assistant."
                
                # Add basic context info
                if username:
                    system_message += f"\nYou are talking to {username}."
                
                if enhanced_context and enhanced_context.get("conversation_summary"):
                    system_message += f"\n\nContext from previous conversation: {enhanced_context['conversation_summary']}"
                
                # Generate a simple response using the ai_utils
                response = await generate_response(
                    prompt=query,
                    system_message=system_message,
                    history=history
                )
                
                metadata = {
                    "method": "basic_fallback",
                    "method_name": "Simple Response",
                    "method_emoji": "💬",
                    "is_fallback": True,
                    "original_error": primary_error
                }
                
                # Store the response in context manager
                if user_id and channel_id:
                    context_manager.add_message(
                        user_id=user_id,
                        channel_id=channel_id,
                        thread_id=thread_id,
                        message={
                            "role": "assistant",
                            "content": response,
                            "timestamp": time.time(),
                            "method": "basic_fallback",
                            "in_response_to": query,
                            "is_fallback": True
                        }
                    )
                
                # Record fallback metrics
                self.metrics["fallback_latency"].append(time.time() - fallback_start_time)
                
                return response, metadata
                
            except Exception as basic_error:
                logger.error(f"Basic fallback also failed: {basic_error}")
                # Fall through to final error handler
        
        except Exception as compound_error:
            logger.error(f"Compound error in fallback handling: {compound_error}")
            # Fall through to final error handler
            
        # If we get here, all fallbacks failed
        error_message = "I'm having technical difficulties right now. Please try again later."
        error_metadata = {
            "method": "error",
            "method_name": "All Fallbacks Failed",
            "method_emoji": "⚠️",
            "original_error": primary_error
        }
        
        return error_message, error_metadata
    
    def register_pre_request_hook(self, hook: Callable):
        """Register a hook to run before processing a request"""
        self.pre_request_hooks.append(hook)
        
    def register_post_request_hook(self, hook: Callable):
        """Register a hook to run after processing a request"""
        self.post_request_hooks.append(hook)
    
    async def get_router_stats(self) -> Dict[str, Any]:
        """
        Get statistics on the router's performance
        
        Returns:
            Dict: Router statistics
        """
        stats = {
            "metrics": self.metrics,
            "primary_router": self.use_sum2act,
            "fallback_enabled": self.fallback_enabled,
            "auto_retry": self.auto_retry,
        }
        
        # Add latency statistics if available
        if self.metrics["primary_latency"]:
            stats["avg_primary_latency"] = sum(self.metrics["primary_latency"]) / len(self.metrics["primary_latency"])
            
        if self.metrics["fallback_latency"]:
            stats["avg_fallback_latency"] = sum(self.metrics["fallback_latency"]) / len(self.metrics["fallback_latency"])
        
        # Add cache stats if available
        if self.response_cache:
            cache_stats = await self.response_cache.get_stats()
            stats["cache"] = cache_stats
            
        return stats
        
    async def invalidate_cache(self, pattern: str = None, channel_id: str = None, guild_id: str = None) -> int:
        """
        Invalidate entries in the response cache
        
        Args:
            pattern: Optional pattern to match against cache keys
            channel_id: Optional channel ID to invalidate
            guild_id: Optional guild ID to invalidate
            
        Returns:
            int: Number of entries invalidated
        """
        if not self.response_cache:
            return 0
            
        if pattern or channel_id or guild_id:
            # Selective invalidation
            return await self.response_cache.invalidate(
                pattern=pattern, 
                channel_id=channel_id,
                guild_id=guild_id
            )
        else:
            # Clear entire cache
            return await self.response_cache.clear()

    def _clean_cache_response(self, response_text: Union[str, Tuple]) -> str:
        """
        Clean up a cached response to remove formatting artifacts
        
        Args:
            response_text: The response text to clean (string or tuple)
            
        Returns:
            str: The cleaned response text
        """
        # Handle tuple responses directly rather than after string conversion
        if isinstance(response_text, tuple):
            if len(response_text) >= 1:
                # Extract just the first element which should be the text
                response_text = response_text[0]
            else:
                return "I couldn't format my response properly."
        
        # Ensure we're working with a string
        if not isinstance(response_text, str):
            try:
                response_text = str(response_text)
            except:
                return "I couldn't format my response properly."
        
        # Check if the string looks like a tuple representation with response and metadata
        # This handles cases where the tuple was converted to a string before reaching this method
        if isinstance(response_text, str) and response_text.startswith("(") and response_text.endswith(")"):
            # Check if this looks like a tuple with metadata dict
            if "{" in response_text and "}" in response_text:
                try:
                    # Extract just the text portion
                    match = re.search(r'^\s*\(\s*[\'"](.+?)[\'"]\s*,\s*\{', response_text, re.DOTALL)
                    if match:
                        response_text = match.group(1)
                except:
                    pass
            # Simpler tuple pattern without metadata
            try:
                match = re.search(r'^\s*\(\s*[\'"](.+?)[\'"]', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)
            except:
                pass
        
        # Remove conclusion artifacts
        response_text = response_text.replace("Conclusion: ", "")
        
        # Clean up metadata patterns if present
        response_text = re.sub(r"\{'method':.*?\}", "", response_text).strip()
        
        # Use safer pattern matching - capture the text in the tuple and replace the whole string with it
        tuple_pattern = r"\('(.*?)', \{'method':.*?\}\)"
        if re.search(tuple_pattern, response_text):
            response_text = re.sub(tuple_pattern, r"\1", response_text).strip()
        
        # Remove any nested parentheses artifacts that might be present
        # But only if they look like tuple artifacts
        tuple_paren_pattern = r"\(['\"](.+?)['\"].*?\)"
        if re.search(tuple_paren_pattern, response_text):
            response_text = re.sub(tuple_paren_pattern, r"\1", response_text)
        
        # Remove stringified tuple patterns at the start of text
        start_tuple_pattern = r"^\s*\(\s*['\"](.+?)['\"].*?$"
        if re.search(start_tuple_pattern, response_text):
            response_text = re.sub(start_tuple_pattern, r"\1", response_text)
        
        # Remove any remaining formatting that might have leaked through
        response_text = response_text.strip('"\'() ')
        
        return response_text

def create_router_adapter(
        use_sum2act: bool = True, 
        fallback_enabled: bool = True, 
        auto_retry: bool = True,
        llm_provider=None,
        state_manager=None,
        response_cache=None,
        config_path: str = "config/router_config.json") -> RouterAdapter:
    """
    Create a router adapter with the specified configuration
    
    Args:
        use_sum2act: Whether to use Sum2Act as primary router
        fallback_enabled: Whether to enable fallback to alternative approaches on failure
        auto_retry: Whether to retry failed requests with fallback methods
        llm_provider: Optional LLM provider instance (will be created if not provided)
        state_manager: Optional state manager instance
        response_cache: Optional response cache instance
        config_path: Optional path to router configuration file
    
    Returns:
        RouterAdapter: Configured router adapter instance
    """
    router = RouterAdapter(
        use_sum2act=use_sum2act,
        fallback_enabled=fallback_enabled,
        auto_retry=auto_retry,
        state_manager=state_manager,
        response_cache=response_cache,
        config_path=config_path
    )
    
    # Immediately initialize with LLM provider if provided
    if llm_provider:
        asyncio.create_task(router.init_routers(llm_provider))
        
    return router 