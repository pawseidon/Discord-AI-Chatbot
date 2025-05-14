"""
Cache integration module for Discord AI Chatbot.

This module provides unified access to different caching strategies
and integrates with the bot's features.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Union, Type

from .cache_interface import CacheInterface
from .context_aware_cache import ContextAwareCache
from .semantic_cache import SemanticCache

logger = logging.getLogger("cache_integration")

# Global instance for singleton pattern
_cache_instance = None

class CacheIntegration:
    """
    Integrated cache system combining multiple caching strategies
    for optimal performance and versatility
    """
    
    def __init__(self, 
                enable_context_cache: bool = True,
                enable_semantic_cache: bool = True,
                max_size: int = 2000,
                ttl: int = 3600):
        """
        Initialize the cache integration
        
        Args:
            enable_context_cache: Whether to enable context-aware caching
            enable_semantic_cache: Whether to enable semantic caching
            max_size: Maximum size per cache type
            ttl: Default time-to-live in seconds
        """
        self.caches = {}
        
        # Initialize standard cache (context-aware cache as base)
        self.caches["standard"] = ContextAwareCache(
            max_size=max_size,
            ttl=ttl,
            context_weight=0.0  # No context weighting for standard cache
        )
        
        # Initialize context-aware cache if enabled
        if enable_context_cache:
            self.caches["context"] = ContextAwareCache(
                max_size=max_size,
                ttl=ttl,
                context_weight=0.7
            )
        
        # Initialize semantic cache if enabled
        if enable_semantic_cache:
            try:
                self.caches["semantic"] = SemanticCache(
                    max_size=max_size,
                    ttl=ttl,
                    similarity_threshold=0.85
                )
            except Exception as e:
                logger.warning(f"Failed to initialize semantic cache: {e}")
        
        logger.info(f"Initialized cache integration with {len(self.caches)} caching strategies")
    
    async def get(self, 
                key: str, 
                default: Any = None, 
                cache_type: str = "default",
                user_id: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None,
                strategy: str = "standard") -> Any:
        """
        Get a value from the cache
        
        Args:
            key: Cache key or query string
            default: Default value if key not found
            cache_type: Type of cache to query
            user_id: Optional user ID for user-specific caches
            context: Optional context for context-aware lookup
            strategy: Caching strategy to use
            
        Returns:
            Cached value or default
        """
        # Check if strategy exists
        if strategy not in self.caches:
            strategy = "standard"
        
        cache = self.caches[strategy]
        
        # Handle different strategy-specific behaviors
        if strategy == "context" and hasattr(cache, "get") and context:
            return await cache.get(key, default, cache_type, user_id, context)
        else:
            return await cache.get(key, default, cache_type, user_id)
    
    async def set(self, 
                key: str, 
                value: Any, 
                ttl: Optional[int] = None,
                cache_type: str = "default",
                user_id: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None,
                strategy: Optional[str] = None) -> bool:
        """
        Set a value in the cache
        
        Args:
            key: Cache key or query string
            value: Value to cache
            ttl: Time-to-live in seconds, None for default
            cache_type: Type of cache to set
            user_id: Optional user ID for user-specific caches
            context: Optional context for context-aware storage
            strategy: Specific caching strategy, None for all
            
        Returns:
            True if successful, False otherwise
        """
        success = True
        
        # If strategy specified, only use that one
        if strategy and strategy in self.caches:
            cache = self.caches[strategy]
            
            if strategy == "context" and hasattr(cache, "set") and context:
                return await cache.set(key, value, ttl, cache_type, user_id, context)
            else:
                return await cache.set(key, value, ttl, cache_type, user_id)
        
        # Otherwise, use all available strategies
        for strategy_name, cache in self.caches.items():
            try:
                if strategy_name == "context" and context:
                    result = await cache.set(key, value, ttl, cache_type, user_id, context)
                else:
                    result = await cache.set(key, value, ttl, cache_type, user_id)
                
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"Error setting cache with strategy {strategy_name}: {e}")
                success = False
        
        return success
    
    async def delete(self, 
                   key: str, 
                   cache_type: str = "default",
                   user_id: Optional[str] = None,
                   strategy: Optional[str] = None) -> bool:
        """
        Delete a value from the cache
        
        Args:
            key: Cache key
            cache_type: Type of cache to delete from
            user_id: Optional user ID for user-specific caches
            strategy: Specific caching strategy, None for all
            
        Returns:
            True if key was deleted from any cache, False otherwise
        """
        found = False
        
        # If strategy specified, only use that one
        if strategy and strategy in self.caches:
            return await self.caches[strategy].delete(key, cache_type, user_id)
        
        # Otherwise, delete from all strategies
        for cache in self.caches.values():
            try:
                if await cache.delete(key, cache_type, user_id):
                    found = True
            except Exception as e:
                logger.error(f"Error deleting from cache: {e}")
        
        return found
    
    async def clear(self, 
                  cache_type: Optional[str] = None,
                  user_id: Optional[str] = None,
                  strategy: Optional[str] = None) -> bool:
        """
        Clear the cache
        
        Args:
            cache_type: Specific cache type to clear, None for all
            user_id: Optional user ID for user-specific caches
            strategy: Specific caching strategy, None for all
            
        Returns:
            True if operation was successful
        """
        success = True
        
        # If strategy specified, only use that one
        if strategy and strategy in self.caches:
            return await self.caches[strategy].clear(cache_type, user_id)
        
        # Otherwise, clear all strategies
        for cache in self.caches.values():
            try:
                result = await cache.clear(cache_type, user_id)
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                success = False
        
        return success
    
    async def get_best_strategy(self, query_type: str, context_size: int = 0) -> str:
        """
        Determine the best caching strategy based on query type
        
        Args:
            query_type: Type of query (e.g., "factual", "conversation", "creative")
            context_size: Size/complexity of available context
            
        Returns:
            Name of the best strategy to use
        """
        if query_type == "conversation" and context_size > 0 and "context" in self.caches:
            return "context"
        elif query_type == "factual" and "semantic" in self.caches:
            return "semantic"
        else:
            return "standard"
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics from all strategies"""
        metrics = {}
        
        for strategy_name, cache in self.caches.items():
            try:
                cache_metrics = await cache.get_metrics()
                metrics[strategy_name] = cache_metrics
            except Exception as e:
                logger.error(f"Error getting metrics from {strategy_name} cache: {e}")
                metrics[strategy_name] = {"error": str(e)}
        
        return metrics
    
    async def optimize(self, max_memory_mb: Optional[int] = None) -> None:
        """
        Optimize cache settings based on available memory
        
        Args:
            max_memory_mb: Maximum memory usage in MB, None for auto-detect
        """
        # If memory limit specified, adjust cache sizes
        if max_memory_mb:
            # Simple heuristic: divide evenly among caches
            cache_count = len(self.caches)
            if cache_count > 0:
                size_per_cache = int((max_memory_mb * 1024 * 1024) / (cache_count * 250))  # ~250 bytes per entry
                
                for cache in self.caches.values():
                    if hasattr(cache, "max_size"):
                        cache.max_size = size_per_cache
                
                logger.info(f"Adjusted cache sizes to {size_per_cache} entries each")

def get_cache_handler(
    enable_context_cache: bool = True,
    enable_semantic_cache: bool = True,
    max_size: int = 2000,
    ttl: int = 3600
) -> CacheIntegration:
    """
    Get or create the global cache integration instance
    
    Args:
        enable_context_cache: Whether to enable context-aware caching
        enable_semantic_cache: Whether to enable semantic caching
        max_size: Maximum size per cache type
        ttl: Default time-to-live in seconds
        
    Returns:
        Cache integration instance
    """
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = CacheIntegration(
            enable_context_cache=enable_context_cache,
            enable_semantic_cache=enable_semantic_cache,
            max_size=max_size,
            ttl=ttl
        )
        
    return _cache_instance 