"""
Base cache interface for Discord AI Chatbot.

This module provides the abstract base class for all cache implementations.
"""

import abc
from typing import Dict, Any, Optional, List, Union
import time
import logging

logger = logging.getLogger("cache_interface")

class CacheInterface(abc.ABC):
    """Abstract base class for cache implementations"""
    
    def __init__(self, 
                 max_size: int = 1000, 
                 ttl: int = 3600, 
                 cache_name: str = "generic"):
        """
        Initialize the cache interface
        
        Args:
            max_size: Maximum number of entries in the cache
            ttl: Default time-to-live in seconds
            cache_name: Name of the cache for logging
        """
        self.max_size = max_size
        self.default_ttl = ttl
        self.cache_name = cache_name
        
        # Metrics
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
        
        logger.info(f"Initialized {cache_name} cache with max_size={max_size}, ttl={ttl}")
    
    @abc.abstractmethod
    async def get(self, 
                key: str, 
                default: Any = None, 
                cache_type: str = "default",
                user_id: Optional[str] = None) -> Any:
        """
        Get a value from the cache
        
        Args:
            key: Cache key
            default: Default value if key not found
            cache_type: Type of cache to query
            user_id: Optional user ID for user-specific caches
            
        Returns:
            Cached value or default
        """
        pass
    
    @abc.abstractmethod
    async def set(self, 
                key: str, 
                value: Any, 
                ttl: Optional[int] = None,
                cache_type: str = "default",
                user_id: Optional[str] = None) -> bool:
        """
        Set a value in the cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds, None for default
            cache_type: Type of cache to set
            user_id: Optional user ID for user-specific caches
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def delete(self, 
                   key: str, 
                   cache_type: str = "default",
                   user_id: Optional[str] = None) -> bool:
        """
        Delete a value from the cache
        
        Args:
            key: Cache key
            cache_type: Type of cache to delete from
            user_id: Optional user ID for user-specific caches
            
        Returns:
            True if key was deleted, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def clear(self, 
                  cache_type: Optional[str] = None,
                  user_id: Optional[str] = None) -> bool:
        """
        Clear the cache
        
        Args:
            cache_type: Specific cache type to clear, None for all
            user_id: Optional user ID for user-specific caches
            
        Returns:
            True if operation was successful
        """
        pass
    
    async def has_key(self, 
                    key: str, 
                    cache_type: str = "default",
                    user_id: Optional[str] = None) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key
            cache_type: Type of cache to query
            user_id: Optional user ID for user-specific caches
            
        Returns:
            True if key exists, False otherwise
        """
        result = await self.get(key, cache_type=cache_type, user_id=user_id)
        return result is not None
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        return {
            "name": self.cache_name,
            "metrics": dict(self.metrics),
            "max_size": self.max_size,
            "ttl": self.default_ttl
        }
    
    def _update_metric(self, metric_name: str, increment: int = 1) -> None:
        """Update a metric counter"""
        if metric_name in self.metrics:
            self.metrics[metric_name] += increment 