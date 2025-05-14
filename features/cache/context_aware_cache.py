"""
Context-aware cache for Discord AI Chatbot.

This module provides a caching system that considers conversation context
when determining cache relevance.
"""

import asyncio
import time
import json
import hashlib
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from .cache_interface import CacheInterface

logger = logging.getLogger("context_aware_cache")

class ContextAwareCache(CacheInterface):
    """
    Context-aware caching system that accounts for conversation context
    when determining cache relevance
    """
    
    def __init__(self, 
                max_size: int = 1000, 
                ttl: int = 3600,
                context_weight: float = 0.7):
        """
        Initialize the context-aware cache
        
        Args:
            max_size: Maximum number of entries in the cache
            ttl: Default time-to-live in seconds
            context_weight: Weight of context in cache key generation (0-1)
        """
        super().__init__(max_size=max_size, ttl=ttl, cache_name="context_aware")
        self.context_weight = max(0.0, min(1.0, context_weight))
        
        # Main cache storage
        self._caches = defaultdict(dict)  # type: Dict[str, Dict[str, Dict[str, Any]]]
        self._user_caches = defaultdict(lambda: defaultdict(dict))  # type: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]
        
        # Cache metadata
        self._timestamps = defaultdict(dict)  # type: Dict[str, Dict[str, float]]
        self._user_timestamps = defaultdict(lambda: defaultdict(dict))  # type: Dict[str, Dict[str, Dict[str, float]]]
        
        # Context tracking
        self._context_keys = defaultdict(list)  # type: Dict[str, List[str]]
        self._user_context_keys = defaultdict(lambda: defaultdict(list))  # type: Dict[str, Dict[str, List[str]]]
        
        logger.info(f"Initialized context-aware cache with context_weight={context_weight}")
    
    async def get(self, 
                key: str, 
                default: Any = None, 
                cache_type: str = "default",
                user_id: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Get a value from the cache
        
        Args:
            key: Cache key or query string
            default: Default value if key not found
            cache_type: Type of cache to query
            user_id: Optional user ID for user-specific caches
            context: Optional context for context-aware lookup
            
        Returns:
            Cached value or default
        """
        # Generate context-aware key if context provided
        if context and self.context_weight > 0:
            key = self._generate_context_key(key, context)
        
        # Check user-specific cache first if user_id provided
        if user_id:
            if cache_type in self._user_caches[user_id] and key in self._user_caches[user_id][cache_type]:
                # Check if expired
                timestamp = self._user_timestamps[user_id][cache_type].get(key, 0)
                if time.time() - timestamp <= self.default_ttl:
                    self._update_metric("hits")
                    return self._user_caches[user_id][cache_type][key]
                else:
                    # Expired, remove it
                    await self.delete(key, cache_type, user_id)
        
        # Check global cache
        if cache_type in self._caches and key in self._caches[cache_type]:
            # Check if expired
            timestamp = self._timestamps[cache_type].get(key, 0)
            if time.time() - timestamp <= self.default_ttl:
                self._update_metric("hits")
                return self._caches[cache_type][key]
            else:
                # Expired, remove it
                await self.delete(key, cache_type)
        
        self._update_metric("misses")
        return default
    
    async def set(self, 
                key: str, 
                value: Any, 
                ttl: Optional[int] = None,
                cache_type: str = "default",
                user_id: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set a value in the cache
        
        Args:
            key: Cache key or query string
            value: Value to cache
            ttl: Time-to-live in seconds, None for default
            cache_type: Type of cache to set
            user_id: Optional user ID for user-specific caches
            context: Optional context for context-aware storage
            
        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl
        
        # Generate context-aware key if context provided
        original_key = key
        if context and self.context_weight > 0:
            key = self._generate_context_key(key, context)
            
            # Track context keys for cleanup
            if user_id:
                self._user_context_keys[user_id][original_key].append(key)
            else:
                self._context_keys[original_key].append(key)
        
        # Ensure we don't exceed max size
        await self._check_size(cache_type, user_id)
        
        # Store in appropriate cache
        if user_id:
            self._user_caches[user_id][cache_type][key] = value
            self._user_timestamps[user_id][cache_type][key] = time.time()
        else:
            self._caches[cache_type][key] = value
            self._timestamps[cache_type][key] = time.time()
        
        self._update_metric("sets")
        return True
    
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
        found = False
        
        # Check if we need to delete context variants
        if user_id and key in self._user_context_keys[user_id]:
            for context_key in self._user_context_keys[user_id][key]:
                if context_key in self._user_caches[user_id][cache_type]:
                    del self._user_caches[user_id][cache_type][context_key]
                    del self._user_timestamps[user_id][cache_type][context_key]
                    found = True
            del self._user_context_keys[user_id][key]
        
        if key in self._context_keys:
            for context_key in self._context_keys[key]:
                if context_key in self._caches[cache_type]:
                    del self._caches[cache_type][context_key]
                    del self._timestamps[cache_type][context_key]
                    found = True
            del self._context_keys[key]
        
        # Delete direct key
        if user_id and key in self._user_caches[user_id][cache_type]:
            del self._user_caches[user_id][cache_type][key]
            del self._user_timestamps[user_id][cache_type][key]
            found = True
        
        if key in self._caches[cache_type]:
            del self._caches[cache_type][key]
            del self._timestamps[cache_type][key]
            found = True
        
        if found:
            self._update_metric("deletes")
        
        return found
    
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
        if user_id:
            if cache_type:
                if cache_type in self._user_caches[user_id]:
                    self._user_caches[user_id][cache_type].clear()
                    self._user_timestamps[user_id][cache_type].clear()
            else:
                self._user_caches[user_id].clear()
                self._user_timestamps[user_id].clear()
                self._user_context_keys[user_id].clear()
        else:
            if cache_type:
                if cache_type in self._caches:
                    self._caches[cache_type].clear()
                    self._timestamps[cache_type].clear()
            else:
                self._caches.clear()
                self._timestamps.clear()
                self._context_keys.clear()
        
        return True
    
    async def get_cached_items(self, 
                             cache_type: str = "default",
                             user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all cached items of a specific type
        
        Args:
            cache_type: Type of cache to query
            user_id: Optional user ID for user-specific cache
            
        Returns:
            Dictionary of cached items
        """
        result = {}
        
        if user_id:
            if cache_type in self._user_caches[user_id]:
                # Filter out expired items
                current_time = time.time()
                for key, value in self._user_caches[user_id][cache_type].items():
                    timestamp = self._user_timestamps[user_id][cache_type].get(key, 0)
                    if current_time - timestamp <= self.default_ttl:
                        result[key] = value
        else:
            if cache_type in self._caches:
                # Filter out expired items
                current_time = time.time()
                for key, value in self._caches[cache_type].items():
                    timestamp = self._timestamps[cache_type].get(key, 0)
                    if current_time - timestamp <= self.default_ttl:
                        result[key] = value
        
        return result
    
    def _generate_context_key(self, key: str, context: Dict[str, Any]) -> str:
        """
        Generate a context-aware cache key
        
        Args:
            key: Original key
            context: Context dictionary
            
        Returns:
            Context-enhanced key
        """
        context_str = self._extract_context_signature(context)
        
        # Hash the context to keep keys manageable
        context_hash = hashlib.md5(context_str.encode()).hexdigest()[:10]
        
        # Mix original key with context based on context_weight
        if self.context_weight >= 1.0:
            return f"{key}:{context_hash}"
        elif self.context_weight <= 0.0:
            return key
        else:
            # Partial mixing based on weight
            key_hash = hashlib.md5(key.encode()).hexdigest()
            mixed_hash = hashlib.md5(
                (key_hash[:int(32 * (1 - self.context_weight))] + 
                 context_hash[:int(10 * self.context_weight)]).encode()
            ).hexdigest()[:10]
            return f"{key}:{mixed_hash}"
    
    def _extract_context_signature(self, context: Dict[str, Any]) -> str:
        """
        Extract a signature string from context
        
        Args:
            context: Context dictionary
            
        Returns:
            Context signature string
        """
        if not context:
            return ""
            
        # Extract key elements from context
        signature_parts = []
        
        # Check for conversation history
        if "conversation_history" in context:
            # Use last few messages only
            history = context["conversation_history"]
            if isinstance(history, list) and history:
                last_messages = history[-3:]  # Last 3 messages
                for msg in last_messages:
                    if isinstance(msg, dict):
                        if "user" in msg:
                            signature_parts.append(f"u:{msg['user'][:50]}")
                        if "bot" in msg:
                            signature_parts.append(f"b:{msg['bot'][:50]}")
        
        # Check for user profile/preferences
        if "user_profile" in context and isinstance(context["user_profile"], dict):
            profile = context["user_profile"]
            if "preferences" in profile and isinstance(profile["preferences"], dict):
                prefs = profile["preferences"]
                for key in sorted(prefs.keys())[:5]:  # Limit to 5 preferences
                    signature_parts.append(f"p:{key}:{prefs[key]}")
        
        # Check for current channel or guild
        if "channel_id" in context:
            signature_parts.append(f"c:{context['channel_id']}")
        if "guild_id" in context:
            signature_parts.append(f"g:{context['guild_id']}")
        
        return "|".join(signature_parts)
    
    async def _check_size(self, cache_type: str, user_id: Optional[str] = None) -> None:
        """
        Check if cache has reached max size and evict if needed
        
        Args:
            cache_type: Cache type to check
            user_id: Optional user ID for user-specific cache
        """
        # Check size of appropriate cache
        if user_id:
            cache_size = len(self._user_caches[user_id][cache_type])
            if cache_size >= self.max_size:
                await self._evict_oldest_entries(cache_type, user_id)
        else:
            cache_size = len(self._caches[cache_type])
            if cache_size >= self.max_size:
                await self._evict_oldest_entries(cache_type)
    
    async def _evict_oldest_entries(self, cache_type: str, user_id: Optional[str] = None) -> None:
        """
        Evict oldest entries from cache
        
        Args:
            cache_type: Cache type to evict from
            user_id: Optional user ID for user-specific cache
        """
        # Get timestamps in sorted order
        if user_id:
            timestamps = self._user_timestamps[user_id][cache_type]
            sorted_items = sorted(timestamps.items(), key=lambda x: x[1])
            
            # Remove oldest 10% or at least 1 entry
            to_remove = max(1, int(len(sorted_items) * 0.1))
            for key, _ in sorted_items[:to_remove]:
                del self._user_caches[user_id][cache_type][key]
                del self._user_timestamps[user_id][cache_type][key]
        else:
            timestamps = self._timestamps[cache_type]
            sorted_items = sorted(timestamps.items(), key=lambda x: x[1])
            
            # Remove oldest 10% or at least 1 entry
            to_remove = max(1, int(len(sorted_items) * 0.1))
            for key, _ in sorted_items[:to_remove]:
                del self._caches[cache_type][key]
                del self._timestamps[cache_type][key] 