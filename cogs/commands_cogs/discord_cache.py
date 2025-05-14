# bot_utilities/discord_cache.py
import logging
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import discord
from collections import defaultdict

logger = logging.getLogger("discord_cache")
logger.setLevel(logging.INFO)

class DiscordCache:
    """
    Specialized cache for Discord interactions that integrates with reasoning methods
    and provides context-aware storage and retrieval
    """
    def __init__(self, 
                 redis_client=None,
                 in_memory_ttl: int = 600,
                 user_context_ttl: int = 3600,
                 guild_cache_ttl: int = 86400,
                 channel_cache_ttl: int = 43200,
                 memory_limit: int = 10000,
                 enable_distributed: bool = False):
        """
        Initialize Discord-optimized cache
        
        Args:
            redis_client: Optional Redis client for distributed caching
            in_memory_ttl: Default TTL for in-memory cache items (10 min)
            user_context_ttl: TTL for user conversation context (1 hour)
            guild_cache_ttl: TTL for guild-specific data (24 hours)
            channel_cache_ttl: TTL for channel-specific data (12 hours)
            memory_limit: Maximum number of items in memory cache
            enable_distributed: Whether to enable distributed caching
        """
        self.redis_client = redis_client
        self.enable_distributed = enable_distributed and redis_client is not None
        
        # Cache TTLs
        self.ttls = {
            "default": in_memory_ttl,
            "user_context": user_context_ttl,
            "guild": guild_cache_ttl,
            "channel": channel_cache_ttl,
            "reasoning_state": 1800,  # 30 min for reasoning state
            "response": 3600,  # 1 hour for generated responses
            "verified_response": 7200,  # 2 hours for verified responses
        }
        
        # In-memory caches with different purposes
        self.memory_caches = {
            "interaction": {},  # Interaction-specific cache
            "user_context": {},  # User conversation context
            "guild": {},  # Guild-specific data
            "channel": {},  # Channel-specific data
            "reasoning_state": {},  # Reasoning state for different methods
            "response": {},  # Generated responses
        }
        
        # Cache metadata
        self.cache_stats = defaultdict(int)
        self.memory_limit = memory_limit
        
        # Set up periodic cleaning task
        self._setup_cache_maintenance()
    
    def _setup_cache_maintenance(self):
        """Set up periodic cache maintenance tasks"""
        self.maintenance_task = asyncio.create_task(self._run_maintenance())
    
    async def _run_maintenance(self):
        """Run periodic cache maintenance"""
        while True:
            try:
                # Clean expired items every 5 minutes
                await asyncio.sleep(300)
                self._clean_expired_items()
                
                # Run LRU eviction if memory limit exceeded
                if sum(len(cache) for cache in self.memory_caches.values()) > self.memory_limit:
                    self._evict_lru_items()
                
                # Log cache stats every hour
                self.cache_stats["maintenance_runs"] += 1
                if self.cache_stats["maintenance_runs"] % 12 == 0:
                    logger.info(f"Cache stats: {dict(self.cache_stats)}")
            except Exception as e:
                logger.error(f"Error in cache maintenance: {e}")
    
    def _clean_expired_items(self):
        """Remove expired items from all caches"""
        now = time.time()
        items_removed = 0
        
        for cache_name, cache in self.memory_caches.items():
            ttl = self.ttls.get(cache_name, self.ttls["default"])
            
            expired_keys = []
            for key, data in cache.items():
                if now > data.get("expiry", 0):
                    expired_keys.append(key)
            
            for key in expired_keys:
                cache.pop(key, None)
                items_removed += 1
        
        if items_removed > 0:
            logger.debug(f"Removed {items_removed} expired cache items")
            self.cache_stats["expired_items_removed"] += items_removed
    
    def _evict_lru_items(self, count: int = 100):
        """Evict least recently used items when memory limit is exceeded"""
        # Collect all items with their last access time
        all_items = []
        for cache_name, cache in self.memory_caches.items():
            for key, data in cache.items():
                all_items.append((cache_name, key, data.get("last_access", 0)))
        
        # Sort by last access time (oldest first)
        all_items.sort(key=lambda x: x[2])
        
        # Remove oldest items
        items_to_remove = min(count, len(all_items) - self.memory_limit)
        if items_to_remove <= 0:
            return
        
        for i in range(items_to_remove):
            cache_name, key, _ = all_items[i]
            self.memory_caches[cache_name].pop(key, None)
        
        logger.debug(f"Evicted {items_to_remove} LRU cache items")
        self.cache_stats["lru_items_evicted"] += items_to_remove
    
    async def get(self, key: str, cache_type: str = "interaction", 
                 user_id: str = None, guild_id: str = None, channel_id: str = None) -> Optional[Any]:
        """
        Get item from cache with Discord context awareness
        
        Args:
            key: Base cache key
            cache_type: Type of cache to use
            user_id: Optional user ID for user-specific cache
            guild_id: Optional guild ID for guild-specific cache
            channel_id: Optional channel ID for channel-specific cache
        
        Returns:
            Cached data or None if not found
        """
        # Build context-aware cache key
        full_key = self._build_key(key, user_id, guild_id, channel_id)
        
        # Try memory cache first
        cache = self.memory_caches.get(cache_type, {})
        if full_key in cache:
            item = cache[full_key]
            now = time.time()
            
            # Check if expired
            if now > item.get("expiry", 0):
                cache.pop(full_key, None)
                self.cache_stats["memory_misses"] += 1
                return None
            
            # Update access metadata
            item["last_access"] = now
            item["access_count"] = item.get("access_count", 0) + 1
            
            self.cache_stats["memory_hits"] += 1
            return item.get("data")
        
        self.cache_stats["memory_misses"] += 1
        
        # If distributed cache enabled, try Redis
        if self.enable_distributed:
            redis_key = f"{cache_type}:{full_key}"
            try:
                data = await self.redis_client.get(redis_key)
                if data:
                    try:
                        parsed_data = json.loads(data)
                        # Store in memory cache for faster future access
                        ttl = self.ttls.get(cache_type, self.ttls["default"])
                        await self.set(key, parsed_data, cache_type, ttl, user_id, guild_id, channel_id)
                        
                        self.cache_stats["redis_hits"] += 1
                        return parsed_data
                    except json.JSONDecodeError:
                        self.cache_stats["redis_decode_errors"] += 1
                
                self.cache_stats["redis_misses"] += 1
            except Exception as e:
                logger.error(f"Redis error in get: {e}")
                self.cache_stats["redis_errors"] += 1
        
        return None
    
    async def set(self, key: str, data: Any, cache_type: str = "interaction", 
                 ttl: int = None, user_id: str = None, guild_id: str = None, 
                 channel_id: str = None) -> bool:
        """
        Store item in cache with Discord context awareness
        
        Args:
            key: Base cache key
            data: Data to cache
            cache_type: Type of cache to use
            ttl: Time-to-live in seconds (overrides default for cache type)
            user_id: Optional user ID for user-specific cache
            guild_id: Optional guild ID for guild-specific cache
            channel_id: Optional channel ID for channel-specific cache
        
        Returns:
            True if successful, False otherwise
        """
        # Build context-aware cache key
        full_key = self._build_key(key, user_id, guild_id, channel_id)
        
        # Set TTL based on cache type if not specified
        if ttl is None:
            ttl = self.ttls.get(cache_type, self.ttls["default"])
        
        # Store in memory cache
        now = time.time()
        cache = self.memory_caches.setdefault(cache_type, {})
        
        cache[full_key] = {
            "data": data,
            "created": now,
            "expiry": now + ttl,
            "last_access": now,
            "access_count": 0,
            "user_id": user_id,
            "guild_id": guild_id,
            "channel_id": channel_id
        }
        
        self.cache_stats["memory_sets"] += 1
        
        # If distributed cache enabled, store in Redis
        if self.enable_distributed:
            redis_key = f"{cache_type}:{full_key}"
            try:
                serialized = json.dumps(data)
                await self.redis_client.setex(redis_key, ttl, serialized)
                self.cache_stats["redis_sets"] += 1
            except Exception as e:
                logger.error(f"Redis error in set: {e}")
                self.cache_stats["redis_errors"] += 1
                return False
        
        return True
    
    async def delete(self, key: str, cache_type: str = "interaction",
                    user_id: str = None, guild_id: str = None, channel_id: str = None) -> bool:
        """
        Delete item from cache
        
        Args:
            key: Base cache key
            cache_type: Type of cache to use
            user_id: Optional user ID for user-specific cache
            guild_id: Optional guild ID for guild-specific cache
            channel_id: Optional channel ID for channel-specific cache
        
        Returns:
            True if deleted, False if not found
        """
        # Build context-aware cache key
        full_key = self._build_key(key, user_id, guild_id, channel_id)
        
        # Delete from memory cache
        cache = self.memory_caches.get(cache_type, {})
        was_in_memory = full_key in cache
        
        if was_in_memory:
            cache.pop(full_key, None)
            self.cache_stats["memory_deletes"] += 1
        
        # If distributed cache enabled, delete from Redis
        if self.enable_distributed:
            redis_key = f"{cache_type}:{full_key}"
            try:
                deleted = await self.redis_client.delete(redis_key)
                if deleted:
                    self.cache_stats["redis_deletes"] += 1
                    return True
            except Exception as e:
                logger.error(f"Redis error in delete: {e}")
                self.cache_stats["redis_errors"] += 1
        
        return was_in_memory
    
    async def get_by_pattern(self, pattern: str, cache_type: str = "interaction") -> List[Tuple[str, Any]]:
        """
        Get items with keys matching a pattern
        
        Args:
            pattern: Pattern to match (simple substring match for memory, glob pattern for Redis)
            cache_type: Type of cache to use
        
        Returns:
            List of (key, data) tuples for matching items
        """
        results = []
        
        # Check memory cache
        cache = self.memory_caches.get(cache_type, {})
        now = time.time()
        
        for full_key, item in cache.items():
            # Skip expired items
            if now > item.get("expiry", 0):
                continue
                
            if pattern in full_key:
                # Update access metadata
                item["last_access"] = now
                item["access_count"] = item.get("access_count", 0) + 1
                
                # Add to results
                results.append((full_key, item.get("data")))
        
        # If distributed cache enabled, check Redis
        if self.enable_distributed and not results:
            redis_pattern = f"{cache_type}:{pattern}*"
            try:
                matching_keys = await self.redis_client.keys(redis_pattern)
                
                for redis_key in matching_keys:
                    data = await self.redis_client.get(redis_key)
                    if data:
                        try:
                            parsed_data = json.loads(data)
                            # Extract original key from Redis key
                            original_key = redis_key.decode('utf-8').split(':', 1)[1]
                            results.append((original_key, parsed_data))
                        except (json.JSONDecodeError, UnicodeDecodeError, IndexError):
                            continue
            except Exception as e:
                logger.error(f"Redis error in get_by_pattern: {e}")
        
        return results
    
    async def clear_user_data(self, user_id: str) -> int:
        """
        Clear all data for a specific user (privacy/GDPR)
        
        Args:
            user_id: User ID to clear data for
            
        Returns:
            Number of items cleared
        """
        items_cleared = 0
        
        # Clear from memory caches
        for cache_name, cache in self.memory_caches.items():
            keys_to_remove = []
            
            for key, item in cache.items():
                if item.get("user_id") == user_id:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                cache.pop(key, None)
                items_cleared += 1
        
        # Clear from Redis if enabled
        if self.enable_distributed:
            try:
                # Search patterns for user data
                patterns = [f"*:user:{user_id}*", f"*:user_{user_id}*"]
                
                for pattern in patterns:
                    matching_keys = await self.redis_client.keys(pattern)
                    if matching_keys:
                        await self.redis_client.delete(*matching_keys)
                        items_cleared += len(matching_keys)
            except Exception as e:
                logger.error(f"Redis error in clear_user_data: {e}")
        
        self.cache_stats["user_data_cleared"] += items_cleared
        return items_cleared
    
    def _build_key(self, key: str, user_id: str = None, guild_id: str = None, channel_id: str = None) -> str:
        """Build a context-aware cache key based on Discord IDs"""
        components = [key]
        
        if user_id:
            components.append(f"user:{user_id}")
        
        if guild_id:
            components.append(f"guild:{guild_id}")
        
        if channel_id:
            components.append(f"channel:{channel_id}")
        
        return ":".join(components)
    
    async def store_reasoning_state(self, 
                                   user_id: str, 
                                   reasoning_type: str, 
                                   state_data: Dict[str, Any],
                                   channel_id: str = None,
                                   guild_id: str = None) -> bool:
        """
        Store reasoning state for a specific user and reasoning method
        
        Args:
            user_id: User ID
            reasoning_type: Type of reasoning (sequential, rag, react, etc.)
            state_data: State data to store
            channel_id: Optional channel context
            guild_id: Optional guild context
            
        Returns:
            True if successful
        """
        key = f"reasoning:{reasoning_type}"
        return await self.set(
            key, 
            state_data, 
            cache_type="reasoning_state",
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id
        )
    
    async def get_reasoning_state(self, 
                                user_id: str, 
                                reasoning_type: str,
                                channel_id: str = None,
                                guild_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Get reasoning state for a specific user and reasoning method
        
        Args:
            user_id: User ID
            reasoning_type: Type of reasoning (sequential, rag, react, etc.)
            channel_id: Optional channel context
            guild_id: Optional guild context
            
        Returns:
            State data or None if not found
        """
        key = f"reasoning:{reasoning_type}"
        return await self.get(
            key, 
            cache_type="reasoning_state", 
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id
        )
    
    async def store_user_context(self, 
                               user_id: str, 
                               context_data: Dict[str, Any],
                               channel_id: str = None,
                               guild_id: str = None) -> bool:
        """
        Store user conversation context
        
        Args:
            user_id: User ID
            context_data: Context data to store
            channel_id: Optional channel context
            guild_id: Optional guild context
            
        Returns:
            True if successful
        """
        key = "conversation_context"
        return await self.set(
            key,
            context_data,
            cache_type="user_context",
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id
        )
    
    async def get_user_context(self, 
                             user_id: str,
                             channel_id: str = None,
                             guild_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Get user conversation context
        
        Args:
            user_id: User ID
            channel_id: Optional channel context
            guild_id: Optional guild context
            
        Returns:
            Context data or None if not found
        """
        key = "conversation_context"
        return await self.get(
            key,
            cache_type="user_context",
            user_id=user_id,
            guild_id=guild_id,
            channel_id=channel_id
        )
    
    async def cache_verified_response(self, 
                                    query: str, 
                                    response: Any,
                                    verification_result: Dict[str, Any],
                                    user_id: str = None,
                                    channel_id: str = None) -> str:
        """
        Cache a verified response with its verification metadata
        
        Args:
            query: User query
            response: Response to cache
            verification_result: Verification result from hallucination handler
            user_id: Optional user ID
            channel_id: Optional channel ID
            
        Returns:
            Cache key for the stored response
        """
        import hashlib
        
        # Create fingerprint for the query
        query_hash = hashlib.md5(query.encode()).hexdigest()
        key = f"response:{query_hash}"
        
        # Combine response and verification data
        cache_data = {
            "response": response,
            "verification": verification_result,
            "query": query,
            "cached_at": time.time()
        }
        
        # Use longer TTL for highly verified responses
        confidence = verification_result.get("confidence", 0)
        ttl = self.ttls["verified_response"] if confidence > 0.8 else self.ttls["response"]
        
        await self.set(
            key,
            cache_data,
            cache_type="response",
            ttl=ttl,
            user_id=user_id,
            channel_id=channel_id
        )
        
        return key
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = dict(self.cache_stats)
        
        # Add current cache size information
        stats["current_items"] = {}
        for cache_name, cache in self.memory_caches.items():
            stats["current_items"][cache_name] = len(cache)
        
        stats["total_items"] = sum(len(cache) for cache in self.memory_caches.values())
        
        # Add Redis stats if available
        if self.enable_distributed:
            try:
                stats["redis_info"] = await self.redis_client.info()
            except Exception as e:
                stats["redis_info_error"] = str(e)
        
        return stats

def create_discord_cache(redis_url: str = None, **kwargs) -> DiscordCache:
    """
    Factory function to create a Discord cache
    
    Args:
        redis_url: Redis URL for distributed caching
        **kwargs: Additional options for DiscordCache
        
    Returns:
        Configured DiscordCache instance
    """
    redis_client = None
    
    if redis_url:
        try:
            import redis.asyncio as aioredis
            redis_client = aioredis.from_url(redis_url)
            kwargs["enable_distributed"] = True
        except (ImportError, Exception) as e:
            logger.error(f"Failed to initialize Redis: {e}")
    
    return DiscordCache(redis_client=redis_client, **kwargs)