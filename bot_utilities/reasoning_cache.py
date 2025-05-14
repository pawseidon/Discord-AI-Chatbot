import time
import json
import os
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('reasoning_cache')

class ReasoningCache:
    """
    Caching system for reasoning type detection to improve performance.
    
    Features:
    - Query-based cache: Store reasoning types for similar queries
    - User preference cache: Faster access to user preferences
    - Contextual cache: Cache based on conversation context
    - TTL-based expiration: Cache entries expire after a configurable time
    - Persistent storage: Option to save cache to disk for reuse between restarts
    """
    
    def __init__(self, cache_file: str = "data/reasoning_cache.json", ttl: int = 3600):
        """
        Initialize the reasoning cache.
        
        Args:
            cache_file: Path to persistent cache file
            ttl: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        self.cache_file = cache_file
        self.ttl = ttl
        
        # Main caches
        self.query_cache = {}  # Maps normalized queries to reasoning types
        self.user_cache = {}   # Maps user_id to preferred reasoning type
        self.context_cache = defaultdict(dict)  # Maps conversation_id -> {context_hash -> reasoning_type}
        
        # Metadata for cache entries
        self.timestamps = {}  # Maps cache_key to last access timestamp
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Load existing cache if available
        self._load_cache()
        
    def _load_cache(self):
        """Load cache from disk if available."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.query_cache = cache_data.get('query_cache', {})
                    self.user_cache = cache_data.get('user_cache', {})
                    self.context_cache = defaultdict(dict, cache_data.get('context_cache', {}))
                    self.timestamps = cache_data.get('timestamps', {})
                    
                    # Clean expired entries on load
                    self._clean_expired()
                    logger.info(f"Loaded {len(self.query_cache)} query cache entries")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            # Initialize empty caches if loading fails
            self.query_cache = {}
            self.user_cache = {}
            self.context_cache = defaultdict(dict)
            self.timestamps = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            # Clean expired entries before saving
            self._clean_expired()
            
            cache_data = {
                'query_cache': self.query_cache,
                'user_cache': self.user_cache,
                'context_cache': dict(self.context_cache),
                'timestamps': self.timestamps
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f)
                
            logger.info(f"Saved cache with {len(self.query_cache)} query entries")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _clean_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        # Check query cache
        for key in self.query_cache:
            if key in self.timestamps and current_time - self.timestamps[key] > self.ttl:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self.query_cache[key]
            if key in self.timestamps:
                del self.timestamps[key]
        
        # We don't expire user preferences, as they're intended to be long-lived
        
        # Check context cache
        for conv_id in list(self.context_cache.keys()):
            for ctx_hash in list(self.context_cache[conv_id].keys()):
                cache_key = f"ctx:{conv_id}:{ctx_hash}"
                if cache_key in self.timestamps and current_time - self.timestamps[cache_key] > self.ttl:
                    del self.context_cache[conv_id][ctx_hash]
                    del self.timestamps[cache_key]
            
            # Remove empty conversation entries
            if not self.context_cache[conv_id]:
                del self.context_cache[conv_id]
        
        logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for better cache hit rate.
        
        This removes minor variations like whitespace, capitalization, and punctuation
        that don't affect reasoning type.
        """
        # Convert to lowercase
        query = query.lower()
        
        # Replace multiple spaces with single space
        query = ' '.join(query.split())
        
        # Consider additional normalizations based on your specific use case
        # e.g., removing certain stopwords, normalizing synonyms, etc.
        
        return query
    
    def _generate_context_hash(self, conversation_history: List[str]) -> str:
        """
        Generate a hash for conversation context.
        
        Uses the last few messages to create a context fingerprint.
        """
        if not conversation_history:
            return "empty"
            
        # Take the last 3 messages for context (or all if fewer than 3)
        relevant_history = conversation_history[-3:]
        
        # Create a simplified representation by joining normalized messages
        context_text = " ".join([self._normalize_query(msg) for msg in relevant_history])
        
        # Basic hash function - consider using a more robust hash if needed
        import hashlib
        return hashlib.md5(context_text.encode()).hexdigest()
    
    def get_cached_reasoning(self, 
                           query: str, 
                           user_id: Optional[str] = None,
                           conversation_id: Optional[str] = None,
                           conversation_history: Optional[List[str]] = None) -> Optional[Tuple[str, float]]:
        """
        Attempt to retrieve cached reasoning type for the given query and context.
        
        Returns:
            Optional tuple of (reasoning_type, confidence)
            None if no cache hit
        """
        current_time = time.time()
        
        # Try exact query match first
        normalized_query = self._normalize_query(query)
        if normalized_query in self.query_cache:
            # Update timestamp and return
            self.timestamps[normalized_query] = current_time
            return self.query_cache[normalized_query]
            
        # Try context-based cache if conversation history provided
        if conversation_id and conversation_history:
            context_hash = self._generate_context_hash(conversation_history)
            if context_hash in self.context_cache.get(conversation_id, {}):
                cache_key = f"ctx:{conversation_id}:{context_hash}"
                self.timestamps[cache_key] = current_time
                return self.context_cache[conversation_id][context_hash]
        
        # No cache hit
        return None
    
    def cache_reasoning(self,
                      query: str,
                      reasoning_result: Tuple[str, float],
                      user_id: Optional[str] = None,
                      conversation_id: Optional[str] = None,
                      conversation_history: Optional[List[str]] = None):
        """
        Cache reasoning type detection result.
        
        Args:
            query: The original query
            reasoning_result: Tuple of (reasoning_type, confidence)
            user_id: Optional user ID
            conversation_id: Optional conversation/thread ID
            conversation_history: Optional conversation history
        """
        current_time = time.time()
        
        # Cache by normalized query
        normalized_query = self._normalize_query(query)
        self.query_cache[normalized_query] = reasoning_result
        self.timestamps[normalized_query] = current_time
        
        # Cache by conversation context if provided
        if conversation_id and conversation_history:
            context_hash = self._generate_context_hash(conversation_history)
            self.context_cache[conversation_id][context_hash] = reasoning_result
            cache_key = f"ctx:{conversation_id}:{context_hash}"
            self.timestamps[cache_key] = current_time
        
        # Periodically save cache and clean expired entries
        # Use a random chance to avoid doing this too frequently
        if hash(query) % 10 == 0:  # ~10% chance
            self._save_cache()
    
    def set_user_preference(self, user_id: str, reasoning_type: str):
        """
        Cache user's preferred reasoning type.
        
        Args:
            user_id: User ID
            reasoning_type: Preferred reasoning type
        """
        self.user_cache[user_id] = reasoning_type
        self._save_cache()
    
    def get_user_preference(self, user_id: str) -> Optional[str]:
        """
        Get user's preferred reasoning type from cache.
        
        Args:
            user_id: User ID
        
        Returns:
            Optional preferred reasoning type or None if not set
        """
        return self.user_cache.get(user_id)
    
    def invalidate_query_cache(self, query_pattern: Optional[str] = None):
        """
        Invalidate query cache entries matching a pattern.
        
        Args:
            query_pattern: Optional pattern to match (None to clear all)
        """
        if query_pattern is None:
            self.query_cache.clear()
            # Also clear timestamps related to query cache
            self.timestamps = {k: v for k, v in self.timestamps.items() if not k.startswith("query:")}
            logger.info("Cleared entire query cache")
        else:
            import re
            pattern = re.compile(query_pattern)
            keys_to_remove = []
            for key in self.query_cache:
                if pattern.search(key):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.query_cache[key]
                if key in self.timestamps:
                    del self.timestamps[key]
            
            logger.info(f"Removed {len(keys_to_remove)} matching query cache entries")
            
    def invalidate_conversation_cache(self, conversation_id: str):
        """
        Invalidate cache for a specific conversation.
        
        Args:
            conversation_id: Conversation/thread ID to invalidate
        """
        if conversation_id in self.context_cache:
            # Remove all context cache for this conversation
            del self.context_cache[conversation_id]
            
            # Remove related timestamps
            prefix = f"ctx:{conversation_id}:"
            self.timestamps = {k: v for k, v in self.timestamps.items() if not k.startswith(prefix)}
            
            logger.info(f"Cleared cache for conversation {conversation_id}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            "query_cache_size": len(self.query_cache),
            "user_cache_size": len(self.user_cache),
            "conversation_cache_size": sum(len(ctx) for ctx in self.context_cache.values()),
            "total_cache_entries": len(self.query_cache) + len(self.user_cache) + 
                                 sum(len(ctx) for ctx in self.context_cache.values()),
            "cache_file_size_kb": os.path.getsize(self.cache_file) / 1024 if os.path.exists(self.cache_file) else 0
        } 