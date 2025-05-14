"""
Semantic cache for Discord AI Chatbot.

This module provides a caching system that uses semantic similarity
to match queries with cached responses.
"""

import asyncio
import time
import json
import hashlib
from collections import defaultdict
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from .cache_interface import CacheInterface

# Try to import embedding utilities
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger("semantic_cache")

class SemanticCache(CacheInterface):
    """
    Semantic caching system that uses query embeddings to find
    similar cached items based on semantic similarity
    """
    
    def __init__(self, 
                max_size: int = 1000, 
                ttl: int = 3600,
                similarity_threshold: float = 0.85,
                embed_model: str = "all-MiniLM-L6-v2",
                embedding_function: Optional[Callable] = None):
        """
        Initialize the semantic cache
        
        Args:
            max_size: Maximum number of entries in the cache
            ttl: Default time-to-live in seconds
            similarity_threshold: Threshold for similarity matching (0-1)
            embed_model: Model to use for embeddings if sentence_transformers available
            embedding_function: Optional custom function to generate embeddings
        """
        super().__init__(max_size=max_size, ttl=ttl, cache_name="semantic")
        self.similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        
        # Set up embedding function
        self.embedding_model = None
        self.embedding_function = embedding_function
        
        if embedding_function is None and HAVE_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"Loading sentence transformer model: {embed_model}")
                self.embedding_model = SentenceTransformer(embed_model)
                self.embedding_function = self._generate_embeddings
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        if self.embedding_function is None:
            logger.warning("No embedding function available. Using fallback hashing.")
        
        # Main cache storage
        self._caches = defaultdict(dict)  # type: Dict[str, Dict[str, Dict[str, Any]]]
        self._user_caches = defaultdict(lambda: defaultdict(dict))  # type: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]
        
        # Cache metadata
        self._timestamps = defaultdict(dict)  # type: Dict[str, Dict[str, float]]
        self._user_timestamps = defaultdict(lambda: defaultdict(dict))  # type: Dict[str, Dict[str, Dict[str, float]]]
        
        # Embedding storage
        self._embeddings = defaultdict(dict)  # type: Dict[str, Dict[str, np.ndarray]]
        self._user_embeddings = defaultdict(lambda: defaultdict(dict))  # type: Dict[str, Dict[str, Dict[str, np.ndarray]]]
        
        # Metrics
        self.metrics.update({
            "semantic_hits": 0,
            "fallback_hits": 0
        })
        
        logger.info(f"Initialized semantic cache with similarity_threshold={similarity_threshold}")
    
    async def get(self, 
                key: str, 
                default: Any = None, 
                cache_type: str = "default",
                user_id: Optional[str] = None) -> Any:
        """
        Get a value from the cache
        
        Args:
            key: Cache key or query string
            default: Default value if key not found
            cache_type: Type of cache to query
            user_id: Optional user ID for user-specific caches
            
        Returns:
            Cached value or default
        """
        # Try exact match first
        exact_match = await self._get_exact_match(key, cache_type, user_id)
        if exact_match is not None:
            return exact_match
        
        # Try semantic match if embedding function available
        if self.embedding_function is not None:
            semantic_match = await self._get_semantic_match(key, cache_type, user_id)
            if semantic_match is not None:
                self._update_metric("semantic_hits")
                self._update_metric("hits")
                return semantic_match
        
        # Try fallback to hash-based fuzzy matching
        fallback_match = await self._get_fallback_match(key, cache_type, user_id)
        if fallback_match is not None:
            self._update_metric("fallback_hits")
            self._update_metric("hits")
            return fallback_match
        
        self._update_metric("misses")
        return default
    
    async def set(self, 
                key: str, 
                value: Any, 
                ttl: Optional[int] = None,
                cache_type: str = "default",
                user_id: Optional[str] = None) -> bool:
        """
        Set a value in the cache
        
        Args:
            key: Cache key or query string
            value: Value to cache
            ttl: Time-to-live in seconds, None for default
            cache_type: Type of cache to set
            user_id: Optional user ID for user-specific caches
            
        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl
        
        # Ensure we don't exceed max size
        await self._check_size(cache_type, user_id)
        
        # Generate embedding if available
        if self.embedding_function is not None:
            embedding = await asyncio.to_thread(self.embedding_function, key)
            
            # Store embedding
            if user_id:
                self._user_embeddings[user_id][cache_type][key] = embedding
            else:
                self._embeddings[cache_type][key] = embedding
        
        # Store value
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
        
        # Delete from appropriate cache
        if user_id:
            if key in self._user_caches[user_id][cache_type]:
                del self._user_caches[user_id][cache_type][key]
                del self._user_timestamps[user_id][cache_type][key]
                if key in self._user_embeddings[user_id][cache_type]:
                    del self._user_embeddings[user_id][cache_type][key]
                found = True
        else:
            if key in self._caches[cache_type]:
                del self._caches[cache_type][key]
                del self._timestamps[cache_type][key]
                if key in self._embeddings[cache_type]:
                    del self._embeddings[cache_type][key]
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
                    self._user_embeddings[user_id][cache_type].clear()
            else:
                self._user_caches[user_id].clear()
                self._user_timestamps[user_id].clear()
                self._user_embeddings[user_id].clear()
        else:
            if cache_type:
                if cache_type in self._caches:
                    self._caches[cache_type].clear()
                    self._timestamps[cache_type].clear()
                    self._embeddings[cache_type].clear()
            else:
                self._caches.clear()
                self._timestamps.clear()
                self._embeddings.clear()
        
        return True
    
    async def _get_exact_match(self, 
                            key: str, 
                            cache_type: str,
                            user_id: Optional[str] = None) -> Any:
        """Get exact match from cache"""
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
        
        return None
    
    async def _get_semantic_match(self, 
                                key: str, 
                                cache_type: str,
                                user_id: Optional[str] = None) -> Any:
        """Find semantically similar match in cache"""
        if self.embedding_function is None:
            return None
            
        # Generate embedding for query
        query_embedding = await asyncio.to_thread(self.embedding_function, key)
        
        # Find most similar entry
        max_similarity = -1.0
        best_match = None
        
        # Check in user-specific cache first
        if user_id:
            for cached_key, embedding in self._user_embeddings[user_id][cache_type].items():
                # Skip expired items
                timestamp = self._user_timestamps[user_id][cache_type].get(cached_key, 0)
                if time.time() - timestamp > self.default_ttl:
                    continue
                    
                similarity = self._calculate_similarity(query_embedding, embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = self._user_caches[user_id][cache_type].get(cached_key)
        
        # Check in global cache if no good match found yet
        if max_similarity < self.similarity_threshold:
            for cached_key, embedding in self._embeddings[cache_type].items():
                # Skip expired items
                timestamp = self._timestamps[cache_type].get(cached_key, 0)
                if time.time() - timestamp > self.default_ttl:
                    continue
                    
                similarity = self._calculate_similarity(query_embedding, embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = self._caches[cache_type].get(cached_key)
        
        # Return best match if above threshold
        if max_similarity >= self.similarity_threshold:
            return best_match
            
        return None
    
    async def _get_fallback_match(self, 
                                key: str, 
                                cache_type: str,
                                user_id: Optional[str] = None) -> Any:
        """Fallback to simpler similarity matching"""
        # Normalize and tokenize the key for matching
        key_tokens = self._tokenize_string(key)
        
        # Find best matching entry
        max_overlap = 0.0
        best_match = None
        
        # Check in user-specific cache first
        if user_id:
            for cached_key in self._user_caches[user_id][cache_type].keys():
                # Skip expired items
                timestamp = self._user_timestamps[user_id][cache_type].get(cached_key, 0)
                if time.time() - timestamp > self.default_ttl:
                    continue
                    
                cached_tokens = self._tokenize_string(cached_key)
                overlap = self._calculate_token_overlap(key_tokens, cached_tokens)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = self._user_caches[user_id][cache_type].get(cached_key)
        
        # Check in global cache if no good match found yet
        if max_overlap < 0.8:  # Lower threshold for token overlap
            for cached_key in self._caches[cache_type].keys():
                # Skip expired items
                timestamp = self._timestamps[cache_type].get(cached_key, 0)
                if time.time() - timestamp > self.default_ttl:
                    continue
                    
                cached_tokens = self._tokenize_string(cached_key)
                overlap = self._calculate_token_overlap(key_tokens, cached_tokens)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = self._caches[cache_type].get(cached_key)
        
        # Return best match if above threshold
        if max_overlap >= 0.8:  # Lower threshold for token overlap
            return best_match
            
        return None
    
    def _generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for text using sentence transformer"""
        if self.embedding_model is None:
            # Return empty embedding if model not available
            return np.zeros(384)  # Common embedding size
            
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        # Normalize embeddings for cosine similarity
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        return float(similarity)
    
    def _tokenize_string(self, text: str) -> List[str]:
        """Simple tokenization for fallback similarity"""
        # Convert to lowercase and split on whitespace and punctuation
        import re
        text = text.lower()
        return re.findall(r'\w+', text)
    
    def _calculate_token_overlap(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate Jaccard similarity for token overlap"""
        if not tokens1 or not tokens2:
            return 0.0
            
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
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
                if key in self._user_embeddings[user_id][cache_type]:
                    del self._user_embeddings[user_id][cache_type][key]
        else:
            timestamps = self._timestamps[cache_type]
            sorted_items = sorted(timestamps.items(), key=lambda x: x[1])
            
            # Remove oldest 10% or at least 1 entry
            to_remove = max(1, int(len(sorted_items) * 0.1))
            for key, _ in sorted_items[:to_remove]:
                del self._caches[cache_type][key]
                del self._timestamps[cache_type][key]
                if key in self._embeddings[cache_type]:
                    del self._embeddings[cache_type][key] 