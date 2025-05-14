import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
import re
import logging
import asyncio
import os
import pickle
from collections import Counter
import redis

# Set up logging
logger = logging.getLogger("response_cache")
logger.setLevel(logging.INFO)

class ResponseCache:
    """
    Caches responses with semantic fingerprinting to efficiently retrieve
    responses for semantically similar queries.
    
    Features:
    - Semantic fingerprinting for approximate matching
    - Time-based invalidation for freshness
    - Channel/context-specific caching
    - Success/quality rating tracking
    - Distributed cache support
    - Vector similarity search capability
    """
    def __init__(self, 
                max_cache_size: int = 1000, 
                default_ttl: int = 3600,
                storage_dir: str = "bot_data/cache",
                db_backend: str = "file",
                similarity_threshold: float = 0.7,
                auto_persist: bool = True,
                vector_search_enabled: bool = False,
                connection_string: str = None):
        """
        Initialize the cache
        
        Args:
            max_cache_size: Maximum number of entries to store
            default_ttl: Default time-to-live in seconds
            storage_dir: Directory to persist cache to
            db_backend: Storage backend ('file', 'memory', 'redis')
            similarity_threshold: Default similarity threshold
            auto_persist: Automatically persist cache to storage
            vector_search_enabled: Use vector embeddings for similarity search
            connection_string: Redis connection string
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.semantic_index: Dict[str, List[str]] = {}
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        self.access_counts: Dict[str, int] = {}
        self.storage_dir = storage_dir
        self.db_backend = db_backend
        self.similarity_threshold = similarity_threshold
        self.auto_persist = auto_persist
        self.vector_search_enabled = vector_search_enabled
        self.vector_embeddings = {}
        self.periodic_task = None
        
        # Stat tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "exact_hits": 0,
            "similar_hits": 0,
            "evictions": 0,
            "invalidations": 0
        }
        
        # Initialize storage
        if db_backend == "file" and not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)
            
        # Load existing cache if available
        self._load_cache()
        
        # Don't create tasks here, wait for start_tasks to be called
        
        # Try to connect to Redis if connection string provided
        if connection_string:
            try:
                self.redis = redis.from_url(connection_string)
                self.redis.ping()  # Test connection
                self.enabled = True
                self.in_memory_mode = False
                logger.info("Response cache initialized with Redis")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}")
                logger.info("Using in-memory cache instead")
    
    async def _setup_periodic_tasks(self):
        """Set up periodic cleanup and maintenance tasks"""
        while True:
            # Run cache cleanup every hour
            await asyncio.sleep(3600)
            await self.cleanup_expired()
            
            # Persist cache every 5 minutes
            await asyncio.sleep(300)
            if self.auto_persist:
                self._persist_cache()
    
    def _load_cache(self):
        """Load cache from storage backend"""
        if self.db_backend == "memory":
            return  # No need to load anything
            
        if self.db_backend == "file":
            try:
                cache_path = os.path.join(self.storage_dir, "cache.pkl")
                semantic_path = os.path.join(self.storage_dir, "semantic_index.pkl")
                stats_path = os.path.join(self.storage_dir, "cache_stats.pkl")
                access_path = os.path.join(self.storage_dir, "access_counts.pkl")
                
                if os.path.exists(cache_path):
                    with open(cache_path, "rb") as f:
                        self.cache = pickle.load(f)
                        
                if os.path.exists(semantic_path):
                    with open(semantic_path, "rb") as f:
                        self.semantic_index = pickle.load(f)
                
                if os.path.exists(stats_path):
                    with open(stats_path, "rb") as f:
                        self.stats = pickle.load(f)
                        
                if os.path.exists(access_path):
                    with open(access_path, "rb") as f:
                        self.access_counts = pickle.load(f)
                        
                logger.info(f"Loaded {len(self.cache)} cache entries")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        
        # TODO: Add support for Redis or other distributed cache backends
    
    def _persist_cache(self):
        """Persist cache to storage backend"""
        if not self.auto_persist:
            return
            
        if self.db_backend == "memory":
            return  # No need to persist anything
            
        if self.db_backend == "file":
            try:
                cache_path = os.path.join(self.storage_dir, "cache.pkl")
                semantic_path = os.path.join(self.storage_dir, "semantic_index.pkl")
                stats_path = os.path.join(self.storage_dir, "cache_stats.pkl")
                access_path = os.path.join(self.storage_dir, "access_counts.pkl")
                
                with open(cache_path, "wb") as f:
                    pickle.dump(self.cache, f)
                    
                with open(semantic_path, "wb") as f:
                    pickle.dump(self.semantic_index, f)
                    
                with open(stats_path, "wb") as f:
                    pickle.dump(self.stats, f)
                    
                with open(access_path, "wb") as f:
                    pickle.dump(self.access_counts, f)
            except Exception as e:
                logger.error(f"Error persisting cache: {e}")
        
        # TODO: Add support for Redis or other distributed cache backends
    
    async def generate_fingerprint(self, query: str) -> str:
        """
        Generate a semantic fingerprint for a query
        
        Args:
            query: The query to fingerprint
            
        Returns:
            str: The fingerprint
        """
        # Normalize the query
        normalized = query.lower()
        
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove stop words (simple implementation)
        stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "from", "by", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below",
            "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
            "don't", "should", "now", "i", "me", "my", "myself", "we", "our",
            "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself", "she", "her", "hers", "herself", "it",
            "its", "itself", "they", "them", "their", "theirs", "themselves",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "having", "do", "does", "did", "doing", "would",
            "should", "could", "ought", "i'm", "you're", "he's", "she's",
            "it's", "we're", "they're", "i've", "you've", "we've", "they've",
            "i'd", "you'd", "he'd", "she'd", "we'd", "they'd", "i'll", "you'll",
            "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't",
            "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't",
            "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't",
            "cannot", "couldn't", "mustn't", "let's", "that's", "who's",
            "what's", "here's", "there's", "when's", "where's", "why's", "how's"
        }
        
        words = [word for word in normalized.split() if word not in stop_words]
        
        # Get word frequency for more accurate fingerprinting
        word_counts = Counter(words)
        
        # Extract significant n-grams (2-3 words)
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
        
        # Combine words and significant n-grams with their frequencies
        signature_elements = []
        
        # Add frequent words with their counts
        for word, count in word_counts.most_common(10):  # Limit to top 10 words
            signature_elements.append(f"{word}:{count}")
            
        # Add significant n-grams
        signature_elements.extend(bigrams[:5])  # Limit to top 5 bigrams
        signature_elements.extend(trigrams[:3])  # Limit to top 3 trigrams
        
        # Sort to ensure consistent ordering
        signature_elements.sort()
        
        # Join into a string and hash
        signature = "_".join(signature_elements)
        
        # Create an MD5 hash for compact storage
        return hashlib.md5(signature.encode('utf-8')).hexdigest()
    
    async def get_vector_embedding(self, query: str, ai_provider=None) -> List[float]:
        """
        Get a vector embedding for a query to use for similarity search
        
        Args:
            query: The query to embed
            ai_provider: Optional AI provider for generating embeddings
            
        Returns:
            List[float]: Vector embedding
        """
        if not self.vector_search_enabled:
            return []
            
        # If we have an AI provider that supports embeddings, use it
        if ai_provider and hasattr(ai_provider, 'get_embedding'):
            try:
                return await ai_provider.get_embedding(query)
            except Exception as e:
                logger.error(f"Error getting embedding from AI provider: {e}")
                
        # Otherwise, return empty vector (will fall back to fingerprint matching)
        return []
            
    async def calculate_vector_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate the cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            float: Similarity score (0-1)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
            
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        
        # Calculate cosine similarity
        if mag1 * mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)
    
    async def get_similar_fingerprints(self, fingerprint: str, threshold: float = 0.7) -> List[str]:
        """
        Get similar fingerprints from the semantic index
        
        Args:
            fingerprint: The query fingerprint
            threshold: Similarity threshold (0-1)
            
        Returns:
            List[str]: List of similar fingerprints
        """
        # For a real implementation, this would use advanced techniques
        # like MinHash or other similarity measures
        
        # For this implementation, we'll use a simple prefix matching approach
        # This is a simplified approximation of semantic similarity
        prefix_length = int(len(fingerprint) * threshold)
        prefix = fingerprint[:prefix_length]
        
        similar_fps = []
        for fp in self.cache.keys():
            # Simple prefix comparison
            if fp.startswith(prefix):
                similar_fps.append(fp)
                
        return similar_fps
    
    async def store(self, 
                   query: str, 
                   response: str, 
                   metadata: Dict[str, Any] = None,
                   channel_id: str = None,
                   guild_id: str = None,
                   ttl: int = None,
                   ai_provider=None) -> str:
        """
        Store a response in the cache
        
        Args:
            query: The query that generated the response
            response: The response to cache
            metadata: Additional metadata about the response
            channel_id: Optional channel ID for context-specific caching
            guild_id: Optional guild ID for context-specific caching
            ttl: Time-to-live in seconds (None for default)
            ai_provider: Optional AI provider for vector embeddings
            
        Returns:
            str: The cache key
        """
        # Generate fingerprint
        fingerprint = await self.generate_fingerprint(query)
        
        # Generate vector embedding if enabled
        embedding = []
        if self.vector_search_enabled and ai_provider:
            embedding = await self.get_vector_embedding(query, ai_provider)
        
        # Create cache key (combine fingerprint with context if provided)
        context_parts = []
        if channel_id:
            context_parts.append(f"channel:{channel_id}")
        if guild_id:
            context_parts.append(f"guild:{guild_id}")
            
        context_suffix = "_".join(context_parts) if context_parts else ""
        cache_key = f"{fingerprint}:{context_suffix}" if context_suffix else fingerprint
        
        # Store in cache
        self.cache[cache_key] = {
            "query": query,
            "response": response,
            "metadata": metadata or {},
            "created_at": time.time(),
            "expires_at": time.time() + (ttl or self.default_ttl),
            "access_count": 0,
            "channel_id": channel_id,
            "guild_id": guild_id,
            "fingerprint": fingerprint
        }
        
        # Store vector embedding if available
        if embedding:
            self.vector_embeddings[cache_key] = embedding
        
        # Update access count
        self.access_counts[cache_key] = 0
        
        # Check if cache is too large and evict if needed
        if len(self.cache) > self.max_cache_size:
            await self.evict_entries()
            
        # Persist cache
        if self.auto_persist:
            self._persist_cache()
            
        return cache_key
    
    async def retrieve(self, 
                      query: str, 
                      channel_id: str = None,
                      guild_id: str = None,
                      threshold: float = None,
                      ai_provider=None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a response from the cache
        
        Args:
            query: The query to look up
            channel_id: Optional channel ID for context-specific retrieval
            guild_id: Optional guild ID for context-specific retrieval
            threshold: Similarity threshold for retrieval (None for default)
            ai_provider: Optional AI provider for vector embeddings
            
        Returns:
            Optional[Dict]: The cached entry if found, None otherwise
        """
        # Update statistics
        self.stats["total_requests"] += 1
        
        # Set threshold to default if not provided
        if threshold is None:
            threshold = self.similarity_threshold
        
        # Generate fingerprint
        fingerprint = await self.generate_fingerprint(query)
        
        # Create context suffix for exact matching
        context_parts = []
        if channel_id:
            context_parts.append(f"channel:{channel_id}")
        if guild_id:
            context_parts.append(f"guild:{guild_id}")
            
        context_suffix = "_".join(context_parts) if context_parts else ""
        cache_key = f"{fingerprint}:{context_suffix}" if context_suffix else fingerprint
        
        # Try exact match first
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            # Check if entry is expired
            if entry["expires_at"] > time.time():
                # Update access count
                entry["access_count"] += 1
                self.access_counts[cache_key] += 1
                
                # Update statistics
                self.stats["cache_hits"] += 1
                self.stats["exact_hits"] += 1
                
                return entry
            else:
                # Remove expired entry
                del self.cache[cache_key]
                if cache_key in self.access_counts:
                    del self.access_counts[cache_key]
        
        # For vector search, get embedding and find most similar
        if self.vector_search_enabled and ai_provider:
            query_embedding = await self.get_vector_embedding(query, ai_provider)
            if query_embedding:
                best_match = None
                best_similarity = 0.0
                
                # Find most similar vector
                for key, embedding in self.vector_embeddings.items():
                    if key in self.cache and self.cache[key]["expires_at"] > time.time():
                        similarity = await self.calculate_vector_similarity(query_embedding, embedding)
                        if similarity >= threshold and similarity > best_similarity:
                            best_similarity = similarity
                            best_match = key
                
                if best_match:
                    entry = self.cache[best_match]
                    entry["access_count"] += 1
                    self.access_counts[best_match] += 1
                    entry["similarity"] = best_similarity
                    
                    # Update statistics
                    self.stats["cache_hits"] += 1
                    self.stats["similar_hits"] += 1
                    
                    return entry
        
        # If no exact match or vector match, try similar fingerprints
        similar_fps = await self.get_similar_fingerprints(fingerprint, threshold)
        
        # Filter by context if provided
        if context_suffix:
            similar_fps = [fp for fp in similar_fps if fp.endswith(context_suffix)]
        
        # Find best match
        best_match = None
        for fp in similar_fps:
            entry = self.cache.get(fp)
            if entry and entry["expires_at"] > time.time():
                # Basic similarity check between queries
                query_similarity = self.calculate_similarity(query, entry["query"])
                if query_similarity >= threshold:
                    if best_match is None or query_similarity > best_match[1]:
                        best_match = (fp, query_similarity)
        
        if best_match:
            fp, similarity = best_match
            entry = self.cache[fp]
            
            # Update access count
            entry["access_count"] += 1
            self.access_counts[fp] += 1
            
            # Add similarity score to the entry
            entry["similarity"] = similarity
            
            # Update statistics
            self.stats["cache_hits"] += 1
            self.stats["similar_hits"] += 1
            
            return entry
        
        # No match found
        self.stats["cache_misses"] += 1
        return None
    
    def calculate_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate simple similarity between two queries
        
        Args:
            query1: First query
            query2: Second query
            
        Returns:
            float: Similarity score (0-1)
        """
        # Normalize queries
        q1 = query1.lower()
        q2 = query2.lower()
        
        # Remove punctuation
        q1 = re.sub(r'[^\w\s]', '', q1)
        q2 = re.sub(r'[^\w\s]', '', q2)
        
        # Split into words
        words1 = set(q1.split())
        words2 = set(q2.split())
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union
    
    async def invalidate(self, 
                        query: str = None, 
                        channel_id: str = None,
                        guild_id: str = None,
                        pattern: str = None) -> int:
        """
        Invalidate cache entries based on criteria
        
        Args:
            query: Query to invalidate (exact or similar)
            channel_id: Channel ID to invalidate
            guild_id: Guild ID to invalidate
            pattern: Regex pattern to match against queries
            
        Returns:
            int: Number of entries invalidated
        """
        keys_to_remove = []
        
        # If a specific query is provided, invalidate that query
        if query:
            fingerprint = await self.generate_fingerprint(query)
            
            # Find all keys with this fingerprint
            for key in self.cache.keys():
                if key.startswith(fingerprint):
                    # If channel_id is specified, only invalidate matching channel entries
                    if channel_id and f"channel:{channel_id}" not in key:
                        continue
                        
                    # If guild_id is specified, only invalidate matching guild entries
                    if guild_id and f"guild:{guild_id}" not in key:
                        continue
                        
                    keys_to_remove.append(key)
        
        # If only channel_id is provided, invalidate all entries for that channel
        elif channel_id:
            channel_suffix = f"channel:{channel_id}"
            for key in self.cache.keys():
                if channel_suffix in key:
                    keys_to_remove.append(key)
        
        # If only guild_id is provided, invalidate all entries for that guild
        elif guild_id:
            guild_suffix = f"guild:{guild_id}"
            for key in self.cache.keys():
                if guild_suffix in key:
                    keys_to_remove.append(key)
        
        # If pattern is provided, invalidate entries matching the pattern
        elif pattern:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for key, entry in self.cache.items():
                    if regex.search(entry["query"]):
                        keys_to_remove.append(key)
            except Exception as e:
                logger.error(f"Error compiling regex pattern: {e}")
        
        # Remove the keys
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_counts:
                del self.access_counts[key]
            if key in self.vector_embeddings:
                del self.vector_embeddings[key]
        
        # Update stats
        self.stats["invalidations"] += len(keys_to_remove)
        
        # Persist cache
        if self.auto_persist and keys_to_remove:
            self._persist_cache()
            
        return len(keys_to_remove)
    
    async def update_rating(self, cache_key: str, rating: float) -> bool:
        """
        Update the quality rating for a cached response
        
        Args:
            cache_key: The cache key to update
            rating: Quality rating (0-1)
            
        Returns:
            bool: True if updated, False otherwise
        """
        if cache_key not in self.cache:
            return False
            
        entry = self.cache[cache_key]
        
        # Update metadata with rating
        if "ratings" not in entry["metadata"]:
            entry["metadata"]["ratings"] = []
            
        # Add this rating
        entry["metadata"]["ratings"].append(rating)
        
        # Calculate average rating
        ratings = entry["metadata"]["ratings"]
        entry["metadata"]["avg_rating"] = sum(ratings) / len(ratings)
        
        # Persist cache
        if self.auto_persist:
            self._persist_cache()
            
        return True
    
    async def evict_entries(self, count: int = None) -> int:
        """
        Evict entries to keep cache size within limits
        
        Args:
            count: Number of entries to evict (None for auto-calculation)
            
        Returns:
            int: Number of entries evicted
        """
        # If count is not specified, calculate how many to evict
        if count is None:
            count = max(1, int(self.max_cache_size * 0.1))  # Evict 10% by default
        
        # Get a list of all entries with their access counts
        entries = [(key, self.access_counts.get(key, 0)) for key in self.cache.keys()]
        
        # Sort by access count (ascending) to evict least used entries first
        entries.sort(key=lambda x: x[1])
        
        # Evict the least accessed entries
        to_evict = entries[:count]
        
        # Remove the entries
        for key, _ in to_evict:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_counts:
                del self.access_counts[key]
            if key in self.vector_embeddings:
                del self.vector_embeddings[key]
        
        # Update stats
        self.stats["evictions"] += len(to_evict)
        
        # Persist cache
        if self.auto_persist and to_evict:
            self._persist_cache()
            
        return len(to_evict)
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries
        
        Returns:
            int: Number of entries removed
        """
        current_time = time.time()
        keys_to_remove = []
        
        # Find expired entries
        for key, entry in self.cache.items():
            if entry["expires_at"] <= current_time:
                keys_to_remove.append(key)
        
        # Remove the entries
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_counts:
                del self.access_counts[key]
            if key in self.vector_embeddings:
                del self.vector_embeddings[key]
        
        # Persist cache
        if self.auto_persist and keys_to_remove:
            self._persist_cache()
            
        return len(keys_to_remove)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict: Cache statistics
        """
        # Calculate hit rate
        hit_rate = 0
        if self.stats["total_requests"] > 0:
            hit_rate = self.stats["cache_hits"] / self.stats["total_requests"]
            
        # Calculate memory usage
        memory_usage = self.estimate_memory_usage()
        
        # Return current stats plus calculated ones
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "hit_rate": hit_rate,
            "memory_usage_bytes": memory_usage,
            "memory_usage_mb": memory_usage / (1024 * 1024)
        }
    
    def estimate_memory_usage(self) -> int:
        """
        Estimate the memory usage of the cache
        
        Returns:
            int: Estimated memory usage in bytes
        """
        total_size = 0
        
        # Estimate cache entries size
        for key, entry in self.cache.items():
            # Key size
            total_size += len(key) * 2  # Unicode characters
            
            # Query and response size
            total_size += len(entry["query"]) * 2
            total_size += len(entry["response"]) * 2
            
            # Metadata size (rough estimate)
            metadata_str = json.dumps(entry["metadata"])
            total_size += len(metadata_str)
            
            # Other fields (constants)
            total_size += 100  # Timestamps, counters, etc.
        
        # Vector embeddings size
        for key, embedding in self.vector_embeddings.items():
            # Each float is 8 bytes
            total_size += len(embedding) * 8
        
        # Access counts and other dictionaries
        total_size += len(self.access_counts) * 16  # Key and counter
        
        return total_size
    
    async def clear(self) -> int:
        """
        Clear the entire cache
        
        Returns:
            int: Number of entries cleared
        """
        count = len(self.cache)
        
        self.cache = {}
        self.access_counts = {}
        self.vector_embeddings = {}
        
        # Reset stats
        self.stats["evictions"] += count
        
        # Persist cache
        if self.auto_persist:
            self._persist_cache()
            
        return count

    async def start_tasks(self):
        """Start periodic tasks - must be called in an async context when event loop is running"""
        if self.periodic_task is None:
            self.periodic_task = asyncio.create_task(self._setup_periodic_tasks())
            logger.info("Started response cache periodic tasks")

def create_response_cache(
        max_cache_size: int = 1000, 
        default_ttl: int = 3600,
        storage_dir: str = "bot_data/cache",
        db_backend: str = "file",
        vector_search_enabled: bool = False) -> ResponseCache:
    """Create and initialize a ResponseCache instance"""
    return ResponseCache(
        max_cache_size=max_cache_size,
        default_ttl=default_ttl,
        storage_dir=storage_dir,
        db_backend=db_backend,
        vector_search_enabled=vector_search_enabled
    ) 