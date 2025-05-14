"""
Reasoning Detection Utilities

This module provides utilities for detecting different reasoning patterns
and determining the appropriate reasoning approaches for user queries.
"""

import logging
import re
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('reasoning_utils')

# Ensure cache directory exists
CACHE_DIR = os.path.join("bot_data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class ReasoningCache:
    """Cache for reasoning type detection to avoid repeated computation"""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize the reasoning cache
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.cache = {}
        self.max_size = max_size
        
        # Load cache from disk if available
        self.load_cache()
    
    def get(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Get a cached reasoning type for a query
        
        Args:
            query: The query to check
            
        Returns:
            Tuple of (reasoning_type, confidence) or None if not found
        """
        # Generate a hash key for the query
        key = self._hash_query(query)
        
        return self.cache.get(key)
    
    def set(self, query: str, reasoning_type: str, confidence: float) -> None:
        """
        Cache a reasoning type for a query
        
        Args:
            query: The query to cache
            reasoning_type: The detected reasoning type
            confidence: Confidence level (0-1)
        """
        # Generate a hash key for the query
        key = self._hash_query(query)
        
        # Add to cache
        self.cache[key] = (reasoning_type, confidence)
        
        # Manage cache size
        if len(self.cache) > self.max_size:
            # Remove a random key (simple approach)
            # A more sophisticated approach would use LRU
            self.cache.pop(next(iter(self.cache)))
            
        # Save cache to disk periodically
        # In a production environment, this would use a more efficient approach
        # like saving only on shutdown or after a certain number of updates
        self.save_cache()
    
    def _hash_query(self, query: str) -> str:
        """Generate a hash key for a query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def save_cache(self) -> None:
        """Save cache to disk"""
        try:
            cache_path = os.path.join(CACHE_DIR, "reasoning_cache.json")
            with open(cache_path, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving reasoning cache: {str(e)}")
    
    def load_cache(self) -> None:
        """Load cache from disk"""
        try:
            cache_path = os.path.join(CACHE_DIR, "reasoning_cache.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    self.cache = json.load(f)
        except Exception as e:
            logger.error(f"Error loading reasoning cache: {str(e)}")
            self.cache = {}

class ReasoningDetector:
    """Detects appropriate reasoning types for user queries"""
    
    def __init__(self, enable_cache: bool = True):
        """
        Initialize the reasoning detector
        
        Args:
            enable_cache: Whether to use caching
        """
        self.enable_cache = enable_cache
        self.cache = ReasoningCache() if enable_cache else None
        
        # Initialize reasoning pattern matchers
        self._init_patterns()
    
    def _init_patterns(self) -> None:
        """Initialize the reasoning pattern matchers"""
        # Patterns for each reasoning type
        self.patterns = {
            "rag": [
                r"(what|when|where|who|how|why).+(is|are|was|were|will|do|does)",
                r"(find|search|research|information|tell me|learn|article|data|facts|latest|recent)",
                r"(history of|meaning of|definition of|examples of|information about)"
            ],
            "sequential": [
                r"(step by step|steps to|process for|procedure|sequence|systematic|methodical)",
                r"(analyze|break down|step-by-step|thorough analysis|detailed explanation)",
                r"(first|then|next|finally|afterwards|subsequently|following this)"
            ],
            "verification": [
                r"(verify|fact.?check|confirm|is.+true|validate|check.+accuracy|reliable)",
                r"(correct|incorrect|accurate|inaccurate|trustworthy|credible|evidence)",
                r"(prove|disprove|debunk|refute|authenticate|cross.?check|sources)"
            ],
            "creative": [
                r"(create|generate|write|story|poem|fiction|imagine|creative)",
                r"(novel|innovative|unique|original|artistic|fantasy|design)",
                r"(invent|brainstorm|idea|concept|vision|inspiration)"
            ],
            "calculation": [
                r"(calculate|compute|solve|equation|formula|math|arithmetic)",
                r"(number|sum|difference|product|quotient|value|result)",
                r"(\d+[\+\-\*\/\^]\d+|\d+\s*[\+\-\*\/\^]\s*\d+)"
            ],
            "graph": [
                r"(graph|chart|diagram|visualization|network|connections|relationship)",
                r"(nodes|edges|links|connected|relation between|structure|map)",
                r"(conceptual|mind map|flow chart|hierarchy|organize)"
            ],
            "multi_agent": [
                r"(multiple perspectives|different viewpoints|various angles|debate)",
                r"(team|collaborate|different experts|panel|diverse views)",
                r"(pros and cons|advantages and disadvantages|collaborative)"
            ],
            "cot": [
                r"(logical|reasoning|think through|chain of thought|deduce|infer)",
                r"(conclusion|premise|argument|logic|deduction|derive)",
                r"(follow the logic|rationale|therefore|hence|thus)"
            ],
            "step_back": [
                r"(big picture|broader context|overall view|step back|zoom out)",
                r"(high level|overview|summary|bird's eye view|general idea)",
                r"(conceptual framework|abstraction|meta|higher perspective)"
            ],
            "conversational": [
                r"(chat|talk|converse|casual|informal|friendly|conversation)",
                r"(hello|hi|hey|morning|afternoon|evening)",
                r"(how are you|nice to meet|good to see)"
            ]
        }
        
        # Compile the regex patterns for efficiency
        self.compiled_patterns = {}
        for reasoning_type, patterns in self.patterns.items():
            self.compiled_patterns[reasoning_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def detect_reasoning_type(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        conversation_id: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Detect the most appropriate reasoning type for a query
        
        Args:
            query: The user's query
            conversation_history: Optional conversation history
            conversation_id: Optional conversation ID for context
            
        Returns:
            Tuple of (reasoning_type, confidence)
        """
        # Check cache first if enabled
        if self.enable_cache and self.cache:
            cached_result = self.cache.get(query)
            if cached_result:
                return cached_result
        
        # Initialize scores for each reasoning type
        scores = {reasoning_type: 0.0 for reasoning_type in self.patterns.keys()}
        
        # Normalize the query for better matching
        normalized_query = query.lower().strip()
        
        # Check each reasoning type's patterns against the query
        for reasoning_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(normalized_query):
                    scores[reasoning_type] += 1.0
        
        # Check for math expressions which strongly indicate calculation reasoning
        if re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', normalized_query):
            scores["calculation"] += 3.0
        
        # Check for question formats which indicate RAG
        if re.match(r'^(what|when|where|who|how|why).+\?$', normalized_query):
            scores["rag"] += 2.0
        
        # Consider conversation history if provided
        if conversation_history:
            # Look at the last few messages for context
            recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            
            # Check for conversation continuation patterns
            for msg in recent_history:
                if msg.get("role") == "assistant" and msg.get("reasoning_type"):
                    # Slight boost to continue with the same reasoning type
                    prev_reasoning = msg.get("reasoning_type")
                    if prev_reasoning in scores:
                        scores[prev_reasoning] += 0.5
        
        # Find the highest scoring reasoning type
        max_score = max(scores.values()) if scores else 0
        
        # If no clear signal, default to conversational
        if max_score == 0:
            selected_type = "conversational"
            confidence = 1.0
        else:
            # Get all types with the max score
            top_types = [rt for rt, score in scores.items() if score == max_score]
            
            # If there's a tie, use a preference order
            if len(top_types) > 1:
                # Preference order (higher index = higher preference)
                preference_order = [
                    "conversational",  # Lowest preference 
                    "step_back",
                    "cot",
                    "multi_agent",
                    "graph",
                    "creative",
                    "verification",
                    "sequential",
                    "rag",
                    "calculation"  # Highest preference
                ]
                
                # Find the type with the highest preference
                selected_type = max(top_types, key=lambda t: preference_order.index(t) if t in preference_order else -1)
            else:
                selected_type = top_types[0]
            
            # Calculate confidence based on the score
            confidence = min(max_score / 5.0, 1.0)  # Scale to 0-1 range
        
        # Cache the result if caching is enabled
        if self.enable_cache and self.cache:
            self.cache.set(query, selected_type, confidence)
        
        return selected_type, confidence
    
    def detect_multiple_reasoning_types(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Tuple[str, float]]:
        """
        Detect multiple applicable reasoning types for a query
        
        Args:
            query: The user's query
            conversation_history: Optional conversation history
            
        Returns:
            List of (reasoning_type, confidence) tuples sorted by confidence
        """
        # Initialize scores for each reasoning type
        scores = {reasoning_type: 0.0 for reasoning_type in self.patterns.keys()}
        
        # Normalize the query for better matching
        normalized_query = query.lower().strip()
        
        # Check each reasoning type's patterns against the query
        for reasoning_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(normalized_query):
                    scores[reasoning_type] += 1.0
        
        # Add special case detection
        if re.search(r'\d+\s*[\+\-\*\/\^]\s*\d+', normalized_query):
            scores["calculation"] += 3.0
            
        if re.match(r'^(what|when|where|who|how|why).+\?$', normalized_query):
            scores["rag"] += 2.0
            
        # Consider conversation history if provided
        if conversation_history:
            # Look at the last few messages for context
            recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            
            # Check for conversation continuation patterns
            for msg in recent_history:
                if msg.get("role") == "assistant" and msg.get("reasoning_type"):
                    # Slight boost to continue with the same reasoning type
                    prev_reasoning = msg.get("reasoning_type")
                    if prev_reasoning in scores:
                        scores[prev_reasoning] += 0.5
        
        # Convert scores to (reasoning_type, confidence) tuples
        # Scale the scores to 0-1 confidence values
        max_possible_score = 5.0  # Approximate maximum possible score
        type_confidences = [
            (rt, min(score / max_possible_score, 1.0))
            for rt, score in scores.items() if score > 0
        ]
        
        # Sort by confidence (descending)
        type_confidences.sort(key=lambda x: x[1], reverse=True)
        
        # If no types detected, default to conversational
        if not type_confidences:
            type_confidences = [("conversational", 1.0)]
        
        return type_confidences
    
    def should_combine_reasoning(
        self, 
                         query: str, 
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Determine if a query would benefit from combining multiple reasoning approaches
        
        Args:
            query: The user's query
            conversation_history: Optional conversation history
            
        Returns:
            Boolean indicating if combined reasoning should be used
        """
        # Get the top reasoning types with confidences
        type_confidences = self.detect_multiple_reasoning_types(query, conversation_history)
        
        # Need at least two reasoning types to combine
        if len(type_confidences) < 2:
            return False
        
        # Get the top two types
        top_type, top_conf = type_confidences[0]
        second_type, second_conf = type_confidences[1]
        
        # Set of good combinations
        good_combinations = {
            frozenset(["rag", "verification"]),
            frozenset(["sequential", "rag"]),
            frozenset(["graph", "rag"]),
            frozenset(["sequential", "calculation"]),
            frozenset(["multi_agent", "verification"]),
            frozenset(["creative", "sequential"]),
            frozenset(["cot", "verification"]),
            frozenset(["step_back", "graph"])
        }
        
        # Check if our top two types form a good combination
        current_combo = frozenset([top_type, second_type])
        if current_combo in good_combinations:
            # Check if both types have reasonable confidence
            if second_conf >= 0.3:  # Require reasonable confidence in the second type
                return True
        
        # Look for specific patterns that indicate combined reasoning
        normalized_query = query.lower().strip()
        
        # Pattern: research + verification
        if (re.search(r'(research|find|search).+(verify|fact.?check|confirm)', normalized_query) or
            re.search(r'(verify|fact.?check|confirm).+(research|find|search)', normalized_query)):
            return True
            
        # Pattern: step-by-step + research
        if (re.search(r'(step by step|analyze|explain).+(research|information|find)', normalized_query) or
            re.search(r'(research|information|find).+(step by step|analyze|explain)', normalized_query)):
            return True
            
        # Complex query detection (longer queries often benefit from combined reasoning)
        words = normalized_query.split()
        if len(words) > 25:  # Long, complex query
            # If the top two confidence scores are both high enough
            if top_conf > 0.5 and second_conf > 0.3:
                return True
                
        # Default to single reasoning approach
        return False 