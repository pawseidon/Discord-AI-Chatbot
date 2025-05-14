"""
Caching module for Discord AI Chatbot.

This module provides caching capabilities including context-aware caching,
semantic caching, and memory management to optimize performance.
"""

from .cache_interface import CacheInterface
from .context_aware_cache import ContextAwareCache
from .semantic_cache import SemanticCache
from .cache_integration import get_cache_handler, CacheIntegration

__all__ = [
    'CacheInterface',
    'ContextAwareCache',
    'SemanticCache',
    'get_cache_handler',
    'CacheIntegration'
] 