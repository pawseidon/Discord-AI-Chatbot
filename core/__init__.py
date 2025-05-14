"""
Core module for Discord AI Chatbot.

This module provides essential core functionality including
AI provider integration, Discord integration, and configuration.
"""

from .ai_provider import get_ai_provider
from .config_loader import get_config_loader, get_config, set_config, load_current_language, load_instructions
from .ai_utils import get_ai_provider, get_bot_names_and_triggers, update_timestamp_cache

__all__ = [
    'get_ai_provider',
    'get_config_loader',
    'get_config',
    'set_config',
    'load_current_language', 
    'load_instructions',
    'get_bot_names_and_triggers',
    'update_timestamp_cache'
]
