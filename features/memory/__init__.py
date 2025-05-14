"""
Memory module for Discord AI Chatbot.

This module provides memory management, conversation history, and retention
systems for the bot.
"""

from .memory_utils import (
    MemoryManager,
    create_memory_manager
)

__all__ = [
    'MemoryManager',
    'create_memory_manager'
]
