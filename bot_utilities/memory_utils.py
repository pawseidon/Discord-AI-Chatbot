"""
Memory Utilities Module

This module provides utility functions for memory management.
"""

import os
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Union

async def get_user_preferences(user_id: str) -> Dict[str, Any]:
    """
    Get a user's preferences from the memory service.
    
    Args:
        user_id: The user ID to retrieve preferences for
        
    Returns:
        Dict[str, Any]: The user's preferences
    """
    try:
        # Import here to avoid circular imports
        from bot_utilities.services.memory_service import memory_service
        
        # Get user preferences from memory service
        preferences = await memory_service.get_preferences(user_id)
        
        # Return preferences or empty dict if none found
        return preferences or {}
    except Exception as e:
        print(f"Error getting user preferences: {str(e)}")
        return {}

async def store_user_preferences(user_id: str, preferences: Dict[str, Any]) -> bool:
    """
    Store a user's preferences in the memory service.
    
    Args:
        user_id: The user ID to store preferences for
        preferences: The preferences to store
        
    Returns:
        bool: True if successfully stored, False otherwise
    """
    try:
        # Import here to avoid circular imports
        from bot_utilities.services.memory_service import memory_service
        
        # Store user preferences in memory service
        success = await memory_service.set_preferences(user_id, preferences)
        
        return success
    except Exception as e:
        print(f"Error storing user preferences: {str(e)}")
        return False

async def get_memory(user_id: str, max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Get a user's conversation memory from the memory service.
    
    Args:
        user_id: The user ID to retrieve memory for
        max_items: Maximum number of items to retrieve
        
    Returns:
        List[Dict[str, Any]]: The user's conversation memory
    """
    try:
        # Import here to avoid circular imports
        from bot_utilities.services.memory_service import memory_service
        
        # Get user memory from memory service
        memory = await memory_service.get_memory(user_id, max_items=max_items)
        
        # Return memory or empty list if none found
        return memory or []
    except Exception as e:
        print(f"Error getting user memory: {str(e)}")
        return []

async def clear_user_memory(user_id: str) -> bool:
    """
    Clear a user's conversation memory from the memory service.
    
    Args:
        user_id: The user ID to clear memory for
        
    Returns:
        bool: True if successfully cleared, False otherwise
    """
    try:
        # Import here to avoid circular imports
        from bot_utilities.services.memory_service import memory_service
        
        # Clear user memory from memory service
        success = await memory_service.clear_memory(user_id)
        
        return success
    except Exception as e:
        print(f"Error clearing user memory: {str(e)}")
        return False

async def get_channel_history(user_id: str, channel_id: str, max_items: int = 10) -> List[Dict[str, Any]]:
    """
    Get conversation history for a specific channel from the memory service.
    
    Args:
        user_id: The user ID
        channel_id: The channel ID
        max_items: Maximum number of items to retrieve
        
    Returns:
        List[Dict[str, Any]]: The channel's conversation history
    """
    try:
        # Import here to avoid circular imports
        from bot_utilities.services.memory_service import memory_service
        
        # Get channel history from memory service
        history = await memory_service.get_channel_history(channel_id, max_items=max_items)
        
        # Return history or empty list if none found
        return history or []
    except Exception as e:
        print(f"Error getting channel history: {str(e)}")
        return [] 