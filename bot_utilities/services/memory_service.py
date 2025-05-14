"""
Memory Service

This module provides a centralized service for managing user conversation history,
preferences, and other memory-related functionality.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional
import discord

# Import directly from original module is causing circular imports, so we'll use
# direct file access for user preferences. The original module now uses this service.
# from ..memory_utils import UserPreferences, process_conversation_history

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('memory_service')

# Global storage for message history
message_history = {}

class MemoryService:
    """Service for managing user conversation history and preferences"""
    
    def __init__(self):
        """Initialize the memory service"""
        self.user_data_dir = os.path.join("bot_data", "user_tracking")
        os.makedirs(self.user_data_dir, exist_ok=True)
        self.user_data_file = os.path.join(self.user_data_dir, "user_data.json")
        
        # Set up paths for storing user preferences
        self.USER_PREFS_DIR = os.path.join("bot_data", "user_preferences")
        os.makedirs(self.USER_PREFS_DIR, exist_ok=True)
        
        # Set up directory for memory storage
        self.MEMORY_DIR = os.path.join("bot_data", "memory_summaries")
        os.makedirs(self.MEMORY_DIR, exist_ok=True)
    
    async def load_from_disk(self):
        """
        Load memory data from disk storage
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load any conversation summaries
            if os.path.exists(self.MEMORY_DIR):
                logger.info(f"Found memory directory at {self.MEMORY_DIR}")
                
                # Check and load user preferences
                if os.path.exists(self.USER_PREFS_DIR):
                    pref_files = [f for f in os.listdir(self.USER_PREFS_DIR) if f.endswith('.json')]
                    logger.info(f"Found {len(pref_files)} user preference files")
                
                # Any other initialization can go here
                
            logger.info("Memory service loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading memory from disk: {e}")
            return False
    
    async def get_conversation_history(self, message: discord.Message, bot) -> List[Dict[str, Any]]:
        """
        Get the processed conversation history for a message
        
        Args:
            message: The Discord message
            bot: The Discord bot instance
            
        Returns:
            The processed conversation history
        """
        # Create a key for this user-channel pair
        key = f"{message.author.id}-{message.channel.id}"
        
        # Initialize history if it doesn't exist
        if key not in message_history:
            message_history[key] = []
            
        # Get the conversation history
        history = message_history[key]
        
        # If history is too long, summarize it
        if len(history) > 10:
            # Import ConversationSummarizer only when needed to avoid circular import
            from ..memory_utils import ConversationSummarizer
            history = await ConversationSummarizer.summarize_conversation(history)
            message_history[key] = history
        
        return history
    
    async def add_to_history(self, user_id: str, channel_id: str, entry: Dict[str, str]) -> None:
        """
        Add an entry to the conversation history
        
        Args:
            user_id: The ID of the user
            channel_id: The ID of the channel
            entry: The conversation entry (dict with 'role' and 'content' keys)
        """
        # Create a key for this conversation
        key = f"{user_id}-{channel_id}"
        
        # Initialize history for this key if it doesn't exist
        if key not in message_history:
            message_history[key] = []
        
        # Add the entry
        message_history[key].append(entry)
        
        # Trim history if it gets too long (keeping the most recent entries)
        max_history = 20  # Adjust as needed
        if len(message_history[key]) > max_history:
            message_history[key] = message_history[key][-max_history:]
    
    async def clear_history(self, user_id: str, channel_id: Optional[str] = None) -> None:
        """
        Clear conversation history for a user
        
        Args:
            user_id: The ID of the user
            channel_id: Optional channel ID to clear specific conversation
        """
        if channel_id:
            # Clear specific conversation
            key = f"{user_id}-{channel_id}"
            if key in message_history:
                del message_history[key]
        else:
            # Clear all conversations for this user
            keys_to_delete = []
            for key in message_history.keys():
                if key.startswith(f"{user_id}-"):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del message_history[key]
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get preferences for a user
        
        Args:
            user_id: The ID of the user
            
        Returns:
            The user preferences
        """
        # Get user preferences directly from file
        pref_file = os.path.join(self.USER_PREFS_DIR, f"{user_id}.json")
        preferences = {}
        
        if os.path.exists(pref_file):
            try:
                with open(pref_file, 'r', encoding='utf-8') as f:
                    preferences = json.load(f)
            except Exception as e:
                logger.error(f"Error loading preferences for user {user_id}: {e}")
        
        return preferences
    
    async def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """
        Set all preferences for a user
        
        Args:
            user_id: The ID of the user
            preferences: The complete preferences dictionary
        """
        pref_file = os.path.join(self.USER_PREFS_DIR, f"{user_id}.json")
        
        try:
            with open(pref_file, 'w', encoding='utf-8') as f:
                json.dump(preferences, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving preferences for user {user_id}: {e}")
    
    async def set_user_preference(self, user_id: str, key: str, value: Any) -> None:
        """
        Set a preference for a user
        
        Args:
            user_id: The ID of the user
            key: The preference key
            value: The preference value
        """
        # Get existing preferences
        preferences = await self.get_user_preferences(user_id)
        
        # Update the preference
        preferences[key] = value
        
        # Save updated preferences
        await self.set_user_preferences(user_id, preferences)
    
    async def clear_user_preferences(self, user_id: str) -> None:
        """
        Clear all preferences for a user
        
        Args:
            user_id: The ID of the user
        """
        pref_file = os.path.join(self.USER_PREFS_DIR, f"{user_id}.json")
        
        if os.path.exists(pref_file):
            try:
                os.remove(pref_file)
            except Exception as e:
                logger.error(f"Error clearing preferences for user {user_id}: {e}")
    
    async def clear_user_data(self, user_id: str) -> None:
        """
        Clear all data for a user (conversation history and preferences)
        
        Args:
            user_id: The ID of the user
        """
        # Clear conversation history
        await self.clear_history(user_id)
        
        # Clear preferences
        await self.clear_user_preferences(user_id)

# Create a singleton instance for global access
memory_service = MemoryService() 