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
        """Initialize memory service"""
        # Set paths for memory directory and user preferences
        self.MEMORY_DIR = os.path.join("bot_data", "memory")
        self.USER_PREFS_DIR = os.path.join(self.MEMORY_DIR, "user_prefs")
        self.user_data_file = os.path.join(self.MEMORY_DIR, "user_data.json")
        
        # Create directories if they don't exist
        os.makedirs(self.USER_PREFS_DIR, exist_ok=True)
        
        # Initialize message history dictionary
        self.message_history = {}
        
        # Initialize user last seen data
        self.user_last_seen = {}
        
        # Load existing data
        self._load_user_data()
        logger.info(f"Found {len(os.listdir(self.USER_PREFS_DIR))} user preference files")
        logger.info("Memory service loaded successfully")
    
    def _load_user_data(self):
        """
        Load user data from disk storage
        
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
                
                # Load user data file if it exists
                if os.path.exists(self.user_data_file):
                    try:
                        with open(self.user_data_file, 'r') as f:
                            data = json.load(f)
                            self.user_last_seen = data.get('last_seen', {})
                    except Exception as e:
                        logger.error(f"Error loading user data file: {e}")
                
            logger.info("Memory service loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading memory from disk: {e}")
            return False
    
    async def get_conversation_history(self, user_id: str, channel_id: str) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a user in a specific channel
        
        Args:
            user_id: The ID of the user
            channel_id: The ID of the channel
            
        Returns:
            The conversation history
        """
        # Create a key for this user-channel pair
        key = f"{user_id}-{channel_id}"
        
        # Initialize history if it doesn't exist
        if key not in self.message_history:
            self.message_history[key] = []
            
        # Get the conversation history
        history = self.message_history[key]
        
        # If history is too long, summarize it
        max_history = 20  # Adjust as needed
        if len(history) > max_history:
            # Just trim to keep the most recent entries if we don't have a summarizer
            history = history[-max_history:]
            self.message_history[key] = history
        
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
        if key not in self.message_history:
            self.message_history[key] = []
        
        # Add the entry
        self.message_history[key].append(entry)
        
        # Trim history if it gets too long (keeping the most recent entries)
        max_history = 20  # Adjust as needed
        if len(self.message_history[key]) > max_history:
            self.message_history[key] = self.message_history[key][-max_history:]
    
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
            if key in self.message_history:
                del self.message_history[key]
        else:
            # Clear all conversations for this user
            keys_to_delete = []
            for key in self.message_history.keys():
                if key.startswith(f"{user_id}-"):
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.message_history[key]
    
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
        Clear all data for a user
        
        Args:
            user_id: The ID of the user
        """
        # Clear conversation history
        await self.clear_history(user_id)
        
        # Clear user preferences
        pref_file = os.path.join(self.USER_PREFS_DIR, f"{user_id}.json")
        if os.path.exists(pref_file):
            try:
                os.remove(pref_file)
            except Exception as e:
                logger.error(f"Error removing preferences file for user {user_id}: {e}")
    
    async def reset_conversation(self, conversation_id: str) -> None:
        """
        Reset a specific conversation
        
        Args:
            conversation_id: The ID of the conversation to reset
        """
        # Break apart the conversation ID to get user and channel components
        try:
            parts = conversation_id.split(":")
            if len(parts) == 2:
                guild_or_dm, channel_id = parts
                
                # Check if this is a DM
                if guild_or_dm == "DM":
                    user_id = channel_id  # In DMs, the channel ID is the user ID
                    await self.clear_history(user_id, channel_id)
                else:
                    # For guild channels, we need to clear all users' history in this channel
                    guild_id = guild_or_dm
                    keys_to_delete = []
                    for key in self.message_history.keys():
                        if key.endswith(f"-{channel_id}"):
                            keys_to_delete.append(key)
                    
                    for key in keys_to_delete:
                        del self.message_history[key]
            else:
                logger.warning(f"Invalid conversation ID format: {conversation_id}")
        except Exception as e:
            logger.error(f"Error resetting conversation {conversation_id}: {e}")
    
    async def store_conversation_context(self, conversation_id: str, context_data: Dict[str, Any]) -> None:
        """
        Store context data for a specific conversation
        
        Args:
            conversation_id: The ID of the conversation
            context_data: Dictionary of context data to store
        """
        try:
            # Create a key for context storage
            context_key = f"context_{conversation_id}"
            
            # Initialize the context if it doesn't exist
            if not hasattr(self, 'conversation_contexts'):
                self.conversation_contexts = {}
                
            # Get existing context or create new
            if context_key not in self.conversation_contexts:
                self.conversation_contexts[context_key] = {}
                
            # Update with new context data
            self.conversation_contexts[context_key].update(context_data)
            
            logger.debug(f"Stored context for conversation {conversation_id}: {context_data.keys()}")
        except Exception as e:
            logger.error(f"Error storing conversation context: {e}")
    
    async def get_conversation_context(self, conversation_id: str, key: str = None) -> Any:
        """
        Get context data for a specific conversation
        
        Args:
            conversation_id: The ID of the conversation
            key: Optional specific context key to retrieve
            
        Returns:
            The requested context data or None if not found
        """
        try:
            # Create the context key
            context_key = f"context_{conversation_id}"
            
            # Check if we have context data
            if not hasattr(self, 'conversation_contexts'):
                return None if key else {}
                
            if context_key not in self.conversation_contexts:
                return None if key else {}
                
            # Return specific key or entire context
            if key:
                return self.conversation_contexts[context_key].get(key)
            else:
                return self.conversation_contexts[context_key]
        except Exception as e:
            logger.error(f"Error retrieving conversation context: {e}")
            return None if key else {}
    
    async def update_user_last_seen(self, user_id: str) -> None:
        """
        Update the last seen timestamp for a user
        
        Args:
            user_id: The ID of the user
        """
        import time
        current_time = int(time.time())
        
        if not hasattr(self, 'user_last_seen'):
            self.user_last_seen = {}
            
        self.user_last_seen[user_id] = current_time
        
        # Save to file periodically
        try:
            with open(self.user_data_file, 'w') as f:
                json.dump({
                    'last_seen': self.user_last_seen
                }, f)
        except Exception as e:
            logger.error(f"Error saving user last seen data: {e}")

    async def save_to_disk(self) -> bool:
        """
        Save memory data to disk
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Save user preferences
            for user_id in os.listdir(self.USER_PREFS_DIR):
                if user_id.endswith('.json'):
                    # Already saved directly when setting preferences
                    pass
                    
            # Save any other data that needs persistence
            with open(self.user_data_file, 'w') as f:
                json.dump({
                    'last_seen': getattr(self, 'user_last_seen', {})
                }, f)
                
            logger.info("Memory data saved to disk successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving memory to disk: {e}")
            return False

    async def load_from_disk(self) -> bool:
        """
        Load memory data from disk
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if user data file exists
            if os.path.exists(self.user_data_file):
                with open(self.user_data_file, 'r') as f:
                    data = json.load(f)
                    self.user_last_seen = data.get('last_seen', {})
                    
            # Ensure user preferences directory exists
            if os.path.exists(self.USER_PREFS_DIR):
                logger.info(f"Found {len(os.listdir(self.USER_PREFS_DIR))} user preference files")
                
            logger.info("Memory data loaded from disk successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading memory from disk: {e}")
            return False

# Create a singleton instance for global access
memory_service = MemoryService() 