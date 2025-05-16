"""
Memory Service

This module provides a centralized service for managing user conversation history,
preferences, and other memory-related functionality.
"""

import logging
import json
import os
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import discord
import asyncio
import random

# Import directly from original module is causing circular imports, so we'll use
# direct file access for user preferences. The original module now uses this service.
# from ..memory_utils import UserPreferences, process_conversation_history

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('memory_service')

# Constants
USER_DATA_DIR = os.path.join("bot_data", "user_data")
CONVERSATION_DIR = os.path.join("bot_data", "conversations")
MEMORY_DIR = os.path.join("bot_data", "memory")
USER_TRACKING_DIR = os.path.join("bot_data", "user_tracking")

# Ensure directories exist
for directory in [USER_DATA_DIR, CONVERSATION_DIR, MEMORY_DIR, USER_TRACKING_DIR]:
    os.makedirs(directory, exist_ok=True)

class MemoryService:
    """Service for managing user memory, history, and preferences"""
    
    def __init__(self):
        """Initialize memory service"""
        # Set paths for memory directory and user preferences
        self.MEMORY_DIR = MEMORY_DIR
        self.USER_PREFS_DIR = USER_DATA_DIR
        self.user_data_file = os.path.join(self.MEMORY_DIR, "user_data.json")
        
        # Create directories if they don't exist
        os.makedirs(self.USER_PREFS_DIR, exist_ok=True)
        
        # Initialize message history dictionary
        self.message_history = {}
        
        # Initialize conversation history dictionary
        self.conversation_history = {}
        
        # Define maximum history length
        self.MAX_HISTORY_LEN = 50
        
        # Initialize user last seen data
        self.user_last_seen = {}
        
        # In-memory caches for different types of data
        self.user_cache = {}  # Cache for user data
        self.history_cache = {}  # Cache for conversation history
        self.preference_cache = {}  # Cache for user preferences
        self.last_seen_cache = {}  # Cache for last seen timestamps
        
        # Cache management
        self.cache_ttl = 3600  # Cache time-to-live in seconds (1 hour)
        self.last_cache_cleanup = time.time()
        self.cleanup_interval = 900  # Cleanup interval in seconds (15 minutes)
        
        # Create the user tracking file if it doesn't exist
        self.user_tracking_file = os.path.join(USER_TRACKING_DIR, "user_data.json")
        if not os.path.exists(self.user_tracking_file):
            with open(self.user_tracking_file, 'w') as f:
                json.dump({}, f)
        
        # Load the user tracking data
        self._load_tracking_data()
        
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
    
    def _load_tracking_data(self):
        """Load user tracking data from file"""
        try:
            with open(self.user_tracking_file, 'r') as f:
                self.tracking_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("User tracking file not found or invalid. Creating new tracking data.")
            self.tracking_data = {}
    
    def _save_tracking_data(self):
        """Save user tracking data to file"""
        try:
            with open(self.user_tracking_file, 'w') as f:
                json.dump(self.tracking_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user tracking data: {e}")
            
    def _maybe_cleanup_cache(self):
        """Cleanup expired cache entries if it's time"""
        current_time = time.time()
        if current_time - self.last_cache_cleanup > self.cleanup_interval:
            self._cleanup_cache()
            self.last_cache_cleanup = current_time
            
    def _cleanup_cache(self):
        """Remove expired entries from all caches"""
        current_time = time.time()
        # Clean user cache
        expired_user_keys = [k for k, v in self.user_cache.items() 
                          if current_time - v.get('last_accessed', 0) > self.cache_ttl]
        for key in expired_user_keys:
            del self.user_cache[key]
            
        # Clean history cache
        expired_history_keys = [k for k, v in self.history_cache.items() 
                             if current_time - v.get('last_accessed', 0) > self.cache_ttl]
        for key in expired_history_keys:
            del self.history_cache[key]
            
        # Clean preference cache
        expired_pref_keys = [k for k, v in self.preference_cache.items() 
                          if current_time - v.get('last_accessed', 0) > self.cache_ttl]
        for key in expired_pref_keys:
            del self.preference_cache[key]
            
        logger.info(f"Cache cleanup: Removed {len(expired_user_keys)} user entries, " 
                   f"{len(expired_history_keys)} history entries, and {len(expired_pref_keys)} preference entries")
    
    async def get_conversation_history(self, user_id: str, channel_id: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            user_id: The user ID
            channel_id: The channel ID (optional)
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        self._maybe_cleanup_cache()
        
        # Form the conversation key
        conversation_id = f"{user_id}:{channel_id}" if channel_id else user_id
        
        # Check if we have it in cache
        if conversation_id in self.history_cache:
            self.history_cache[conversation_id]['last_accessed'] = time.time()
            history = self.history_cache[conversation_id]['data']
            return history[-limit:] if limit else history
            
        # Check if we have it in conversation history
        if conversation_id in self.conversation_history:
            history = self.conversation_history[conversation_id]
            
            # Cache the result
            self.history_cache[conversation_id] = {
                'data': history,
                'last_accessed': time.time()
            }
            
            return history[-limit:] if limit else history
            
        # Load from disk if not found in memory
        history_file = os.path.join(CONVERSATION_DIR, f"{conversation_id}.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    
                # Cache the result
                self.history_cache[conversation_id] = {
                    'data': history,
                    'last_accessed': time.time()
                }
                
                # Also store in conversation history
                self.conversation_history[conversation_id] = history
                
                return history[-limit:] if limit else history
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning(f"Error loading history for conversation {conversation_id}")
                
        # Return empty list if not found
        return []
    
    def _schedule_save(self):
        """
        Schedule saving of conversation history to disk
        This is a placeholder that will asynchronously save data without blocking
        """
        # This would ideally use a background task or job queue
        # For now, we'll periodically save based on a random chance to avoid 
        # saving on every message (which would be inefficient)
        if random.random() < 0.1:  # 10% chance to save on each call
            try:
                asyncio.create_task(self.save_to_disk())
            except RuntimeError:
                # We're not in an event loop, save directly
                import threading
                threading.Thread(target=lambda: asyncio.run(self.save_to_disk())).start()
                
    async def add_to_history(self, user_id: str, channel_id: str, entry: Dict[str, str]) -> bool:
        """
        Add a message to the conversation history
        
        Args:
            user_id: The user ID
            channel_id: The channel ID
            entry: The message entry to add
            
        Returns:
            bool: Whether the operation was successful
        """
        try:
            # Get channel conversation key
            conversation_key = f"{user_id}:{channel_id}"
            
            # Initialize history for this conversation if it doesn't exist
            if conversation_key not in self.conversation_history:
                self.conversation_history[conversation_key] = []
                
            # Add the entry to the conversation history
            self.conversation_history[conversation_key].append(entry)
            
            # Trim history if it exceeds max length
            if len(self.conversation_history[conversation_key]) > self.MAX_HISTORY_LEN:
                self.conversation_history[conversation_key] = self.conversation_history[conversation_key][-self.MAX_HISTORY_LEN:]
                
            # Schedule a save to disk
            self._schedule_save()
            return True
        except Exception as e:
            logger.error(f"Error adding to history: {e}")
            return False
    
    async def clear_conversation_history(self, conversation_id: str):
        """
        Clear conversation history
        
        Args:
            conversation_id: The conversation ID to clear
        """
        # Remove from cache
        if conversation_id in self.history_cache:
            del self.history_cache[conversation_id]
            
        # Remove from disk
        history_file = os.path.join(CONVERSATION_DIR, f"{conversation_id}.json")
        if os.path.exists(history_file):
            try:
                os.remove(history_file)
            except Exception as e:
                logger.error(f"Error removing history file for conversation {conversation_id}: {e}")
    
    async def get_user_preferences(self, user_id: str) -> dict:
        """
        Get a user's preferences
        
        Args:
            user_id: The user's Discord ID
            
        Returns:
            dict: The user's preferences
        """
        self._maybe_cleanup_cache()
        
        # Check if we have it in cache
        if user_id in self.preference_cache:
            self.preference_cache[user_id]['last_accessed'] = time.time()
            return self.preference_cache[user_id]['data']
            
        # Load from disk
        prefs_file = os.path.join(self.USER_PREFS_DIR, f"{user_id}_preferences.json")
        if os.path.exists(prefs_file):
            try:
                with open(prefs_file, 'r') as f:
                    prefs = json.load(f)
                    
                # Cache the result
                self.preference_cache[user_id] = {
                    'data': prefs,
                    'last_accessed': time.time()
                }
                return prefs
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning(f"Error loading preferences for user {user_id}")
                
        # Default preferences if not found
        default_prefs = {
            "reasoning_type": "auto",  # Default to auto-detection
            "verbosity": "balanced",  # Default to balanced verbosity
            "language": "en",  # Default to English
            "emoji_style": "standard"  # Default emoji style
        }
        
        # Cache the defaults
        self.preference_cache[user_id] = {
            'data': default_prefs,
            'last_accessed': time.time()
        }
        
        return default_prefs
        
    async def set_user_preferences(self, user_id: str, preferences: dict):
        """
        Set a user's preferences
        
        Args:
            user_id: The user's Discord ID
            preferences: The preferences to set
        """
        # Get existing preferences
        existing_prefs = await self.get_user_preferences(user_id)
        
        # Update with new preferences
        existing_prefs.update(preferences)
        
        # Save to disk
        prefs_file = os.path.join(self.USER_PREFS_DIR, f"{user_id}_preferences.json")
        try:
            with open(prefs_file, 'w') as f:
                json.dump(existing_prefs, f, indent=2)
                
            # Update cache
            self.preference_cache[user_id] = {
                'data': existing_prefs,
                'last_accessed': time.time()
            }
        except Exception as e:
            logger.error(f"Error saving preferences for user {user_id}: {e}")
    
    async def clear_user_data(self, user_id: str):
        """
        Clear all data for a user
        
        Args:
            user_id: The user's Discord ID
        """
        # Remove from caches
        if user_id in self.preference_cache:
            del self.preference_cache[user_id]
            
        if user_id in self.user_cache:
            del self.user_cache[user_id]
            
        # Remove preference file
        prefs_file = os.path.join(self.USER_PREFS_DIR, f"{user_id}_preferences.json")
        if os.path.exists(prefs_file):
            try:
                os.remove(prefs_file)
            except Exception as e:
                logger.error(f"Error removing preferences file for user {user_id}: {e}")
                
        # Flag user as deleted in tracking data
        if "users" in self.tracking_data and user_id in self.tracking_data["users"]:
            self.tracking_data["users"][user_id]["deleted"] = True
            self._save_tracking_data()
    
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
                    await self.clear_conversation_history(conversation_id)
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
    
    async def update_user_last_seen(self, user_id: str):
        """
        Update when a user was last seen
        
        Args:
            user_id: The user's Discord ID
        """
        # Ensure user exists
        if "users" not in self.tracking_data:
            self.tracking_data["users"] = {}
            
        if user_id not in self.tracking_data["users"]:
            # Create bare minimum entry
            self.tracking_data["users"][user_id] = {
                "name": None,
                "joined_at": None,
                "last_seen": None
            }
            
        # Update last seen timestamp
        now = datetime.now().isoformat()
        self.tracking_data["users"][user_id]["last_seen"] = now
        
        # Update in-memory cache
        self.last_seen_cache[user_id] = {
            'timestamp': now,
            'last_accessed': time.time()
        }
        
        # Save periodically, but not on every update to avoid I/O overhead
        # Use a random chance to save, with higher probability as time passes
        if hash(user_id + now) % 20 == 0:  # ~5% chance to save
            self._save_tracking_data()
            
    async def track_user(self, user_id: str, username: str, joined_at: Optional[datetime] = None):
        """
        Track a user in the system
        
        Args:
            user_id: The user's Discord ID
            username: The user's Discord username
            joined_at: When the user joined (optional)
        """
        # Ensure the guild structure exists
        if "users" not in self.tracking_data:
            self.tracking_data["users"] = {}
            
        # Add or update the user
        if user_id not in self.tracking_data["users"]:
            self.tracking_data["users"][user_id] = {
                "name": username,
                "joined_at": joined_at.isoformat() if joined_at else None,
                "last_seen": None
            }
        else:
            # Update the username in case it changed
            self.tracking_data["users"][user_id]["name"] = username
            
        # Save changes
        self._save_tracking_data()
    
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