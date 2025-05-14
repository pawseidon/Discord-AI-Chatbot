"""
Agent Memory Manager

This module manages different types of memory for agents, including
conversation history, agent scratchpads, and user preferences.
"""

import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('agent_memory')

# Define storage directories
DATA_DIR = os.path.join("bot_data", "memory")
USER_DATA_DIR = os.path.join(DATA_DIR, "users")
CONVERSATION_DATA_DIR = os.path.join(DATA_DIR, "conversations")

# Ensure directories exist
os.makedirs(USER_DATA_DIR, exist_ok=True)
os.makedirs(CONVERSATION_DATA_DIR, exist_ok=True)

class AgentMemoryManager:
    """
    Manages different types of memory for agents
    """
    
    def __init__(self):
        """Initialize the memory manager"""
        self.conversation_memories = {}  # In-memory cache of conversation histories
        self.user_memories = {}  # In-memory cache of user-specific data
        self.agent_scratchpads = {}  # Temporary working memory for agents
        
        # Track memory sizes to avoid excessive growth
        self.max_conversation_memory = 100  # Maximum number of messages per conversation
        self.max_user_memory_items = 50  # Maximum number of memory items per user
        
        # Create a memory lock to prevent concurrent writes
        self._memory_lock = asyncio.Lock()
    
    async def load_conversation_memory(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Load a conversation's memory from storage
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of message objects
        """
        # Check in-memory cache first
        if conversation_id in self.conversation_memories:
            return self.conversation_memories[conversation_id]
        
        # Load from disk if not in memory
        try:
            file_path = os.path.join(CONVERSATION_DATA_DIR, f"{conversation_id}.json")
            if os.path.exists(file_path):
                async with self._memory_lock:
                    with open(file_path, "r") as f:
                        memory = json.load(f)
                        self.conversation_memories[conversation_id] = memory
                        return memory
        except Exception as e:
            logger.error(f"Error loading conversation memory: {str(e)}")
        
        # Initialize empty memory if not found
        self.conversation_memories[conversation_id] = []
        return []
    
    async def add_to_conversation_memory(
        self, 
        conversation_id: str, 
        role: str, 
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to conversation memory
        
        Args:
            conversation_id: ID of the conversation
            role: Role of the message sender (user, assistant, system)
            content: Message content
            user_id: Optional ID of the user
            metadata: Optional additional message data
        """
        # Ensure conversation memory is loaded
        if conversation_id not in self.conversation_memories:
            await self.load_conversation_memory(conversation_id)
        
        # Limit the size of conversation memory
        if len(self.conversation_memories[conversation_id]) >= self.max_conversation_memory:
            # Remove the oldest messages, keeping the most recent ones
            self.conversation_memories[conversation_id] = self.conversation_memories[conversation_id][-(self.max_conversation_memory - 1):]
        
        # Create the message object
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if user_id:
            message["user_id"] = user_id
            
        if metadata:
            message["metadata"] = metadata
        
        # Add to memory
        self.conversation_memories[conversation_id].append(message)
        
        # Save to disk
        await self._save_conversation_memory(conversation_id)
    
    async def _save_conversation_memory(self, conversation_id: str) -> None:
        """
        Save conversation memory to disk
        
        Args:
            conversation_id: ID of the conversation to save
        """
        try:
            file_path = os.path.join(CONVERSATION_DATA_DIR, f"{conversation_id}.json")
            async with self._memory_lock:
                with open(file_path, "w") as f:
                    json.dump(self.conversation_memories[conversation_id], f)
        except Exception as e:
            logger.error(f"Error saving conversation memory: {str(e)}")
    
    async def load_user_memory(self, user_id: str) -> Dict[str, Any]:
        """
        Load a user's memory from storage
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary of user memory
        """
        # Check in-memory cache first
        if user_id in self.user_memories:
            return self.user_memories[user_id]
        
        # Load from disk if not in memory
        try:
            file_path = os.path.join(USER_DATA_DIR, f"{user_id}.json")
            if os.path.exists(file_path):
                async with self._memory_lock:
                    with open(file_path, "r") as f:
                        memory = json.load(f)
                        self.user_memories[user_id] = memory
                        return memory
        except Exception as e:
            logger.error(f"Error loading user memory: {str(e)}")
        
        # Initialize empty memory if not found
        self.user_memories[user_id] = {
            "preferences": {},
            "memories": {},
            "last_seen": datetime.now().isoformat()
        }
        return self.user_memories[user_id]
    
    async def _save_user_memory(self, user_id: str) -> None:
        """
        Save user memory to disk
        
        Args:
            user_id: ID of the user to save
        """
        try:
            file_path = os.path.join(USER_DATA_DIR, f"{user_id}.json")
            async with self._memory_lock:
                with open(file_path, "w") as f:
                    json.dump(self.user_memories[user_id], f)
        except Exception as e:
            logger.error(f"Error saving user memory: {str(e)}")
    
    async def store_memory_item(
        self, 
        user_id: str, 
        key: str, 
        value: Any,
        conversation_id: Optional[str] = None
    ) -> None:
        """
        Store an item in user memory
        
        Args:
            user_id: ID of the user
            key: Key for the memory item
            value: Value to store
            conversation_id: Optional conversation context
        """
        # Load user memory
        user_memory = await self.load_user_memory(user_id)
        
        # Store the item
        if "memories" not in user_memory:
            user_memory["memories"] = {}
            
        # Manage memory size
        if len(user_memory["memories"]) >= self.max_user_memory_items:
            # Find the oldest item to remove
            oldest_key = None
            oldest_time = None
            
            for k, v in user_memory["memories"].items():
                item_time = v.get("timestamp")
                if oldest_time is None or (item_time and item_time < oldest_time):
                    oldest_key = k
                    oldest_time = item_time
            
            # Remove the oldest item
            if oldest_key:
                del user_memory["memories"][oldest_key]
        
        # Create memory item
        memory_item = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id
        }
        
        user_memory["memories"][key] = memory_item
        
        # Update last seen time
        user_memory["last_seen"] = datetime.now().isoformat()
        
        # Save to disk
        await self._save_user_memory(user_id)
    
    async def get_memory_item(
        self, 
        user_id: str, 
        key: str,
        conversation_id: Optional[str] = None
    ) -> Optional[Any]:
        """
        Retrieve an item from user memory
        
        Args:
            user_id: ID of the user
            key: Key for the memory item
            conversation_id: Optional conversation context for filtering
            
        Returns:
            The stored value or None if not found
        """
        # Load user memory
        user_memory = await self.load_user_memory(user_id)
        
        # Retrieve the item
        if "memories" in user_memory and key in user_memory["memories"]:
            memory_item = user_memory["memories"][key]
            
            # If conversation_id is provided, filter by it
            if conversation_id and memory_item.get("conversation_id") != conversation_id:
                return None
                
            return memory_item.get("value")
        
        return None
    
    async def set_user_preference(self, user_id: str, preference_key: str, preference_value: Any) -> None:
        """
        Set a user preference
        
        Args:
            user_id: ID of the user
            preference_key: Preference key
            preference_value: Preference value
        """
        # Load user memory
        user_memory = await self.load_user_memory(user_id)
        
        # Set the preference
        if "preferences" not in user_memory:
            user_memory["preferences"] = {}
            
        user_memory["preferences"][preference_key] = preference_value
        
        # Update last seen time
        user_memory["last_seen"] = datetime.now().isoformat()
        
        # Save to disk
        await self._save_user_memory(user_id)
    
    async def get_user_preference(self, user_id: str, preference_key: str) -> Optional[Any]:
        """
        Get a user preference
        
        Args:
            user_id: ID of the user
            preference_key: Preference key
            
        Returns:
            The preference value or None if not found
        """
        # Load user memory
        user_memory = await self.load_user_memory(user_id)
        
        # Get the preference
        if "preferences" in user_memory:
            return user_memory["preferences"].get(preference_key)
        
        return None
    
    async def update_user_last_seen(self, user_id: str) -> None:
        """
        Update the last seen timestamp for a user
        
        Args:
            user_id: ID of the user
        """
        # Load user memory
        user_memory = await self.load_user_memory(user_id)
        
        # Update last seen time
        user_memory["last_seen"] = datetime.now().isoformat()
        
        # Save to disk
        await self._save_user_memory(user_id)
    
    async def add_to_agent_scratchpad(
        self, 
        agent_id: str, 
        conversation_id: str, 
        note: str
    ) -> None:
        """
        Add a note to an agent's scratchpad
        
        Args:
            agent_id: ID of the agent
            conversation_id: ID of the conversation context
            note: Note to add
        """
        # Create keys if needed
        scratch_key = f"{agent_id}:{conversation_id}"
        
        if scratch_key not in self.agent_scratchpads:
            self.agent_scratchpads[scratch_key] = []
            
        # Add the note
        self.agent_scratchpads[scratch_key].append({
            "timestamp": datetime.now().isoformat(),
            "content": note
        })
        
        # Limit size if needed
        if len(self.agent_scratchpads[scratch_key]) > 20:  # Keep only the last 20 notes
            self.agent_scratchpads[scratch_key] = self.agent_scratchpads[scratch_key][-20:]
    
    async def get_agent_scratchpad(
        self, 
        agent_id: str, 
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get an agent's scratchpad notes
        
        Args:
            agent_id: ID of the agent
            conversation_id: ID of the conversation context
            
        Returns:
            List of scratchpad notes
        """
        scratch_key = f"{agent_id}:{conversation_id}"
        return self.agent_scratchpads.get(scratch_key, [])
    
    async def reset_conversation(self, conversation_id: str) -> None:
        """
        Reset a conversation's memory
        
        Args:
            conversation_id: ID of the conversation to reset
        """
        # Clear from memory
        if conversation_id in self.conversation_memories:
            self.conversation_memories[conversation_id] = []
            
        # Clear agent scratchpads for this conversation
        for key in list(self.agent_scratchpads.keys()):
            if key.endswith(f":{conversation_id}"):
                self.agent_scratchpads[key] = []
                
        # Delete from disk
        try:
            file_path = os.path.join(CONVERSATION_DATA_DIR, f"{conversation_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting conversation memory file: {str(e)}")
    
    async def clear_user_data(self, user_id: str) -> None:
        """
        Clear all data for a user
        
        Args:
            user_id: ID of the user
        """
        # Clear from memory
        if user_id in self.user_memories:
            del self.user_memories[user_id]
            
        # Clear from conversation memories
        for conv_id, messages in self.conversation_memories.items():
            self.conversation_memories[conv_id] = [
                msg for msg in messages if msg.get("user_id") != user_id
            ]
            
        # Delete from disk
        try:
            file_path = os.path.join(USER_DATA_DIR, f"{user_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                
            # Update conversation files
            for conv_id in self.conversation_memories:
                await self._save_conversation_memory(conv_id)
        except Exception as e:
            logger.error(f"Error deleting user memory file: {str(e)}")
            
    async def get_all_user_ids(self) -> List[str]:
        """
        Get all user IDs with stored memory
        
        Returns:
            List of user IDs
        """
        user_ids = []
        
        # Get from disk
        for filename in os.listdir(USER_DATA_DIR):
            if filename.endswith(".json"):
                user_ids.append(filename[:-5])  # Remove .json extension
                
        return user_ids 