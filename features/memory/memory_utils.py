"""
Memory utilities module for Discord AI Chatbot.

This module provides memory management, conversation history, and retention
capabilities for enhanced user interactions.
"""

import json
import os
import time
import asyncio
import hashlib
import uuid
import discord
from datetime import datetime
from typing import Dict, List, Any, Optional

# Initialize paths for storing data
DATA_DIR = "bot_data"
USER_PREFS_DIR = os.path.join(DATA_DIR, "user_preferences")
MEMORY_DIR = os.path.join(DATA_DIR, "memory_summaries")

# Ensure directories exist
os.makedirs(USER_PREFS_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

class ConversationSummarizer:
    """Handles summarizing long conversations to maintain context while reducing token usage"""
    
    @staticmethod
    async def summarize_conversation(conversation_history, ai_provider=None, threshold=8):
        """
        Summarize conversation when it exceeds the threshold length
        
        Args:
            conversation_history (list): List of message dictionaries
            ai_provider: AI provider for generating summaries
            threshold (int): Number of messages before summarization is triggered
        
        Returns:
            list: Updated conversation history with summary
        """
        # If conversation is not long enough, don't summarize
        if len(conversation_history) < threshold:
            return conversation_history
            
        # Take the last N messages to summarize
        messages_to_summarize = conversation_history[-(threshold-1):]
        
        # Format the conversation for summarization
        formatted_convo = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages_to_summarize
        ])
        
        # Create a system message for summarization
        system_message = (
            "You are a helpful assistant that summarizes conversations. "
            "Create a concise summary of the key points from this conversation. "
            "Focus on important information, questions, and answers. "
            "Ignore casual greetings or irrelevant details."
        )
        
        try:
            if ai_provider and hasattr(ai_provider, 'async_call'):
                prompt = f"{system_message}\n\nPlease summarize this conversation:\n\n{formatted_convo}"
                summary = await ai_provider.async_call(prompt)
            else:
                # Fallback to simple summarization if no provider available
                summary = f"Summary of the last {len(messages_to_summarize)} messages (no AI provider available)"
            
            # Create a new conversation history with the summary and most recent messages
            new_history = [
                # Keep the very first message which usually contains the user's initial query
                conversation_history[0], 
                # Add our summary as a system message
                {"role": "system", "content": f"[Summary of previous conversation: {summary}]"},
                # Keep the most recent messages (last 3)
                *conversation_history[-3:]
            ]
            
            return new_history
            
        except Exception as e:
            print(f"Error during conversation summarization: {e}")
            # If summarization fails, just return a truncated version of the conversation
            return conversation_history[0:1] + conversation_history[-7:]

class UserPreferences:
    """Manages user preferences and settings"""
    
    @staticmethod
    async def get_user_preferences(user_id):
        """Get preferences for a specific user"""
        pref_file = os.path.join(USER_PREFS_DIR, f"{user_id}.json")
        
        if os.path.exists(pref_file):
            try:
                with open(pref_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading user preferences: {e}")
                return {}
        else:
            # Return default preferences
            return {
                "created_at": time.time(),
                "topics_of_interest": [],
                "preferred_response_length": "medium",  # short, medium, long
                "use_voice": False,
                "use_embeds": False,  # Default to not using embeds
                "use_streaming": True,  # Enable streaming responses by default
                "last_active": time.time()
            }
    
    @staticmethod
    async def save_user_preferences(user_id, preferences):
        """Save user preferences"""
        pref_file = os.path.join(USER_PREFS_DIR, f"{user_id}.json")
        
        # Ensure last_active is updated
        preferences["last_active"] = time.time()
        
        try:
            with open(pref_file, 'w', encoding='utf-8') as f:
                json.dump(preferences, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving user preferences: {e}")
            return False
    
    @staticmethod
    async def update_user_preference(user_id, key, value):
        """Update a single preference for a user"""
        preferences = await UserPreferences.get_user_preferences(user_id)
        preferences[key] = value
        return await UserPreferences.save_user_preferences(user_id, preferences)
    
    @staticmethod
    async def track_topic_interest(user_id, message_content):
        """
        Track topics a user is interested in based on their messages
        This is a simple implementation - could be enhanced with NLP
        """
        # List of common topics to track
        common_topics = [
            "crypto", "bitcoin", "ethereum", "programming", "python", 
            "javascript", "gaming", "music", "sports", "news",
            "weather", "tech", "science", "history", "movies",
            "books", "politics", "food", "travel", "health"
        ]
        
        preferences = await UserPreferences.get_user_preferences(user_id)
        
        # Extract topics of interest from the message
        message_lower = message_content.lower()
        for topic in common_topics:
            if topic in message_lower and topic not in preferences["topics_of_interest"]:
                # Add topic to interests if mentioned and not already tracked
                preferences["topics_of_interest"].append(topic)
                # Keep the list to a reasonable size
                if len(preferences["topics_of_interest"]) > 10:
                    preferences["topics_of_interest"] = preferences["topics_of_interest"][-10:]
        
        await UserPreferences.save_user_preferences(user_id, preferences)

class ConversationMemory:
    """Manages long-term memory of conversations"""
    
    def __init__(self, storage_dir='bot_data/memories'):
        """Initialize with storage directory"""
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    async def store_interaction(self, user_id: str, channel_id: str, query: str, response: str):
        """Store an interaction in the memory"""
        # Create user directory if it doesn't exist
        user_dir = os.path.join(self.storage_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Create a memory record
        memory = {
            "timestamp": datetime.now().isoformat(),
            "channel_id": channel_id,
            "query": query,
            "response": response,
            "memory_id": str(uuid.uuid4())
        }
        
        # Save to a file named with the timestamp for easy sorting
        filename = f"{int(time.time())}.json"
        with open(os.path.join(user_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2)
            
        return memory
    
    async def get_recent_conversations(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent conversations for a user"""
        user_dir = os.path.join(self.storage_dir, user_id)
        if not os.path.exists(user_dir):
            return []
            
        # Get all memory files and sort by timestamp (newest first)
        memory_files = sorted(
            [f for f in os.listdir(user_dir) if f.endswith('.json')],
            key=lambda x: int(x.split('.')[0]),
            reverse=True
        )
        
        # Load the most recent ones up to the limit
        memories = []
        for i, filename in enumerate(memory_files):
            if i >= limit:
                break
                
            with open(os.path.join(user_dir, filename), 'r', encoding='utf-8') as f:
                memories.append(json.load(f))
                
        return memories
    
    async def format_memory_for_context(self, user_id: str, limit: int = 5) -> str:
        """Format memories for inclusion in context"""
        memories = await self.get_recent_conversations(user_id, limit)
        if not memories:
            return ""
            
        # Create a formatted memory string
        memory_str = "Previous interactions:\n\n"
        for memory in memories:
            timestamp = datetime.fromisoformat(memory["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            memory_str += f"[{timestamp}]\n"
            memory_str += f"User: {memory['query']}\n"
            memory_str += f"Assistant: {memory['response']}\n\n"
            
        return memory_str

class MemoryManager:
    """
    Integrated memory management system combining conversation history,
    user preferences, and long-term memory
    """
    
    def __init__(self):
        """Initialize the memory manager"""
        self.conversation_history = {}  # Key: user_id-channel_id, Value: list of messages
        self.memory = ConversationMemory()
        self.max_history_length = 15  # Default max history length
        
    async def add_message(self, user_id: str, channel_id: str, message: Dict[str, Any]) -> None:
        """Add a message to the conversation history"""
        key = f"{user_id}-{channel_id}"
        
        if key not in self.conversation_history:
            self.conversation_history[key] = []
            
        self.conversation_history[key].append(message)
        
        # If it's a user message, track interests
        if message.get("role") == "user":
            await UserPreferences.track_topic_interest(user_id, message.get("content", ""))
    
    async def process_history(self, user_id: str, channel_id: str, ai_provider=None) -> List[Dict[str, Any]]:
        """Process and potentially summarize conversation history"""
        key = f"{user_id}-{channel_id}"
        
        # If no history exists, return empty list
        if key not in self.conversation_history or not self.conversation_history[key]:
            return []
        
        conversation = self.conversation_history[key]
        
        # If conversation is getting long, summarize it
        if len(conversation) > self.max_history_length + 2:  # Allow some buffer
            conversation = await ConversationSummarizer.summarize_conversation(
                conversation, 
                ai_provider=ai_provider,
                threshold=self.max_history_length
            )
            # Update the history with the summarized version
            self.conversation_history[key] = conversation
        
        return conversation
    
    async def get_context(self, user_id: str, channel_id: str, query: str, ai_provider=None) -> Dict[str, Any]:
        """Get comprehensive context for a new query"""
        # Process and get conversation history
        conversation = await self.process_history(user_id, channel_id, ai_provider)
        
        # Get user preferences
        preferences = await UserPreferences.get_user_preferences(user_id)
        
        # Get long-term memories
        memories = await self.memory.format_memory_for_context(user_id, limit=3)
        
        # Combine into a context object
        context = {
            "user_id": user_id,
            "channel_id": channel_id,
            "query": query,
            "conversation_history": conversation,
            "user_preferences": preferences,
            "long_term_memories": memories
        }
        
        return context
    
    async def store_interaction(self, user_id: str, channel_id: str, query: str, response: str) -> None:
        """Store an interaction in long-term memory"""
        await self.memory.store_interaction(user_id, channel_id, query, response)
    
    async def clear_history(self, user_id: str, channel_id: str) -> bool:
        """Clear conversation history for a user-channel pair"""
        key = f"{user_id}-{channel_id}"
        if key in self.conversation_history:
            del self.conversation_history[key]
            return True
        return False
    
    def set_max_history_length(self, length: int) -> None:
        """Set the maximum history length before summarization"""
        if length > 0:
            self.max_history_length = length

# Global instance for singleton pattern
_memory_manager = None

def create_memory_manager() -> MemoryManager:
    """Create or get the global memory manager instance"""
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = MemoryManager()
        
    return _memory_manager 