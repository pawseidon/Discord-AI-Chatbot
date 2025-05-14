import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict

# Set up logging
logger = logging.getLogger('context_manager')

class ConversationThread:
    """Represents a single conversation thread with full context history"""
    
    def __init__(self, user_id: str, channel_id: str, thread_id: Optional[str] = None):
        self.user_id = user_id
        self.channel_id = channel_id
        self.thread_id = thread_id
        self.messages = []  # List of messages in chronological order
        self.last_active = time.time()
        self.context_map = {}  # Mapping of context keys to values
        self.thinking_history = []  # History of sequential thinking processes
        self.related_threads = []  # Related conversation threads
        self.summary = ""  # Current conversation summary
        self.topic = ""  # Detected conversation topic
        
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the conversation thread"""
        self.messages.append(message)
        self.last_active = time.time()
        
    def add_thinking(self, thinking: Dict[str, Any]) -> None:
        """Add a sequential thinking process to the history"""
        self.thinking_history.append(thinking)
        
    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages in the thread"""
        return self.messages[-limit:] if len(self.messages) > limit else self.messages
        
    def get_related_thinking(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get the most relevant sequential thinking processes"""
        # Simple keyword matching for now, could be enhanced with embeddings later
        relevant_thinking = []
        query_keywords = set(query.lower().split())
        
        for thinking in self.thinking_history:
            problem = thinking.get("problem", "").lower()
            problem_words = set(problem.split())
            # Calculate keyword overlap
            overlap = len(query_keywords.intersection(problem_words))
            if overlap > 0:
                relevant_thinking.append((thinking, overlap))
        
        # Sort by relevance (keyword overlap)
        relevant_thinking.sort(key=lambda x: x[1], reverse=True)
        
        # Return the most relevant thinking processes
        return [thinking for thinking, _ in relevant_thinking[:limit]]
        
    def update_context_map(self, key: str, value: Any) -> None:
        """Update a value in the context map"""
        self.context_map[key] = value
        
    def build_context_for_query(self, query: str) -> Dict[str, Any]:
        """Build a comprehensive context object for a query"""
        # Get recent messages, increasing from 5 to 10 for better continuity
        recent_messages = self.get_recent_messages(10)
        
        # Create a summary of the most recent conversation flow
        conversation_flow = ""
        if len(recent_messages) >= 2:
            # Create a brief flow of the last few message exchanges
            for msg in recent_messages[-5:]:  # Last 5 messages
                role = msg.get("role", "unknown")
                # Get the content and handle if it's a list
                raw_content = msg.get("content", "")
                
                # Convert content to string if it's a list
                if isinstance(raw_content, list):
                    content_str = str(raw_content[0]) if raw_content else ""
                else:
                    content_str = str(raw_content)
                
                # Include only the first 50 chars of each message for the flow
                content = content_str[:50] + ("..." if len(content_str) > 50 else "")
                conversation_flow += f"{role}: {content}\n"
        
        # Check if this is from the master/owner
        is_master = any(msg.get("is_from_master", False) for msg in recent_messages if msg.get("role") == "user")
        
        context = {
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "last_active": self.last_active,
            "query": query,
            "recent_messages": recent_messages,
            "conversation_flow": conversation_flow,
            "context_values": self.context_map,
            "is_master": is_master
        }
        
        # Add conversation summary if available
        if self.summary:
            context["conversation_summary"] = self.summary
            
        # Add topic if detected
        if self.topic:
            context["conversation_topic"] = self.topic
            
        # Add relevant thinking history
        relevant_thinking = self.get_related_thinking(query)
        if relevant_thinking:
            context["relevant_thinking"] = relevant_thinking
            
        return context

class ContextManager:
    """Manages conversation context across multiple channels and users"""
    
    def __init__(self, storage_dir: str = "bot_data/context"):
        self.storage_dir = storage_dir
        self.conversations = {}  # Map of user-channel keys to conversation threads
        self.thread_map = {}  # Map of thread IDs to user-channel keys
        self.thinking_index = defaultdict(list)  # Keyword-based index for finding relevant thinking processes
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
    def get_conversation_key(self, user_id: str, channel_id: str) -> str:
        """Generate a key for the conversation dictionary"""
        return f"{user_id}:{channel_id}"
        
    def get_or_create_conversation(self, user_id: str, channel_id: str, thread_id: Optional[str] = None) -> ConversationThread:
        """Get an existing conversation or create a new one"""
        key = self.get_conversation_key(user_id, channel_id)
        
        # If thread_id is provided, check if we have a mapping
        if thread_id and thread_id in self.thread_map:
            key = self.thread_map[thread_id]
        
        # Create a new conversation if it doesn't exist
        if key not in self.conversations:
            self.conversations[key] = ConversationThread(user_id, channel_id, thread_id)
            
            # Add thread mapping if thread_id is provided
            if thread_id:
                self.thread_map[thread_id] = key
                
        return self.conversations[key]
        
    def add_message(self, user_id: str, channel_id: str, message: Dict[str, Any], thread_id: Optional[str] = None) -> None:
        """Add a message to a conversation thread"""
        conversation = self.get_or_create_conversation(user_id, channel_id, thread_id)
        conversation.add_message(message)
        
    def add_thinking_process(self, user_id: str, channel_id: str, thinking: Dict[str, Any], thread_id: Optional[str] = None) -> None:
        """Add a sequential thinking process to a conversation"""
        conversation = self.get_or_create_conversation(user_id, channel_id, thread_id)
        conversation.add_thinking(thinking)
        
        # Index the thinking process by keywords for future retrieval
        if "problem" in thinking:
            problem = thinking["problem"].lower()
            keywords = set(problem.split())
            for keyword in keywords:
                if len(keyword) > 3:  # Only index meaningful keywords
                    self.thinking_index[keyword].append((user_id, channel_id, thread_id, len(conversation.thinking_history)-1))
        
    def update_context_value(self, user_id: str, channel_id: str, key: str, value: Any, thread_id: Optional[str] = None) -> None:
        """Update a context value for a conversation"""
        conversation = self.get_or_create_conversation(user_id, channel_id, thread_id)
        conversation.update_context_map(key, value)
        
    def update_conversation_summary(self, user_id: str, channel_id: str, summary: str, thread_id: Optional[str] = None) -> None:
        """Update the conversation summary"""
        conversation = self.get_or_create_conversation(user_id, channel_id, thread_id)
        conversation.summary = summary
        
    def update_conversation_topic(self, user_id: str, channel_id: str, topic: str, thread_id: Optional[str] = None) -> None:
        """Update the detected conversation topic"""
        conversation = self.get_or_create_conversation(user_id, channel_id, thread_id)
        conversation.topic = topic
        
    def get_conversation_context(self, user_id: str, channel_id: str, query: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive context for a new query"""
        conversation = self.get_or_create_conversation(user_id, channel_id, thread_id)
        return conversation.build_context_for_query(query)
        
    def find_related_thinking(self, query: str, limit: int = 3) -> List[Tuple[str, str, Optional[str], Dict[str, Any]]]:
        """Find thinking processes related to the query across all conversations"""
        query_keywords = set(query.lower().split())
        matches = defaultdict(int)
        
        # Score thinking processes by keyword matches
        for keyword in query_keywords:
            if len(keyword) > 3 and keyword in self.thinking_index:
                for user_id, channel_id, thread_id, thinking_idx in self.thinking_index[keyword]:
                    key = (user_id, channel_id, thread_id)
                    matches[key] += 1
        
        # Sort by number of keyword matches
        sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        results = []
        for (user_id, channel_id, thread_id), _ in sorted_matches:
            key = self.get_conversation_key(user_id, channel_id)
            if key in self.conversations:
                conversation = self.conversations[key]
                # Ensure thinking_idx is within bounds
                if 'thinking_idx' in locals() and thinking_idx < len(conversation.thinking_history):
                    thinking = conversation.thinking_history[thinking_idx]
                    results.append((user_id, channel_id, thread_id, thinking))
        
        return results
        
    def link_conversations(self, user_id1: str, channel_id1: str, user_id2: str, channel_id2: str) -> None:
        """Link two conversations as related"""
        conv1 = self.get_or_create_conversation(user_id1, channel_id1)
        conv2 = self.get_or_create_conversation(user_id2, channel_id2)
        
        # Add bidirectional relationship
        key1 = self.get_conversation_key(user_id1, channel_id1)
        key2 = self.get_conversation_key(user_id2, channel_id2)
        
        if key2 not in conv1.related_threads:
            conv1.related_threads.append(key2)
        
        if key1 not in conv2.related_threads:
            conv2.related_threads.append(key1)
        
    async def save_to_disk(self) -> None:
        """Save the context manager state to disk"""
        try:
            # Create a serializable representation of the state
            state = {
                "conversations": {},
                "thread_map": self.thread_map,
                "timestamp": time.time()
            }
            
            # Convert conversation objects to dictionaries
            for key, conversation in self.conversations.items():
                state["conversations"][key] = {
                    "user_id": conversation.user_id,
                    "channel_id": conversation.channel_id,
                    "thread_id": conversation.thread_id,
                    "messages": conversation.messages,
                    "last_active": conversation.last_active,
                    "context_map": conversation.context_map,
                    "thinking_history": conversation.thinking_history,
                    "related_threads": conversation.related_threads,
                    "summary": conversation.summary,
                    "topic": conversation.topic
                }
                
            # Save to file
            filename = os.path.join(self.storage_dir, "context_state.json")
            async with asyncio.Lock():
                with open(filename, "w") as f:
                    json.dump(state, f, indent=2)
                    
            logger.info(f"Context state saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving context state: {e}")
            return False
            
    async def load_from_disk(self) -> bool:
        """Load the context manager state from disk"""
        try:
            filename = os.path.join(self.storage_dir, "context_state.json")
            if not os.path.exists(filename):
                logger.info(f"No context state file found at {filename}")
                return False
                
            async with asyncio.Lock():
                with open(filename, "r") as f:
                    state = json.load(f)
                    
            # Process thread map
            self.thread_map = state.get("thread_map", {})
            
            # Process conversations
            for key, conv_data in state.get("conversations", {}).items():
                # Create conversation object
                conversation = ConversationThread(
                    conv_data["user_id"],
                    conv_data["channel_id"],
                    conv_data.get("thread_id")
                )
                
                # Restore conversation data
                conversation.messages = conv_data.get("messages", [])
                conversation.last_active = conv_data.get("last_active", time.time())
                conversation.context_map = conv_data.get("context_map", {})
                conversation.thinking_history = conv_data.get("thinking_history", [])
                conversation.related_threads = conv_data.get("related_threads", [])
                conversation.summary = conv_data.get("summary", "")
                conversation.topic = conv_data.get("topic", "")
                
                # Add to conversations dictionary
                self.conversations[key] = conversation
                
                # Rebuild thinking index
                for i, thinking in enumerate(conversation.thinking_history):
                    if "problem" in thinking:
                        problem = thinking["problem"].lower()
                        keywords = set(problem.split())
                        for keyword in keywords:
                            if len(keyword) > 3:
                                self.thinking_index[keyword].append((
                                    conversation.user_id,
                                    conversation.channel_id,
                                    conversation.thread_id,
                                    i
                                ))
                
            logger.info(f"Context state loaded from {filename} ({len(self.conversations)} conversations)")
            return True
        except Exception as e:
            logger.error(f"Error loading context state: {e}")
            return False
            
    async def scheduled_save(self, interval: int = 300) -> None:
        """Periodically save context to disk"""
        while True:
            await asyncio.sleep(interval)
            await self.save_to_disk()
            
    async def cleanup_old_conversations(self, max_age: int = 86400) -> None:
        """Remove old conversations to prevent memory bloat"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, conversation in self.conversations.items():
            if current_time - conversation.last_active > max_age:
                keys_to_remove.append(key)
                
                # Also remove thread mapping if it exists
                if conversation.thread_id and conversation.thread_id in self.thread_map:
                    del self.thread_map[conversation.thread_id]
        
        # Remove the identified conversations
        for key in keys_to_remove:
            del self.conversations[key]
            
        if keys_to_remove:
            logger.info(f"Removed {len(keys_to_remove)} old conversations")
            
        # Schedule next cleanup
        await asyncio.sleep(3600)  # Check once per hour
        asyncio.create_task(self.cleanup_old_conversations(max_age))

# Global instance for singleton pattern
_context_manager = None

def create_context_manager(storage_dir: str = "bot_data/context") -> ContextManager:
    """Create or get the global context manager instance"""
    global _context_manager
    
    if _context_manager is None:
        _context_manager = ContextManager(storage_dir=storage_dir)
        
    return _context_manager

async def start_context_manager_tasks(manager: Optional[ContextManager] = None) -> None:
    """Start background tasks for the context manager"""
    if manager is None:
        manager = create_context_manager()
        
    # Load existing context
    await manager.load_from_disk()
    
    # Start scheduled tasks
    asyncio.create_task(manager.scheduled_save())
    asyncio.create_task(manager.cleanup_old_conversations()) 