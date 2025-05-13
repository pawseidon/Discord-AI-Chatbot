import logging
import json
import time
import asyncio
import os
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import uuid

# Set up logging
logger = logging.getLogger("state_manager")
logger.setLevel(logging.INFO)

class StateManager:
    """
    Manages and persists state across user interactions and reasoning steps.
    
    Responsibilities:
    - Track reasoning progress during complex queries
    - Store conversation history with metadata
    - Summarize conversation state for context-aware responses
    - Maintain cross-session state for ongoing conversations
    - Support persistence to different storage backends
    """
    def __init__(self, storage_dir: str = "bot_data/states", 
                 db_backend: str = "file",
                 max_history_length: int = 20,
                 auto_persist: bool = True,
                 auto_cleanup: bool = True):
        """
        Initialize the state manager
        
        Args:
            storage_dir: Directory to persist state to
            db_backend: Storage backend ('file', 'sqlite', 'memory')
            max_history_length: Maximum history entries per session
            auto_persist: Automatically persist state to storage
            auto_cleanup: Automatically clean up old sessions and reasoning
        """
        # Sessions keyed by session_id
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Active reasoning states for ongoing complex queries
        self.reasoning_states: Dict[str, Dict[str, Any]] = {}
        
        # Cache of recent states for quick access
        self.state_cache: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.storage_dir = storage_dir
        self.db_backend = db_backend
        self.max_history_length = max_history_length
        self.auto_persist = auto_persist
        self.auto_cleanup = auto_cleanup
        
        # Ensure storage directory exists
        if db_backend == "file" and not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)
            
        # Load existing sessions if available
        self._load_state()
        
        # Set up auto-cleanup if enabled
        if auto_cleanup:
            asyncio.create_task(self._setup_cleanup_task())
        
    async def _setup_cleanup_task(self):
        """Set up periodic cleanup task"""
        while True:
            await asyncio.sleep(3600)  # Run every hour
            await self.cleanup_old_sessions(24)
            await self.cleanup_completed_reasoning(1)
            
    def _load_state(self):
        """Load state from storage backend"""
        if self.db_backend == "memory":
            return  # No need to load anything
            
        if self.db_backend == "file":
            try:
                sessions_path = os.path.join(self.storage_dir, "sessions.pkl")
                reasoning_path = os.path.join(self.storage_dir, "reasoning.pkl")
                
                if os.path.exists(sessions_path):
                    with open(sessions_path, "rb") as f:
                        self.sessions = pickle.load(f)
                        
                if os.path.exists(reasoning_path):
                    with open(reasoning_path, "rb") as f:
                        self.reasoning_states = pickle.load(f)
                        
                logger.info(f"Loaded {len(self.sessions)} sessions and {len(self.reasoning_states)} reasoning states")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        
        # TODO: Add support for SQLite or other database backends
                
    def _persist_state(self):
        """Persist state to storage backend"""
        if not self.auto_persist:
            return
            
        if self.db_backend == "memory":
            return  # No need to persist anything
            
        if self.db_backend == "file":
            try:
                sessions_path = os.path.join(self.storage_dir, "sessions.pkl")
                reasoning_path = os.path.join(self.storage_dir, "reasoning.pkl")
                
                with open(sessions_path, "wb") as f:
                    pickle.dump(self.sessions, f)
                    
                with open(reasoning_path, "wb") as f:
                    pickle.dump(self.reasoning_states, f)
            except Exception as e:
                logger.error(f"Error persisting state: {e}")
        
        # TODO: Add support for SQLite or other database backends
        
    async def create_session(self, user_id: str, channel_id: str = None, guild_id: str = None, session_id: str = None) -> str:
        """Create a new session and return its ID"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Initialize the session data
        self.sessions[session_id] = {
            "user_id": user_id,
            "channel_id": channel_id,
            "guild_id": guild_id,
            "created_at": timestamp,
            "last_active": timestamp,
            "history": [],
            "metadata": {},
            "state": {},
            "active_reasoning": None,
            "reasoning_history": [],
            "tags": set()
        }
        
        logger.info(f"Created new session {session_id} for user {user_id}")
        
        # Persist state
        self._persist_state()
        
        return session_id
    
    async def get_or_create_session(self, user_id: str, channel_id: str = None, guild_id: str = None) -> str:
        """Get an existing session for a user or create a new one"""
        # Check for existing active session for this user in this channel
        for session_id, session in self.sessions.items():
            if (session["user_id"] == user_id and 
                session["channel_id"] == channel_id and
                time.time() - session["last_active"] < 3600):  # Active in last hour
                
                # Update last active timestamp
                session["last_active"] = time.time()
                return session_id
        
        # No active session found, create new one
        return await self.create_session(user_id, channel_id, guild_id)
    
    async def add_message(self, 
                     session_id: str, 
                     message: str, 
                     role: str = "user", 
                     metadata: Dict[str, Any] = None) -> None:
        """Add a message to the session history"""
        if session_id not in self.sessions:
            # Try to auto-create the session using session_id format (user_id:channel_id)
            try:
                if ":" in session_id:
                    user_id, channel_id = session_id.split(":", 1)
                    logger.info(f"Auto-creating session {session_id} for message")
                    await self.create_session(user_id, channel_id, session_id=session_id)
                else:
                    logger.warning(f"Attempted to add message to unknown session {session_id}")
                    return
            except Exception as e:
                logger.error(f"Failed to auto-create session: {e}")
                return
        
        # Get the session
        session = self.sessions[session_id]
        
        # Update last active timestamp
        session["last_active"] = time.time()
        
        # Create message entry
        message_entry = {
            "role": role,
            "content": message,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Add to history
        session["history"].append(message_entry)
        
        # Trim history if it exceeds max length
        if len(session["history"]) > self.max_history_length:
            session["history"] = session["history"][-self.max_history_length:]
            
        # Persist state
        self._persist_state()
    
    async def get_history(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        if session_id not in self.sessions:
            logger.warning(f"Attempted to get history for unknown session {session_id}")
            return []
        
        # Get the full history
        history = self.sessions[session_id]["history"]
        
        # Apply limit if provided
        if limit and limit > 0:
            history = history[-limit:]
            
        return history
    
    async def format_history_for_llm(self, session_id: str, limit: int = None) -> List[Dict[str, str]]:
        """Format history in a format suitable for LLM context"""
        raw_history = await self.get_history(session_id, limit)
        
        # Format for LLM consumption
        formatted_history = []
        for entry in raw_history:
            formatted_entry = {
                "role": entry["role"],
                "content": entry["content"]
            }
            formatted_history.append(formatted_entry)
            
        return formatted_history
    
    async def start_reasoning(self, 
                         session_id: str, 
                         query: str, 
                         method: str = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Start a reasoning process and return a reasoning ID"""
        reasoning_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Initialize reasoning state
        self.reasoning_states[reasoning_id] = {
            "session_id": session_id,
            "query": query,
            "method": method,
            "started_at": timestamp,
            "last_updated": timestamp,
            "steps": [],
            "status": "in_progress",
            "result": None,
            "metadata": metadata or {},
            "memory": {},
            "tools_used": []
        }
        
        # Link reasoning process to session
        if session_id in self.sessions:
            self.sessions[session_id]["active_reasoning"] = reasoning_id
            if "reasoning_history" not in self.sessions[session_id]:
                self.sessions[session_id]["reasoning_history"] = []
            self.sessions[session_id]["reasoning_history"].append(reasoning_id)
        
        logger.info(f"Started reasoning {reasoning_id} for session {session_id} using method {method}")
        
        # Persist state
        self._persist_state()
        
        return reasoning_id
    
    async def add_reasoning_step(self, 
                           reasoning_id: str, 
                           step_type: str, 
                           content: str, 
                           metadata: Dict[str, Any] = None) -> None:
        """Add a step to an ongoing reasoning process"""
        if reasoning_id not in self.reasoning_states:
            logger.warning(f"Attempted to add step to unknown reasoning process {reasoning_id}")
            return
        
        # Get the reasoning state
        reasoning = self.reasoning_states[reasoning_id]
        
        # Update last updated timestamp
        reasoning["last_updated"] = time.time()
        
        # Create step entry
        step_entry = {
            "type": step_type,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Add to steps
        reasoning["steps"].append(step_entry)
        
        # Track tool usage if step is a tool
        if step_type == "tool":
            tool_name = metadata.get("tool_name") if metadata else None
            if tool_name and tool_name not in reasoning["tools_used"]:
                reasoning["tools_used"].append(tool_name)
        
        # Persist state
        self._persist_state()
    
    async def update_reasoning_memory(self, reasoning_id: str, key: str, value: Any) -> None:
        """Update the memory for an ongoing reasoning process"""
        if reasoning_id not in self.reasoning_states:
            logger.warning(f"Attempted to update memory for unknown reasoning process {reasoning_id}")
            return
            
        self.reasoning_states[reasoning_id]["memory"][key] = value
        self.reasoning_states[reasoning_id]["last_updated"] = time.time()
        
        # Persist state
        self._persist_state()
    
    async def get_reasoning_memory(self, reasoning_id: str, key: str) -> Any:
        """Get a memory value from a reasoning process"""
        if reasoning_id not in self.reasoning_states:
            logger.warning(f"Attempted to get memory for unknown reasoning process {reasoning_id}")
            return None
            
        return self.reasoning_states[reasoning_id]["memory"].get(key)
    
    async def complete_reasoning(self, 
                           reasoning_id: str, 
                           result: str, 
                           success: bool = True) -> None:
        """Mark a reasoning process as complete with result"""
        if reasoning_id not in self.reasoning_states:
            logger.warning(f"Attempted to complete unknown reasoning process {reasoning_id}")
            return
        
        # Get the reasoning state
        reasoning = self.reasoning_states[reasoning_id]
        
        # Update status and result
        reasoning["status"] = "completed" if success else "failed"
        reasoning["result"] = result
        reasoning["completed_at"] = time.time()
        
        # Update session to clear active reasoning if this was the active one
        session_id = reasoning["session_id"]
        if session_id in self.sessions:
            if self.sessions[session_id].get("active_reasoning") == reasoning_id:
                self.sessions[session_id]["active_reasoning"] = None
        
        logger.info(f"Completed reasoning {reasoning_id} with status {reasoning['status']}")
        
        # Persist state
        self._persist_state()
    
    async def get_reasoning_process(self, reasoning_id: str) -> Dict[str, Any]:
        """Get the current state of a reasoning process"""
        if reasoning_id not in self.reasoning_states:
            logger.warning(f"Attempted to get unknown reasoning process {reasoning_id}")
            return None
        
        return self.reasoning_states[reasoning_id]
    
    async def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get the current state for a session"""
        if session_id not in self.sessions:
            logger.warning(f"Attempted to get state for unknown session {session_id}")
            return {}
        
        return self.sessions[session_id]["state"]
    
    async def set_session_state(self, session_id: str, state: Dict[str, Any]) -> None:
        """Set the state for a session"""
        if session_id not in self.sessions:
            logger.warning(f"Attempted to set state for unknown session {session_id}")
            return
        
        self.sessions[session_id]["state"] = state
        self.sessions[session_id]["last_active"] = time.time()
        
        # Persist state
        self._persist_state()
    
    async def update_session_state(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update parts of the session state"""
        if session_id not in self.sessions:
            logger.warning(f"Attempted to update state for unknown session {session_id}")
            return
        
        # Get current state
        current_state = self.sessions[session_id]["state"]
        
        # Apply updates
        current_state.update(updates)
        self.sessions[session_id]["last_active"] = time.time()
        
        # Persist state
        self._persist_state()
        
    async def add_session_tag(self, session_id: str, tag: str) -> None:
        """Add a tag to a session for categorization"""
        if session_id not in self.sessions:
            logger.warning(f"Attempted to add tag to unknown session {session_id}")
            return
            
        if "tags" not in self.sessions[session_id]:
            self.sessions[session_id]["tags"] = set()
            
        self.sessions[session_id]["tags"].add(tag)
        
        # Persist state
        self._persist_state()
        
    async def get_sessions_by_tag(self, tag: str) -> List[str]:
        """Get all session IDs with a specific tag"""
        matching_sessions = []
        
        for session_id, session in self.sessions.items():
            tags = session.get("tags", set())
            if tag in tags:
                matching_sessions.append(session_id)
                
        return matching_sessions
    
    async def summarize_conversation(self, session_id: str, max_length: int = 200) -> str:
        """Generate a summary of the conversation so far"""
        if session_id not in self.sessions:
            logger.warning(f"Attempted to summarize unknown session {session_id}")
            return ""
        
        history = self.sessions[session_id]["history"]
        
        # If history is short, no need to summarize
        if len(history) <= 3:
            return " ".join([entry["content"] for entry in history])
        
        # Otherwise, create a summary of the conversation
        # For now we'll do a simple truncation, but this could be replaced
        # with a call to an LLM to generate a more sophisticated summary
        
        combined_text = " ".join([
            f"{entry['role']}: {entry['content']}" 
            for entry in history
        ])
        
        if len(combined_text) <= max_length:
            return combined_text
        
        return combined_text[:max_length] + "..."
    
    async def get_active_reasoning_for_session(self, session_id: str) -> Optional[str]:
        """Get the currently active reasoning process for a session"""
        if session_id not in self.sessions:
            return None
            
        return self.sessions[session_id].get("active_reasoning")
    
    async def get_session_for_user(self, user_id: str, channel_id: str = None) -> Optional[str]:
        """Get the most recent session for a user, optionally in a specific channel"""
        matching_sessions = []
        
        for session_id, session in self.sessions.items():
            if session["user_id"] == user_id:
                if channel_id is None or session["channel_id"] == channel_id:
                    matching_sessions.append((session_id, session["last_active"]))
        
        if not matching_sessions:
            return None
            
        # Return the most recent session
        matching_sessions.sort(key=lambda x: x[1], reverse=True)
        return matching_sessions[0][0]
    
    async def get_recent_messages(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from a session's history"""
        if session_id not in self.sessions:
            # Try to auto-create the session using session_id format (user_id:channel_id)
            try:
                if ":" in session_id:
                    user_id, channel_id = session_id.split(":", 1)
                    logger.info(f"Auto-creating session {session_id} for get_recent_messages")
                    await self.create_session(user_id, channel_id, session_id=session_id)
                else:
                    logger.warning(f"Attempted to get messages from unknown session {session_id}")
                    return []
            except Exception as e:
                logger.error(f"Failed to auto-create session: {e}")
                return []
            
        history = self.sessions[session_id].get("history", [])
        return history[-limit:] if limit > 0 else history
    
    async def set_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> None:
        """Set metadata for a session"""
        if session_id not in self.sessions:
            # Try to auto-create the session using session_id format (user_id:channel_id)
            try:
                if ":" in session_id:
                    user_id, channel_id = session_id.split(":", 1)
                    logger.info(f"Auto-creating session {session_id} for set_session_metadata")
                    await self.create_session(user_id, channel_id, session_id=session_id)
                else:
                    logger.warning(f"Attempted to set metadata for unknown session {session_id}")
                    return
            except Exception as e:
                logger.error(f"Failed to auto-create session: {e}")
                return
            
        self.sessions[session_id]["metadata"] = metadata
        self.sessions[session_id]["last_active"] = time.time()
        
        # Persist state
        self._persist_state()
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up sessions older than the specified age"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if current_time - session["last_active"] > max_age_seconds:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
            self._persist_state()
            
        return len(sessions_to_remove)
    
    async def cleanup_completed_reasoning(self, max_age_hours: int = 1) -> int:
        """Clean up completed reasoning processes older than the specified age"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        reasoning_to_remove = []
        
        for reasoning_id, reasoning in self.reasoning_states.items():
            if reasoning["status"] in ["completed", "failed"]:
                if "completed_at" in reasoning and current_time - reasoning["completed_at"] > max_age_seconds:
                    reasoning_to_remove.append(reasoning_id)
        
        for reasoning_id in reasoning_to_remove:
            del self.reasoning_states[reasoning_id]
        
        if reasoning_to_remove:
            logger.info(f"Cleaned up {len(reasoning_to_remove)} old reasoning processes")
            self._persist_state()
            
        return len(reasoning_to_remove)
        
    async def export_session_data(self, session_id: str) -> Dict[str, Any]:
        """Export session data for transfer or analysis"""
        if session_id not in self.sessions:
            logger.warning(f"Attempted to export unknown session {session_id}")
            return {}
            
        session_data = self.sessions[session_id].copy()
        
        # Get all reasoning processes for this session
        reasoning_processes = {}
        for reasoning_id, reasoning in self.reasoning_states.items():
            if reasoning["session_id"] == session_id:
                reasoning_processes[reasoning_id] = reasoning.copy()
                
        return {
            "session": session_data,
            "reasoning_processes": reasoning_processes,
            "exported_at": time.time()
        }
        
    async def import_session_data(self, data: Dict[str, Any]) -> Optional[str]:
        """Import session data from export"""
        try:
            session_data = data["session"]
            reasoning_processes = data.get("reasoning_processes", {})
            
            # Generate new IDs to avoid collisions
            old_session_id = session_data["id"] if "id" in session_data else None
            new_session_id = str(uuid.uuid4())
            
            # Update session ID in session data
            session_data["id"] = new_session_id
            
            # Store session
            self.sessions[new_session_id] = session_data
            
            # Map of old reasoning IDs to new ones
            reasoning_id_map = {}
            
            # Store reasoning processes with new IDs
            for old_reasoning_id, reasoning_data in reasoning_processes.items():
                new_reasoning_id = str(uuid.uuid4())
                reasoning_id_map[old_reasoning_id] = new_reasoning_id
                
                # Update session ID reference
                reasoning_data["session_id"] = new_session_id
                
                # Store with new ID
                self.reasoning_states[new_reasoning_id] = reasoning_data
            
            # Update references in session
            if "active_reasoning" in session_data and session_data["active_reasoning"] in reasoning_id_map:
                session_data["active_reasoning"] = reasoning_id_map[session_data["active_reasoning"]]
                
            if "reasoning_history" in session_data:
                new_history = []
                for old_id in session_data["reasoning_history"]:
                    if old_id in reasoning_id_map:
                        new_history.append(reasoning_id_map[old_id])
                session_data["reasoning_history"] = new_history
            
            # Persist changes
            self._persist_state()
            
            logger.info(f"Imported session data to new session ID {new_session_id}")
            return new_session_id
            
        except Exception as e:
            logger.error(f"Error importing session data: {e}")
            return None

def create_state_manager(storage_dir: str = "bot_data/states", 
                        db_backend: str = "file",
                        max_history_length: int = 20) -> StateManager:
    """Create and initialize a StateManager instance"""
    return StateManager(
        storage_dir=storage_dir,
        db_backend=db_backend,
        max_history_length=max_history_length
    ) 