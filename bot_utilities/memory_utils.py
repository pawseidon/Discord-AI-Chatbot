import json
import os
import time
import asyncio
from openai import AsyncOpenAI
from bot_utilities.config_loader import config
from bot_utilities.ai_utils import get_local_client, get_remote_client

# Initialize paths for storing data
DATA_DIR = "bot_data"
USER_PREFS_DIR = os.path.join(DATA_DIR, "user_preferences")
MEMORY_DIR = os.path.join(DATA_DIR, "memory_summaries")

# Ensure directories exist
os.makedirs(USER_PREFS_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

# Initialize the clients
local_client = None
remote_client = None

class ConversationSummarizer:
    """Handles summarizing long conversations to maintain context while reducing token usage"""
    
    @staticmethod
    async def summarize_conversation(conversation_history, threshold=8):
        """
        Summarize conversation when it exceeds the threshold length
        Args:
            conversation_history (list): List of message dictionaries
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
        
        # Get the summarization model
        global local_client, remote_client
        if config.get('USE_LOCAL_MODEL', False):
            if local_client is None:
                local_client = get_local_client()
            client = local_client
        else:
            if remote_client is None:
                remote_client = get_remote_client()
            client = remote_client
            
        try:
            # Generate summary using the LLM
            response = await client.chat.completions.create(
                model=config.get('LOCAL_MODEL_ID', 'mistral-nemo-instruct-2407') if config.get('USE_LOCAL_MODEL', False) else config['MODEL_ID'],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Please summarize this conversation:\n\n{formatted_convo}"}
                ]
            )
            
            summary = response.choices[0].message.content
            
            # Create a new conversation history with the summary and most recent messages
            new_history = [
                # Keep the very first message which usually contains the user's initial query
                conversation_history[0], 
                # Add our summary as a system message
                {"role": "system", "content": f"[Summary of previous conversation: {summary}]"},
                # Keep the most recent messages (last 3)
                *conversation_history[-3:]
            ]
            
            print(f"Summarized conversation from {len(conversation_history)} messages to {len(new_history)}")
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
                "use_embeds": False,  # Default to not using embeds for more conversational feel
                "use_streaming": True,  # Enable streaming responses by default for better UX
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

async def process_conversation_history(message_history, user_id, channel_id):
    """
    Process and potentially summarize conversation history
    Also track user interests based on the conversation
    
    Args:
        message_history (dict): The full message history dictionary
        user_id (str): The user's ID
        channel_id (str): The channel ID
    
    Returns:
        list: The updated conversation history for this user-channel pair
    """
    key = f"{user_id}-{channel_id}"
    
    # If no history exists, return empty list
    if key not in message_history or not message_history[key]:
        return []
    
    # Get the conversation history for this user-channel pair
    conversation = message_history[key]
    
    # Track user interests based on their last message (if it's from the user)
    if conversation and conversation[-1]["role"] == "user":
        await UserPreferences.track_topic_interest(user_id, conversation[-1]["content"])
    
    # If conversation is getting long, summarize it
    max_history = config.get('MAX_HISTORY', 8)
    if len(conversation) > max_history + 2:  # Allow some buffer over the max
        conversation = await ConversationSummarizer.summarize_conversation(conversation, threshold=max_history)
        # Update the history with the summarized version
        message_history[key] = conversation
    
    return conversation

def get_enhanced_instructions(instructions, user_id):
    """
    Enhance instructions with user preferences and personalization
    
    Args:
        instructions (str): The base instructions
        user_id (str): The user's ID
    
    Returns:
        str: Enhanced instructions with user preferences
    """
    # This will be run synchronously, so we use a helper function to load preferences
    def load_prefs():
        pref_file = os.path.join(USER_PREFS_DIR, f"{user_id}.json")
        if os.path.exists(pref_file):
            try:
                with open(pref_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    preferences = load_prefs()
    
    # If we have preferences, enhance the instructions
    if preferences:
        topics = ", ".join(preferences.get("topics_of_interest", []))
        response_length = preferences.get("preferred_response_length", "medium")
        
        # Add personalization based on preferences
        personalization = "\n\nAdditional personalization information:"
        
        if topics:
            personalization += f"\n- The user has shown interest in these topics: {topics}"
        
        if response_length == "short":
            personalization += "\n- The user prefers shorter, concise responses. Be direct and to the point."
        elif response_length == "long":
            personalization += "\n- The user prefers detailed explanations. You can be more thorough in your responses."
        
        return instructions + personalization
    
    # If no preferences, return the original instructions
    return instructions 