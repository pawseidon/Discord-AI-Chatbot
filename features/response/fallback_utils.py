import os
import json
import time
import random
import sqlite3
import re
import discord
from pathlib import Path

# Set up fallback storage
FALLBACK_DIR = "bot_data/fallbacks"
os.makedirs(FALLBACK_DIR, exist_ok=True)
FALLBACK_DB = os.path.join(FALLBACK_DIR, "fallback.db")

# Custom REGEXP function for SQLite
def regexp(pattern, text):
    try:
        return re.search(pattern, text, re.IGNORECASE) is not None
    except Exception:
        return False

# Create database if it doesn't exist
def initialize_fallback_db():
    """Initialize the fallback database"""
    conn = sqlite3.connect(FALLBACK_DB)
    
    # Register the REGEXP function
    try:
        conn.create_function("REGEXP", 2, regexp)
    except Exception as e:
        print(f"Warning: Could not register REGEXP function: {e}")
        
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS common_responses (
        id INTEGER PRIMARY KEY,
        query_pattern TEXT NOT NULL,
        response TEXT NOT NULL,
        category TEXT,
        created_at REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cached_responses (
        id INTEGER PRIMARY KEY,
        query_hash TEXT NOT NULL UNIQUE,
        query TEXT NOT NULL,
        response TEXT NOT NULL,
        used_count INTEGER DEFAULT 1,
        last_used REAL
    )
    ''')
    
    # Create index for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS query_pattern_idx ON common_responses(query_pattern)')
    cursor.execute('CREATE INDEX IF NOT EXISTS query_hash_idx ON cached_responses(query_hash)')
    
    # Insert some default common responses if table is empty
    cursor.execute('SELECT COUNT(*) FROM common_responses')
    if cursor.fetchone()[0] == 0:
        default_responses = [
            ("hello|hi|hey|greetings", "Hello! I'm here to help. What can I assist you with today?", "greeting", time.time()),
            ("how are you|how do you feel|how are you doing", "I'm just a bot, but I'm functioning well and ready to assist you!", "greeting", time.time()),
            ("thank you|thanks|thx", "You're welcome! Is there anything else I can help with?", "gratitude", time.time()),
            ("help|assistance|support", "I'm here to help! You can ask me questions, and I'll do my best to assist you.", "help", time.time()),
            ("bye|goodbye|see you|farewell", "Goodbye! Feel free to reach out if you need assistance in the future.", "farewell", time.time())
        ]
        cursor.executemany(
            'INSERT INTO common_responses (query_pattern, response, category, created_at) VALUES (?, ?, ?, ?)',
            default_responses
        )
    
    conn.commit()
    conn.close()

# Initialize the database
initialize_fallback_db()

def simple_hash(text):
    """Create a simple hash of text for caching purposes"""
    text = text.lower().strip()
    return str(hash(text))

async def get_fallback_response(query, username=None):
    """
    Get a fallback response when the LLM is unavailable
    
    Args:
        query (str): The user's query
        username (str, optional): The username for personalization
        
    Returns:
        str: A fallback response
    """
    query = query.lower().strip()
    query_hash = simple_hash(query)
    
    # Try to find a cached exact response first
    conn = sqlite3.connect(FALLBACK_DB)
    
    # Register the REGEXP function
    try:
        conn.create_function("REGEXP", 2, regexp)
    except Exception as e:
        print(f"Warning: Could not register REGEXP function in query: {e}")
        
    cursor = conn.cursor()
    
    cursor.execute('SELECT response FROM cached_responses WHERE query_hash = ?', (query_hash,))
    cached = cursor.fetchone()
    
    if cached:
        # Update usage stats
        cursor.execute(
            'UPDATE cached_responses SET used_count = used_count + 1, last_used = ? WHERE query_hash = ?',
            (time.time(), query_hash)
        )
        conn.commit()
        conn.close()
        return cached[0]
    
    # Try to match against common patterns using REGEXP or fallback to simple matching
    pattern_match = None
    try:
        cursor.execute('SELECT response, query_pattern FROM common_responses WHERE ? REGEXP query_pattern LIMIT 1', (query,))
        pattern_match = cursor.fetchone()
    except sqlite3.OperationalError as e:
        if "no such function: REGEXP" in str(e):
            # Fallback to simple substring matching if REGEXP is not available
            print("REGEXP function not available, falling back to simple matching")
            cursor.execute('SELECT response, query_pattern FROM common_responses')
            for row in cursor.fetchall():
                response, pattern = row
                patterns = pattern.split('|')
                if any(p in query for p in patterns):
                    pattern_match = (response, pattern)
                    break
        else:
            raise e
    
    if pattern_match:
        response = pattern_match[0]
        if username:
            response = response.replace("{username}", username)
        conn.close()
        return response
        
    # No match found, use a generic response
    conn.close()
    
    generic_responses = [
        "I'm having trouble connecting to my language model at the moment. Could you try again later?",
        "I apologize, but my main AI service seems to be unavailable right now. I've noted your question and will try to respond properly when the service is back online.",
        "It seems I'm experiencing some technical difficulties. Could you please rephrase your question or try again in a few minutes?",
        "I'm currently in fallback mode due to connection issues with my main AI system. I can only provide basic responses at the moment.",
        f"I'm sorry, but I can't provide a complete answer right now due to technical limitations. {('I have made a note of your question, ' + username + '.') if username else 'Please try again later.'}"
    ]
    
    return random.choice(generic_responses)

def cache_successful_response(query, response):
    """
    Cache a successful query-response pair for future fallbacks
    
    Args:
        query (str): The user's query
        response (str): The successful response
    """
    query = query.strip()
    if not query or not response or len(query) < 5:
        return False
        
    query_hash = simple_hash(query)
    
    conn = sqlite3.connect(FALLBACK_DB)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            'INSERT OR REPLACE INTO cached_responses (query_hash, query, response, used_count, last_used) VALUES (?, ?, ?, 1, ?)',
            (query_hash, query, response, time.time())
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error caching response: {e}")
        return False
    finally:
        conn.close()

def is_offline_mode():
    """Check if we should be in offline mode (i.e., LLM unavailable)"""
    # Check for manual offline flag file
    offline_flag = os.path.join(FALLBACK_DIR, "offline_mode")
    if os.path.exists(offline_flag):
        return True
        
    # Check if we've had recent failed attempts
    failed_attempts_file = os.path.join(FALLBACK_DIR, "failed_attempts.json")
    if os.path.exists(failed_attempts_file):
        try:
            with open(failed_attempts_file, 'r') as f:
                failed_data = json.load(f)
                
            # If we've had multiple recent failures, enable offline mode
            recent_failures = failed_data.get('recent_failures', 0)
            last_failure = failed_data.get('last_failure', 0)
            
            # If we've had 3+ failures in the last 5 minutes, go offline
            if recent_failures >= 3 and (time.time() - last_failure) < 300:
                return True
                
        except Exception:
            pass
    
    return False

def record_llm_failure():
    """Record a failed attempt to connect to the LLM"""
    failed_attempts_file = os.path.join(FALLBACK_DIR, "failed_attempts.json")
    
    failed_data = {
        'recent_failures': 1,
        'last_failure': time.time()
    }
    
    if os.path.exists(failed_attempts_file):
        try:
            with open(failed_attempts_file, 'r') as f:
                existing_data = json.load(f)
                
            # If last failure was within 5 minutes, increment counter
            if (time.time() - existing_data.get('last_failure', 0)) < 300:
                failed_data['recent_failures'] = existing_data.get('recent_failures', 0) + 1
            # Otherwise start fresh
        except Exception:
            pass
    
    with open(failed_attempts_file, 'w') as f:
        json.dump(failed_data, f)

def record_llm_success():
    """Record a successful connection to the LLM"""
    failed_attempts_file = os.path.join(FALLBACK_DIR, "failed_attempts.json")
    
    # Reset failure counter
    failed_data = {
        'recent_failures': 0,
        'last_failure': 0
    }
    
    with open(failed_attempts_file, 'w') as f:
        json.dump(failed_data, f)
        
    # Remove offline flag if it exists
    offline_flag = os.path.join(FALLBACK_DIR, "offline_mode")
    if os.path.exists(offline_flag):
        try:
            os.remove(offline_flag)
        except Exception:
            pass

class FallbackHandler:
    """Handler for providing fallback responses when the LLM is unavailable"""
    
    @staticmethod
    async def handle_message(message, bot_name=None):
        """
        Process a message and return a fallback response
        
        Args:
            message (discord.Message): The Discord message
            bot_name (str, optional): The bot's name
            
        Returns:
            str: A fallback response
        """
        query = message.content
        username = message.author.display_name if message.author else None
        
        # Check if this is addressed to the bot
        is_addressed_to_bot = False
        if bot_name and bot_name.lower() in query.lower():
            is_addressed_to_bot = True
            query = query.lower().replace(bot_name.lower(), "").strip()
            
        if is_addressed_to_bot or is_offline_mode():
            response = await get_fallback_response(query, username)
            
            if bot_name:
                # Add a prefix to make it clear we're in fallback mode
                return f"[{bot_name} - Fallback Mode] {response}"
            else:
                return response
                
        return None  # No fallback needed 