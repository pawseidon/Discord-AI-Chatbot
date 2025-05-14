"""
AI utilities module for Discord AI Chatbot.

This module provides AI client management, API connectivity,
and utility functions for AI integration.
"""

import aiohttp
import io
import time
import os
import random
import json
import asyncio
import re
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from urllib.parse import quote

# Third-party imports
try:
    from langdetect import detect
except ImportError:
    detect = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    import httpx
except ImportError:
    httpx = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from dotenv import load_dotenv

# Local imports (using relative imports for the new structure)
from .config_loader import get_config, load_current_language, load_instructions
from ..utils.token_utils import count_tokens

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("ai_utils")

# Cache for storing the current timestamp
timestamp_cache = {
    "last_updated": time.time(),
    "current_date": datetime.now().strftime("%Y-%m-%d"),
    "current_time": datetime.now().strftime("%H:%M:%S"),
    "current_year": datetime.now().year,
    "current_month": datetime.now().month,
    "current_day": datetime.now().day,
    "update_interval": 60  # Update every 60 seconds
}

def update_timestamp_cache() -> Dict[str, Any]:
    """
    Update the timestamp cache if it's older than the update interval
    
    Returns:
        Updated timestamp cache dictionary
    """
    current_time = time.time()
    if current_time - timestamp_cache["last_updated"] > timestamp_cache["update_interval"]:
        now = datetime.now()
        timestamp_cache["last_updated"] = current_time
        timestamp_cache["current_date"] = now.strftime("%Y-%m-%d")
        timestamp_cache["current_time"] = now.strftime("%H:%M:%S")
        timestamp_cache["current_year"] = now.year
        timestamp_cache["current_month"] = now.month
        timestamp_cache["current_day"] = now.day
        logger.info(f"Updated timestamp cache: {timestamp_cache['current_date']} {timestamp_cache['current_time']}")
    
    return timestamp_cache

def get_bot_names_and_triggers() -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Get the configured bot names and trigger words
    
    Returns:
        Tuple: (list of bot names, list of trigger words, list of prefixes, list of suffixes)
    """
    # Names should include the configured display name and common variants
    bot_names = [
        get_config('DISPLAY_NAME', 'Assistant').lower(),
        'assistant',
        'bot',
        'ai',
        'you',
        'hand',  # Add the default bot name from config (Hand)
    ]
    
    # Add custom names if configured
    custom_names = get_config('BOT_NAMES', [])
    if isinstance(custom_names, list) and custom_names:
        bot_names.extend([name.lower() for name in custom_names])
    
    # Deduplicate names
    bot_names = list(set(bot_names))
    
    # Trigger words that might indicate the bot is being addressed
    trigger_words = [
        'hey',
        'hi',
        'hello', 
        'help',
        'explain',
        'tell me',
        'could you',
        'can you',
        'answer',
        'solve',
        'what is',
        'how do',
        'why is',
        'assist'
    ]
    
    # Command prefixes
    prefixes = [
        '!ask',
        '/ask',
        '.ask',
        '!bot',
        '/bot',
        '!ai',
        '/ai',
        '!assistant',
        '/assistant'
    ]
    
    # Command suffixes
    suffixes = [
        '?',
        'please',
        'thanks',
        'thank you'
    ]
    
    # Add custom triggers if configured
    custom_triggers = get_config('TRIGGER_WORDS', [])
    if isinstance(custom_triggers, list) and custom_triggers:
        trigger_words.extend([trigger.lower() for trigger in custom_triggers])
    
    # Add items from TRIGGER config list too (for backward compatibility)
    trigger_config = get_config('TRIGGER', [])
    if isinstance(trigger_config, list):
        for trigger in trigger_config:
            # Skip placeholder triggers as they'll be processed separately
            if '%' not in trigger:
                trigger_words.append(trigger.lower())
    
    # Deduplicate triggers
    trigger_words = list(set(trigger_words))
    
    return bot_names, trigger_words, prefixes, suffixes

def get_local_client() -> Optional[Any]:
    """
    Initialize a client for a local LLM server
    
    Returns:
        AsyncOpenAI client configured for local use, or None if not available
    """
    if AsyncOpenAI is None:
        logger.warning("AsyncOpenAI not installed, cannot create local client")
        return None
        
    # For WSL to Windows, use the host.docker.internal hostname or the Windows IP
    host = get_config('LOCAL_MODEL_HOST', 'host.docker.internal')
    port = get_config('LOCAL_MODEL_PORT', '1234')
    base_url = f"http://{host}:{port}/v1"
    
    logger.info(f"Connecting to local model at: {base_url}")
    
    return AsyncOpenAI(
        base_url=base_url,
        api_key="not-needed"  # Local servers typically don't require an API key
    )
    
def get_remote_client() -> Optional[Any]:
    """
    Initialize a client for a remote LLM API
    
    Returns:
        AsyncOpenAI client configured for remote use, or None if not available
    """
    if AsyncOpenAI is None:
        logger.warning("AsyncOpenAI not installed, cannot create remote client")
        return None
        
    if httpx is None:
        logger.warning("httpx not installed, cannot create remote client with robust configuration")
    
    api_key = os.environ.get("API_KEY")
    logger.debug(f"API_KEY found: {bool(api_key)}")
    
    if not api_key:
        logger.warning("API_KEY not found in environment variables. Cannot use remote API.")
        return None
        
    # Get the base URL and ensure it's properly formatted
    api_base = get_config('API_BASE_URL', 'https://api.openai.com/v1')
    
    # Remove trailing slash if present
    if api_base.endswith('/'):
        api_base = api_base[:-1]
    
    # If it ends with /chat/completions, strip that off
    if api_base.endswith('/chat/completions'):
        api_base = api_base.rsplit('/chat/completions', 1)[0]
        
    logger.info(f"Connecting to remote API at: {api_base}")
    
    # Configure a robust HTTP client with retries and timeouts if httpx is available
    http_client = None
    if httpx:
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,    # 5 seconds to connect
                read=30.0,      # 30 seconds to read response
                write=10.0,     # 10 seconds to write request
                pool=5.0        # 5 seconds for connection from pool
            ),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0  # 30 seconds keepalive
            ),
            follow_redirects=True
        )
    
    return AsyncOpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=http_client, 
    )

class AIProvider:
    """Unified AI provider interface for Discord AI Chatbot"""
    
    def __init__(self, client=None, model=None):
        """
        Initialize the AI provider
        
        Args:
            client: Optional AsyncOpenAI client
            model: Optional model name to use
        """
        self.client = client
        self.model = model or get_config('MODEL_NAME', 'gpt-3.5-turbo')
        self.local_client = None
        self.remote_client = None
        self.fallback_to_remote = get_config('FALLBACK_TO_REMOTE', True)
        self.latest_response = None  # Store the latest response for diagnostics
        
    async def ensure_client(self) -> bool:
        """
        Ensure a client is available, attempting to create one if needed
        
        Returns:
            True if a client is available, False otherwise
        """
        if self.client:
            return True
            
        # Try to use local client first if specified in config
        use_local = get_config('USE_LOCAL_MODEL', False)
        
        if use_local:
            if not self.local_client:
                self.local_client = get_local_client()
            
            if self.local_client:
                self.client = self.local_client
                logger.info(f"Using local model: {self.model}")
                return True
            elif self.fallback_to_remote:
                logger.warning("Local model not available, falling back to remote")
            else:
                logger.error("Local model not available and fallback disabled")
                return False
        
        # Use remote client if local not available or not configured
        if not self.remote_client:
            self.remote_client = get_remote_client()
            
        if self.remote_client:
            self.client = self.remote_client
            logger.info(f"Using remote model: {self.model}")
            return True
            
        logger.error("No AI client available")
        return False
    
    async def async_call(self, 
                       prompt: str, 
                       temperature: float = 0.7, 
                       max_tokens: int = 1000, 
                       system_prompt: Optional[str] = None) -> str:
        """
        Make an asynchronous call to the AI model
        
        Args:
            prompt: User prompt
            temperature: Model temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            Generated response as string
        """
        if not await self.ensure_client():
            return "AI service is currently unavailable. Please try again later."
        
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            self.latest_response = response
            
            # Extract and return the message content
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message'):
                    content = response.choices[0].message.content
                    return content.strip()
                    
            # For different API formats
            if isinstance(response, dict):
                if 'choices' in response and len(response['choices']) > 0:
                    if 'message' in response['choices'][0]:
                        content = response['choices'][0]['message'].get('content', '')
                        return content.strip()
            
            # Fallback for unknown response format
            return str(response)
            
        except Exception as e:
            logger.error(f"Error calling AI API: {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"
    
    async def stream_response(self, 
                           messages: List[Dict[str, str]], 
                           temperature: float = 0.7, 
                           max_tokens: int = 1000) -> str:
        """
        Stream a response from the AI model
        
        Args:
            messages: List of message dictionaries
            temperature: Model temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generator yielding response chunks
        """
        if not await self.ensure_client():
            yield "AI service is currently unavailable. Please try again later."
            return
            
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            collected_content = []
            
            async for chunk in stream:
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    if hasattr(chunk.choices[0], 'delta'):
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            content = delta.content
                            collected_content.append(content)
                            yield content
                
                # For different API formats
                if isinstance(chunk, dict):
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        if 'delta' in chunk['choices'][0]:
                            delta = chunk['choices'][0]['delta']
                            if 'content' in delta and delta['content']:
                                content = delta['content']
                                collected_content.append(content)
                                yield content
            
            # Save the complete response for diagnostics
            self.latest_response = ''.join(collected_content)
            
        except Exception as e:
            logger.error(f"Error streaming from AI API: {str(e)}")
            yield f"An error occurred while processing your request: {str(e)}"
            
    async def search_internet(self, query: str, max_results: int = 5) -> str:
        """
        Search the internet for information
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Formatted search results
        """
        if DDGS is None:
            return "Web search functionality is not available (duckduckgo_search package not installed)."
            
        if httpx is None:
            return "Web search functionality is not available (httpx package not installed)."
            
        if BeautifulSoup is None:
            return "Web search functionality is not available (beautifulsoup4 package not installed)."
            
        try:
            # Update timestamp to ensure search has current time info
            update_timestamp_cache()
            
            # Format the search query with date info if needed for freshness
            current_year = timestamp_cache["current_year"]
            current_date = timestamp_cache["current_date"]
            
            # For time-sensitive queries, add the current year/date
            if any(term in query.lower() for term in ["latest", "news", "recent", "current", "today", "now"]):
                if str(current_year) not in query:
                    query = f"{query} {current_date}"
            
            logger.info(f"Searching internet for: {query}")
            
            # Perform search
            search_results = []
            
            # Define an async search function that will run in a thread
            async def perform_search():
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=max_results))
                    return results
                except Exception as e:
                    logger.error(f"DuckDuckGo search error: {str(e)}")
                    return []
            
            # Run search in a thread to avoid blocking
            search_results = await asyncio.to_thread(lambda: perform_search())
            
            if not search_results:
                return f"I couldn't find any relevant information for '{query}'. Please try a different search term."
            
            # Format the results
            formatted_results = f"**Search results for: {query}**\n\n"
            
            for i, result in enumerate(search_results, 1):
                title = result.get('title', 'No title')
                body = result.get('body', 'No content')
                href = result.get('href', '#')
                
                formatted_results += f"**{i}. {title}**\n"
                formatted_results += f"{body}\n"
                formatted_results += f"Source: {href}\n\n"
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching internet: {str(e)}")
            return f"An error occurred while searching for '{query}': {str(e)}"
    
    async def get_crypto_price(self, crypto_name: str) -> str:
        """
        Get current cryptocurrency price information
        
        Args:
            crypto_name: Name or symbol of cryptocurrency
            
        Returns:
            Formatted price information
        """
        try:
            # Standardize crypto name
            crypto_name = crypto_name.lower().strip()
            
            # Convert common names to symbols if needed
            name_to_symbol = {
                "bitcoin": "btc",
                "ethereum": "eth",
                "litecoin": "ltc",
                "ripple": "xrp",
                "cardano": "ada",
                "dogecoin": "doge",
                "polkadot": "dot",
                "solana": "sol",
                "binance coin": "bnb",
                "binance": "bnb",
                "tether": "usdt"
            }
            
            # Use symbol if provided crypto_name maps to one
            symbol = name_to_symbol.get(crypto_name, crypto_name)
            
            # Create API URL (using CoinGecko's public API)
            if len(symbol) < 6:  # Likely a symbol
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_name}&vs_currencies=usd,eur&include_24hr_change=true"
            else:  # Likely a name
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_name}&vs_currencies=usd,eur&include_24hr_change=true"
            
            # Make request
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        # Try alternative lookup if direct lookup failed
                        if len(symbol) < 6:
                            search_url = f"https://api.coingecko.com/api/v3/search?query={symbol}"
                        else:
                            search_url = f"https://api.coingecko.com/api/v3/search?query={crypto_name}"
                            
                        async with session.get(search_url) as search_response:
                            if search_response.status == 200:
                                search_data = await search_response.json()
                                if search_data.get("coins") and len(search_data["coins"]) > 0:
                                    first_result = search_data["coins"][0]
                                    coin_id = first_result.get("id")
                                    
                                    # Get price with the found coin ID
                                    price_url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd,eur&include_24hr_change=true"
                                    async with session.get(price_url) as price_response:
                                        if price_response.status == 200:
                                            data = await price_response.json()
                                        else:
                                            return f"Could not retrieve price for {crypto_name}. API response: {price_response.status}"
                                else:
                                    return f"Could not find cryptocurrency: {crypto_name}"
                            else:
                                return f"Could not search for {crypto_name}. API response: {search_response.status}"
                    else:
                        data = await response.json()
            
            # Format response
            if not data or len(data) == 0:
                return f"No price information found for {crypto_name}"
                
            for coin_id, price_data in data.items():
                usd_price = price_data.get("usd", "N/A")
                eur_price = price_data.get("eur", "N/A")
                usd_change = price_data.get("usd_24h_change", "N/A")
                
                # Format 24h change
                if isinstance(usd_change, (int, float)):
                    change_str = f"{usd_change:.2f}%"
                    change_direction = "📈" if usd_change >= 0 else "📉"
                else:
                    change_str = "N/A"
                    change_direction = ""
                
                result = f"**{coin_id.upper()}** Price Information:\n"
                result += f"USD: ${usd_price:,.2f}\n" if isinstance(usd_price, (int, float)) else f"USD: {usd_price}\n"
                result += f"EUR: €{eur_price:,.2f}\n" if isinstance(eur_price, (int, float)) else f"EUR: {eur_price}\n"
                result += f"24h Change: {change_str} {change_direction}\n"
                result += f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                return result
                
            return f"No price information found for {crypto_name}"
            
        except Exception as e:
            logger.error(f"Error retrieving crypto price: {str(e)}")
            return f"An error occurred while retrieving the price for {crypto_name}: {str(e)}"
    
    async def text_to_speech(self, text: str) -> Optional[io.BytesIO]:
        """
        Convert text to speech
        
        Args:
            text: Text to convert
            
        Returns:
            BytesIO object containing the audio, or None if failed
        """
        if gTTS is None:
            logger.warning("gTTS not installed, cannot perform text-to-speech")
            return None
            
        try:
            # Detect language (default to English if detection fails)
            lang = 'en'
            if detect:
                try:
                    detected_lang = detect(text)
                    # Map detected language to gTTS supported languages if needed
                    lang_mapping = {
                        'zh-cn': 'zh-CN',
                        'zh-tw': 'zh-TW',
                        'zh': 'zh-CN'
                    }
                    lang = lang_mapping.get(detected_lang, detected_lang)
                except:
                    pass
            
            # Create gTTS object
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save to BytesIO object
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            return fp
            
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {str(e)}")
            return None

# Global instance for singleton pattern
_ai_provider = None

async def get_ai_provider(client=None, model=None) -> AIProvider:
    """
    Get or create the global AI provider instance
    
    Args:
        client: Optional AsyncOpenAI client to use
        model: Optional model name to use
        
    Returns:
        AIProvider instance
    """
    global _ai_provider
    
    if _ai_provider is None:
        _ai_provider = AIProvider(client=client, model=model)
        
    return _ai_provider 