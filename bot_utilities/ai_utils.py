import aiohttp
import io
import time
import os
import random
import json
import asyncio
import re
from datetime import datetime, timezone
from langdetect import detect
from gtts import gTTS
from urllib.parse import quote
from bot_utilities.config_loader import load_current_language, config, load_instructions
from openai import AsyncOpenAI
from duckduckgo_search import DDGS  # Use current DDGS API
from dotenv import load_dotenv
from bot_utilities.news_utils import get_news_context, detect_news_query
from bot_utilities.fallback_utils import is_offline_mode, record_llm_failure, record_llm_success, cache_successful_response, get_fallback_response
import httpx
import urllib.parse
from bs4 import BeautifulSoup

load_dotenv()

current_language = load_current_language()
internet_access = config['INTERNET_ACCESS']

# Get instruction name and content for bot's persona
default_instruction_name = config.get('DEFAULT_INSTRUCTION', 'hand')
instructions_content = load_instructions()
bot_instructions = instructions_content.get(default_instruction_name, "")

# Cache for storing the current timestamp
# This will be updated periodically to ensure the bot always has current time
timestamp_cache = {
    "last_updated": time.time(),
    "current_date": datetime.now().strftime("%Y-%m-%d"),
    "current_time": datetime.now().strftime("%H:%M:%S"),
    "current_year": datetime.now().year,
    "current_month": datetime.now().month,
    "current_day": datetime.now().day,
    "update_interval": 60  # Update every 60 seconds
}

def update_timestamp_cache():
    """Update the timestamp cache if it's older than the update interval"""
    current_time = time.time()
    if current_time - timestamp_cache["last_updated"] > timestamp_cache["update_interval"]:
        now = datetime.now()
        timestamp_cache["last_updated"] = current_time
        timestamp_cache["current_date"] = now.strftime("%Y-%m-%d")
        timestamp_cache["current_time"] = now.strftime("%H:%M:%S")
        timestamp_cache["current_year"] = now.year
        timestamp_cache["current_month"] = now.month
        timestamp_cache["current_day"] = now.day
        print(f"Updated timestamp cache: {timestamp_cache['current_date']} {timestamp_cache['current_time']}")

# Extract bot name and triggers from configuration and instructions
def get_bot_names_and_triggers():
    """
    Get the configured bot names and trigger words
    
    Returns:
        Tuple: (list of bot names, list of trigger words, list of prefixes, list of suffixes)
    """
    # Names should include the configured display name and common variants
    bot_names = [
        config.get('DISPLAY_NAME', 'Assistant').lower(),
        'assistant',
        'bot',
        'ai',
        'you',
        'hand',  # Add the default bot name from config (Hand)
    ]
    
    # Add custom names if configured
    custom_names = config.get('BOT_NAMES', [])
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
    custom_triggers = config.get('TRIGGER_WORDS', [])
    if isinstance(custom_triggers, list) and custom_triggers:
        trigger_words.extend([trigger.lower() for trigger in custom_triggers])
    
    # Add items from TRIGGER config list too (for backward compatibility)
    if 'TRIGGER' in config and isinstance(config['TRIGGER'], list):
        for trigger in config['TRIGGER']:
            # Skip placeholder triggers as they'll be processed separately
            if '%' not in trigger:
                trigger_words.append(trigger.lower())
    
    # Deduplicate triggers
    trigger_words = list(set(trigger_words))
    
    return bot_names, trigger_words, prefixes, suffixes

# Initialize the client based on configuration
def get_local_client():
    # For WSL to Windows, use the host.docker.internal hostname or the Windows IP
    host = config.get('LOCAL_MODEL_HOST', 'host.docker.internal')
    port = config.get('LOCAL_MODEL_PORT', '1234')
    base_url = f"http://{host}:{port}/v1"
    
    print(f"Connecting to LM Studio at: {base_url}")
    
    return AsyncOpenAI(
        base_url=base_url,
        api_key="not-needed"  # LM Studio doesn't require an API key
    )
    
def get_remote_client():
    api_key = os.environ.get("API_KEY")
    print(f"DEBUG - API_KEY found: {bool(api_key)}")
    print(f"DEBUG - API_KEY length: {len(api_key) if api_key else 0}")
    print(f"DEBUG - API_KEY first chars: {api_key[:4]}*** last chars: ***{api_key[-4:] if api_key else ''}")
    
    if not api_key:
        print("WARNING: API_KEY not found in environment variables. Cannot use remote API.")
        return None
        
    # Get the base URL and ensure it's properly formatted
    api_base = config['API_BASE_URL']
    
    # Remove trailing slash if present
    if api_base.endswith('/'):
        api_base = api_base[:-1]
    
    # If it ends with /chat/completions, strip that off
    if api_base.endswith('/chat/completions'):
        api_base = api_base.rsplit('/chat/completions', 1)[0]
        
    print(f"Connecting to remote API at: {api_base}")
    
    # Configure a robust HTTP client with retries and timeouts
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

# Create client when needed (not at module import time)
local_client = None
remote_client = None

# Create a simple AI provider class for sequential thinking
class AIProvider:
    def __init__(self, client=None, model=None):
        self.client = client
        self.model = model or config.get('DEFAULT_MODEL', 'meta-llama/llama-4-maverick-17b-128e-instruct')
        
    async def async_call(self, prompt, temperature=0.2, max_tokens=2000):
        """Call the LLM with a prompt and return the text response"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in AIProvider.async_call: {e}")
            raise

async def get_ai_provider():
    """
    Get the appropriate AI provider client with error handling
    
    Returns:
        AIProvider: Configured AI provider client
    """
    global local_client, remote_client
    use_local = config.get('USE_LOCAL_MODEL', False)
    
    # Always update timestamp cache when getting a provider
    update_timestamp_cache()
    
    try:
        # Check for offline mode
        if is_offline_mode():
            print("Operating in offline mode. Using fallback responses.")
            return AIProvider(None, "fallback")

        if use_local:
            # Local LLM setup - lazily initialize to avoid startup delay
            if local_client is None:
                try:
                    local_client = get_local_client()
                    print("Successfully initialized local LLM client")
                except Exception as e:
                    print(f"Error initializing local client, falling back to remote: {e}")
                    use_local = False
            
            if local_client is not None:
                model = config.get('LOCAL_MODEL_NAME', 'local')
                return AIProvider(local_client, model)
        
        # Remote setup with exponential backoff for rate limits
        max_retries = 3
        base_delay = 2  # Start with 2 second delay
        
        for retry in range(max_retries):
            try:
                remote_client = get_remote_client()
                if remote_client:
                    model = config.get('REMOTE_MODEL_NAME', 'gpt-3.5-turbo-16k')
                    return AIProvider(remote_client, model)
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() and retry < max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s
                    delay = base_delay * (2 ** retry)
                    print(f"Rate limited, retrying in {delay}s (attempt {retry+1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    print(f"Error getting remote client: {e}")
                    break

        # If we got here with no working client, use fallback
        print("No working AI provider found. Using fallback responses.")
        record_llm_failure("api_unreachable")
        return AIProvider(None, "fallback")
        
    except Exception as e:
        print(f"Critical error in get_ai_provider: {e}")
        # Emergency fallback
        record_llm_failure("critical_error")
        return AIProvider(None, "fallback")

# Common cryptocurrency names and their CoinGecko IDs
CRYPTO_MAPPING = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "sol": "solana",
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
    "cardano": "cardano",
    "ada": "cardano",
    "binance coin": "binancecoin",
    "bnb": "binancecoin",
    "xrp": "ripple",
    "polkadot": "polkadot",
    "dot": "polkadot",
    "avalanche": "avalanche-2",
    "avax": "avalanche-2",
    "polygon": "matic-network",
    "matic": "matic-network"
}

async def get_crypto_price(crypto_name):
    """Get real-time cryptocurrency price from CoinGecko API"""
    # Update timestamp cache to ensure we have current date and time
    update_timestamp_cache()
    
    crypto_name = crypto_name.lower().strip()
    
    # Clean the crypto name (remove words like "price", "value", etc.)
    clean_terms = ["price", "value", "worth", "current", "now", "today", "chart", "market", "cap"]
    for term in clean_terms:
        crypto_name = crypto_name.replace(term, "").strip()
    
    # Try to match with known cryptocurrencies
    coin_id = None
    for key, value in CRYPTO_MAPPING.items():
        if key in crypto_name:
            coin_id = value
            break
    
    # If no match found, use the input as-is
    if not coin_id:
        # Remove special characters and get first word
        coin_id = re.sub(r'[^a-zA-Z0-9\s]', '', crypto_name).split()[0]
    
    try:
        async with aiohttp.ClientSession() as session:
            # Get price in multiple currencies
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd,eur,gbp&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"
            
            # Add a timestamp parameter to prevent caching
            url += f"&_t={int(time.time())}"
            
            # Add headers to identify our bot and prevent rate limiting
            headers = {
                'User-Agent': 'Discord-AI-Chatbot/1.0',
                'Accept': 'application/json'
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if coin_id in data:
                        price_data = data[coin_id]
                        result = f"**{coin_id.capitalize()} Price Information**\n"
                        result += f"USD: ${price_data.get('usd', 'N/A'):,.2f}\n"
                        result += f"EUR: €{price_data.get('eur', 'N/A'):,.2f}\n"
                        result += f"GBP: £{price_data.get('gbp', 'N/A'):,.2f}\n"
                        
                        if 'usd_24h_change' in price_data:
                            change = price_data['usd_24h_change']
                            change_symbol = "📈" if change > 0 else "📉"
                            result += f"24h Change: {change_symbol} {change:.2f}%\n"
                            
                        if 'usd_market_cap' in price_data:
                            result += f"Market Cap: ${price_data['usd_market_cap']:,.0f}\n"
                        
                        # Use the timestamp from our cache to ensure current time    
                        current_datetime = f"{timestamp_cache['current_date']} {timestamp_cache['current_time']} UTC"
                        result += f"\nData from CoinGecko at {current_datetime}"
                        return result
                    else:
                        return f"Could not find price data for '{crypto_name}'. Please check the cryptocurrency name and try again."
                else:
                    # If API rate limit is hit or other error, fall back to search
                    print(f"CoinGecko API error: {response.status} - {await response.text()}")
                    return None
    except Exception as e:
        print(f"Error fetching crypto price: {e}")
        return None

async def search_internet(query):
    """Perform internet search using DuckDuckGo with fallback options."""
    if not config['INTERNET_ACCESS']:
        return "Internet access has been disabled by user"
    
    # Update timestamp cache
    update_timestamp_cache()
    
    # Add current year/date to query for time-sensitive searches
    if any(term in query.lower() for term in ["current", "latest", "today", "now", "price"]):
        # Append current year if not already in query
        if str(timestamp_cache["current_year"]) not in query:
            query += f" {timestamp_cache['current_year']}"
            
        # Append current date for very time-sensitive queries
        if any(term in query.lower() for term in ["today", "now", "current"]):
            query += f" {timestamp_cache['current_date']}"
    
    print(f"Searching the internet for: {query}")
    
    # Track the search methods we've tried
    tried_methods = []
    max_retries = 3
    
    # First try: DuckDuckGo with retries and exponential backoff
    try:
        tried_methods.append("DuckDuckGo")
        
        # Run duckduckgo search in a threadpool since it's synchronous
        def perform_ddg_search():
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=6):
                    results.append(r)
                return results
        
        # Try with exponential backoff
        for attempt in range(max_retries):
            try:
                results = await asyncio.to_thread(perform_ddg_search)
                
                if results:
                    search_results = ""
                    for index, result in enumerate(results[:6]):
                        search_results += f'[{index}] Title: {result["title"]}\nURL: {result.get("href", "No URL")}\nSnippet: {result["body"]}\n\n'
                    
                    return search_results
                else:
                    # If no results but no error, try next method
                    print(f"DuckDuckGo returned no results for: {query}")
                    break
                    
            except Exception as e:
                # Check if it's a rate limit error (could be different error messages)
                if any(error_term in str(e).lower() for error_term in ["rate", "limit", "429", "too many requests"]):
                    # If it's the last retry, move on to the next method
                    if attempt == max_retries - 1:
                        print(f"DuckDuckGo rate limited after {max_retries} attempts, trying alternative methods")
                        break
                    
                    # Otherwise, wait with exponential backoff
                    wait_time = (2 ** attempt) + random.random()
                    print(f"DuckDuckGo rate limited, retrying in {wait_time:.2f} seconds (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    # For other errors, just try the next method
                    print(f"DuckDuckGo search error: {e}")
                    break
    except Exception as e:
        print(f"Error setting up DuckDuckGo search: {e}")

    # If we reach here, DuckDuckGo has failed or been rate limited
    
    # Second try: Use Google search via alternative method
    try:
        tried_methods.append("Google")
        print(f"Trying Google search for: {query}")
        
        async with aiohttp.ClientSession() as session:
            # Create a Google-like search URL
            encoded_query = urllib.parse.quote(query)
            url = f"https://www.google.com/search?q={encoded_query}"
            
            # Use a realistic user agent
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1"
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Simple parsing to extract search results
                    # This is a basic implementation and could be improved with better parsing
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract search results
                    search_results = ""
                    result_count = 0
                    
                    # Look for result blocks
                    for div in soup.select("div.g"):
                        try:
                            title_element = div.select_one("h3")
                            link_element = div.select_one("a")
                            snippet_element = div.select_one("div.VwiC3b")
                            
                            if title_element and link_element and snippet_element:
                                title = title_element.get_text().strip()
                                link = link_element.get("href")
                                if link.startswith("/url?"):
                                    # Extract the actual URL from Google's redirect URL
                                    link = link.split("?q=")[1].split("&")[0]
                                snippet = snippet_element.get_text().strip()
                                
                                search_results += f"[{result_count}] Title: {title}\nURL: {link}\nSnippet: {snippet}\n\n"
                                result_count += 1
                                
                                if result_count >= 5:
                                    break
                        except Exception as e:
                            continue
                    
                    if search_results:
                        return search_results
                    
                    # If parsing failed but we got a 200 response, return a simplified response
                    if not search_results:
                        return f"Successfully accessed search results for '{query}', but couldn't parse the results. Please try a more specific query."
                else:
                    print(f"Google search returned status code: {response.status}")
                    # Continue to the next method if this fails
    except Exception as e:
        print(f"Google search error: {e}")

    # Third try: Simple web search using a different endpoint
    try:
        tried_methods.append("Alternative Search")
        print(f"Trying alternative search for: {query}")
        
        # You could implement various fallbacks here
        # For example, using a simple API like SerpAPI (with API key) or a different search engine
        
        # This is a placeholder for demonstration
        # In a real implementation, you would integrate with another search provider
        
        # For now, we'll return a message about the failure of previous methods
        return f"Unable to perform a search for '{query}'. I tried {', '.join(tried_methods)}, but encountered rate limits or other issues. Please try again later with a more specific query."
        
    except Exception as e:
        print(f"Alternative search error: {e}")
    
    # If all methods fail, return an error message
    return f"I encountered difficulties searching for '{query}'. I tried {', '.join(tried_methods)} but wasn't able to get results. Please try again later or rephrase your query."

async def stream_response(messages, model, client):
    """
    Stream a response from the LLM model
    
    Args:
        messages (list): The message history
        model (str): The model ID to use
        client (AsyncOpenAI): The OpenAI client (local or remote)
        
    Returns:
        async generator: An async generator that yields response chunks
    """
    try:
        print(f"Attempting to stream response from model: {model}")
        # Add timeout parameters and more robust configuration
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            timeout=60.0,  # Add a 60-second timeout
        )
        
        collected_chunks = []
        collected_content = ""
        
        # Process the stream
        async for chunk in stream:
            if not hasattr(chunk.choices[0].delta, 'content'):
                continue
                
            content = chunk.choices[0].delta.content
            if content:
                collected_chunks.append(content)
                collected_content += content
                yield content
                
        # Record successful LLM call
        record_llm_success()
        
        # Cache the successful response
        if len(messages) > 1:
            user_message = messages[-1]["content"]
            cache_successful_response(user_message, collected_content)
    except httpx.TimeoutException as e:
        print(f"Timeout error connecting to LLM: {e}")
        print(f"Model: {model}")
        if hasattr(client, 'base_url'):
            print(f"Client base URL: {client.base_url}")
        
        # Record the failure
        record_llm_failure()
        
        # Yield a fallback message
        fallback_msg = "The AI service took too long to respond. Please try again later."
        yield fallback_msg
    except httpx.ConnectError as e:
        print(f"Connection error communicating with LLM: {e}")
        print(f"Model: {model}")
        if hasattr(client, 'base_url'):
            print(f"Client base URL: {client.base_url}")
        
        # Record the failure
        record_llm_failure()
        
        # Yield a fallback message
        fallback_msg = "I couldn't connect to the AI service. Please check your internet connection and try again."
        yield fallback_msg
    except Exception as e:
        print(f"Error streaming from LLM: {e}")
        print(f"Model: {model}")
        if hasattr(client, 'base_url'):
            print(f"Client base URL: {client.base_url}")
        
        # Record the failure
        record_llm_failure()
        
        # Yield a fallback message
        fallback_msg = "I encountered an error connecting to the AI service. Please try again later."
        yield fallback_msg

async def generate_response(instructions, history=None, model=None, stream=False, temperature=None, max_tokens=None):
    """
    Generate a response using either OpenAI or local model
    
    Args:
        instructions: Instructions or prompt (can be string or coroutine)
        history: Optional conversation history
        model: Optional model override
        stream: Whether to stream the response
        temperature: Optional temperature setting
        max_tokens: Optional max tokens setting
        
    Returns:
        str or StreamingResponse: The generated response
    """
    global local_client, remote_client
    
    # Handle empty history
    if history is None:
        history = []
    
    # Prepare timestamp info for context
    # Using a cache to avoid repeated date/time lookups
    if 'timestamp_cache' not in globals():
        global timestamp_cache
        timestamp_cache = {}
    
    # Update timestamp cache every 60 seconds
    current_time = time.time()
    if 'last_update' not in timestamp_cache or current_time - timestamp_cache.get('last_update', 0) > 60:
        now = datetime.now(timezone.utc)
        timestamp_cache = {
            'last_update': current_time,
            'current_date': now.strftime('%Y-%m-%d'),
            'current_time': now.strftime('%H:%M:%S'),
            'current_year': now.year,
            'current_month': now.month,
            'current_day': now.day,
        }
    
    # Explicitly add current date/time info to the instructions
    current_time_info = f"\n\nThe current date and time is {timestamp_cache['current_date']} {timestamp_cache['current_time']} UTC. Today is {timestamp_cache['current_year']}-{timestamp_cache['current_month']:02d}-{timestamp_cache['current_day']:02d}."
    
    # Handle instructions that might be a coroutine
    if asyncio.iscoroutine(instructions):
        instructions = await instructions
    
    # Now that we have the instructions string, add all context
    enhanced_instructions = instructions + current_time_info
    
    # Check if we're in offline mode (LLM unavailable)
    if is_offline_mode():
        print("LLM is unavailable, using fallback mode")
        
        # Get the latest user message
        if history and len(history) > 0:
            last_user_message = history[-1]["content"]
            # Get a fallback response
            username = None
            try:
                # Try to extract a username from the interaction if available
                for msg in reversed(history):
                    if msg["role"] == "user" and "my name is" in msg["content"].lower():
                        potential_name = msg["content"].lower().split("my name is")[1].strip().split()[0]
                        if potential_name and len(potential_name) > 1:
                            username = potential_name.capitalize()
                            break
            except:
                pass
                
            fallback_response = await get_fallback_response(last_user_message, username=username)
            return fallback_response
        else:
            return "I'm currently operating in fallback mode due to unavailability of my primary language model. I can only provide basic responses right now."
    
    # Check if the latest message contains a query that might need real-time information
    latest_message = history[-1]["content"].lower() if history else ""
    
    # Common triggers for real-time information
    real_time_triggers = [
        "current price", "bitcoin price", "crypto price",
        "latest news", "weather in", "what is the price of",
        "how much is", "stock price", "current event",
        "today's", "recent", "latest", "now", "current status"
    ]
    
    # Simple check for real-time queries
    needs_real_time_info = any(trigger in latest_message for trigger in real_time_triggers)
    
    # Additional context from internet search if needed
    internet_context = ""
    news_context = ""
    
    # Check if this is a news-related query
    is_news_query, _ = await detect_news_query(latest_message)
    
    if is_news_query:
        # If it's a news query, get relevant news articles
        news_context = await get_news_context(latest_message)
        if news_context:
            print("Found relevant news articles for the query")
    
    if needs_real_time_info and internet_access:
        print("Detected request for real-time information. Performing internet search...")
        
        # Extract relevant search terms by removing bot's name and keeping keywords
        search_query = latest_message
        
        # Get bot name and triggers dynamically
        bot_names, trigger_words, prefixes, suffixes = get_bot_names_and_triggers()
        
        # Remove bot name references from the query
        for name in bot_names:
            search_query = search_query.replace(name, "").strip()
        
        # Check if this is a cryptocurrency price query
        is_crypto_query = any(term in search_query for term in ["crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "price of", "how much is"])
        
        crypto_terms = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "doge", "cardano", "ada", "bnb", "xrp", "polkadot", "dot", "avalanche", "avax", "polygon", "matic"]
        found_crypto = None
        
        for crypto in crypto_terms:
            if crypto in search_query:
                found_crypto = crypto
                break
        
        if is_crypto_query and found_crypto:
            print(f"Detected cryptocurrency price query for: {found_crypto}")
            # Extract just the crypto name for better search
            for prefix in ["how much is", "what is the price of", "current price of", "price of"]:
                if prefix in search_query:
                    search_query = search_query.replace(prefix, "").strip()
            
            # Try to get price from CoinGecko API first
            crypto_price_data = await get_crypto_price(search_query)
            
            if crypto_price_data:
                # Include current date/time info in the context to make it clear this is current data
                current_datetime = f"{timestamp_cache['current_date']} {timestamp_cache['current_time']} UTC"
                internet_context = f"\n\nHere is current cryptocurrency information as of {current_datetime}:\n{crypto_price_data}\n\nUse this information to provide an up-to-date response about the cryptocurrency price."
            else:
                # Fall back to regular search if API fails
                print(f"Falling back to regular search for: {search_query} price")
                # Add current date to ensure recent results
                search_query += f" price today {timestamp_cache['current_year']} {timestamp_cache['current_month']}/{timestamp_cache['current_day']}"
                internet_results = await search_internet(search_query)
                if internet_results and "Error performing internet search" not in internet_results:
                    current_datetime = f"{timestamp_cache['current_date']} {timestamp_cache['current_time']} UTC"
                    internet_context = f"\n\nHere is some real-time information from the internet that might help you answer this query as of {current_datetime}:\n{internet_results}\n\nUse this information to provide an up-to-date response about current prices. The current date is {timestamp_cache['current_date']}."
        else:
            # Don't do a web search if we already have news context
            if not news_context:
                # Extract just the important part for searching
                # For queries like "how much is X" or "what is the price of X", extract just the X part
                for prefix in ["how much is", "what is the price of", "current price of", "price of"]:
                    if prefix in search_query:
                        search_query = search_query.replace(prefix, "").strip()
                        search_query += " price"  # Add "price" to get more relevant results
                
                # If it contains multiple phrases with "in", like "weather in New York", keep just that part
                if "weather in" in search_query:
                    search_query = "weather in" + search_query.split("weather in")[1].strip()
                    
                # Clean up any remaining question marks or other punctuation
                search_query = search_query.replace("?", "").replace("!", "").strip()
                
                # Add current date info for time-sensitive queries
                if any(term in search_query for term in ["today", "current", "now", "latest"]):
                    search_query += f" {timestamp_cache['current_date']}"
                
                print(f"Searching the internet for: {search_query}")
                internet_results = await search_internet(search_query)
                if internet_results and "Error performing internet search" not in internet_results:
                    current_datetime = f"{timestamp_cache['current_date']} {timestamp_cache['current_time']} UTC"
                    internet_context = f"\n\nHere is some real-time information from the internet that might help you answer this query:\n{internet_results}\n\nUse this information to provide an up-to-date response. The current date and time is {current_datetime}."
    
    # Add news context if found
    if news_context:
        enhanced_instructions += f"\n\n{news_context}\nUse this real-time news information to provide an up-to-date response about current events."
    
    # Add internet context if found
    if internet_context:
        enhanced_instructions += internet_context
    
    messages = [
            {"role": "system", "content": enhanced_instructions},
        ]
        
    # Convert user messages to correct format - GROQ API doesn't support 'user' field, only 'name'
    for msg in history:
        message_copy = dict(msg)  # Create a copy of the message
        # Convert 'user' field to 'name' field for user messages
        if "user" in message_copy and message_copy["role"] == "user":
            message_copy["name"] = message_copy.pop("user")
        # Convert 'metadata' field to 'name' field for assistant messages
        if "metadata" in message_copy and message_copy["role"] == "assistant":
            message_copy["name"] = message_copy.pop("metadata")
        messages.append(message_copy)
    
    # Prepare to handle potential failures
    try:
        # Check if we should use remote API (GROQ) or fall back to local model
        use_local_model = config.get('USE_LOCAL_MODEL', False)
        api_key_available = os.environ.get("API_KEY") is not None
        
        # If API key isn't available but remote model is requested, fall back to local model
        if not use_local_model and not api_key_available:
            print("API_KEY not found in environment but remote model requested. Falling back to local model.")
            use_local_model = True
            
        if use_local_model:
            print("Using local LLM model as configured in settings")
            if local_client is None:
                local_client = get_local_client()
                
            # If streaming is requested, return a streamed response
            if stream:
                return stream_response(messages, config.get('LOCAL_MODEL_ID', 'mistral-nemo-instruct-2407'), local_client)
                
            try:
                print(f"Using local model: {config.get('LOCAL_MODEL_ID', 'mistral-nemo-instruct-2407')}")
                response = await local_client.chat.completions.create(
                    model=config.get('LOCAL_MODEL_ID', 'mistral-nemo-instruct-2407'),
                    messages=messages,
                )
                
                # Record successful LLM call
                record_llm_success()
                
                # Cache the successful response for future fallback
                if history and len(history) > 0:
                    last_message = history[-1]["content"]
                    cache_successful_response(last_message, response.choices[0].message.content)
                    
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error with local LLM: {e}")
                
                # Record the failure
                record_llm_failure()
                
                # If we get here, try using the fallback
                if history and len(history) > 0:
                    last_message = history[-1]["content"]
                    fallback_response = await get_fallback_response(last_message)
                    return fallback_response
                    
                return "I encountered an error connecting to the local language model. Please make sure LM Studio is running and accessible from WSL."
        else:
            # Using remote API with tools
            print("Using remote LLM (GROQ API)")
            # Initialize remote_client if it's None
            if remote_client is None:
                remote_client = get_remote_client()
                
            # Double-check if remote_client was created successfully
            if remote_client is None:
                print("Failed to create remote client. Falling back to local model.")
                if local_client is None:
                    local_client = get_local_client()
                
                # If streaming is requested, return a streamed response
                if stream:
                    return stream_response(messages, config.get('LOCAL_MODEL_ID', 'mistral-nemo-instruct-2407'), local_client)
                
                try:
                    print(f"Using local model as fallback")
                    response = await local_client.chat.completions.create(
                        model=config.get('LOCAL_MODEL_ID', 'mistral-nemo-instruct-2407'),
                        messages=messages,
                        timeout=30.0,  # Add timeout
                    )
                    
                    # Record successful LLM call
                    record_llm_success()
                    
                    # Cache the successful response for future fallback
                    if history and len(history) > 0:
                        last_message = history[-1]["content"]
                        cache_successful_response(last_message, response.choices[0].message.content)
                        
                    return response.choices[0].message.content
                except httpx.TimeoutException as e:
                    print(f"Timeout error with local LLM: {e}")
                    record_llm_failure()
                except httpx.ConnectError as e:
                    print(f"Connection error with local LLM: {e}")
                    record_llm_failure()
                except Exception as e:
                    print(f"Error with local LLM fallback: {e}")
                    record_llm_failure()
                    
                # If we get here, try using the fallback
                if history and len(history) > 0:
                    last_message = history[-1]["content"]
                    fallback_response = await get_fallback_response(last_message)
                    return fallback_response
                    
                return "I encountered an error connecting to both remote and local language models. Please check your configuration."
            
            # If streaming is requested and not using tools, return a streamed response
            if stream and not needs_real_time_info:
                print(f"Requesting streaming response from GROQ API model: {config['MODEL_ID']}")
                return stream_response(messages, config['MODEL_ID'], remote_client)
            
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "searchtool",
                        "description": "Searches the internet.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The query for search engine",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ]
            
            try:
                print(f"Requesting response from GROQ API model: {config['MODEL_ID']}")
                response = await remote_client.chat.completions.create(
                    model=config['MODEL_ID'],
                    messages=messages,        
                    tools=tools,
                    tool_choice="auto",
                    timeout=45.0,  # Add timeout
                )
                
                # Record successful LLM call
                record_llm_success()
                print("Successfully received response from GROQ API")
                
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                if tool_calls:
                    print("Response includes tool calls - executing them")
                    available_functions = {
                        "searchtool": duckduckgotool,
                    }
                    messages.append(response_message)

                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_to_call = available_functions[function_name]
                        function_args = json.loads(tool_call.function.arguments)
                        function_response = await function_to_call(
                            query=function_args.get("query")
                        )
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )
                    
                    # If streaming is requested for the final response
                    if stream:
                        print("Streaming final response after tool calls")
                        return stream_response(messages, config['MODEL_ID'], remote_client)
                    
                    print("Requesting second response after tool calls")
                    second_response = await remote_client.chat.completions.create(
                        model=config['MODEL_ID'],
                        messages=messages,
                        timeout=45.0,  # Add timeout
                    ) 
                    
                    # Cache successful response
                    if history and len(history) > 0:
                        last_message = history[-1]["content"]
                        cache_successful_response(last_message, second_response.choices[0].message.content)
                        
                    return second_response.choices[0].message.content
                
                # Cache successful response
                if history and len(history) > 0:
                    last_message = history[-1]["content"]
                    cache_successful_response(last_message, response_message.content)
                    
                return response_message.content
                
            except httpx.TimeoutException as e:
                print(f"Timeout error with remote LLM: {e}")
                print(f"API Base URL: {config['API_BASE_URL']}")
                print(f"Model ID: {config['MODEL_ID']}")
                record_llm_failure()
                
                # Try falling back to local model if available
                if local_client is not None:
                    try:
                        print("Trying local model after remote timeout...")
                        local_response = await local_client.chat.completions.create(
                            model=config.get('LOCAL_MODEL_ID', 'mistral-nemo-instruct-2407'),
                            messages=messages,
                            timeout=30.0,
                        )
                        return local_response.choices[0].message.content
                    except Exception as local_error:
                        print(f"Local fallback also failed: {local_error}")
                        
            except httpx.ConnectError as e:
                print(f"Connection error with remote LLM: {e}")
                print(f"API Base URL: {config['API_BASE_URL']}")
                print(f"Model ID: {config['MODEL_ID']}")
                record_llm_failure()
                
            except Exception as e:
                print(f"Error with remote LLM: {e}")
                print(f"API Base URL: {config['API_BASE_URL']}")
                print(f"API Key available: {api_key_available}")
                print(f"Model ID: {config['MODEL_ID']}")
                
                # Record the failure
                record_llm_failure()
                
                # If we get here, try using the fallback
                if history and len(history) > 0:
                    last_message = history[-1]["content"]
                    fallback_response = await get_fallback_response(last_message)
                    return fallback_response
                    
                return "I encountered an error connecting to the AI service. Please try again later."
            
    except Exception as e:
        print(f"Unexpected error in generate_response: {e}")
        
        # Use fallback response as a last resort
        if history and len(history) > 0:
            last_message = history[-1]["content"]
            fallback_response = await get_fallback_response(last_message)
            return fallback_response
            
        return "I'm experiencing technical difficulties. Please try again later."

async def duckduckgotool(query) -> str:
    """Legacy search tool for OpenAI's tool calling capability"""
    result = await search_internet(query)
    return result

async def poly_image_gen(session, prompt):
    seed = random.randint(1, 100000)
    image_url = f"https://image.pollinations.ai/prompt/{prompt}?seed={seed}"
    async with session.get(image_url) as response:
        image_data = await response.read()
        return io.BytesIO(image_data)

async def generate_image_prodia(prompt, model, sampler, seed, neg):
    print("\033[1;32m(Prodia) Creating image for :\033[0m", prompt)
    start_time = time.time()
    async def create_job(prompt, model, sampler, seed, neg):
        if neg is None:
            negative = "(nsfw:1.5),verybadimagenegative_v1.3, ng_deepnegative_v1_75t, (ugly face:0.8),cross-eyed,sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy, DeepNegative, facing away, tilted head, {Multiple people}, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot, repeating hair, nsfw, [[[[[bad-artist-anime, sketch by bad-artist]]]]], [[[mutation, lowres, bad hands, [text, signature, watermark, username], blurry, monochrome, grayscale, realistic, simple background, limited palette]]], close-up, (swimsuit, cleavage, armpits, ass, navel, cleavage cutout), (forehead jewel:1.2), (forehead mark:1.5), (bad and mutated hands:1.3), (worst quality:2.0), (low quality:2.0), (blurry:2.0), multiple limbs, bad anatomy, (interlocked fingers:1.2),(interlocked leg:1.2), Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), crown braid, (deformed fingers:1.2), (long fingers:1.2)"
        else:
            negative = neg
        url = 'https://api.prodia.com/generate'
        params = {
            'new': 'true',
            'prompt': f'{quote(prompt)}',
            'model': model,
            'negative_prompt': f"{negative}",
            'steps': '100',
            'cfg': '9.5',
            'seed': f'{seed}',
            'sampler': sampler,
            'upscale': 'True',
            'aspect_ratio': 'square'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data['job']
            
    job_id = await create_job(prompt, model, sampler, seed, neg)
    url = f'https://api.prodia.com/job/{job_id}'
    headers = {
        'authority': 'api.prodia.com',
        'accept': '*/*',
    }

    async with aiohttp.ClientSession() as session:
        while True:
            async with session.get(url, headers=headers) as response:
                json = await response.json()
                if json['status'] == 'succeeded':
                    async with session.get(f'https://images.prodia.xyz/{job_id}.png?download=1', headers=headers) as response:
                        content = await response.content.read()
                        img_file_obj = io.BytesIO(content)
                        duration = time.time() - start_time
                        print(f"\033[1;34m(Prodia) Finished image creation\n\033[0mJob id : {job_id}  Prompt : ", prompt, "in", duration, "seconds.")
                        return img_file_obj

async def text_to_speech(text):
    bytes_obj = io.BytesIO()
    detected_language = detect(text)
    tts = gTTS(text=text, lang=detected_language)
    tts.write_to_fp(bytes_obj)
    bytes_obj.seek(0)
    return bytes_obj

async def create_prompt(message, username, user_id, channel, conversation_history=None, enhanced_instructions=None):
    """
    Create a prompt for the AI model
    
    Args:
        message (str): The user's message
        username (str): The username
        user_id (str): The user's ID
        channel (discord.Channel): The channel where the message was sent
        conversation_history (list, optional): History of the conversation
        enhanced_instructions (str, optional): Enhanced instructions for the AI
        
    Returns:
        str: The formatted prompt
    """
    # Get base instructions from config
    default_instruction_name = config.get('DEFAULT_INSTRUCTION', 'hand')
    
    # Get base instructions
    if not enhanced_instructions:
        # Use the loaded instructions from the beginning of the file or a default
        base_instructions = instructions_content.get(default_instruction_name, "You are an AI Discord bot. Be helpful, engaged, and conversational.")
    else:
        # Use enhanced instructions if available
        base_instructions = enhanced_instructions
    
    # Current time info
    current_time_str = timestamp_cache["current_date"] + " " + timestamp_cache["current_time"]
    
    # Add specialized capabilities info
    specialized_capabilities = """
You have the following specialized capabilities:
1. You can search for recent news by using NEWS_SEARCH(query) in your thinking.
2. You can get cryptocurrency prices using CRYPTO_PRICE(symbol) in your thinking.
3. You can get the current time using TIME() in your thinking.
4. You can tell time differences using TIME_DIFF(time1, time2) in your thinking.
5. You can mention Discord users when the user explicitly asks you to mention someone. Look for phrases like "mention [user]", "tag [user]", or "ping [user]" in the user's request. When mentioning users:
   - IMPORTANT: When referring to users by name in your response, use the exact username format as requested
   - Simply write the username directly (like "Hello username" or "@username") 
   - Do not use any special formatting for mentions - the system will automatically convert them to proper Discord mentions
   - Never use <@USER_ID> format or any other custom formatting
   - Only mention users when specifically requested to do so

For example, if you want to get crypto prices, you can think: "Let me check CRYPTO_PRICE(BTC)"
Don't include these function calls in your final answer - they're just for your reasoning.
"""
    
    # Add Discord formatting information
    discord_formatting = """
Important: You're talking in a Discord server, so you can use Discord message formatting:
- **bold** for emphasis
- *italics* for light emphasis
- __underline__ for titles or important points
- ~~strikethrough~~ for corrections
- `code` for short code snippets or commands
- ```python
code blocks
``` for multi-line code (replace python with the appropriate language)
"""
    
    # Combine everything into the full system prompt
    system_prompt = f"{base_instructions}\n\n{specialized_capabilities}\n\n{discord_formatting}\n\nCurrent time: {current_time_str}"
    
    # Get the conversation history string
    if conversation_history:
        history_str = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_history])
    else:
        history_str = "No conversation history available."
    
    # Combine everything for the final prompt
    prompt = f"""System: {system_prompt}

Conversation History:
{history_str}

User ({username}): {message}"""
    
    # Add special context for the bot owner if applicable
    prompt = await add_owner_context(prompt, user_id)
    
    return prompt

async def should_use_sequential_thinking(prompt, user_id=None, channel_id=None):
    """
    Determine if a query should use sequential thinking based on its complexity or nature.
    
    Args:
        prompt: The user's prompt/query
        user_id: The ID of the user who sent the message (optional)
        channel_id: The ID of the channel where the message was sent (optional)
        
    Returns:
        Tuple: (use_sequential, complexity_score, reasoning)
    """
    # Normalize prompt to lowercase for easier pattern matching
    prompt_lower = prompt.lower()
    
    # Direct indicators that sequential thinking should be used
    explicit_indicators = [
        "step by step", 
        "sequential", 
        "in sequence", 
        "one by one",
        "first", "second", "third", "fourth", "fifth", 
        "break down", 
        "detailed explanation",
        "explain thoroughly",
        "walk me through",
        "walkthrough",
        "how to",
        "procedure",
        "instructions",
        "guide me",
        "tutorial",
        "analyze",
        "analyse",
        "examine",
        "investigate",
        "explore",
        "develop",
        "explain",
        "elaborate",
        "describe in detail",
        "evaluate",
        "assess"
    ]
    
    # Topics that typically benefit from sequential thinking
    topic_indicators = [
        # Technical procedures
        "solve", "algorithm", "code", "program", "implement", "function", "method",
        
        # Mathematics/puzzles
        "equation", "problem", "formula", "calculate", "math", "puzzle", "riddle", "rubik", "cube", "sudoku",
        
        # DIY/manual tasks
        "build", "create", "make", "construct", "assemble", "install", "fix", "repair", "troubleshoot", 
        
        # Recipes/cooking
        "recipe", "cook", "bake", "prepare", "mix", "blend", "food",
        
        # Learning/educational
        "learn", "understand", "explain", "concept", "principle", "theory", "framework",
        
        # Planning
        "plan", "strategy", "approach", "organize", "outline", "blueprint", "roadmap",
        
        # Analysis
        "review", "critique", "interpret", "breakdown", "dissect", "deconstruct",
        
        # Problem-solving
        "resolve", "solution", "tackle", "address", "overcome", "handle",
        
        # Decision making
        "decide", "choice", "option", "alternative", "recommendation", "pros and cons",
        
        # Complex tasks
        "complex", "complicated", "intricate", "sophisticated", "advanced"
    ]
    
    # Complex query indicators
    complexity_indicators = {
        "multi-step": ["step", "phase", "stage", "part", "section"],
        "detail-oriented": ["detail", "specific", "precisely", "exactly", "thoroughly", "comprehensive", "complete"],
        "comparative": ["compare", "contrast", "versus", "vs", "difference", "similarity", "distinction"],
        "process-focused": ["process", "procedure", "workflow", "pipeline", "sequence", "methodology"],
        "analytical": ["analyze", "examine", "investigate", "assess", "evaluate", "consider", "reflect"],
        "reasoning": ["reason", "logic", "rationale", "justification", "basis", "grounds"],
        "systematic": ["system", "framework", "structure", "organization", "arrangement"]
    }
    
    # Calculate a complexity score
    complexity_score = 0.0
    
    # Check for explicit sequential thinking indicators (highest weight)
    for indicator in explicit_indicators:
        # Look for whole word matches or phrases
        if re.search(r'\b' + re.escape(indicator) + r'\b', prompt_lower) or indicator in prompt_lower:
            # Strong signal, high weight
            complexity_score += 2.0
            reasoning = f"Sequential thinking recommended due to complexity indicators"
            return True, complexity_score, reasoning
    
    # Check for topic-specific indicators
    topic_matches = []
    for indicator in topic_indicators:
        if re.search(r'\b' + re.escape(indicator) + r'\b', prompt_lower) or indicator in prompt_lower:
            topic_matches.append(indicator)
            complexity_score += 0.5  # Medium weight
    
    if len(topic_matches) >= 2:  # At least two topic indicators
        reasoning = f"Sequential thinking recommended for complex topic"
        return True, complexity_score, reasoning
            
    # Check for complexity indicators
    complexity_matches = []
    for category, indicators in complexity_indicators.items():
        for indicator in indicators:
            if re.search(r'\b' + re.escape(indicator) + r'\b', prompt_lower) or indicator in prompt_lower:
                complexity_matches.append(f"{category}:{indicator}")
                complexity_score += 0.3  # Lower weight
    
    # Additional heuristics
    
    # Long prompts tend to be complex queries
    word_count = len(prompt.split())
    if word_count > 15:  # Lower threshold for long query
        complexity_score += 0.1 * min(5, (word_count - 15) / 10)  # Cap at +0.5
    
    # Questions with multiple parts or sub-questions
    question_count = prompt.count("?")
    if question_count > 1:  # Multiple questions
        complexity_score += 0.2 * min(5, question_count)  # Cap at +1.0
    
    # Common structural indicators for complex queries
    structural_patterns = [
        r"(\d+)[.)]", # Numbered lists
        r"(first|second|third|finally)[\s:,]", # Ordered steps 
        r"(before|after|then|next|subsequently)[\s,]", # Sequential terms
        r"(why|how|what if|when)[\s\?]", # Analytical questions
        r"(because|due to|as a result|therefore|consequently)[\s,]" # Reasoning terms
    ]
    
    for pattern in structural_patterns:
        matches = re.findall(pattern, prompt_lower)
        if matches:
            complexity_score += 0.3
            complexity_matches.append(f"structure:{matches[0]}")
            
    # Decision threshold: lower threshold to trigger sequential thinking more often
    threshold = 0.8
    use_sequential = complexity_score >= threshold
    
    if complexity_matches:
        reasoning = f"Sequential thinking evaluation based on query complexity"
    elif topic_matches:
        reasoning = f"Sequential thinking evaluation based on topic indicators"
    else:
        reasoning = f"General complexity analysis"
        
    return use_sequential, complexity_score, reasoning

async def is_bot_master(user_id):
    """
    Check if the user is the bot's master/owner
    
    Args:
        user_id: The user ID to check
        
    Returns:
        bool: True if this is the bot's master
    """
    from bot_utilities.config_loader import config
    
    # Get owner ID from config
    owner_id = config.get('BOT_OWNER', "818806194608013313")  # Default to Paws ID if not configured
    
    # Convert both to strings to ensure comparison works
    user_id = str(user_id)
    owner_id = str(owner_id)
    
    return user_id == owner_id

async def add_owner_context(prompt, user_id):
    """
    Add context awareness when talking to the bot owner
    
    Args:
        prompt: The original prompt
        user_id: The user ID
        
    Returns:
        str: Enhanced prompt with owner context if applicable
    """
    # Check if this is the bot owner
    is_owner = await is_bot_master(user_id)
    if not is_owner:
        return prompt
    
    # Add natural context for the owner (Paws)
    owner_context = """
This conversation is with Paws, the creator and developer of this bot.
Paws appreciates technical, detailed, and thorough responses.
Incorporate insights about AI development, programming concepts, and technical details when relevant.
Remember previous conversations and be consistent with your responses to maintain context.
"""
    
    # Check if prompt has a system section we can add to
    if "System:" in prompt:
        # Find and enhance the system section
        parts = prompt.split("System:", 1)
        system_parts = parts[1].split("\n\n", 1)
        
        # Add owner context to the system part
        enhanced_system = system_parts[0] + "\n" + owner_context
        
        # Rebuild the prompt
        if len(system_parts) > 1:
            return parts[0] + "System:" + enhanced_system + "\n\n" + system_parts[1]
        else:
            return parts[0] + "System:" + enhanced_system
    else:
        # Add as a prefix if no System section found
        return owner_context + "\n\n" + prompt
