import aiohttp
import io
import time
import os
import random
import json
import asyncio
import re
import datetime
from langdetect import detect
from gtts import gTTS
from urllib.parse import quote
from bot_utilities.config_loader import load_current_language, config, load_instructions
from openai import AsyncOpenAI
from duckduckgo_search import DDGS  # Use current DDGS API
from dotenv import load_dotenv
from bot_utilities.news_utils import get_news_context, detect_news_query
from bot_utilities.fallback_utils import is_offline_mode, record_llm_failure, record_llm_success, cache_successful_response, get_fallback_response

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
    "current_date": datetime.datetime.now().strftime("%Y-%m-%d"),
    "current_time": datetime.datetime.now().strftime("%H:%M:%S"),
    "current_year": datetime.datetime.now().year,
    "current_month": datetime.datetime.now().month,
    "current_day": datetime.datetime.now().day,
    "update_interval": 60  # Update every 60 seconds
}

def update_timestamp_cache():
    """Update the timestamp cache if it's older than the update interval"""
    current_time = time.time()
    if current_time - timestamp_cache["last_updated"] > timestamp_cache["update_interval"]:
        now = datetime.datetime.now()
        timestamp_cache["last_updated"] = current_time
        timestamp_cache["current_date"] = now.strftime("%Y-%m-%d")
        timestamp_cache["current_time"] = now.strftime("%H:%M:%S")
        timestamp_cache["current_year"] = now.year
        timestamp_cache["current_month"] = now.month
        timestamp_cache["current_day"] = now.day
        print(f"Updated timestamp cache: {timestamp_cache['current_date']} {timestamp_cache['current_time']}")

# Extract bot name and triggers from configuration and instructions
def get_bot_names_and_triggers():
    """Extract bot name and triggers for filtering search queries"""
    bot_names = []
    
    # Get triggers from config
    if 'TRIGGER' in config and config['TRIGGER']:
        bot_names.extend([trigger.lower() for trigger in config['TRIGGER']])
    
    # Try to extract name from instruction content
    if bot_instructions:
        # Look for common patterns where bot name might be defined
        instruction_lines = bot_instructions.split('\n')
        for line in instruction_lines:
            line = line.lower()
            # Look for patterns like 'You are "Bot Name"' or 'You are Bot Name'
            if 'you are' in line and ('"' in line or "'" in line):
                # Extract text between quotes if present
                for quote_type in ['"', "'"]:
                    if quote_type in line:
                        parts = line.split(quote_type)
                        if len(parts) >= 3:  # At least one quoted string
                            quoted_text = parts[1].strip()
                            if quoted_text and quoted_text.lower() not in bot_names:
                                bot_names.append(quoted_text.lower())
                                # Also add without "the" if it starts with "the "
                                if quoted_text.lower().startswith("the "):
                                    bot_names.append(quoted_text.lower()[4:])

    # Add some fallbacks if no names were found
    if not bot_names:
        bot_names = ["chatbot", "assistant", "bot"]
    
    return bot_names

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
    
    return AsyncOpenAI(
        base_url=api_base,
        api_key=api_key,
    )

# Create client when needed (not at module import time)
local_client = None
remote_client = None

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
    """Perform internet search using DuckDuckGo."""
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
    try:
        # Run duckduckgo search in a threadpool since it's synchronous
        def perform_search():
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=6):
                    results.append(r)
                return results
        
        results = await asyncio.to_thread(perform_search)
        
        if results:
            search_results = ""
            for index, result in enumerate(results[:6]):
                search_results += f'[{index}] Title: {result["title"]}\nURL: {result.get("href", "No URL")}\nSnippet: {result["body"]}\n\n'
            
            return search_results
        else:
            return "No search results found for the given query."
    except Exception as e:
        print(f"Search error: {e}")
        return f"Error performing internet search: {e}"

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
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
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

async def generate_response(instructions, history, stream=False):
    global local_client, remote_client
    
    # Update timestamp cache to ensure current date/time information
    update_timestamp_cache()
    
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
        bot_names = get_bot_names_and_triggers()
        
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
    
    # Explicitly add current date/time info to the instructions
    current_time_info = f"\n\nThe current date and time is {timestamp_cache['current_date']} {timestamp_cache['current_time']} UTC. Today is {timestamp_cache['current_year']}-{timestamp_cache['current_month']:02d}-{timestamp_cache['current_day']:02d}."
    
    # Add all context to instructions
    enhanced_instructions = instructions + current_time_info
    
    # Add news context if found
    if news_context:
        enhanced_instructions += f"\n\n{news_context}\nUse this real-time news information to provide an up-to-date response about current events."
    
    # Add internet context if found
    if internet_context:
        enhanced_instructions += internet_context
    
    messages = [
            {"role": "system", "name": "instructions", "content": enhanced_instructions},
            *history,
        ]
    
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
                    )
                    
                    # Record successful LLM call
                    record_llm_success()
                    
                    # Cache the successful response for future fallback
                    if history and len(history) > 0:
                        last_message = history[-1]["content"]
                        cache_successful_response(last_message, response.choices[0].message.content)
                        
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"Error with local LLM fallback: {e}")
                    
                    # Record the failure
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
                        messages=messages
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

def create_prompt(message, username, user_id, channel, conversation_history=None, enhanced_instructions=None):
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
    # Get base instructions
    if not enhanced_instructions:
        base_instructions = instructions.get(instruc_config, "You are an AI Discord bot. Be helpful, engaged, and conversational.")
    else:
        # Use enhanced instructions if available
        base_instructions = enhanced_instructions
    
    # Current time info
    current_time_str = get_current_time()
    
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
    
    return prompt
