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
import httpx
import urllib.parse
from bs4 import BeautifulSoup
import traceback

# NOTE: Avoid importing agent_service here to prevent circular imports
# Instead, use lazy imports inside methods when needed

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
    """Extract bot name and triggers for filtering search queries and smart triggers"""
    bot_names = []
    
    # Get triggers from config
    triggers = []
    if 'TRIGGER' in config and config['TRIGGER']:
        triggers.extend([trigger.lower() for trigger in config['TRIGGER']])
    
    # Try to extract name from instruction content
    if bot_instructions:
        # Look for common patterns where bot name might be defined
        instruction_lines = bot_instructions.split('\n')
        for line in instruction_lines:
            line = line.lower()
            
            # Look for patterns like 'You are "Bot Name"' or 'You are Bot Name' or 'Your name is Bot Name'
            name_patterns = [
                r'you are ["\']([^"\']+)["\']',  # You are "Name"
                r'you are (?:an?|the) ["\']([^"\']+)["\']',  # You are a/an/the "Name" 
                r'you are (?:an?|the) ([A-Z][a-zA-Z0-9_\s]+)',  # You are a/an/the Name (capitalized)
                r'your name is ["\']([^"\']+)["\']',  # Your name is "Name"
                r'your name is ([A-Z][a-zA-Z0-9_\s]+)',  # Your name is Name (capitalized)
                r'called ["\']([^"\']+)["\']',  # called "Name"
                r'known as ["\']([^"\']+)["\']',  # known as "Name"
            ]
            
            for pattern in name_patterns:
                matches = re.findall(pattern, line)
                if matches:
                    for match in matches:
                        # Clean the extracted name and add it if not already in list
                        extracted_name = match.strip()
                        if extracted_name and extracted_name.lower() not in [n.lower() for n in bot_names]:
                            bot_names.append(extracted_name)
                            
                            # If the name starts with "the", also add version without "the"
                            if extracted_name.lower().startswith("the "):
                                bot_names.append(extracted_name[4:])

    # Add some fallbacks if no names were found
    if not bot_names:
        instruction_name = config.get('DEFAULT_INSTRUCTION', 'bot')
        bot_names = [instruction_name, "chatbot", "assistant", "bot"]
    
    # Process triggers to replace placeholders with actual values
    processed_triggers = []
    for trigger in triggers:
        if "%BOT_NAME%" in trigger:
            # Replace with each extracted bot name
            for name in bot_names:
                processed_triggers.append(trigger.replace("%BOT_NAME%", name.lower()))
        elif "%BOT_NICKNAME%" in trigger or "%BOT_USERNAME%" in trigger:
            # These will be handled dynamically at runtime in the message processing
            processed_triggers.append(trigger)
        else:
            # Regular trigger word
            processed_triggers.append(trigger)
    
    return bot_names, processed_triggers

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
    """Get an AI provider instance for sequential thinking"""
    global remote_client, local_client
    
    # Use local model if enabled in config
    if config.get('USE_LOCAL_MODEL', False):
        if local_client is None:
            local_client = get_local_client()
        return AIProvider(client=local_client, model=config.get('LOCAL_MODEL', 'local'))
    else:
        # Use remote model (default)
        if remote_client is None:
            remote_client = get_remote_client()
        return AIProvider(client=remote_client, model=config.get('DEFAULT_MODEL', 'meta-llama/llama-4-maverick-17b-128e-instruct'))

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
    """Perform internet search using Serper.dev API with robust fallbacks."""
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
    
    # 1. Try Serper.dev API first (if API key is available)
    serper_api_key = os.environ.get("SERPER_API_KEY")
    if serper_api_key:
        try:
            tried_methods.append("Serper.dev API")
            print("Using Serper.dev API for search")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": serper_api_key, "Content-Type": "application/json"},
                    json={"q": query, "num": 5}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process organic search results
                        search_results = ""
                        result_count = 0
                        
                        if "organic" in data:
                            for index, result in enumerate(data["organic"][:5]):
                                title = result.get("title", "No Title")
                                link = result.get("link", "No URL")
                                snippet = result.get("snippet", "No description available.")
                                
                                search_results += f"[{result_count}] Title: {title}\nURL: {link}\nSnippet: {snippet}\n\n"
                                result_count += 1
                        
                        # Add knowledge graph if available
                        if "knowledgeGraph" in data:
                            kg = data["knowledgeGraph"]
                            if "title" in kg:
                                search_results += f"Knowledge Graph: {kg.get('title')}\n"
                                if "description" in kg:
                                    search_results += f"Description: {kg.get('description')}\n"
                                if "attributes" in kg:
                                    search_results += "Attributes:\n"
                                    for key, value in kg["attributes"].items():
                                        search_results += f"- {key}: {value}\n"
                                search_results += "\n"
                        
                        if search_results:
                            return search_results
        except Exception as e:
            print(f"Serper.dev API error: {e}")
    
    # 2. Try SerpAPI for Google (if API key is available)
    serpapi_key = os.environ.get("SERPAPI_KEY")
    if serpapi_key:
        try:
            tried_methods.append("SerpAPI")
            print("Using SerpAPI for search")
            
            async with aiohttp.ClientSession() as session:
                params = {
                    "engine": "google",
                    "q": query,
                    "api_key": serpapi_key,
                    "num": "5"
                }
                async with session.get("https://serpapi.com/search", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        search_results = ""
                        result_count = 0
                        
                        if "organic_results" in data:
                            for index, result in enumerate(data["organic_results"][:5]):
                                title = result.get("title", "No Title")
                                link = result.get("link", "No URL")
                                snippet = result.get("snippet", "No description available.")
                                
                                search_results += f"[{result_count}] Title: {title}\nURL: {link}\nSnippet: {snippet}\n\n"
                                result_count += 1
                        
                        if search_results:
                            return search_results
        except Exception as e:
            print(f"SerpAPI error: {e}")
    
    # 3. Try direct Google scraping with improved resilience and random delays
    try:
        tried_methods.append("Google Scraping")
        print(f"Trying Google scraping for: {query}")
        
        # Add a random delay before scraping to avoid pattern detection
        await asyncio.sleep(random.uniform(1, 3))
        
        async with aiohttp.ClientSession() as session:
            # Create a Google-like search URL
            encoded_query = urllib.parse.quote(query)
            # Use a random Google domain to distribute requests
            google_domains = ["www.google.com", "www.google.co.uk", "www.google.ca", "www.google.com.au"]
            domain = random.choice(google_domains)
            url = f"https://{domain}/search?q={encoded_query}&num=8"
            
            # Rotate user agents to appear more natural
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
            ]
            
            headers = {
                "User-Agent": random.choice(user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                # Add a random referer to appear more natural
                "Referer": random.choice([
                    "https://www.google.com/",
                    "https://www.bing.com/",
                    "https://duckduckgo.com/"
                ])
            }
            
            async with session.get(url, headers=headers, timeout=15) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # More robust parsing to extract search results
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Initialize search results
                    search_results = ""
                    result_count = 0
                    
                    # Try multiple selector patterns to find search results
                    # Google changes their HTML structure frequently
                    selectors = [
                        "div.g", 
                        "div.Gx5Zad", 
                        "div.tF2Cxc",
                        "div.egMi0",
                        ".v7W49e > div" # Newer Google layout
                    ]
                    
                    # Try all selectors until we find results
                    results = []
                    for selector in selectors:
                        results = soup.select(selector)
                        if results and len(results) >= 3:  # Make sure we have meaningful results
                            break
                    
                    # If none of the specific selectors worked, try a more general approach
                    if not results or len(results) < 3:
                        # Look for links with titles and snippets near them
                        results = []
                        for h3 in soup.select("h3"):
                            parent_div = h3.find_parent("div")
                            if parent_div:
                                results.append(parent_div)
                    
                    # Process the results
                    for result_div in results[:8]:  # Limit to top 8 results
                        try:
                            # Try different patterns for title
                            title_element = (
                                result_div.select_one("h3") or 
                                result_div.select_one(".DKV0Md") or
                                result_div.select_one(".LC20lb")
                            )
                            
                            # Try different patterns for link
                            link_element = result_div.select_one("a")
                            
                            # Try different patterns for snippet
                            snippet_element = (
                                result_div.select_one(".VwiC3b") or 
                                result_div.select_one(".lEBKkf") or
                                result_div.select_one(".s3v9rd") or
                                result_div.select_one(".st")
                            )
                            
                            # Extract data if we found the elements
                            if title_element and link_element:
                                title = title_element.get_text().strip()
                                link = link_element.get("href")
                                
                                # Clean Google's redirect links
                                if link and link.startswith("/url?"):
                                    link = link.split("?q=")[1].split("&")[0]
                                elif link and not link.startswith("http"):
                                    continue  # Skip non-web results
                                
                                # Extract snippet if available, or use a placeholder
                                snippet = "No description available."
                                if snippet_element:
                                    snippet = snippet_element.get_text().strip()
                                
                                # Add to results
                                search_results += f"[{result_count}] Title: {title}\nURL: {link}\nSnippet: {snippet}\n\n"
                                result_count += 1
                                
                                if result_count >= 5:
                                    break
                        except Exception as e:
                            print(f"Error parsing a search result: {e}")
                            continue
                    
                    # If we found any valid results, return them
                    if search_results and result_count >= 3:  # Ensure we have at least 3 meaningful results
                        return search_results
                    
                    # Check if the page contains CAPTCHA or other blocks
                    if "captcha" in html.lower() or "unusual traffic" in html.lower():
                        print("Google search is asking for CAPTCHA verification")
                    else:
                        print("Google search returned HTML but couldn't extract results")
        
        # Add a delay before trying the next method
        await asyncio.sleep(random.uniform(1, 2))
    except Exception as e:
        print(f"Google scraping error: {e}")
        traceback.print_exc()

    # 4. Try Bing with improved scraping techniques
    try:
        tried_methods.append("Bing")
        print(f"Trying Bing search for: {query}")
        
        # Add a random delay to appear more natural
        await asyncio.sleep(random.uniform(1, 3))
        
        async with aiohttp.ClientSession() as session:
            # Bing search URL
            encoded_query = urllib.parse.quote(query)
            url = f"https://www.bing.com/search?q={encoded_query}&count=10"
            
            # Rotate user agents
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            ]
            
            headers = {
                "User-Agent": random.choice(user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://www.bing.com/",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                # Add cookies to appear like a returning visitor
                "Cookie": "SRCHHPGUSR=HV=1641717770; SRCHUSR=DOB=20220109; _EDGE_S=ui=en; _EDGE_V=1"
            }
            
            # Use a proxy if needed
            proxy = None
            proxy_url = os.environ.get("HTTP_PROXY")
            if proxy_url:
                proxy = proxy_url
                print(f"Using proxy: {proxy}")
            
            async with session.get(url, headers=headers, proxy=proxy, timeout=15) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Parse the HTML
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Initialize search results
                    search_results = ""
                    result_count = 0
                    
                    # Bing search results are contained in <li class="b_algo"> elements
                    results = soup.select("li.b_algo")
                    
                    # If the default selector didn't work, try some alternatives
                    if not results or len(results) < 3:
                        # Try alternative selectors
                        results = soup.select("div.b_title") or soup.select(".b_algo")
                        
                        # If still no results, look for any links with proper titles
                        if not results or len(results) < 3:
                            results = []
                            for h2 in soup.select("h2"):
                                if h2.select_one("a"):
                                    results.append(h2.find_parent("div") or h2)
                    
                    for result in results[:6]:
                        try:
                            # Find the title and link
                            title_element = result.select_one("h2") or result.select_one(".b_title") or result.select_one("a")
                            
                            if title_element:
                                # If title_element is directly an <a> tag
                                if title_element.name == "a":
                                    link_element = title_element
                                else:
                                    # Otherwise look for the first <a> tag inside title_element
                                    link_element = title_element.select_one("a")
                                
                                # Find description/snippet
                                snippet_element = (
                                    result.select_one(".b_caption p") or 
                                    result.select_one(".b_snippet") or
                                    result.select_one("p")
                                )
                                
                                if link_element:
                                    title = title_element.get_text().strip()
                                    link = link_element.get("href")
                                    
                                    # Ensure the link is absolute
                                    if link and not link.startswith(("http:", "https:")):
                                        if link.startswith("/"):
                                            link = f"https://www.bing.com{link}"
                                        else:
                                            continue  # Skip invalid links
                                    
                                    # Extract snippet if available
                                    snippet = "No description available."
                                    if snippet_element:
                                        snippet = snippet_element.get_text().strip()
                                    
                                    search_results += f"[{result_count}] Title: {title}\nURL: {link}\nSnippet: {snippet}\n\n"
                                    result_count += 1
                                    
                                    if result_count >= 5:
                                        break
                        except Exception as e:
                            print(f"Error parsing Bing result: {e}")
                            continue
                    
                    if search_results and result_count >= 3:  # Ensure we have at least 3 meaningful results
                        return search_results
    except Exception as e:
        print(f"Bing search error: {e}")
        traceback.print_exc()
    
    # 5. Try free search API service as last resort
    try:
        tried_methods.append("Free Search API")
        print("Trying Free Search API")
        
        async with aiohttp.ClientSession() as session:
            # Use a free search API as last resort
            encoded_query = urllib.parse.quote(query)
            url = f"https://ddg-api.herokuapp.com/search?query={encoded_query}&limit=5"
            
            async with session.get(url, timeout=15) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    search_results = ""
                    result_count = 0
                    
                    if "results" in data:
                        for index, result in enumerate(data["results"][:5]):
                            title = result.get("title", "No Title")
                            link = result.get("link", "No URL")
                            snippet = result.get("snippet", "No description available.")
                            
                            search_results += f"[{result_count}] Title: {title}\nURL: {link}\nSnippet: {snippet}\n\n"
                            result_count += 1
                    
                    if search_results:
                        return search_results
    except Exception as e:
        print(f"Free Search API error: {e}")
    
    # If all methods fail, return an error message
    return f"I encountered difficulties searching for '{query}'. I tried {', '.join(tried_methods)} but wasn't able to get results. Search engines might be rate-limiting our requests. Please try again later with a more specific query or consider adding a search API key for better results."

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

async def generate_response(instructions, history, stream=False):
    """
    Generate a response using the LLM
    
    DEPRECATED: Use agent_service.process_query() instead.
    This function is maintained for backward compatibility.
    
    Args:
        instructions: System instructions for the model
        history: Conversation history
        stream: Whether to stream the response
        
    Returns:
        str: The generated response
    """
    # Lazy import to avoid circular dependencies
    from bot_utilities.services.agent_service import agent_service
    
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
    
    # Update timestamp cache to ensure current date/time information
    update_timestamp_cache()
    
    # Extract the query from history (last user message)
    query = history[-1]["content"] if history and len(history) > 0 else ""
    
    try:
        # Ensure agent_service is initialized
        await agent_service.ensure_initialized()
        
        # Create a temporary conversation ID
        conversation_id = f"temp-{time.time()}"
        
        # Process using agent_service
        response = await agent_service.process_query(
            query=query,
            user_id=None,
            conversation_id=conversation_id,
            context={
                "system_instructions": instructions,
                "history": history,
                "current_date": timestamp_cache['current_date'],
                "current_time": timestamp_cache['current_time'],
                "current_year": timestamp_cache['current_year'],
                "current_month": timestamp_cache['current_month'],
                "current_day": timestamp_cache['current_day'],
                "stream": stream
            }
        )
        
        # Cache the successful response for future fallback
        if query:
            cache_successful_response(query, response)
            
        return response
    
    except Exception as e:
        print(f"Unexpected error in generate_response: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        
        # Fallback to original implementation if agent_service fails
        global local_client, remote_client
        
        try:
            # Legacy code starts here
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
                bot_names, trigger_words = get_bot_names_and_triggers()
                
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
                    print(f"Error with local model: {e}")
                    print(f"Stack trace: {traceback.format_exc()}")
                    
                    # Record the failure
                    record_llm_failure()
                    
                    # Retry with remote API if available
                    if api_key_available:
                        print("Trying remote API as fallback...")
                        use_local_model = False
            else:
                # Return a fallback response
                last_message = history[-1]["content"] if history and len(history) > 0 else ""
                return await get_fallback_response(last_message)
            
            # Using remote API (GROQ)
            if not use_local_model:
                print("Using remote LLM model")
                if remote_client is None:
                    remote_client = get_remote_client()
                    
                # If streaming is requested, return a streamed response
                if stream:
                    return stream_response(messages, config['MODEL_ID'], remote_client)
                
                try:
                    print(f"Using remote model: {config['MODEL_ID']}")
                    response = await remote_client.chat.completions.create(
                        model=config['MODEL_ID'],
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
                    print(f"Error with remote model: {e}")
                    
                    # Record the failure
                    record_llm_failure()
                    
                    # Attempt with local model if available
                    if local_client:
                        print("Trying local model as fallback...")
                        try:
                            response = await local_client.chat.completions.create(
                                model=config.get('LOCAL_MODEL_ID', 'mistral-nemo-instruct-2407'),
                                messages=messages,
                            )
                            return response.choices[0].message.content
                        except Exception as local_error:
                            print(f"Local model fallback also failed: {local_error}")
                    
                    # Get a fallback response from cached/pre-defined responses
                    last_message = history[-1]["content"] if history and len(history) > 0 else ""
                    return await get_fallback_response(last_message)
        except Exception as e:
            print(f"Unexpected error in generate_response: {e}")
            print(f"Stack trace: {traceback.format_exc()}")
            
            # Use fallback response as a last resort
            if history and len(history) > 0:
                last_message = history[-1]["content"]
                fallback_response = await get_fallback_response(last_message)
                return fallback_response
            
            return "I'm experiencing technical difficulties. Please try again later."
    except Exception as e:
        print(f"Unexpected error in generate_response: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        
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
    
    return prompt

async def should_use_sequential_thinking(prompt):
    """
    Determine if a query should use sequential thinking based on its complexity or nature.
    
    Args:
        prompt: The user's prompt/query
        
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
        "tutorial"
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
        "plan", "strategy", "approach", "organize", "outline", "blueprint", "roadmap"
    ]
    
    # Complex query indicators
    complexity_indicators = {
        "multi-step": ["step", "phase", "stage", "part", "section"],
        "detail-oriented": ["detail", "specific", "precisely", "exactly", "thoroughly"],
        "comparative": ["compare", "contrast", "versus", "vs", "difference", "similarity"],
        "process-focused": ["process", "procedure", "workflow", "pipeline", "sequence"],
        "analytical": ["analyze", "examine", "investigate", "assess", "evaluate"]
    }
    
    # Calculate a complexity score
    complexity_score = 0.0
    
    # Check for explicit sequential thinking indicators (highest weight)
    for indicator in explicit_indicators:
        if indicator in prompt_lower:
            # Strong signal, high weight
            complexity_score += 2.0
            reasoning = f"Contains explicit sequential indicator: '{indicator}'"
            return True, complexity_score, reasoning
    
    # Check for topic-specific indicators
    topic_matches = []
    for indicator in topic_indicators:
        if indicator in prompt_lower:
            topic_matches.append(indicator)
            complexity_score += 0.5  # Medium weight
    
    if len(topic_matches) >= 2:  # At least two topic indicators
        reasoning = f"Contains multiple sequential topic indicators: {', '.join(topic_matches[:3])}"
        return True, complexity_score, reasoning
            
    # Check for complexity indicators
    complexity_matches = []
    for category, indicators in complexity_indicators.items():
        for indicator in indicators:
            if indicator in prompt_lower:
                complexity_matches.append(f"{category}:{indicator}")
                complexity_score += 0.3  # Lower weight
    
    # Additional heuristics
    
    # Long prompts tend to be complex queries
    word_count = len(prompt.split())
    if word_count > 20:  # Long query
        complexity_score += 0.1 * min(5, (word_count - 20) / 10)  # Cap at +0.5
    
    # Questions with multiple parts or sub-questions
    question_count = prompt.count("?")
    if question_count > 1:  # Multiple questions
        complexity_score += 0.2 * min(5, question_count)  # Cap at +1.0
    
    # Common structural indicators for complex queries
    structural_patterns = [
        r"(\d+)[.)]", # Numbered lists
        r"(first|second|third|finally)[\s:,]", # Ordered steps 
        r"(before|after|then|next|subsequently)[\s,]" # Sequential terms
    ]
    
    for pattern in structural_patterns:
        matches = re.findall(pattern, prompt_lower)
        if matches:
            complexity_score += 0.3
            complexity_matches.append(f"structure:{matches[0]}")
            
    # Decision threshold: if score is high enough, use sequential thinking
    threshold = 1.0
    use_sequential = complexity_score >= threshold
    
    if complexity_matches:
        reasoning = f"Complexity indicators: {', '.join(complexity_matches[:3])}"
    elif topic_matches:
        reasoning = f"Topic indicators: {', '.join(topic_matches[:3])}"
    else:
        reasoning = f"General complexity score: {complexity_score:.2f}"
        
    return use_sequential, complexity_score, reasoning
