"""
AI Utilities Module

This module provides utilities for working with AI providers.
"""

import os
import re
import json
import time
import uuid
import aiohttp
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Union
import datetime
from dateutil import parser

# Import configuration
from bot_utilities.config import config

# Global clients
remote_client = None
local_client = None

# Cache for timestamp data
timestamp_cache = {
    "last_update": 0,
    "current_date": "",
    "current_time": "",
    "current_year": 0,
    "current_month": "",
    "current_day": 0,
    "day_of_week": "",
}

# Define crypto mapping
CRYPTO_MAPPING = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
    "xrp": "ripple",
    "ripple": "ripple",
    "cardano": "cardano",
    "ada": "cardano",
    "solana": "solana",
    "sol": "solana",
    "bnb": "binancecoin",
    "binance": "binancecoin",
    "polkadot": "polkadot",
    "dot": "polkadot",
    "litecoin": "litecoin",
    "ltc": "litecoin",
    "chainlink": "chainlink",
    "link": "chainlink",
    "polygon": "matic-network",
    "matic": "matic-network",
    "avalanche": "avalanche-2",
    "avax": "avalanche-2",
}


def update_timestamp_cache():
    """Update the timestamp cache with current date and time"""
    # Only update once per minute to avoid excessive calls
    current_time = time.time()
    if current_time - timestamp_cache["last_update"] < 60:
        return

    # Get current UTC time
    now = datetime.datetime.now(datetime.timezone.utc)

    # Update cache
    timestamp_cache["last_update"] = current_time
    timestamp_cache["current_date"] = now.strftime("%Y-%m-%d")
    timestamp_cache["current_time"] = now.strftime("%H:%M:%S")
    timestamp_cache["current_year"] = now.year
    timestamp_cache["current_month"] = now.strftime("%B")
    timestamp_cache["current_day"] = now.day
    timestamp_cache["day_of_week"] = now.strftime("%A")


def get_bot_names_and_triggers():
    """Get the configured bot name, triggers, and ID"""
    # Default values
    bot_name = "Assistant"
    bot_names_list = ["assistant", "ai"]
    triggers = []
    bot_id = 0

    try:
        # Get custom name if defined
        bot_name = config.get("DEFAULT_BOT_NAME", "Assistant")

        # Get bot trigger words if configured
        if "TRIGGER" in config and config["TRIGGER"]:
            if isinstance(config["TRIGGER"], list):
                # Process special triggers
                for trigger in config["TRIGGER"]:
                    if (
                        trigger == "%BOT_NAME%"
                        or trigger == "%BOT_NICKNAME%"
                        or trigger == "%BOT_USERNAME%"
                    ):
                        # These are dynamic placeholders that will be replaced at runtime
                        # with the actual bot name, nickname, or username
                        continue
                    else:
                        triggers.append(trigger.lower())
            else:
                # Single trigger as string
                triggers.append(config["TRIGGER"].lower())

        # Add special case trigger
        if "%BOT_NAME%" in [t.upper() for t in triggers]:
            if trigger not in bot_names_list:
                bot_names_list.append(trigger.lower())

        # Convert bot name to list of variations (lowercase)
        bot_names_list = [bot_name.lower()]

        # Add common variations
        if " " in bot_name:
            # If name has spaces, add version without spaces
            bot_names_list.append(bot_name.lower().replace(" ", ""))

        if "-" in bot_name:
            # If name has hyphens, add version without hyphens
            bot_names_list.append(bot_name.lower().replace("-", ""))

        # Add triggers to the list if not already included
        for trigger in triggers:
            if trigger not in bot_names_list:
                bot_names_list.append(trigger.lower())

        # Get bot ID if available
        bot_id = config.get("BOT_ID", 0)
    except Exception as e:
        print(f"Error getting bot names and triggers: {e}")

    return {
        "name": bot_name,
        "names": bot_names_list,
        "triggers": triggers,
        "id": bot_id,
    }


def get_local_client():
    """Create a client for local LLM"""
    # For WSL to Windows, use the host.docker.internal hostname or the Windows
    # IP
    local_host = config.get("LOCAL_MODEL_HOST", "127.0.0.1")
    local_port = config.get("LOCAL_MODEL_PORT", "1234")
    local_url = f"http://{local_host}:{local_port}/v1"

    # Extract the API key if provided
    api_key = os.environ.get("API_KEY", "")

    # Create and return the local client
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=local_url, api_key=api_key)


def get_remote_client():
    """Create a client for remote API"""
    # Get the base URL and API key
    base_url = config.get("API_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = os.environ.get("API_KEY", "")

    # Catch common configuration errors
    if not api_key:
        print(
            "⚠️ WARNING: No API key found in environment variables. Please set API_KEY in your .env file."
        )

    # Create and return the remote client
    try:
        from openai import AsyncOpenAI

        return AsyncOpenAI(base_url=base_url, api_key=api_key)
    except Exception as e:
        print(f"Error creating remote client: {e}")
        return None


class AIProvider:
    """Provider for AI capabilities using either local or remote LLM"""

    def __init__(self, client=None, model=None):
        self.client = client
        self.model = model or config.get(
            "DEFAULT_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"
        )

    async def async_call(
        self, prompt=None, messages=None, temperature=0.2, max_tokens=2000
    ):
        """Make an async call to the LLM provider"""
        if not self.client:
            return "Error: No AI provider client configured."

        model_params = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            if messages:
                completion = await self.client.chat.completions.create(
                    messages=messages, **model_params
                )
                return completion.choices[0].message.content
            elif prompt:
                # For plain text completions
                completion = await self.client.completions.create(
                    prompt=prompt, **model_params
                )
                return completion.choices[0].text
            else:
                return "Error: No input provided to AI provider."
        except Exception as e:
            error_msg = f"Error in async_call: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return f"I encountered an error: {str(e)}"

    async def generate_text(
            self,
            messages=None,
            prompt=None,
            temperature=0.2,
            max_tokens=2000,
            **kwargs):
        """
        Generate text using the LLM provider

        This method provides compatibility with code that expects a generate_text method

        Args:
            messages: List of message objects
            prompt: String prompt (alternative to messages)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Generated text
        """
        try:
            if messages:
                # Log message structure for debugging
                print(f"AIProvider.generate_text: Using {len(messages)} messages")
                return await self.async_call(
                    messages=messages, temperature=temperature, max_tokens=max_tokens
                )
            elif prompt:
                print(
                    f"AIProvider.generate_text: Using prompt of length {len(prompt)}")
                return await self.async_call(
                    prompt=prompt, temperature=temperature, max_tokens=max_tokens
                )
            else:
                error_msg = (
                    "Either messages or prompt must be provided to generate_text"
                )
                print(f"ERROR: {error_msg}")
                traceback.print_exc()
                return f"I encountered an error: {error_msg}"
        except Exception as e:
            error_msg = f"Error in AIProvider.generate_text: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return f"I encountered an error while generating a response: {str(e)}"


async def get_ai_provider():
    """Get an AI provider instance for sequential thinking"""
    global remote_client, local_client

    # Use local model if enabled in config
    if config.get("USE_LOCAL_MODEL", False):
        if local_client is None:
            local_client = get_local_client()
        return AIProvider(
            client=local_client,
            model=config.get(
                "LOCAL_MODEL",
                "local"))
    else:
        # Use remote model (default)
        if remote_client is None:
            remote_client = get_remote_client()
        return AIProvider(
            client=remote_client,
            model=config.get(
                "DEFAULT_MODEL",
                "meta-llama/llama-4-maverick-17b-128e-instruct"),
        )


async def get_crypto_price(crypto_name):
    """Get real-time cryptocurrency price from CoinGecko API"""
    # Update timestamp cache to ensure we have current date and time
    update_timestamp_cache()

    crypto_name = crypto_name.lower().strip()

    # Clean the crypto name (remove words like "price", "value", etc.)
    clean_terms = [
        "price",
        "value",
        "worth",
        "current",
        "now",
        "today",
        "chart",
        "market",
        "cap",
    ]
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
        # If we can't identify the cryptocurrency, return None
        print(f"Could not identify cryptocurrency: {crypto_name}")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            # Get price in multiple currencies
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd,eur,gbp&include_market_cap=true&include_24hr_vol=true&include_24hr_change=true"

            # Add a timestamp parameter to prevent caching
            url += f"&_t={int(time.time())}"

            # Add headers to identify our bot and prevent rate limiting
            headers = {
                "User-Agent": "Discord-AI-Chatbot/1.0",
                "Accept": "application/json",
            }

            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()

                    if coin_id in data:
                        price_data = data[coin_id]

                        # Return structured data for direct use in on_message
                        # handler
                        return {
                            "price": price_data.get(
                                "usd", 0), "eur_price": price_data.get(
                                "eur", 0), "gbp_price": price_data.get(
                                "gbp", 0), "change_24h": price_data.get(
                                "usd_24h_change", 0), "market_cap": price_data.get(
                                "usd_market_cap", 0), "coin_id": coin_id, "timestamp": f"{
                                timestamp_cache['current_date']} {
                                timestamp_cache['current_time']} UTC", }
                    else:
                        print(
                            f"Coin ID {coin_id} not found in CoinGecko response")
                        return None
                else:
                    # If API rate limit is hit or other error, fall back to
                    # search
                    print(
                        f"CoinGecko API error: {response.status} - {await response.text()}"
                    )
                    return None
    except asyncio.TimeoutError:
        print(f"Timeout fetching {coin_id} price")
        return None
    except Exception as e:
        print(f"Error fetching crypto price: {e}")
        return None


async def search_internet(query):
    """Perform internet search using multiple APIs with robust fallbacks."""
    if not config["INTERNET_ACCESS"]:
        return "Internet access has been disabled by user"

    # Clean up the query first to improve search quality
    query = query.strip()

    # Clean up query indicators like opinion requests that might confuse search
    opinion_phrases = [
        "give me your honest opinion",
        "and give me your honest opinion",
        "what do you think",
        "what's your take",
        "please provide your thoughts",
        "analyze this",
    ]

    for phrase in opinion_phrases:
        if phrase in query.lower():
            query = query.lower().replace(phrase, "").strip()

    # Update timestamp cache
    update_timestamp_cache()

    # Log the final search query
    print(f"📚 Web Search Query: '{query}'")

    # Add current year/date to query for time-sensitive searches
    if any(
        term in query.lower()
        for term in ["current", "latest", "today", "now", "recent", "ongoing"]
    ):
        # Append current year if not already in query
        if str(timestamp_cache["current_year"]) not in query:
            query += f" {timestamp_cache['current_year']}"

        # Append current date for very time-sensitive queries
        if any(term in query.lower() for term in ["today", "now", "current"]):
            query += f" {timestamp_cache['current_date']}"

    # Track the search methods we've tried
    tried_methods = []

    # Result formatting function
    def format_results(results_list):
        formatted = ""
        for i, result in enumerate(results_list):
            title = result.get("title", "No Title")
            url = result.get("url", result.get("link", "No URL"))
            snippet = result.get(
                "snippet", result.get(
                    "body", "No description available."))
            formatted += f"[{i}] Title: {title}\nURL: {url}\nSnippet: {snippet}\n\n"
        return formatted

    # 1. Try Tavily API (if API key is available)
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if tavily_api_key:
        try:
            tried_methods.append("Tavily API")
            print("Using Tavily API for search")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.tavily.com/search",
                    headers={"Content-Type": "application/json"},
                    json={
                        "api_key": tavily_api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "include_domains": [],
                        "exclude_domains": [],
                        "max_results": 5,
                    },
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        if results:
                            print(
                                f"Tavily search successful, found {len(results)} results")
                            return format_results(results)
                        else:
                            print("Tavily search returned no results")
                    else:
                        err_text = await response.text()
                        print(
                            f"Tavily API error: {response.status} - {err_text}")
        except Exception as e:
            print(f"Tavily search error: {e}")

    # 2. Try DuckDuckGo search (as first fallback)
    try:
        tried_methods.append("DuckDuckGo")
        print("Using DuckDuckGo for search")
        async def run_ddg_search():
            from duckduckgo_search import AsyncDDGS

            # Create an async DuckDuckGo search client
            async_ddgs = AsyncDDGS()

            # Perform the search with proper error handling
            try:
                results = await async_ddgs.text(query, max_results=5)
                await async_ddgs.close()
                return results
            except Exception as e:
                print(f"Error: {e}")
                if async_ddgs:
                    await async_ddgs.close()
                return None

        # Execute the search with a timeout to avoid hanging
        try:
            # Longer timeout for DuckDuckGo as it can sometimes be slower
            ddg_results = await asyncio.wait_for(run_ddg_search(), timeout=10)
            if ddg_results:
                print(
                    f"DuckDuckGo search successful, found {len(ddg_results)} results")
                return format_results(ddg_results)
            else:
                print("DuckDuckGo search returned no results")
        except asyncio.TimeoutError:
            print("DuckDuckGo search timed out")
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")

    # 3. Try Google Search API using Serper.dev (if API key is available)
    serper_api_key = os.environ.get("SERPER_API_KEY")
    if serper_api_key:
        try:
            tried_methods.append("Serper.dev")
            print("Using Serper.dev (Google) for search")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": serper_api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "num": 5},
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        organic_results = data.get("organic", [])
                        if organic_results:
                            print(
                                f"Serper.dev search successful, found {len(organic_results)} results")
                            return format_results(organic_results)
                        else:
                            print("Serper.dev search returned no results")
                    else:
                        err_text = await response.text()
                        print(
                            f"Serper.dev API error: {response.status} - {err_text}")
        except Exception as e:
            print(f"Serper.dev search error: {e}")

    # If we've tried multiple search methods and all failed, return a response
    # mentioning the attempts
    if len(tried_methods) > 0:
        methods_tried = ", ".join(tried_methods)
        return f"I tried searching for information about '{query}' using {methods_tried}, but couldn't retrieve any results. Could you try rephrasing your question or being more specific?"
    else:
        return f"I couldn't search for information about '{query}' because no search providers are currently available. Please try again later."


async def stream_response(messages, model, client):
    """Stream a response from the AI provider"""
    try:
        # Create the streaming chat completion
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        # Initialize variables for tracking the response
        collected_chunks = []
        collected_messages = []

        # Process each chunk as it arrives
        async for chunk in stream:
            collected_chunks.append(chunk)
            chunk_message = chunk.choices[0].delta.content

            # Skip chunks without content
            if not chunk_message:
                continue

            collected_messages.append(chunk_message)
            yield chunk_message, False

        # Signal completion
        yield "", True
    except Exception as e:
        print(f"Error in stream_response: {e}")
        yield f"I encountered an error: {str(e)}", True


async def generate_response(instructions, history, stream=False):
    """
    Generate a response from an AI provider

    Args:
        instructions: The system instructions or prompt
        history: List of {'role': role, 'content': content} dicts
        stream: Whether to stream the response

    Returns:
        The generated response
    """
    try:
        # Create the client
        client = None
        
        # Prepare messages with system instructions and history
        messages = [{"role": "system", "content": instructions}]
        if history:
            messages.extend(history)

        # Determine whether to use local or remote LLM
        if config.get("USE_LOCAL_MODEL", False):
            client = get_local_client()
            model = config.get("LOCAL_MODEL_ID", "local")
        else:
            client = get_remote_client()
            model = config.get("MODEL_ID", "meta-llama/llama-4-maverick-17b-128e-instruct")

        if not client:
            return "Error: Could not initialize AI client. Please check your API key and base URL."

        # Generate the response
        if stream:
            return stream_response(messages, model, client)
        else:
            # Create the chat completion
            completion = await client.chat.completions.create(
                model=model, messages=messages
            )

            # Return the response
            return completion.choices[0].message.content
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return f"I encountered an error: {str(e)}"


async def chat_completion(
        messages,
        model=None,
        client=None,
        temperature=0.7,
        max_tokens=2000,
        **kwargs):
    """
    Generate a chat completion

    Args:
        messages: List of message objects
        model: Optional model ID
        client: Optional client to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters

    Returns:
        The generated text
    """
    # Get or create client
    if not client:
        if config.get("USE_LOCAL_MODEL", False):
            client = get_local_client()
            model = model or config.get("LOCAL_MODEL_ID", "local")
        else:
            client = get_remote_client()
            model = model or config.get(
                "MODEL_ID", "meta-llama/llama-4-maverick-17b-128e-instruct"
            )

    # Generate response
    try:
        if not client:
            return "Error: Could not initialize AI client. Please check your API key and base URL."

        # Create the chat completion
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Return the response
        return completion.choices[0].message.content
    except Exception as e:
        error_msg = f"Error in chat_completion: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return f"I encountered an error: {str(e)}"


async def duckduckgotool(query) -> str:
    """Legacy DuckDuckGo search tool"""
    # This is a wrapper for backward compatibility
    return await search_internet(query)


async def poly_image_gen(session, prompt):
    """Legacy image generation tool (not implemented)"""
    return "Image generation is not available in this version."


async def generate_image_prodia(prompt, model, sampler, seed, neg):
    """Generate an image using Prodia API"""
    # This function is a stub for backward compatibility
    return "Image generation is not available in this version."

    async def create_job(prompt, model, sampler, seed, neg):
        """Create an image generation job"""
        return {"job_id": None, "status": "failed"}

    try:
        # Create a job
        job = await create_job(prompt, model, sampler, seed, neg)
        job_id = job.get("job_id")

        if not job_id:
            return "Failed to create image generation job."

        # Return placeholder for now
        return f"Image would be generated with ID: {job_id}"
    except Exception as e:
        print(f"Error generating image: {e}")
        return f"Error generating image: {str(e)}"


async def text_to_speech(text):
    """Convert text to speech"""
    # This function is a stub for backward compatibility
    print("Text-to-speech requested but not implemented")
    return "Text-to-speech is not available in this version."


def create_prompt(
    message,
    username,
    user_id,
    channel,
    conversation_history=None,
    enhanced_instructions=None,
):
    """
    Generate a prompt for the chat

    Args:
        message: The user's message
        username: The username
        user_id: The user ID
        channel: The channel
        conversation_history: Optional conversation history
        enhanced_instructions: Optional enhanced instructions

    Returns:
        The formatted prompt
    """
    # Get the current timestamp
    update_timestamp_cache()
    current_date = timestamp_cache["current_date"]
    current_time = timestamp_cache["current_time"]

    # Get bot information
    bot_info = get_bot_names_and_triggers()
    bot_name = bot_info["name"]

    if not enhanced_instructions:
        instruction_file = config.get("DEFAULT_INSTRUCTION", "hand")

        # Load the instruction file
        try:
            instruction_file_path = f"instructions/{instruction_file}.txt"
            with open(instruction_file_path, "r", encoding="utf-8") as f:
                enhanced_instructions = f.read()
        except Exception as e:
            print(f"Error loading instruction file: {e}")
            enhanced_instructions = f"You are {bot_name}, a helpful assistant."

    # Get language setting
    language = config.get("LANGUAGE", "en")

    # Format the prompt
    prompt = f"{enhanced_instructions}\n\n"

    # Add language instructions
    if language != "en":
        prompt += f"Important: Respond in {language} language.\n\n"

    # Add timestamp information
    prompt += f"Current Date: {current_date}\n"
    prompt += f"Current Time: {current_time} UTC\n"

    # Add user and channel information
    prompt += f"User: {username} (ID: {user_id})\n"
    prompt += f"Channel: {channel}\n\n"

    # Add conversation history
    if conversation_history and len(conversation_history) > 0:
        prompt += "=== Previous Conversation ===\n"
        max_history = 20  # Limit history length
        for i, exchange in enumerate(conversation_history[-max_history:]):
            prompt += f"{exchange['role'].title()}: {exchange['content']}\n"
        prompt += "=== Current Conversation ===\n"

    # Add the user's current message
    prompt += f"User: {message}\n"
    prompt += f"{bot_name}:"

    return prompt


async def should_use_sequential_thinking(prompt):
    """
    Determine if sequential thinking should be used for the prompt

    Args:
        prompt: The user's prompt

    Returns:
        bool: Whether to use sequential thinking
    """
    # Keywords/phrases indicating complex queries that benefit from sequential
    # thinking
    complex_indicators = [
        "explain",
        "analyze",
        "evaluate",
        "compare",
        "contrast",
        "find the mistake",
        "step by step",
        "solution",
        "solve",
        "reasoning",
        "think through",
        "breakdown",
        "in detail",
        "comprehensive",
        "thoroughly",
        "elaborate",
        "debug",
        "diagnose",
        "pros and cons",
        "advantages and disadvantages",
        "implications",
        "consequences",
        "calculate",
        "determine",
        "derive",
        "predict",
        "forecast",
        "sequence of events",
        "series of steps",
    ]

    # Check for the presence of these indicators
    for indicator in complex_indicators:
        if indicator.lower() in prompt.lower():
            print(f"Sequential thinking indicator found: '{indicator}'")
            return True

    # Check the length of the prompt (longer prompts often need sequential
    # thinking)
    words = prompt.split()
    if len(words) > 30:  # Longer messages typically benefit from sequential thinking
        print("Long message detected, using sequential thinking")
        return True

    # Check for question marks (multiple questions suggest sequential thinking)
    if prompt.count("?") > 1:
        print("Multiple questions detected, using sequential thinking")
        return True

    # Check for lists or enumerations (often need sequential thinking)
    list_patterns = [
        r"\b\d+\.\s",  # Numbered lists like "1. Item"
        r"\n\s*[-*•]\s",  # Bullet points
        r"(firstly|secondly|finally)\b",  # Ordinal markers
        r"(first|second|third|next|then|last)\b",  # Sequential words
    ]

    for pattern in list_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            print(f"List pattern detected: '{pattern}'")
            return True

    # Default to not using sequential thinking for shorter/simpler prompts
    return False
