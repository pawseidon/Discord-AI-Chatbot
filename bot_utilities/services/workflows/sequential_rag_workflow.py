"""
Sequential RAG Workflow Module

This module implements a workflow that combines sequential thinking with RAG (Retrieval-Augmented Generation).
"""

import os
import re
import json
import time
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import aiohttp

# Import workflow result and helper functions
from bot_utilities.services.workflow_helper import (
    WorkflowResult,
    standardize_workflow_output,
    record_workflow_usage
)

# Import AI provider utilities
from bot_utilities.ai_utils import get_ai_provider


async def execute_sequential_rag_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None,
    search_results: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Union[str, Dict[str, Any]]:
    """
    Execute a workflow that combines sequential thinking with information retrieval (RAG)
    
    Args:
        query: The user query
        user_id: User ID for memory lookup
        conversation_id: Conversation ID for context
        update_callback: Optional callback for streaming updates
        search_results: Optional pre-fetched search results
        context: Optional additional context for the workflow
        
    Returns:
        WorkflowResult or str: Result of the workflow
    """
    # Get AI provider
    llm_provider = await get_ai_provider()
    
    # Set default context if none provided
    if context is None:
        context = {}
    
    # Extract context parameters
    display_raw_results = context.get("display_raw_results", False)
    retrieved_information = context.get("retrieved_information", search_results)
    
    # Start timing
    start_time = time.time()
    
    # Special handlers for certain query types
    is_crypto_query = is_crypto_price_query(query)
    is_social_query = is_social_relationship_query(query)
    
    try:
        # First, check if it's a crypto price query
        if is_crypto_query:
            # Extract cryptocurrency name
            crypto_name = extract_crypto_name(query)
            
            # Notify about processing
        if update_callback:
                await update_callback("thinking", {"thinking": "💰 Fetching latest cryptocurrency price data..."})
            
            # Handle crypto price query
        result = await handle_crypto_price_query(crypto_name)
            
            # Calculate execution time
        execution_time = time.time() - start_time
            
            # Record workflow usage
        await record_workflow_usage(
                workflow_name="sequential_rag",
                query=query,
                user_id=user_id,
                execution_time=execution_time
            )
            
            # Return the result
        if result:
                return result
            # If crypto handler failed, continue with standard processing
        
        # Next, check if it's a social/relationship query
        elif is_social_query:
            # Notify about processing
            if update_callback:
                await update_callback("thinking", {"thinking": "👥 Processing social relationship query..."})
            
            # Handle social relationship query
            result = await handle_social_relationship_query(query, user_id)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Record workflow usage
            await record_workflow_usage(
                workflow_name="sequential_rag",
                query=query,
                user_id=user_id,
                execution_time=execution_time
            )
            
            # Return the result
            if result:
                return result
            # If social query handler failed, continue with standard processing
        
        # Notify about reasoning switch
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["rag", "sequential"],
                "is_combined": True
            })
            
            # Notify starting retrieval
            await update_callback("thinking", {"thinking": "🔄→📚 Gathering information..."})
        
        # If search results weren't provided or retrieved from context, search for them
        if not retrieved_information:
            if update_callback:
                await update_callback("thinking", {"thinking": "📚 Searching for relevant information..."})
            
            # Search the web
            from bot_utilities.ai_utils import search_internet
            retrieved_information = await search_internet(query)
        
        # Notify about moving to sequential processing
        if update_callback:
            await update_callback("thinking", {"thinking": "📚→🔄 Organizing information sequentially..."})
        
        # If display_raw_results is True, just return the search results
        if display_raw_results:
            execution_time = time.time() - start_time
            await record_workflow_usage(
                workflow_name="sequential_rag",
                query=query,
                user_id=user_id,
                execution_time=execution_time
            )
            return f"## Search Results: {query}\n\n{retrieved_information}"
        
        # Generate sequential thinking response using the LLM
        sequential_prompt = [
            {"role": "system", "content": f"""You are an AI assistant specialized in breaking down information systematically and sequentially.

First analyze the user's original query: "{query}"

Then review the search results below to extract key information, organize it logically, and present a comprehensive response.

Follow these steps in your thinking:
1. Identify the main question/topic from the query
2. Extract relevant facts, data, and perspectives from the search results
3. Organize information into a coherent structure with clear headings
4. Present information in a systematic, sequential manner
5. Focus on accuracy and completeness

IMPORTANT: Format your response with appropriate Markdown formatting including:
- Clear section headings with # or ## for main sections
- Bullet points for lists
- Bold or italics for emphasis
- Code blocks for any technical content
- Tables when presenting comparative data

Base your response solely on the information in the search results and your general knowledge. If the search results are insufficient, acknowledge limitations.

Focus on creating a comprehensive, well-structured answer."""},
            {"role": "user", "content": f"Here is my query: {query}\n\nHere are the search results:\n\n{retrieved_information}\n\nPlease provide a comprehensive, sequential analysis based on this information."}
        ]
        
        # Generate the sequential thinking response
        sequential_response = await llm_provider.generate_text(
            messages=sequential_prompt,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Execution time
        execution_time = time.time() - start_time
        
        # Record workflow usage
        await record_workflow_usage(
            workflow_name="sequential_rag",
            query=query,
            user_id=user_id,
            execution_time=execution_time
        )
        
        # Return the result
        return sequential_response
        
    except Exception as e:
        # Log the error
        print(f"Error in sequential_rag_workflow: {e}")
        traceback.print_exc()
        
        # Record workflow usage even on error
        execution_time = time.time() - start_time
        await record_workflow_usage(
            workflow_name="sequential_rag",
            query=query,
            user_id=user_id,
            execution_time=execution_time
        )
        
        # Return a fallback response
        if retrieved_information:
            return f"I was unable to process the information in a sequential manner due to an error, but here are the raw search results:\n\n{retrieved_information}"
        else:
            return "I encountered an error processing your request with sequential thinking and couldn't retrieve the necessary information. Please try rephrasing your query."


def is_crypto_price_query(query: str) -> bool:
    """
    Detect if a query is asking for cryptocurrency price information
    
    Args:
        query: The user query
        
    Returns:
        bool: True if the query is about cryptocurrency prices
    """
    # List of common cryptocurrencies
    crypto_terms = [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency", 
        "dogecoin", "doge", "xrp", "ripple", "cardano", "ada", "solana", "sol",
        "bnb", "binance", "polkadot", "dot", "litecoin", "ltc", "chainlink", "link"
    ]
    
    # Price-related terms
    price_terms = [
        "price", "value", "worth", "cost", "market", "trading at", "going for",
        "current", "now", "today", "rate", "exchange", "usd", "dollar", "eur", "euro"
    ]
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Check if query contains both crypto terms and price terms
    has_crypto = any(term in query_lower for term in crypto_terms)
    has_price = any(term in query_lower for term in price_terms)
    
    return has_crypto and has_price


async def handle_crypto_price_query(query: str) -> str:
    """
    Handle a cryptocurrency price query by fetching current price data
    
    Args:
        query: The user query about cryptocurrency prices
        
    Returns:
        str: Formatted cryptocurrency price information
    """
    # Extract which cryptocurrency to look up
    crypto_map = {
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
        "link": "chainlink"
    }
    
    # Determine which crypto to fetch based on the query
    query_lower = query.lower()
    crypto_id = None
    crypto_name = None
    
    for term, coin_id in crypto_map.items():
        if term in query_lower:
            crypto_id = coin_id
            crypto_name = term
            break
    
    # If we couldn't identify a specific cryptocurrency, default to Bitcoin
    if not crypto_id:
        crypto_id = "bitcoin"
        crypto_name = "Bitcoin"
    
    try:
        # Use CoinGecko API to fetch current price data
        async with aiohttp.ClientSession() as session:
            # Add a custom user agent to prevent 403 errors
            headers = {
                "User-Agent": "Discord-AI-Chatbot/1.0",
                "Accept": "application/json"
            }
            
            url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false"
            
            # Add a timeout to the request to prevent hanging
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract relevant data
                    current_price = data["market_data"]["current_price"]["usd"]
                    price_change_24h_percentage = data["market_data"]["price_change_percentage_24h"]
                    market_cap = data["market_data"]["market_cap"]["usd"]
                    high_24h = data["market_data"]["high_24h"]["usd"]
                    low_24h = data["market_data"]["low_24h"]["usd"]
                    
                    # Format the response
                    crypto_display_name = data["name"]
                    symbol = data["symbol"].upper()
                    
                    # Ensure the symbol is not empty
                    if not symbol:
                        # Map of coin IDs to their ticker symbols
                        ticker_symbols = {
                            "bitcoin": "BTC",
                            "ethereum": "ETH",
                            "dogecoin": "DOGE",
                            "ripple": "XRP",
                            "cardano": "ADA",
                            "solana": "SOL",
                            "binancecoin": "BNB",
                            "polkadot": "DOT",
                            "litecoin": "LTC",
                            "chainlink": "LINK",
                            "matic-network": "MATIC",
                            "avalanche-2": "AVAX"
                        }
                        symbol = ticker_symbols.get(crypto_id, crypto_id.upper() if len(crypto_id) <= 4 else "")
                    
                    # Determine if price is up or down with appropriate emoji
                    trend_emoji = "🟢" if price_change_24h_percentage >= 0 else "🔴"
                    
                    response_text = f"""# 💰 {crypto_display_name} ({symbol}) Price Information

## Current Price
**${current_price:,.2f}** {trend_emoji} {price_change_24h_percentage:.2f}% (24h)

## 24-Hour Range
- **High:** ${high_24h:,.2f}
- **Low:** ${low_24h:,.2f}

## Market Data
- **Market Cap:** ${market_cap:,.0f}

*Data sourced from CoinGecko as of {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}*

> This is real-time market data and prices may change rapidly. This information should not be considered financial advice.
"""
                    return response_text
                elif response.status == 429:
                    # Rate limited - provide a meaningful error message
                    return f"I'm getting too many requests to the cryptocurrency price API. Please try again in a few minutes."
                else:
                    # Log the specific error for debugging
                    error_text = await response.text()
                    print(f"CoinGecko API error (status {response.status}): {error_text}")
                    return None  # Return None to fall back to regular search results
    
    except asyncio.TimeoutError:
        print(f"Timeout fetching {crypto_name} price")
        return None  # Return None to fall back to regular search results
    except Exception as e:
        print(f"Error fetching crypto price: {str(e)}")
        return None  # Return None to fall back to regular search results


def is_social_relationship_query(query: str) -> bool:
    """
    Detect if a query is about social relationships, dating, or meeting people
    
    Args:
        query: The user query
        
    Returns:
        bool: True if the query is about social relationships
    """
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Social/relationship-related keywords
    relationship_terms = [
        "find women", "find men", "find a girlfriend", "find a boyfriend", 
        "get a girlfriend", "get a boyfriend", "dating", "relationship", 
        "attract", "impress", "approach", "ask out", "talk to girls", "talk to women",
        "talk to guys", "talk to men", "find a date", "get a date", "get matches",
        "dating app", "tinder", "bumble", "hinge", "find love", "find partner"
    ]
    
    # Check if query contains relationship patterns
    for term in relationship_terms:
        if term in query_lower:
            return True
    
    # Check for question patterns about meeting people
    question_patterns = [
        r"how (do|can|should) (i|you|one|we) (find|get|meet|attract|date|approach|talk to) (women|men|girls|guys|dates|partners)",
        r"where (do|can|should) (i|you|one|we) (find|get|meet|attract|date) (women|men|girls|guys|dates|partners)",
        r"ways to (find|get|meet|attract|date|approach|talk to) (women|men|girls|guys|dates|partners)"
    ]
    
    for pattern in question_patterns:
        if re.search(pattern, query_lower):
            return True
            
    return False


async def handle_social_relationship_query(query: str, llm_provider) -> str:
    """
    Handle a social/relationship query with appropriate guidance
    
    Args:
        query: The user query about social relationships
        llm_provider: AI provider for generating response
        
    Returns:
        str: Thoughtful guidance on social relationships
    """
    try:
        # Create a prompt for generating a respectful, thoughtful response
        relationship_prompt = f"""
        The user has asked: "{query}"
        
        This appears to be a question about social relationships, dating, or meeting people.
        
        Provide thoughtful, respectful guidance that:
        1. Emphasizes healthy social connection based on mutual respect and shared interests
        2. Focuses on personal growth, confidence, and authentic communication
        3. Suggests community involvement, social activities, and interest-based groups
        4. Avoids any manipulative tactics or viewing relationships as transactional
        5. Includes specific, actionable advice the person can implement
        
        Structure your response with:
        - A thoughtful perspective on healthy social connections
        - Practical suggestions for meeting people in various contexts
        - Guidance on communication and authentic relationship building
        - A section on personal development that builds confidence
        - Emphasis on respect, consent, and treating others as individuals
        
        Keep the tone supportive, respectful, and inclusive.
        """
        
        # Generate a tailored response
        response = await llm_provider.generate_text(
            messages=[
                {"role": "system", "content": "You are providing thoughtful guidance on healthy social connections and relationships."},
                {"role": "user", "content": relationship_prompt}
            ],
            temperature=0.4,
            max_tokens=1200
        )
        
        # Format the final response with appropriate framing
        formatted_response = f"""# 👥 Building Healthy Social Connections

{response}

> Remember that meaningful connections are built on mutual respect, shared interests, and authentic communication."""
        
        return formatted_response
        
    except Exception as e:
        print(f"Error handling social relationship query: {str(e)}")
        return None  # Return None to fall back to regular search results 

def extract_crypto_name(query: str) -> str:
    """
    Extract cryptocurrency name from a query
    
    Args:
        query: The user query
        
    Returns:
        str: The extracted cryptocurrency name
    """
    # Convert to lowercase and strip
    query = query.lower().strip()
    
    # Clean the query by removing price-related terms
    clean_terms = [
        "price", "value", "worth", "current", "now", "today", "chart", 
        "market", "cap", "cost", "trading at", "going for", "rate"
    ]
    
    # Remove terms from query
    for term in clean_terms:
        query = query.replace(term, "").strip()
    
    # List of common cryptocurrencies to check against
    crypto_terms = [
        "bitcoin", "btc", "ethereum", "eth", "dogecoin", "doge", 
        "xrp", "ripple", "cardano", "ada", "solana", "sol",
        "bnb", "binance", "polkadot", "dot", "litecoin", "ltc", 
        "chainlink", "link"
    ]
    
    # Find which crypto term is in the query
    for term in crypto_terms:
        if term in query:
            return term
    
    # If no specific match, return the cleaned query
    return query 