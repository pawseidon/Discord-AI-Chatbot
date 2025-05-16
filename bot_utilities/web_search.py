"""
Web Search Module

This module provides utilities for searching the web and retrieving information.
"""

import os
import json
import urllib.parse
import asyncio
import aiohttp
import traceback
from typing import Dict, Any, List, Optional, Union

async def search_web(query: str, num_results: int = 5, safe_search: bool = True) -> str:
    """
    Search the web for information on a specific query.
    
    Args:
        query: The search query
        num_results: Number of search results to retrieve
        safe_search: Whether to enable safe search filtering
        
    Returns:
        str: Formatted search results
    """
    try:
        # Clean up the query to preserve all important words
        cleaned_query = query
        
        # Remove common prefixes that don't add context
        for prefix in ["find", "search for", "tell me about", "info on", "information about"]:
            if cleaned_query.lower().startswith(prefix):
                cleaned_query = cleaned_query[len(prefix):].strip()
        
        # Ensure we don't strip it down to just one word due to overzealous cleaning
        if len(cleaned_query.split()) < 2 and len(query.split()) >= 2:
            # Revert to original query if cleaned version is too short
            cleaned_query = query
            
        # Try DuckDuckGo search first
        results = await search_duckduckgo(cleaned_query, num_results)
        
        if not results or not results.strip():
            # Fallback to secondary search method
            results = await search_fallback(cleaned_query, num_results)
            
        if not results or not results.strip():
            # Final fallback message
            return "I couldn't find any relevant information on this topic. Let me answer based on my existing knowledge."
            
        return results
    except Exception as e:
        print(f"Error in search_web: {str(e)}")
        traceback.print_exc()
        return f"I encountered an error while searching for information: {str(e)}"

async def search_duckduckgo(query: str, num_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo API.
    
    Args:
        query: The search query
        num_results: Number of search results to retrieve
        
    Returns:
        str: Formatted search results
    """
    # Get API key from environment variable
    api_key = os.getenv("DUCKDUCKGO_API_KEY", "")
    
    # If no API key is available, return empty string
    if not api_key:
        print("DuckDuckGo API key not configured")
        return ""
        
    try:
        # Format the search URL
        encoded_query = urllib.parse.quote(query)
        search_url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&pretty=1&kl=us-en&kp=-1&t=Discord-AI-Chatbot"
        
        # Make the API request
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url) as response:
                if response.status != 200:
                    print(f"DuckDuckGo search failed with status {response.status}")
                    return ""
                    
                data = await response.json()
                
        # Extract and format the results
        formatted_results = []
        
        # Add abstract text if available
        if data.get('Abstract'):
            formatted_results.append(f"Abstract: {data['Abstract']}")
            
        # Add definition if available
        if data.get('Definition'):
            formatted_results.append(f"Definition: {data['Definition']}")
            
        # Add related topics
        for topic in data.get('RelatedTopics', [])[:num_results]:
            if 'Text' in topic:
                formatted_results.append(f"• {topic['Text']}")
                
        # Join results with newlines
        return "\n\n".join(formatted_results)
    except Exception as e:
        print(f"Error in search_duckduckgo: {str(e)}")
        traceback.print_exc()
        return ""
        
async def search_fallback(query: str, num_results: int = 5) -> str:
    """
    Fallback search method when primary search fails.
    
    Args:
        query: The search query
        num_results: Number of search results to retrieve
        
    Returns:
        str: Formatted search results
    """
    # Get API key from environment variable
    api_key = os.getenv("TAVILY_API_KEY", "")
    
    # If no API key is available, return empty string
    if not api_key:
        print("Tavily API key not configured")
        return ""
        
    try:
        print(f"📚 Web Search Query: '{query}'")
        
        # Format the search URL
        search_url = "https://api.tavily.com/search"
        headers = {
            "content-type": "application/json",
            "x-api-key": api_key
        }
        
        # Create request payload - always searching with full query, not just keywords
        payload = {
            "query": query,
            "search_depth": "advanced",
            "max_results": num_results,
            "include_domains": [],
            "exclude_domains": [],
            "include_answer": True,
            "include_raw_content": False,  # Save bandwidth
            "include_images": False,       # No images needed
            "include_similar_results": False
        }
        
        # Make the API request
        async with aiohttp.ClientSession() as session:
            async with session.post(search_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    print(f"Tavily search failed with status {response.status}")
                    return ""
                    
                data = await response.json()
                
        # Extract and format the results
        formatted_results = []
        
        # Add direct answer if available
        if data.get('answer'):
            formatted_results.append(f"Answer: {data['answer']}")
            
        # Add search results
        for result in data.get('results', []):
            if result.get('content'):
                formatted_results.append(f"• {result['title']}: {result['content']}")
                
        # Join results with newlines
        return "\n\n".join(formatted_results)
    except Exception as e:
        print(f"Error in search_fallback: {str(e)}")
        traceback.print_exc()
        return ""

async def research_topic(query: str, depth: int = 2) -> str:
    """
    Perform deep research on a topic by querying multiple sources.
    
    Args:
        query: The research query
        depth: Research depth (1-3, higher means more thorough)
        
    Returns:
        str: Formatted research results
    """
    try:
        # Formulate research sub-queries
        sub_queries = [
            query,
            f"latest developments in {query}",
            f"{query} detailed explanation",
            f"{query} pros and cons"
        ]
        
        # Limit the number of sub-queries based on depth
        sub_queries = sub_queries[:depth+2]
        
        # Search for each sub-query
        results = []
        for sub_query in sub_queries:
            sub_result = await search_web(sub_query, num_results=3)
            if sub_result and sub_result.strip():
                results.append(f"Research on '{sub_query}':\n{sub_result}")
                
        # Join all results
        combined_results = "\n\n---\n\n".join(results)
        
        if not combined_results or not combined_results.strip():
            return "I couldn't find detailed research on this topic. Let me answer based on my existing knowledge."
            
        return combined_results
    except Exception as e:
        print(f"Error in research_topic: {str(e)}")
        traceback.print_exc()
        return f"I encountered an error while researching this topic: {str(e)}" 