import aiohttp
import asyncio
import json
import time
import os
from datetime import datetime, timedelta
from bot_utilities.config_loader import config

# News cache to avoid repeated API calls for the same topics
NEWS_CACHE_DIR = "bot_data/news_cache"
os.makedirs(NEWS_CACHE_DIR, exist_ok=True)

# Define supported news categories
NEWS_CATEGORIES = [
    "world", "nation", "business", "technology", "entertainment", 
    "sports", "science", "health", "politics"
]

# Initialize with your API key(s) - supports multiple services if needed
# You can use free-tier API keys or replace with services you prefer
NEWS_API_KEYS = {
    "newsapi": os.environ.get("NEWS_API_KEY", ""),  # NewsAPI.org
    "gnews": os.environ.get("GNEWS_API_KEY", "")    # GNews
}

async def get_news_for_topic(topic, limit=5, force_refresh=False):
    """
    Get news articles for a specific topic or category
    
    Args:
        topic (str): News topic or category to search for
        limit (int): Maximum number of articles to return
        force_refresh (bool): Force refresh the cache
        
    Returns:
        list: List of news article dictionaries
    """
    # Clean up topic and convert to lowercase
    topic = topic.strip().lower()
    
    # First check if we have cached results that are still fresh
    cache_file = os.path.join(NEWS_CACHE_DIR, f"{topic.replace(' ', '_')}.json")
    
    if os.path.exists(cache_file) and not force_refresh:
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # Check if cache is still fresh (less than 30 minutes old)
            cache_time = cache_data.get('timestamp', 0)
            if (time.time() - cache_time) < 1800:  # 30 minutes in seconds
                print(f"Using cached news for '{topic}'")
                return cache_data.get('articles', [])[:limit]
        except Exception as e:
            print(f"Error reading news cache: {e}")
    
    # If we reach here, we need to fetch fresh data
    articles = []
    
    # Try NewsAPI first if key is available
    if NEWS_API_KEYS["newsapi"]:
        articles = await fetch_from_newsapi(topic, limit)
    
    # Fall back to GNews if NewsAPI failed or returned no results
    if not articles and NEWS_API_KEYS["gnews"]:
        articles = await fetch_from_gnews(topic, limit)
    
    # If both failed, try scraping approach without API key
    if not articles:
        articles = await fetch_news_without_api(topic, limit)
    
    # Cache the results if we got any
    if articles:
        cache_data = {
            'timestamp': time.time(),
            'articles': articles
        }
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error writing news cache: {e}")
    
    return articles[:limit]

async def fetch_from_newsapi(topic, limit=5):
    """Fetch news from NewsAPI.org"""
    api_key = NEWS_API_KEYS["newsapi"]
    if not api_key:
        return []
    
    # Format today's date and 7 days ago for the API
    today = datetime.now().strftime('%Y-%m-%d')
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # Determine if topic is a category or a search term
    if topic in NEWS_CATEGORIES:
        url = f"https://newsapi.org/v2/top-headlines?category={topic}&language=en&apiKey={api_key}"
    else:
        url = f"https://newsapi.org/v2/everything?q={topic}&language=en&from={week_ago}&to={today}&sortBy=relevancy&apiKey={api_key}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('status') == 'ok':
                        articles = data.get('articles', [])
                        return [
                            {
                                'title': article.get('title', 'No title'),
                                'description': article.get('description', 'No description'),
                                'url': article.get('url', ''),
                                'source': article.get('source', {}).get('name', 'Unknown'),
                                'published': article.get('publishedAt', '')
                            }
                            for article in articles
                        ]
                print(f"NewsAPI error: {response.status}")
                return []
    except Exception as e:
        print(f"Error fetching from NewsAPI: {e}")
        return []

async def fetch_from_gnews(topic, limit=5):
    """Fetch news from GNews API"""
    api_key = NEWS_API_KEYS["gnews"]
    if not api_key:
        return []
    
    url = f"https://gnews.io/api/v4/search?q={topic}&lang=en&max={limit}&apikey={api_key}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])
                    return [
                        {
                            'title': article.get('title', 'No title'),
                            'description': article.get('description', 'No description'),
                            'url': article.get('url', ''),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'published': article.get('publishedAt', '')
                        }
                        for article in articles
                    ]
                print(f"GNews API error: {response.status}")
                return []
    except Exception as e:
        print(f"Error fetching from GNews: {e}")
        return []

async def fetch_news_without_api(topic, limit=5):
    """
    Fallback method to get news without API keys
    Uses a simple HTTP request to a news aggregator
    """
    # This is a fallback that uses a public RSS feed converted to JSON
    url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        async with aiohttp.ClientSession() as session:
            # First try directly with RSS
            async with session.get(url) as response:
                if response.status == 200:
                    # Parse the RSS feed (simplified)
                    content = await response.text()
                    articles = []
                    
                    # Very basic parsing - in a real implementation, use proper XML parsing
                    for item in content.split("<item>")[1:]:
                        if len(articles) >= limit:
                            break
                        
                        try:
                            title = item.split("<title>")[1].split("</title>")[0]
                            link = item.split("<link>")[1].split("</link>")[0]
                            pub_date = item.split("<pubDate>")[1].split("</pubDate>")[0]
                            
                            articles.append({
                                'title': title,
                                'description': 'No description available',
                                'url': link,
                                'source': 'Google News',
                                'published': pub_date
                            })
                        except:
                            continue
                    
                    return articles
    except Exception as e:
        print(f"Error in fallback news fetching: {e}")
        return []

def format_news_for_prompt(articles):
    """Format news articles for inclusion in a prompt"""
    if not articles:
        return "No recent news found on this topic."
    
    formatted = "Here are some recent news headlines on this topic:\n\n"
    for i, article in enumerate(articles):
        pub_date = article.get('published', '')
        # Try to parse and format the date if possible
        try:
            if pub_date:
                dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                pub_date = dt.strftime("%Y-%m-%d %H:%M")
        except:
            # If date parsing fails, use as is
            pass
            
        formatted += f"{i+1}. {article['title']}\n"
        formatted += f"   Source: {article['source']} ({pub_date})\n"
        if article.get('description'):
            formatted += f"   {article['description']}\n"
        formatted += f"   URL: {article['url']}\n\n"
    
    return formatted

async def detect_news_query(message):
    """Detect if a message is asking about news or current events"""
    message_lower = message.lower()
    news_indicators = [
        "news", "latest news", "recent events", "what's happening", 
        "current events", "today's headlines", "breaking news",
        "what happened", "latest developments", "update on",
        "recent news about", "headlines about"
    ]
    
    # Check if message contains news indicators
    is_news_query = any(indicator in message_lower for indicator in news_indicators)
    
    if is_news_query:
        # Try to extract the topic from the query
        topic = None
        for indicator in sorted(news_indicators, key=len, reverse=True):
            if indicator in message_lower:
                parts = message_lower.split(indicator)
                if len(parts) > 1:
                    # Look for topic after the indicator
                    if parts[1].strip():
                        topic = parts[1].strip().split("?")[0].strip()
                    # If not found after, look before the indicator
                    elif parts[0].strip():
                        topic = parts[0].strip().split()[-1]
                break
        
        # If no specific topic found, default to general news
        if not topic or len(topic) < 3:
            topic = "world news"
            
        return True, topic
    
    return False, None

async def get_news_context(message_content):
    """
    Parse a message for news queries and return relevant news context
    
    Args:
        message_content (str): The user's message
    
    Returns:
        str: Formatted news context or empty string if not a news query
    """
    is_news_query, topic = await detect_news_query(message_content)
    
    if is_news_query and topic:
        print(f"Detected news query about: {topic}")
        articles = await get_news_for_topic(topic)
        if articles:
            return format_news_for_prompt(articles)
    
    return "" 