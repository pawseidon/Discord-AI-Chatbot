# Enhanced Discord AI Chatbot Features

This document outlines the advanced features added to the Discord AI Chatbot.

## Memory Management

The bot now has improved memory management capabilities:

- **Conversation Summarization**: Long conversations are automatically summarized to maintain context while reducing token usage
- **User Preferences**: The bot remembers user preferences and topics of interest

### Commands for Memory Management

- `/preferences` - View your current preferences
- `/preferences response_length:[short|medium|long]` - Set your preferred response length
- `/preferences voice_enabled:[true|false]` - Enable or disable voice responses
- `/preferences use_embeds:[true|false]` - Enable or disable rich embeds in responses

## Smart Response Formatting

Responses are now formatted intelligently based on content type:

- **Code blocks** are automatically detected and formatted with syntax highlighting
- **Data/lists** are presented in a clean, organized manner
- **Long responses** are split properly across multiple messages
- **Rich embeds** make information more visually appealing

## Message Threading

The bot can now create threads for complex conversations:

- Automatically creates threads for complex or long responses
- Maintains context between the original message and thread
- Makes multi-turn conversations easier to follow
- Reduces channel clutter for detailed explanations

## Fallback Systems

The bot includes fallback mechanisms for when the LLM is unavailable:

- **Automatic fallback**: When connection issues occur with the LLM, the bot switches to fallback mode
- **Cached responses**: Common responses are cached for offline use
- **Admin control**: Use `/toggleoffline` to manually enable/disable fallback mode

## Enhanced Real-Time Data

The bot now has specialized handlers for different types of real-time data:

- **Cryptocurrency prices**: Gets accurate, up-to-date crypto prices with market data
- **News**: Fetches and summarizes current news on topics
- **Time-aware**: All responses include current date/time awareness

## Voice Interaction

Voice features have been enhanced:

- **User preference**: Enable/disable voice per user
- **Selective usage**: Only convert to speech when explicitly enabled

## Configuration Tips

- For optimal performance, ensure the `bot_data` directory exists and is writable
- The bot automatically creates necessary subdirectories on first run
- To reset user preferences, delete the corresponding file in `bot_data/user_preferences`
- To clear the news cache, delete files in `bot_data/news_cache`

## Database Structure

- **User preferences**: Stored in JSON files in `bot_data/user_preferences`
- **Memory summaries**: Managed in `bot_data/memory_summaries`
- **News cache**: Stored in `bot_data/news_cache`
- **Fallback system**: Uses SQLite database in `bot_data/fallbacks` 