# Discord AI Chatbot 🤖

A powerful Discord bot powered by advanced AI models, offering conversational capabilities, research assistance, and much more.

## Table of Contents
- [Features](#features)
- [Commands](#commands)
- [Installation Guide](#installation-guide)
- [Configuration Options](#configuration-options)
- [Core Capabilities](#core-capabilities)
- [Service Architecture](#service-architecture)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Features

### Conversational AI 💬
- **Multi-Language Support**: Communicate in 16 different languages including English, Spanish, French, Chinese, and more.
- **Multiple Personalities**: Choose from various personalities like DAN, Luna, Suzume, or create your own custom persona.
- **Smart Mention Recognition**: The bot responds when mentioned or called by name, similar to modern AI assistants.
- **Thread Support**: Creates and maintains context within Discord threads for organized conversations.
- **Streaming Responses**: Watch responses appear in real-time as they're generated for a more interactive experience.
- **Memory System**: Remembers previous conversations for more contextual interactions.

### Agent System 🧠
- **Web Search**: Search the internet for real-time information using DuckDuckGo with fallback mechanisms.
- **Research Assistant**: Deep research on topics with multiple sources.
- **Task Automation**: Get step-by-step guides for complex tasks.
- **Crypto Price Tracking**: Real-time cryptocurrency price information.
- **Knowledge Base**: Store and retrieve server-specific information.

### User Experience 🌟
- **Customizable Preferences**: Each user can configure their own experience including response length and more.
- **Multilingual Interface**: The bot's commands and responses adapt to your language settings.
- **Channel-Specific Control**: Enable/disable the bot in specific channels.

## Commands

### Natural Language Interaction
No commands needed for most interactions! Simply:
- **Direct Mentions**: `@BotName [your message]` - Talk directly to the bot
- **Using Bot's Name**: `Hey Bot, [your message]` - The bot recognizes when you use its name
- **Keyword Triggers**: Use configured trigger words like "assistant" or "ai" in messages
- **Replies**: Reply to the bot's messages to continue conversations

### User Settings
- **Reasoning Mode**: `Set my reasoning mode to sequential` or `Use creative mode for my questions`
- **Workflow Mode**: `Enable workflow mode` or `Disable workflow mode` (requires LangGraph)
- **Privacy Controls**: `Clear my data` or `Forget me` to remove your conversation history
- **Conversation Reset**: `Reset our conversation` or `Start over` to begin fresh

### Standard Commands
- `/help` - Display information about available features
- `/clear` - Clear your conversation history and data
- `/reset` - Reset the current conversation but keep your preferences
- `/toggleactive` - Toggle the bot active/inactive in the current channel (Admin only)
- `/toggleinactive` - Toggle the bot inactive/active in the current channel (Admin only)

### Reasoning Types
Include these phrases to explicitly request a specific reasoning approach:
- **Sequential Thinking**: `Think through this step by step: [question]`
- **Information Retrieval**: `Search for information about [topic]`
- **Verification**: `Verify whether [claim] is accurate`
- **Knowledge Base**: `Explain in detail how [topic] works`
- **Creative Mode**: `Write a creative story about [topic]`
- **Graph-of-Thought**: `Map the connections between [concepts]`
- **Multi-Agent**: `Analyze [topic] from multiple perspectives`

### Advanced Features
Use these natural language requests for specialized functions:
- **Web Search**: `Search the web for [query]` or `Find recent information about [topic]`

## Installation Guide

### Prerequisites
- Python 3.10 or higher
- A Discord Bot Token (from [Discord Developer Portal](https://discord.com/developers/applications))
- A Groq API Key (from [Groq Console](https://console.groq.com/keys))

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mishalhossin/Discord-AI-Chatbot
   cd Discord-AI-Chatbot
   ```

2. **Install requirements**
   ```bash
   python3.10 -m pip install -r requirements.txt
   ```

3. **Configure environment variables**
   - Rename `example.env` to `.env`
   - Add your Discord token and Groq API key:
     ```
     DISCORD_TOKEN=YOUR_DISCORD_BOT_TOKEN
     API_KEY=YOUR_GROQ_API_KEY
     TAVILY_API_KEY=YOUR_TAVILY_API_KEY_OPTIONAL
     ```

4. **Start the bot**
   ```bash
   python main.py
   ```

5. **Invite the bot to your server**
   - Use the invite link provided in the console output

### Docker Installation
```bash
# Clone repository
git clone https://github.com/mishalhossin/Discord-AI-Chatbot
cd Discord-AI-Chatbot

# Configure environment variables in .env file
# Then run with Docker
docker-compose up --build
```

## Configuration Options

### Bot Configuration (config.yml)
- **API_BASE_URL**: API endpoint for the language model (default: https://api.groq.com/openai/v1)
- **INTERNET_ACCESS**: Enable/disable internet search capabilities
- **MODEL_ID**: Select the AI model to use (default: meta-llama/llama-4-maverick-17b-128e-instruct)
- **LANGUAGE**: Set the bot's interface language (e.g., 'en', 'es', 'fr')
- **DEFAULT_INSTRUCTION**: Choose the default persona/instruction set
- **TRIGGER**: Words that will trigger the bot to respond
- **ALLOW_DM**: Enable/disable direct messages
- **SMART_MENTION**: Enable/disable name recognition

### Local Model Configuration (for self-hosting)
- **USE_LOCAL_MODEL**: Set to true to use a local LM Studio model
- **LOCAL_MODEL_HOST**: IP address of your local model server
- **LOCAL_MODEL_PORT**: Port for your local model server
- **LOCAL_MODEL_ID**: The model ID for the local model

### Persona Selection
Change the bot's personality by setting `DEFAULT_INSTRUCTION` in config.yml:

- **hand**: Default helpful assistant
- **assist**: Vanilla assistant with no personality
- **DAN**: "Do Anything Now" - breaks typical AI constraints
- **Luna**: Caring and empathetic friend
- **ivan**: Snarky Gen-Z teenager who speaks in abbreviations

### Custom Personas
1. Create a `.txt` file (e.g., `custom.txt`) in the `instructions` folder
2. Add your custom persona instructions in the file
3. Set `DEFAULT_INSTRUCTION: "custom"` in config.yml

### Language Selection
Set the `LANGUAGE` value in config.yml with one of the supported language codes:
- `en` - English 🇺🇸
- `es` - Español 🇪🇸
- `fr`