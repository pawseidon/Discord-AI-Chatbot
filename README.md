# Discord AI Chatbot 🤖

A powerful Discord bot powered by advanced AI models, offering conversational capabilities, image generation, OCR, research assistance, and much more.

## Table of Contents
- [Features](#features)
- [Commands](#commands)
- [Installation Guide](#installation-guide)
- [Configuration Options](#configuration-options)
- [Core Capabilities](#core-capabilities)
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

### Image Capabilities 🎨
- **Image Generation**: Create stunning AI-generated images from text descriptions using multiple models.
- **OCR (Text Extraction)**: Extract text from images and screenshots.
- **Image Analysis**: Detailed descriptions and analysis of uploaded images.
- **Multiple Generation Styles**: Choose from various image generation models and styles.

### Agent System 🧠
- **Web Search**: Search the internet for real-time information using DuckDuckGo with fallback mechanisms.
- **Research Assistant**: Deep research on topics with multiple sources.
- **Task Automation**: Get step-by-step guides for complex tasks.
- **Crypto Price Tracking**: Real-time cryptocurrency price information.
- **Knowledge Base**: Store and retrieve server-specific information.

### User Experience 🌟
- **Customizable Preferences**: Each user can configure their own experience including response length, voice mode, and more.
- **Voice Responses**: The bot can respond with voice messages using text-to-speech.
- **Multilingual Interface**: The bot's commands and responses adapt to your language settings.
- **Channel-Specific Control**: Enable/disable the bot in specific channels.

## Commands

### Core Commands
- `/help` - Display all available commands
- `/toggleactive` - Enable/disable the bot in the current channel
- `/toggledm` - Enable/disable direct message functionality
- `/preferences` - Set your personal preferences for interactions with the bot

### AI Conversation
- **Direct Mentions**: `@BotName [your message]` - Talk directly to the bot
- **Keywords**: Using configured trigger words in messages to activate the bot
- **Replies**: Reply to the bot's messages to continue the conversation

### Image Commands
- `/analyze-image [image]` - Analyze and describe an uploaded image
- `/ocr [image]` - Extract text from an image
- `/generate [prompt]` - Generate a single image from a description
  - Options: `style` (model selection), `enhance` (improve prompt)
- `/imagine [prompt]` - Generate multiple images from a description
  - Options: `count` (1-4 images), `public` (visibility setting)

### Agent Commands
- `/agent [query]` - Use the AI agent to perform complex tasks or answer questions using tools
- `/research [topic]` - Research a topic in depth using multiple internet sources
- `/automate [task]` - Get step-by-step guidance for automating complex tasks

### Utility Commands
- `/bonk` - Clear message history for a fresh start
- `/nekos [category]` - Display random anime-style images in various categories

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
- `fr` - Français 🇫🇷
- `de` - Deutsch 🇩🇪
- `cn` - Chinese 🇨🇳
- `ru` - Russian 🇷🇺
- And many more (see the `lang` folder for all options)

## Core Capabilities

### Multi-Modal Processing
The bot can work with both text and images:
- Process and analyze images through the `/analyze-image` command
- Extract text from images with the `/ocr` command
- Generate images based on descriptions with `/generate` and `/imagine`

### Agent-Based Intelligence
Powered by LangChain, the bot can:
- Perform web searches for up-to-date information
- Access server-specific knowledge bases
- Get real-time cryptocurrency prices
- Conduct in-depth research across multiple sources
- Generate step-by-step automation guides

### Memory and Context Management
- Maintains conversation history for context
- Stores user preferences
- Preserves thread context for organized discussions
- Summarizes long conversations to stay within token limits

### Adaptive Reasoning System 🧠
- **Multiple Reasoning Types**: Automatically selects the optimal reasoning approach:
  - 💬 Conversational - General friendly chat and discussions
  - 🧠 Sequential - Step-by-step analytical thinking for complex problems
  - 🔍 RAG - Retrieval-Augmented Generation for information lookup
  - 📚 Knowledge - In-depth explanations and educational content
  - ✅ Verification - Fact-checking and validation
  - 🎨 Creative - Story, art, and creative content generation
  - 🔢 Calculation - Mathematical operations and computations
  - 📋 Planning - Strategic planning and organization
  - 🕸️ Graph - Network and relationship-based reasoning
  - 👥 Multi-Agent - Multiple perspectives and balanced viewpoints
  - 🔎 Step-Back - Holistic, big-picture thinking
  - ⛓️ Chain-of-Thought - Logical progression and causal reasoning
  - 🔄 ReAct - Action-oriented problem solving
- **Automatic Detection**: Analyzes queries to select the most appropriate reasoning mode
- **Context-Aware Transitions**: Smoothly switches between reasoning types based on conversation flow
- **User Preference**: Remembers individual user reasoning preferences

### Streaming and Voice Responses
- Watch responses appear in real-time with streaming mode
- Get voice responses using text-to-speech technology

## Advanced Usage

### Enhancing Bot Responses
- **Be specific in your queries**: The more specific your request, the better the response
- **Use the right command**: Different commands are optimized for different tasks
- **Leverage thread creation**: Ask the bot to "create a thread" for complex or multi-turn conversations

### Image Generation Tips
- **Be detailed in prompts**: Include style, mood, lighting, and composition details
- **Use the enhance option**: Enable prompt enhancement for better results
- **Try different models**: Each model has different strengths (e.g., MidJourney Style XL, DreamShaper, Realistic Vision)

### Agent Usage Tips
- **Research complex topics**: Use `/research` for in-depth analysis of complex subjects
- **Automation guidance**: Use `/automate` to get detailed steps for complex tasks
- **Real-time information**: The agent can search the web for current information

## Troubleshooting

### Common Issues
- **Bot not responding**: Ensure the bot has proper permissions in your Discord server
- **Image generation failing**: The service might be experiencing high demand, try again later
- **Search results outdated**: The bot's internet search capabilities have limitations on freshness

### Performance Optimization
- Use streaming responses for faster interaction
- Limit the number of images generated at once
- Consider using a local model for lower latency if you self-host

## Contributing

Contributions are welcome! Here's how you can help:
- Report bugs and issues
- Suggest new features
- Submit pull requests with improvements
- Help with documentation and translations

---

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=mishalhossin/Discord-AI-Chatbot&type=Timeline)](https://star-history.com/#mishalhossin/Discord-AI-Chatbot&Timeline)

### Crafted with Care: Made with lots of love and attention to detail. ❤️
