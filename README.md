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

### Audio & Voice Features 🎙️
- **Voice Message Transcription**: Automatically transcribe voice messages to text.
- **Voice Commands**: Use voice messages with text commands for hands-free operation.
- **Text-to-Speech Responses**: Get voice responses using text-to-speech technology.

### Analytical Features 📊
- **Sentiment Analysis**: Analyze the emotional tone of messages with detailed breakdowns.
- **Emotion Detection**: Identify specific emotions in text with confidence ratings.
- **Contextual Understanding**: The bot understands the emotional context of conversations.

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
- `/help` - Display all available commands and their descriptions
- `/toggleactive` - Enable/disable the bot in the current channel (Admin only)
- `/toggledm` - Enable/disable direct message functionality
- `/preferences` - Set your personal preferences for interactions with the bot
  - Set response length (short, medium, long)
  - Enable/disable voice responses
  - Enable/disable rich embeds
  - Enable/disable streaming responses
- `/clear` - Clear your conversation history with the bot in the current channel
- `/toggleoffline` - Toggle offline mode for the bot (Admin only)

### AI Conversation
- **Direct Mentions**: `@BotName [your message]` - Talk directly to the bot
- **Keywords**: Just say any of the trigger words configured in config.yml (default: "hey hand", "okay hand") to activate the bot
- **Replies**: Reply to any of the bot's messages to continue the conversation with full context
- **Threads**: The bot maintains separate conversation histories in different threads

### Image Commands
- `/analyze-image [image]` - Analyze and describe an uploaded image in detail
- `/ocr [image]` - Extract text from an image (works with screenshots, documents, signs, etc.)
- `/generate [prompt]` - Generate a single image from a text description
  - Options: `style` (model selection), `enhance` (improve prompt automatically)
- `/imagine [prompt]` - Generate multiple images from a description
  - Options: `count` (1-4 images), `public` (visibility setting)
  - Example: `/imagine a magical forest with glowing mushrooms, count:4, public:true`

### Voice & Audio Commands
- `/transcribe [voice_message]` - Transcribe a voice message to text
  - Can be used with an uploaded voice file or in reply to a message with a voice attachment
  - Example: `/transcribe` (in reply to a voice message)
- **Text Command Transcription**: Add `!transcribe` when sending a voice message to automatically transcribe it
  - Example: Send a voice message with the text "!transcribe" to get an immediate transcription

### Analytical Commands
- `/sentiment [text]` - Analyze the sentiment and emotions in a message
  - Options: `text` (message to analyze), `public` (visibility setting)
  - Can be used on your own text or in reply to another message
  - Example: `/sentiment How's everyone doing today?`
  - Example: `/sentiment` (in reply to a message)

### Agent Commands
- `/agent [query]` - Use the AI agent to perform complex tasks using tools
  - Can search the web, use knowledge bases, and more
  - Example: `/agent What were the top news stories from yesterday?`
- `/research [topic]` - Research a topic in depth using multiple internet sources
  - Example: `/research The history and development of quantum computing`
- `/mcp-agent [query]` - Advanced agent with specialized tools for complex tasks
  - Example: `/mcp-agent Create a plan for developing a small e-commerce website`
- `/sequential-thinking [problem]` - Solve complex problems step-by-step with detailed reasoning
  - Breaks down problems into logical steps and solves each part in sequence
  - Great for math problems, logic puzzles, or complex reasoning tasks
  - Example: `/sequential-thinking How would I approach designing a garden for a small backyard?`

### Knowledge Base Commands
- `/kb-add [name] [content]` - Add information to the server's knowledge base
- `/kb-search [query]` - Search the server's knowledge base
- `/kb-delete [name]` - Delete an entry from the knowledge base
- `/kb-list` - List all entries in the knowledge base

### Fun Commands
- `/gif [category]` - Get a fun anime-style reaction GIF
  - Categories include: hug, pat, kiss, dance, etc.
  - Example: `/gif hug` will send a cute anime hug GIF

## How to Interact with Each Feature

### Regular Conversations
1. **Starting a conversation**:
   - Simply mention the bot: `@BotName Hey, how are you today?`
   - Use a trigger word: `Hey hand, can you explain quantum computing?`
   - Reply to a previous bot message

2. **Continuing conversations**:
   - The bot remembers your recent conversation history in the same channel
   - For a fresh start, use `/clear` to reset the conversation

3. **Customizing your experience**:
   - Use `/preferences` to set your preferred response length, format, etc.
   - These settings apply to all your interactions with the bot

### Using Voice Transcription
1. **Direct transcription**:
   - Use the `/transcribe` command when uploading a voice message:
     ```
     /transcribe [attach voice message]
     ```
   - Or reply to a voice message with `/transcribe`

2. **Automatic transcription**:
   - When sending a voice message, include `!transcribe` in the text field for automatic transcription
   - The bot will immediately process and transcribe your voice message

3. **Working with transcriptions**:
   - Transcriptions appear in an embedded format with a link to the original voice message
   - You can copy the text, reply to it, or use it as input for other commands

### Using Sentiment Analysis
1. **Analyze any text**:
   ```
   /sentiment The service was excellent and I'm very happy with my purchase!
   ```

2. **Analyze existing messages**:
   - Reply to any message with `/sentiment` to analyze its emotional content
   - Make analysis private (default) or public with the `public` option

3. **Understanding results**:
   - View overall sentiment (positive, negative, neutral)
   - See detailed emotion breakdown with intensity percentages
   - Read a summary of the sentiment analysis
   - Results include confidence ratings to indicate reliability

### Using the Agent for Research
1. Use `/agent` for quick answers that might require internet search:
   ```
   /agent What's the current price of Bitcoin?
   ```

2. Use `/research` for in-depth information on a topic:
   ```
   /research The impact of climate change on ocean ecosystems
   ```
   
3. Use `/sequential-thinking` for problems that need step-by-step reasoning:
   ```
   /sequential-thinking How would I approach designing a garden for a small backyard?
   ```
   The bot will break down the problem into logical steps and address each part in sequence.
   
   The system automatically selects the best reasoning approach:
   - **Sequential Thinking**: For straightforward problems with clear steps
   - **Chain-of-Thought**: For more complex problems requiring deeper analysis
   - **Chain-of-Verification**: For fact-checking and reducing hallucinations
   - **Graph-of-Thought**: For non-linear problems with multiple perspectives and interconnected concepts

### Generating Images
1. Quick single image:
   ```
   /generate a futuristic city with flying cars and neon lights
   ```

2. Multiple images with options:
   ```
   /imagine a medieval castle on a mountain, count:4, public:true
   ```

3. For best results with image generation:
   - Be specific about style, lighting, mood, and composition
   - Use the `enhance` option to let the AI improve your prompt
   - Try different models if you don't get the desired result

### Using the Knowledge Base
1. Add information that you want the bot to remember for your server:
   ```
   /kb-add server-rules Our server has the following rules: 1) Be respectful 2) No spam 3) Have fun!
   ```

2. Later, anyone can retrieve this information:
   ```
   /kb-search rules
   ```

3. The knowledge base is specific to each server, so it's perfect for:
   - Server rules and guidelines
   - FAQ information
   - Team documentation
   - Project information

### Voice Interaction
1. Enable voice responses in your preferences:
   ```
   /preferences voice_enabled:true
   ```

2. The bot will now respond with voice messages in addition to text

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
     OPENAI_API_KEY=YOUR_OPENAI_API_KEY_FOR_VOICE_TRANSCRIPTION
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
- **Automation guidance**: Use `/sequential-thinking` to get detailed steps for complex tasks
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
