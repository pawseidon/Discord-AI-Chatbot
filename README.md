# Discord AI Chatbot 🤖

A powerful Discord bot powered by advanced AI models, featuring a multi-agent reasoning system with specialized agents for different types of queries.

## Table of Contents
- [Features](#features)
- [Commands](#commands)
- [Installation Guide](#installation-guide)
- [Configuration Options](#configuration-options)
- [Multi-Agent Architecture](#multi-agent-architecture)
- [Emoji Reactions](#emoji-reactions)
- [Core Capabilities](#core-capabilities)
- [Service Architecture](#service-architecture)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Features

### Multi-Agent Reasoning System 🧠
- **Specialized Agents**: Different reasoning agents for different types of tasks
- **Dynamic Switching**: Automatically selects the best reasoning approach for each query
- **Emoji Reactions**: Visual indicators showing which reasoning type is being used
- **Workflow Mode**: Complex multi-step reasoning using LangGraph (optional)
- **Symbolic Reasoning**: Deterministic math and logic operations for precise calculations

### Conversational AI 💬
- **Multi-Language Support**: Communicate in 16 different languages including English, Spanish, French, Chinese, and more
- **Multiple Personalities**: Choose from various personalities like DAN, Luna, Suzume, or create your own custom persona
- **Smart Mention Recognition**: The bot responds when mentioned or called by name, similar to modern AI assistants
- **Thread Support**: Creates and maintains context within Discord threads for organized conversations
- **Streaming Responses**: Watch responses appear in real-time as they're generated for a more interactive experience
- **Memory System**: Remembers previous conversations for more contextual interactions

### Agent Capabilities 🛠️
- **Web Search**: Search the internet for real-time information using DuckDuckGo with fallback mechanisms
- **Research Assistant**: Deep research on topics with multiple sources
- **Task Automation**: Get step-by-step guides for complex tasks
- **Crypto Price Tracking**: Real-time cryptocurrency price information
- **Knowledge Base**: Store and retrieve server-specific information

### User Experience 🌟
- **Customizable Preferences**: Each user can configure their own experience including reasoning mode preferences
- **Multilingual Interface**: The bot's commands and responses adapt to your language settings
- **Channel-Specific Control**: Enable/disable the bot in specific channels

## Commands

### Natural Language Interaction
No commands needed for most interactions! Simply:
- **Direct Mentions**: `@BotName [your message]` - Talk directly to the bot
- **Using Bot's Name**: `Hey Bot, [your message]` - The bot recognizes when you use its name
- **Keyword Triggers**: Use configured trigger words like "assistant" or "ai" in messages
- **Replies**: Reply to the bot's messages to continue conversations

The bot automatically detects the most appropriate reasoning mode based on your query:
- "What's the capital of France?" → Conversational or RAG mode (simple factual query)
- "Explain how quantum computing works step by step" → Sequential thinking mode (complex explanation)
- "Find recent information about climate change" → RAG mode (information retrieval)
- "Verify if drinking 8 glasses of water daily is necessary" → Verification mode (fact-checking)
- "Create a short story about a robot learning to paint" → Creative mode (generative content)
- "Calculate the compound interest on $1000 at 5% for 5 years" → Calculation mode (math)
- "Map the relationships between characters in Hamlet" → Graph mode (relationship analysis)
- "Analyze the pros and cons of remote work from different perspectives" → Multi-agent mode (multiple viewpoints)

### User Settings
You can customize your experience with natural language commands:

#### Reasoning Preferences
- **Set Default Mode**: `Set my reasoning mode to sequential` or `I prefer creative mode for all my questions`
- **Temporary Mode Switch**: `Answer this in verification mode: [question]` or `Use graph thinking for this query`
- **Reset Preferences**: `Reset my reasoning preferences` or `Go back to automatic mode selection`
- **Check Current Setting**: `What reasoning mode am I using?` or `Show my current preferences`

#### Workflow Controls
- **Enable Workflow**: `Enable workflow mode for complex reasoning` or `Use LangGraph for my questions`
- **Disable Workflow**: `Disable workflow mode` or `Use standard reasoning instead of workflows`
- **Specific Workflow Request**: `Use the research workflow for this query` or `Apply multi-step analysis to this problem`
- **Ask About Workflows**: `What workflows are available?` or `Explain how the verification workflow works`

#### Memory & Privacy
- **Clear Data**: `Clear my data` or `Forget me` to remove all your conversation history and preferences
- **Reset Conversation**: `Reset our conversation` or `Start over` to begin fresh while keeping preferences
- **Reset Specific Context**: `Forget what we discussed about [topic]` or `Let's change the subject` 
- **Memory Recall**: `What did we discuss earlier about [topic]?` or `Remember when I mentioned [subject]?`

#### Persona & Style
- **Tone Adjustment**: `Please be more technical in your responses` or `Keep explanations simple`
- **Verbosity Control**: `Give me concise answers` or `I'd like detailed explanations`
- **Expertise Area**: `I need help with programming questions` or `Focus on scientific topics`

The bot learns from your interactions and will adjust to your preferences over time. You can always reset these learned preferences with `Reset what you've learned about me`.

### Slash Commands
- `/help` - Display information about available features and reasoning types
- `/clear` - Clear your conversation history and data
- `/reset` - Reset the current conversation but keep your preferences
- `/toggleactive` - Toggle the bot active/inactive in the current channel (Admin only)
- `/toggleinactive` - Toggle the bot inactive/active in the current channel (Admin only)

### Reasoning Types
Include these phrases to explicitly request a specific reasoning approach:
- **Sequential Thinking** 🔄: `Think through this step by step: [question]`
- **RAG (Retrieval-Augmented Generation)** 📚: `Search for information about [topic]`
- **Verification** ✅: `Verify whether [claim] is accurate`
- **Knowledge Base** 📚: `Explain in detail how [topic] works`
- **Creative Mode** 🎨: `Write a creative story about [topic]`
- **Graph-of-Thought** 📊: `Map the connections between [concepts]`
- **Multi-Agent** 👥: `Analyze [topic] from multiple perspectives`
- **Calculation** 🧮: `Calculate [mathematical expression]`

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

## Multi-Agent Architecture

The Discord AI Chatbot uses a sophisticated multi-agent architecture that dynamically selects different reasoning approaches based on the query type:

### Agent Types

| Agent Type | Emoji | Description | Best For |
|------------|-------|-------------|----------|
| Sequential | 🔄 | Step-by-step analytical thinking | Complex problems, detailed analysis |
| RAG | 📚 | Retrieval-Augmented Generation | Information seeking, fact-based queries |
| Conversational | 💬 | Natural, friendly dialogue | Casual chat, simple exchanges |
| Knowledge | 📚 | Detailed educational content | Learning, concept explanation |
| Verification | ✅ | Fact-checking and validation | Claims evaluation, truth assessment |
| Creative | 🎨 | Imaginative content generation | Stories, art, creative writing |
| Calculation | 🧮 | Mathematical operations | Computations, numerical analysis |
| Planning | 📋 | Structured strategy development | Project plans, task organization |
| Graph | 📊 | Relationship mapping | Concept networks, interconnections |
| Multi-Agent | 👥 | Multiple perspectives | Balanced viewpoints, debates |

### How It Works

The bot uses a sophisticated orchestration system to determine the most appropriate reasoning method for each query:

1. **Automatic Intent Detection**: When you send a message, the bot analyzes your query using natural language understanding to determine what you're asking for:
   - Questions about facts or recent information trigger the RAG system
   - Step-by-step requests activate Sequential reasoning
   - Creative requests like stories or poems use Creative mode
   - Mathematical questions route to the Calculation agent
   - Multiple-perspective requests engage the Multi-agent workflow

2. **Smart Context Analysis**: Beyond keywords, the bot examines:
   - Complexity of the question (simple → Conversational, complex → Sequential)
   - Question patterns (who/what/when → RAG, how/why → Sequential)
   - Specific verbs ("verify," "calculate," "create," "explain," "map")
   - Request structure (step-by-step indicators, fact-checking cues)

3. **Dynamic Reasoning Selection**: The bot doesn't just use one reasoning mode:
   - It may combine multiple reasoning types for complex queries
   - For example: "Research recent advances in AI and analyze their potential impacts step by step" 
     would combine RAG (for research) with Sequential (for step-by-step analysis)
   - The reasoning icon (emoji) shows which mode is being used

4. **User Preference Adaptation**: Your settings and history influence mode selection:
   - If you've set a preferred reasoning mode, the bot prioritizes it
   - The bot learns from your interactions and adjusts over time
   - You can override automatic selection for any specific query

5. **Workflow Orchestration**: For complex multi-step queries, the bot may activate workflow mode:
   - Breaks down complex tasks into manageable sub-tasks
   - Routes each sub-task to the appropriate specialized agent
   - Maintains a shared state to pass information between agents
   - Synthesizes all agent outputs into a cohesive response

This autonomous selection happens behind the scenes - you don't need to specify reasoning modes unless you want to override the bot's choices.

### Workflow Mode

For advanced users, the bot supports a workflow mode that uses LangGraph to orchestrate complex reasoning flows:

1. **Workflow Activation**: Enable with `Enable workflow mode` or through user preferences.
   
2. **Graph-Based Reasoning**: The workflow mode creates a directed graph of reasoning steps, where each node represents an agent or processing step.
   
3. **State Management**: LangGraph maintains a shared state object that passes information between agents, allowing for explicit control over the reasoning flow.
   
4. **Complex Multi-Step Processing**:
   - For example, a research query might flow through: Query Analysis → RAG Retrieval → Information Verification → Sequential Synthesis → Final Response.
   - Each step can use a different specialized agent optimized for that task.
   
5. **Fallback Logic**: The workflow includes graceful error handling with fallback paths if a particular reasoning approach fails.

6. **Tools Integration**: Specialized tools like web search, calculators, and memory retrieval are integrated into the workflow as needed.

7. **Installation**: Workflow mode requires LangGraph to be installed (`pip install langgraph`).

## Emoji Reactions

The bot uses emoji reactions to indicate which reasoning type is being used:

| Emoji | Reasoning Type | Description |
|-------|---------------|-------------|
| 🔄 | Sequential | Step-by-step analytical thinking |
| 📚 | RAG | Retrieval-Augmented Generation (information lookup) |
| 💬 | Conversational | Natural dialogue mode |
| ✅ | Verification | Fact-checking and validation |
| 🎨 | Creative | Imaginative content generation |
| 🧮 | Calculation | Mathematical operations |
| 📊 | Graph | Relationship mapping and analysis |
| 👥 | Multi-Agent | Multiple specialized approaches |
| 📋 | Planning | Structured strategy development |
| 🔍 | Step-Back | Big-picture perspective |
| ⛓️ | Chain-of-Thought | Logical progression |
| 🔄 | ReAct | Reasoning with actions |

## Core Capabilities

The Discord AI Chatbot combines several advanced capabilities:

### 1. Multi-Agent Orchestration
The bot uses an orchestration system to coordinate different specialized agents, selecting the most appropriate reasoning approach for each query.

### 2. Hybrid Neural-Symbolic Reasoning
For mathematical and logical operations, the bot combines neural language models with deterministic symbolic reasoning for increased accuracy.

### 3. Service-Oriented Architecture
The codebase uses a modular service architecture to separate concerns and improve maintainability.

### 4. Memory Management
A sophisticated memory system tracks conversation history, user preferences, and agent scratchpads.

### 5. Tool Integration
Agents can use various tools including web search, calculators, and knowledge bases to enhance their capabilities.

## Service Architecture

The bot is built on a service-oriented architecture with these key services:

1. **agent_service**: Core orchestration and reasoning type detection
2. **memory_service**: Conversation history and user preferences
3. **message_service**: Message formatting and processing
4. **workflow_service**: Graph-based reasoning workflows
5. **symbolic_reasoning_service**: Deterministic operations

## Advanced Usage

### Combining Reasoning Approaches
You can explicitly request multiple reasoning types together:

```
@BotName search for recent information about quantum computing and then verify the key claims
```

This combines RAG (search) with verification reasoning.

### Accessing Specialized Tools
Certain phrases trigger specialized tool usage:

```
@BotName calculate the derivative of 3x^2 + 2x with respect to x
```

This triggers the symbolic math calculator for precise mathematical operations.

## Troubleshooting

### Bot Not Responding
- Check if the bot has the correct permissions in your server
- Verify that your API keys are correctly set in the .env file
- Check if the bot is active in the current channel

### Incorrect Reasoning Type
- Be more explicit in your query about the type of reasoning you need
- Use clear trigger phrases like "think step by step" or "search for information about"
- Set your default reasoning preference using `set my reasoning mode to [type]`

### Memory Issues
- If the bot seems to forget context, try using `reset conversation` to start fresh
- For persistent issues, check the bot's logs for memory-related errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.