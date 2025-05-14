# Context-Aware AI Discord Bot

This Discord bot provides a rich, context-aware conversation experience through its advanced multi-agent architecture and service-oriented design. The bot can understand and respond to a wide range of queries, from simple conversations to complex reasoning tasks, by dynamically selecting the most appropriate AI agent for each interaction.

## Key Features

### Multi-Agent Architecture

The bot utilizes a sophisticated multi-agent system that includes specialized agents for different types of tasks:

- **Conversational Agent**: For casual, friendly dialogue
- **Sequential Agent**: Step-by-step analytical thinking for complex problems
- **RAG Agent**: Retrieval-augmented generation for factual information
- **Knowledge Agent**: In-depth explanations and educational content
- **Verification Agent**: Fact-checking and validation
- **Creative Agent**: Story, art, and creative content generation
- **Calculation Agent**: Mathematical operations and computations
- **Planning Agent**: Strategic planning and organization
- **Graph Agent**: Network and relationship-based reasoning
- **ReAct Agent**: Reasoning with action capabilities

The system dynamically selects the most appropriate agent based on your query or allows you to explicitly request a specific reasoning approach.

### Service-Oriented Architecture

The bot's codebase follows a modern service-oriented architecture with centralized services for key functionality:

- **Intent Detection Service**: Identifies the purpose and intent behind user messages
- **Agent Service**: Manages the multi-agent system and orchestrates agent selection
- **Memory Service**: Handles conversation history and user preferences
- **Message Service**: Manages message formatting and delivery
- **Workflow Service**: Coordinates complex multi-agent workflows using LangGraph

This design improves maintainability, reduces code duplication, and provides consistent interfaces.

### Context Awareness

The bot maintains context across conversations, remembering previous interactions and adapting its responses accordingly. It can:

- Track conversation history for more coherent dialogue
- Remember user preferences
- Understand references to previous messages
- Share context between specialized agents

### Privacy Controls

You can manage your data with simple commands:

- **Clear your data**: Just say "clear my data" or "forget me" to remove your conversation history and preferences
- **Reset conversation**: Say "reset our conversation" or "start over" to begin fresh while keeping your preferences

## Using the Bot

### Basic Interaction

Simply mention the bot or use its name to start a conversation. The bot will automatically engage the appropriate agent based on your query.

Examples:
- `@BotName What's the weather like today?`
- `Hey BotName, can you tell me about quantum physics?`

### Explicit Agent Selection

You can explicitly request a specific reasoning approach:

- **Multi-Agent**: `Use multi-agent for analyzing the impact of climate change`
- **Sequential Thinking**: `Think through this step by step: how does a quantum computer work?`
- **Creative**: `Write a short story about a robot discovering emotions`
- **Calculation**: `Calculate the compound interest on $1000 at 5% for 10 years`
- **Knowledge**: `Explain in detail how photosynthesis works`

### Advanced Features

- **Image Generation**: `Generate an image of a sunset over mountains`
- **Image Analysis**: Upload an image with a prompt like `What's in this image?`
- **Voice Transcription**: Upload a voice message with `Transcribe this`
- **Sentiment Analysis**: `Analyze the sentiment of this tweet: [text]`
- **Web Search**: `Search for recent news about space exploration`

## How It Works

The bot leverages a sophisticated architecture that:

1. Uses the Intent Detection Service to identify what the user wants
2. Selects the most appropriate agent through the Agent Service
3. Retrieves conversation context via the Memory Service
4. Delegates sub-tasks to specialized agents when needed
5. Formats and delivers responses through the Message Service
6. Orchestrates complex workflows with the Workflow Service

This architecture enables more accurate, helpful, and contextually appropriate responses than a single-model approach.

## Privacy

The bot respects user privacy:

- Data is only used to maintain conversation context
- You can clear your data at any time with simple commands
- No conversation data is used for training or shared with third parties

Say "clear my data" or "forget me" at any time to remove your information from the system.

## Core Features

### Context-Aware Reasoning

The bot automatically selects the most appropriate reasoning method for each query:

- 🧠 **Sequential Thinking** - Step-by-step problem solving with thought revision
- 🔍 **Information Retrieval** - Searches for facts and information
- 💬 **Conversational** - Natural, flowing conversation
- 📚 **Knowledge Base** - Educational and explanatory content
- ✅ **Verification** - Fact-checking and validating claims
- 🕸️ **Graph-of-Thought** - Mapping relationships between concepts
- ⛓️ **Chain-of-Thought** - Logical reasoning progression
- 🔄 **ReAct Reasoning** - Reasoning with action capabilities
- 🎨 **Creative Mode** - Imagination and creative content
- 🔎 **Step-Back Analysis** - High-level perspective on problems
- 👥 **Multi-Agent** - Multiple perspectives on a topic

### Natural Language Interface

Interact with the bot using natural language:

- Mention the bot: `@Bot help me understand quantum computing`
- Use the bot's name: `Hey Bot, can you explain climate change?`
- Reply to the bot's messages: `Tell me more about that`

### User Settings

You can manage your preferences with natural language:

- `Set my reasoning mode to sequential` - Use sequential thinking by default
- `Enable workflow mode` - Use LangGraph for advanced reasoning flows
- `Change reasoning mode to creative` - Switch to creative mode
- `Reset our conversation` - Start fresh while keeping your preferences

## Usage Examples

### Sequential Thinking
```
@Bot step by step, analyze the implications of quantum computing on cryptography
```

### Information Retrieval
```
@Bot search for information about climate change solutions
```

### Verification
```
@Bot verify whether coffee is beneficial for health
```

### Graph-of-Thought
```
@Bot map the connections between artificial intelligence and ethics
```

### Creative Mode
```
@Bot write a short story about a robot discovering emotions
```

### Step-Back Analysis
```
@Bot what's the broader perspective on social media's impact on society?
```

### Multi-Agent Reasoning
```
@Bot analyze cryptocurrency from multiple perspectives
```

## Getting Help

Simply ask the bot for help to learn more about its capabilities:

```
@Bot help
```
or
```
@Bot what can you do?
```
or
```
@Bot explain your reasoning capabilities
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your `.env` file with your Discord token and other API keys
4. Run the bot:
   ```bash
   python main.py
   ```

## Architecture

The bot uses a modern service-oriented architecture with these key components:

1. **Intent Detection Service**: Analyzes queries to determine user intent and the appropriate reasoning mode
2. **Agent Service**: Manages the multi-agent system and handles agent selection, delegation, and collaboration
3. **Memory Service**: Maintains conversation history, user preferences, and agent state
4. **Message Service**: Handles message formatting, splitting, and delivery
5. **Workflow Service**: Orchestrates complex multi-agent workflows using LangGraph

This modular design makes the code more maintainable, improves testability, and provides consistent interfaces for future extensions.

## Development

To extend the bot's capabilities:

1. Use the service interfaces rather than direct API calls
2. Add new reasoning methods by extending the agent system
3. Add new tools by registering them with the agent tools manager
4. Improve context retention by enhancing the memory service

## Troubleshooting

- **Bot doesn't respond**: Make sure you're mentioning the bot or using its name
- **Incorrect reasoning mode**: Be more explicit in your query or set your reasoning preference
- **API rate limits**: The bot may hit rate limits with external APIs during heavy usage
- **"Cannot send an empty message" error**: The model may have returned an empty response; try rephrasing your query

## License

This project is licensed under the MIT License - see the LICENSE file for details. 