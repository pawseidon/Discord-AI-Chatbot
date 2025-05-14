# Context-Aware Discord AI Bot

This Discord bot leverages advanced context-aware reasoning to automatically detect the most appropriate reasoning method for each query. The bot operates entirely through natural language, with no slash commands.

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

### Reasoning Preferences

You can manually select reasoning modes:

- `Set my reasoning mode to sequential` - Use sequential thinking
- `Change reasoning mode to creative` - Switch to creative mode
- Include the emoji at the start of your message (e.g., 🔍 for search)
- Ask `Explain reasoning modes` to learn more

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
5. To remove all slash commands from an existing bot, run:
   ```bash
   python remove_slash_commands.py
   ```

## Removing Slash Commands

If you've previously used this bot with slash commands, you can remove them using the included script:

1. Make sure your bot token is in the `.env` file
2. Run the removal script:
   ```bash
   python remove_slash_commands.py
   ```
3. Restart your bot for the changes to take effect

## Architecture

The bot's architecture is designed for seamless reasoning transitions:

1. **Reasoning Detection**: Analyzes queries to determine the most appropriate reasoning method
2. **Reasoning Manager**: Manages transitions between reasoning modes and maintains context
3. **Sequential Thinking**: Provides step-by-step reasoning with thought revision capabilities
4. **Agent Tools**: Integrates various capabilities like web search, knowledge retrieval, sentiment analysis
5. **Natural Language Processing**: All interactions are processed through natural language

## Development

To extend the bot's capabilities:

1. Add new reasoning methods by extending the `ReasoningDetector` class
2. Add new agent tools by extending the `AgentTools` class
3. Improve context retention by enhancing the conversation memory system

## Troubleshooting

- **Bot doesn't respond**: Make sure you're mentioning the bot or using its name
- **Incorrect reasoning mode**: Be more explicit in your query or set your reasoning preference
- **API rate limits**: The bot may hit rate limits with external APIs during heavy usage

## License

This project is licensed under the MIT License - see the LICENSE file for details. 