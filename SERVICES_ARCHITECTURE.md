# Service-Oriented Architecture

This document describes the service-oriented architecture implemented in the Discord AI Chatbot to reduce code duplication and improve maintainability.

## Overview

The Discord AI Chatbot is built on a service-oriented architecture that centralizes core functionality into dedicated service modules. This approach improves maintainability, reduces duplication, and provides a consistent interface for all bot components. Each service has a well-defined responsibility and exposes a clear API for other parts of the application to use.

## Service Modules

The services are located in the `bot_utilities/services` directory and include:

### Agent Service (`agent_service.py`)

Centralizes the multi-agent orchestration logic:

- Provides methods for processing queries through the agent system
- Manages agent initialization and configuration
- Handles agent delegation and reasoning type detection
- Exposes methods for clearing user data

```python
# Example usage
from bot_utilities.services.agent_service import agent_service

# Process a query with the agent system
response = await agent_service.process_query(
    query="How do neural networks work?",
    user_id="123456789",
    conversation_id="987654321",
    reasoning_type="sequential"  # Optional reasoning type
)
```

### Intent Detection Service (`intent_detection.py`)

Handles the detection of user intent from message content:

- Analyzes message content using regular expressions
- Detects intents like web search, sequential thinking, symbolic reasoning
- Provides a centralized system for all intent detection

```python
# Example usage
from bot_utilities.services.intent_detection import intent_service

# Detect intent from a message
intent_data = await intent_service.detect_intent(message_content, message)
if intent_data["intent"] == "web_search":
    # Handle web search
    query = intent_data["query"]
```

### Memory Service (`memory_service.py`)

Manages user conversation history and preferences:

- Stores and retrieves conversation history
- Handles user preferences for reasoning modes, etc.
- Tracks user activity and last seen timestamps
- Provides methods for clearing user data

```python
# Example usage
from bot_utilities.services.memory_service import memory_service

# Get user preferences
prefs = await memory_service.get_user_preferences(user_id)

# Store conversation entry
await memory_service.add_to_history(
    conversation_id="server_id:channel_id",
    message={"role": "user", "content": "Hello, how can you help me?"}
)
```

### Message Service (`message_service.py`)

Utilities for message formatting and sending:

- Handles message splitting for long responses
- Manages message formatting with agent emoji
- Provides consistent error handling for message sending
- Handles typing indicators
- Formats code blocks and complex content

```python
# Example usage
from bot_utilities.services.message_service import message_service

# Format a response with the appropriate agent emoji
formatted_response, emoji = await message_service.format_with_agent_emoji(
    response="Here's the answer...", 
    agent_type="sequential"
)

# Send a potentially long response with proper error handling
await message_service.send_response(message, "This is a very long response...")
```

### Workflow Service (`workflow_service.py`)

Manages complex multi-agent workflows using LangGraph:

- Handles workflow creation and configuration
- Manages state transitions between agents
- Provides orchestration for complex multi-step reasoning
- Supports callbacks for progress updates

```python
# Example usage
from bot_utilities.services.workflow_service import workflow_service

# Create and run a workflow
response = await workflow_service.create_and_run_default_workflow(
    user_query="Analyze the impact of AI on jobs",
    user_id="123456789",
    conversation_id="987654321",
    update_callback=my_update_callback
)
```

### Symbolic Reasoning Service (`symbolic_reasoning_service.py`)

Provides deterministic reasoning capabilities for mathematical problems and logical verification:

- Solves mathematical expressions and equations with step-by-step solutions
- Verifies logical statements using rule-based analysis
- Analyzes graph structures and relationships
- Complements neural reasoning with deterministic computation

```python
# Example usage
from bot_utilities.services.symbolic_reasoning_service import symbolic_reasoning_service

# Solve a mathematical problem
result = await symbolic_reasoning_service.solve_math_problem("2x + 5 = 15")
if result["success"]:
    # Use the solution and steps
    solution = result["result"] 
    steps = result["steps"]
    
# Verify a logical statement
verification = await symbolic_reasoning_service.verify_logical_statement(
    "If all humans are mortal and Socrates is human, then Socrates is mortal"
)
```

## Integration Points

The services are integrated at key points in the codebase:

1. **Event Handlers** - `on_message.py` uses services to process incoming messages
2. **Command Cogs** - ReasoningCog and HelpCog use services to handle commands
3. **Agent Orchestration** - The agent orchestration system uses services for memory management
4. **Bot Initialization** - Services are initialized during bot startup in main.py

## Benefits of this Architecture

1. **Reduced Duplication** - Common functionality like intent detection is now in one place
2. **Consistent Behavior** - All components use the same logic for operations like memory management
3. **Easier Testing** - Services can be tested independently of the Discord interface
4. **Simpler Maintenance** - Updates to core functionality only need to be made in one place
5. **Cleaner Code** - Command handlers and event listeners are more focused on their specific tasks

## Adding New Services

To add a new service:

1. Create a new file in `bot_utilities/services/`
2. Define a class with the required functionality
3. Create a singleton instance at the bottom of the file
4. Import the service in `bot_utilities/services/__init__.py`
5. Use the service where needed in cogs and event handlers

```python
# Example of creating a new service
class MyNewService:
    def __init__(self):
        # Initialize the service
        pass
        
    async def some_method(self):
        # Implement functionality
        pass
        
# Create a singleton instance
my_new_service = MyNewService() 
```

## Service Initialization

The services are initialized in main.py in the following order:

1. agent_service - The core agent orchestration service
2. workflow_service - Depends on agent_service for agent functionality
3. symbolic_reasoning_service - Independent service for mathematical operations
4. memory_service - Loads user preferences and conversation history from disk

This initialization order ensures that dependencies are properly handled and that services are ready when needed by the bot.

## Core Capabilities Maintained

The service architecture maintains the following core capabilities:

1. **RAG (Retrieval-Augmented Generation)** - Web search and information retrieval
2. **Sequential Thinking** - Step-by-step reasoning for complex problems
3. **Conversational AI** - Natural, contextual conversation
4. **Graph of Thoughts** - Network and relationship analysis
5. **Symbolic Reasoning** - Mathematical and logical operations
6. **Multi-Agent Orchestration** - Coordinating specialized agent types
