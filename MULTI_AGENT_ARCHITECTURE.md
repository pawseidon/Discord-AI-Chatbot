# Multi-Agent Architecture Documentation

This document provides a comprehensive overview of the multi-agent architecture implemented in this Discord AI Chatbot. The architecture allows for sophisticated problem-solving through specialized agents, intelligent orchestration, and hybrid neural-symbolic reasoning.

## Architecture Overview

The Discord AI Chatbot is designed as a **modular multi-agent system**, with each "agent" (or reasoning mode) specialized for a task (e.g., creative writing, factual retrieval, logical deduction). This multi-agent architecture divides complex problems into tractable sub-tasks.

Each agent runs its own LLM prompt (persona, instructions, tools) and memory. For example, one agent excels at sequential chain-of-thought reasoning, another at web search and retrieval, another at creative brainstorming.

A top-level **Orchestrator** (Planner/Supervisor agent) decomposes user queries and routes sub-tasks to the appropriate agent. This "supervisor" can itself be an LLM agent using tools that include other agents.

## Core Components

### 1. Agent Service (`services/agent_service.py`)

The Agent Service serves as the central coordination point for reasoning detection and agent delegation. It:

- Analyzes incoming user queries and determines appropriate reasoning approaches
- Manages specialized agents for different reasoning types
- Handles emoji reactions to indicate reasoning types
- Provides a unified interface for agent interaction
- Manages memory and context for multi-turn conversations

```python
# Example usage
from bot_utilities.services.agent_service import agent_service

response = await agent_service.process_query(
    query="Explain how rainbows form and include some scientific details.",
    user_id="user_456",
    conversation_id="conversation_123",
    reasoning_type="sequential"  # Optional explicit reasoning type
)
```

### 2. Agent Orchestrator (`agent_orchestrator.py`)

The Agent Orchestrator is the implementation layer that handles the orchestration mechanics. It:

- Instantiates specialized agents based on detected reasoning requirements
- Handles agent delegation and inter-agent communication
- Manages conversation context and history
- Executes tool calls from agents
- Ensures coherent responses back to the user

### 3. Memory Service (`services/memory_service.py`)

The Memory Service provides multi-level memory capabilities:

- Conversation memory for tracking dialogue history
- Agent-specific scratchpads for working memory
- User preferences and personalization storage
- Vector-based semantic memory for relevant context retrieval

```python
# Example memory usage
from bot_utilities.services.memory_service import memory_service

# Store user preference
await memory_service.set_user_preference(
    user_id="user_123",
    preference_key="default_reasoning",
    preference_value="sequential"
)

# Retrieve conversation history
history = await memory_service.get_conversation_history(
    conversation_id="server_id:channel_id",
    limit=10
)
```

### 4. Agent Tools Manager (`agent_tools_manager.py`)

The Tools Manager provides access to external tools and capabilities:

- Web search and information retrieval
- Calculator and mathematical operations
- Fact checking and verification
- Memory management tools
- Knowledge base access

### 5. Workflow Service (`services/workflow_service.py`)

The Workflow Service implements LangGraph integration for complex agent workflows:

- Graph-based workflow definitions with nodes and edges
- State management between reasoning steps
- Conditional branching and routing
- Cycles and feedback loops for iterative reasoning

```python
# Example workflow usage
from bot_utilities.services.workflow_service import workflow_service

response = await workflow_service.create_and_run_default_workflow(
    user_query="Compare the economic policies of the last three presidents",
    user_id="user_123",
    conversation_id="conversation_456"
)
```

### 6. Symbolic Reasoning Service (`services/symbolic_reasoning_service.py`)

The Symbolic Reasoning Service adds deterministic reasoning capabilities:

- Mathematical problem solving with step-by-step solutions
- Logical rule application and inference
- Expression evaluation and verification
- Formal mathematical operations

```python
# Example symbolic reasoning
from bot_utilities.services.symbolic_reasoning_service import symbolic_reasoning_service

result = await symbolic_reasoning_service.evaluate_expression("(15 * 7) + (22 / 2)")
```

## Agent Types and Reasoning Modes

The system supports multiple specialized reasoning modes, each implemented as a distinct agent type:

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

## Reasoning Mode Selection

The system implements a **mode-selection logic** that maps user intent and context to one or more reasoning modes:

1. **Pattern Detection**: The system analyzes the query for specific patterns that indicate reasoning types
2. **Task Classification**: LLM-based classification labels queries (e.g., "fact-check," "generate story," "compute answer")
3. **Context Cues**: Keywords or conversation context trigger specific modes (e.g., "Write a poem" → creative mode)
4. **Dynamic Chaining**: Agents can hand off to other agents when needed (e.g., RAG → verification for fact-checking)
5. **Combination Detection**: Some queries benefit from multiple reasoning types working together

## Multi-Agent Workflow

### Query Processing Flow

1. **Query Analysis**: The orchestrator analyzes the user query to determine required reasoning type(s)
2. **Agent Selection**: Appropriate agent(s) are selected based on reasoning types
3. **Context Preparation**: Relevant memory and context are gathered for the agent
4. **Task Processing**: Agent processes the task, potentially using tools or delegating subtasks
5. **Response Generation**: Agent generates a response, which may include reasoning steps
6. **Emoji Reaction**: The bot adds emoji reactions indicating which reasoning types were used
7. **Memory Update**: Conversation history and agent memory are updated

### Workflow Mode (Advanced)

For complex queries, the system can use a graph-based workflow:

1. **Graph Creation**: A directed graph of reasoning steps is created
2. **Node Execution**: Each node (agent or reasoning step) is executed in sequence
3. **State Management**: Information is passed between nodes via a shared state
4. **Conditional Routing**: Different paths can be taken based on intermediate results
5. **Terminal Nodes**: Final nodes generate the response to the user

## Passing Data Between Agents

Agents share data via an **internal state/memory** or via explicit **handoffs**:

* **Shared Memory**: All agents can access the conversation history and user preferences
* **Agent Scratchpads**: Each agent has its own working memory for intermediate results
* **Command Protocol**: Agents communicate using structured commands with clear intent
* **Tool Results**: Results from tool usage are shared between agents

## Hybrid Symbolic & Neural Methods

The system combines LLMs with deterministic components:

1. **Retrieval-Augmented Generation (RAG)**: Queries are augmented with retrieved information
2. **Symbolic Reasoning Engine**: Mathematical operations use a deterministic solver
3. **Tool Integration**: Specialized tools like calculators, search, and code execution
4. **Verification Methods**: Fact-checking mechanisms to validate information

## Service Architecture Integration

The multi-agent system is implemented using a service-oriented architecture:

1. **Agent Service**: Central interface for agent coordination
2. **Memory Service**: Handles all memory operations
3. **Message Service**: Formats and processes Discord messages
4. **Workflow Service**: Creates and manages reasoning workflows
5. **Symbolic Reasoning Service**: Handles deterministic reasoning

All services are accessed through singleton instances to ensure consistent state:

```python
# Service imports
from bot_utilities.services.agent_service import agent_service
from bot_utilities.services.memory_service import memory_service
from bot_utilities.services.message_service import message_service
from bot_utilities.services.workflow_service import workflow_service
from bot_utilities.services.symbolic_reasoning_service import symbolic_reasoning_service
```

## EmojiReactionCog Integration

The bot uses the EmojiReactionCog to provide visual feedback about reasoning types:

1. Detects the reasoning types used in a response
2. Adds appropriate emoji reactions to messages
3. Updates reactions when reasoning types change during processing
4. Supports multiple reasoning types on a single message

```python
# Example emoji mapping
emoji_map = {
    "sequential": "🔄",
    "rag": "📚",
    "conversational": "💬",
    "verification": "✅",
    "creative": "🎨",
    "calculation": "🧮",
    "graph": "📊",
    "multi_agent": "👥",
    "planning": "📋"
}
```

## Best Practices

1. **Specialized Agents**: Keep agents specialized - they should excel at one type of reasoning
2. **Clear Delegation**: Use explicit delegation patterns when crossing domains
3. **Shared Context**: Ensure critical context is passed between agents
4. **Symbolic Verification**: Use symbolic reasoning to verify numerical or logical outputs
5. **Emoji Indications**: Use emoji reactions to indicate reasoning types for transparency

## Agent Communication Protocol

Agents communicate using structured `AgentCommand` objects with these command types:

- `RESPONSE`: Final response to user
- `DELEGATE`: Request for another agent to handle a subtask
- `TOOL_CALL`: Request to use an external tool
- `SEARCH`: Request for information retrieval
- `VERIFY`: Request for fact verification
- `REFLECT`: Self-reflection or thinking step
- `PLAN`: Strategic planning step
- `MERGE`: Combine multiple outputs
- `ERROR`: Error notification

## Extending the Architecture

### Adding New Agent Types

To add a new agent type:

1. Define the agent's system prompt and configuration
2. Add detection patterns to `ReasoningDetector` class
3. Register any specialized tools the agent needs
4. Update the agent emoji map for visualization

### Adding New Tools

To add a new tool:

1. Implement the tool function in `AgentToolsManager`
2. Register the tool with description and parameter schema
3. Update documentation for agents to know when to use it

### Creating Custom Workflows

To define a new workflow:

1. Use `WorkflowService` to define nodes (agents)
2. Define edges with conditional functions
3. Set up state management for information passing
4. Register the workflow with the orchestrator 