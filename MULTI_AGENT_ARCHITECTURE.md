# Multi-Agent Architecture Documentation

This document provides a detailed overview of the multi-agent architecture implemented in this Discord AI Chatbot. The architecture allows for sophisticated problem-solving through specialized agents, intelligent orchestration, and hybrid neural-symbolic reasoning.

## Core Components

### 1. Agent Orchestrator (`agent_orchestrator.py`)

The Agent Orchestrator serves as the central coordinator for all agent interactions. It:

- Analyzes incoming user queries and determines the most appropriate reasoning approach
- Instantiates specialized agents based on detected reasoning requirements
- Handles agent delegation and inter-agent communication
- Manages conversation context and history
- Ensures coherent responses back to the user

```python
# Example usage
orchestrator = AgentOrchestrator(llm_provider=llm, tools_manager=tools_manager)
response = await orchestrator.process_query(
    query="Explain how rainbows form and include some scientific details.",
    conversation_id="conversation_123",
    user_id="user_456"
)
```

### 2. Agent Configuration (`agent_config.py`)

Each agent is defined by its configuration, which specifies:

- Agent ID and reasoning type
- System prompt that defines its behavior
- Available tools and capabilities
- Temperature and other generation parameters

```python
# Example agent configuration
config = AgentConfig(
    agent_id="sequential",
    system_prompt="You are a Sequential Thinking agent that breaks down complex problems...",
    tools=["calculator", "search", "step_analyzer"],
    temperature=0.3,
    max_tokens=2000
)
```

### 3. Agent Memory System (`agent_memory.py`)

The memory system provides multi-level memory capabilities:

- Conversation memory for tracking dialogue history
- Agent-specific scratchpads for working memory
- Shared memory contexts for cross-agent knowledge sharing
- Vector-based semantic memory for relevant context retrieval
- User preferences and personalization storage

```python
# Example memory usage
memory_manager = AgentMemoryManager()
await memory_manager.store_in_agent_memory(
    agent_id="sequential",
    user_id="user_123",
    conversation_id="conv_456",
    key="problem_decomposition",
    value=["Step 1: Understand the problem", "Step 2: Break it down"]
)
```

### 4. Agent Tools Manager (`agent_tools_manager.py`)

The Tools Manager provides access to external tools and capabilities:

- Calculator and mathematical operations
- Web search and information retrieval
- Fact checking and verification
- Data analysis and visualization
- Knowledge graph creation
- Symbolic reasoning engines

```python
# Example tool usage
tools_manager = AgentToolsManager()
result = await tools_manager.execute_tool(
    name="calculator",
    params={"expression": "2 * (3 + 4)"}
)
```

### 5. Workflow Manager (`agent_workflow_manager.py`)

The Workflow Manager implements LangGraph integration for complex agent workflows:

- Graph-based workflow definitions
- State management between steps
- Conditional branching and routing
- Cycles and feedback loops
- Parallel agent execution

```python
# Example workflow creation
workflow = AgentWorkflowManager()
graph = workflow.create_workflow(
    nodes=["planner", "researcher", "writer"],
    edges=[
        ("planner", "researcher", lambda state: state["needs_research"]),
        ("researcher", "writer", lambda state: True)
    ]
)
```

### 6. Symbolic Reasoning (`symbolic_reasoning.py`)

The Symbolic Reasoning module adds deterministic reasoning capabilities:

- Mathematical problem solving with step-by-step solutions
- Logical rule application and inference
- Data structure analysis
- Fact verification against knowledge bases

```python
# Example symbolic reasoning
reasoner = symbolic_reasoning_registry.get_reasoner()
result = await reasoner.solve_math_problem("(15 * 7) + (22 / 2)")
```

## Agent Types and Reasoning Modes

The system supports various specialized reasoning modes, each implemented as a distinct agent type:

| Agent Type | Description | Best For |
|------------|-------------|----------|
| 🧠 Sequential | Step-by-step analytical thinking | Complex problems, detailed analysis |
| 🔍 RAG | Retrieval-Augmented Generation | Information seeking, fact-based queries |
| 💬 Conversational | Natural, friendly dialogue | Casual chat, simple exchanges |
| 📚 Knowledge | Detailed educational content | Learning, concept explanation |
| ✅ Verification | Fact-checking and validation | Claims evaluation, truth assessment |
| 🎨 Creative | Imaginative content generation | Stories, art, creative writing |
| 🔢 Calculation | Mathematical operations | Computations, numerical analysis |
| 📋 Planning | Structured strategy development | Project plans, task organization |
| 🕸️ Graph | Relationship mapping | Concept networks, interconnections |
| 👥 Multi-Agent | Multiple perspectives | Balanced viewpoints, debates |
| 🔎 Step-Back | Big-picture perspective | Holistic analysis, context understanding |
| ⛓️ Chain-of-Thought | Logical progression | Deductive reasoning, cause-effect |
| 🔄 ReAct | Reasoning with actions | Tool usage, action-based problem solving |

## How It Works

### Query Processing Flow

1. **Query Analysis**: The orchestrator analyzes the user query to determine the required reasoning type
2. **Agent Selection**: The appropriate agent is selected or created based on reasoning type
3. **Context Preparation**: Relevant memory and context are gathered for the agent
4. **Task Processing**: The agent processes the task, potentially using tools or delegating subtasks
5. **Response Generation**: The agent generates a response, which may include reasoning steps
6. **Memory Update**: Conversation history and agent memory are updated

### Agent Delegation Flow

When an agent encounters a subtask better suited for another specialist:

1. **Delegation Decision**: Agent determines another agent type would be better
2. **Command Creation**: Agent creates a delegation command with task description
3. **Orchestrator Handling**: Orchestrator receives command and routes to target agent
4. **Subtask Processing**: Target agent processes the subtask
5. **Result Integration**: Original agent receives result and incorporates it
6. **Final Response**: Complete response with integrated specialist knowledge

### Symbolic Reasoning Integration

For tasks requiring deterministic reasoning:

1. **Pattern Detection**: System identifies mathematical or logical patterns
2. **Symbolic Handler**: Task is routed to appropriate symbolic reasoning function
3. **Deterministic Processing**: Task is solved using rule-based algorithms
4. **Step Documentation**: Each reasoning step is documented
5. **Result Integration**: Results are formatted for user presentation

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

Example command structure:
```python
{
    "command_type": "DELEGATE",
    "content": "What are the key factors in climate change?",
    "target_agent": "knowledge",
    "metadata": {
        "origin_agent": "sequential",
        "reason": "Need detailed factual knowledge"
    }
}
```

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

1. Use `AgentWorkflowManager` to define nodes (agents)
2. Define edges with conditional functions
3. Set up state management for information passing
4. Register the workflow with the orchestrator

## Best Practices

1. **Specialized Agents**: Keep agents specialized - they should excel at one type of reasoning
2. **Clear Delegation**: Use explicit delegation patterns when crossing domains
3. **Shared Context**: Ensure critical context is passed between agents
4. **Symbolic Verification**: Use symbolic reasoning to verify numerical or logical outputs
5. **Fallback Mechanisms**: Always implement fallbacks for when primary approaches fail

## Architecture Diagram

```
User Query
   │
   ▼
┌─────────────────┐
│                 │
│  ORCHESTRATOR   │◄───────┐
│                 │        │
└────────┬────────┘        │
         │                 │
         ▼                 │
┌─────────────────┐        │
│  REASONING      │        │
│  DETECTOR       │        │
└────────┬────────┘        │
         │                 │
         ▼                 │
┌─────────────────┐        │
│                 │        │
│  PRIMARY AGENT  │        │ Delegation
│                 │        │
└────┬─────┬──────┘        │
     │     │               │
     │     ▼               │
     │  ┌──────────────┐   │
     │  │ SPECIALIZED  │   │
     │  │ AGENT        ├───┘
     │  └──────────────┘
     │
     ▼
┌─────────────────┐    ┌──────────────┐
│                 │    │              │
│  TOOLS MANAGER  │◄───┤ MEMORY       │
│                 │    │ MANAGER      │
└────┬─────┬──────┘    └──────────────┘
     │     │
     │     ▼
     │  ┌──────────────┐
     │  │ SYMBOLIC     │
     │  │ REASONING    │
     │  └──────────────┘
     ▼
┌─────────────────┐
│                 │
│  RESPONSE       │
│                 │
└─────────────────┘
```

## Resources

- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)
- [Discord.py Documentation](https://discordpy.readthedocs.io/en/stable/)
- [Agent Patterns in LangChain](https://python.langchain.com/docs/expression_language/cookbook/agent)
- [Symbolic AI Overview](https://en.wikipedia.org/wiki/Symbolic_artificial_intelligence) 