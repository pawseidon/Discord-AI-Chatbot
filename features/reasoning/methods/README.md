# Reasoning Methods

This directory contains modular implementations of different reasoning methods used by the Discord AI chatbot.

## Structure

Each reasoning method is implemented as a separate module with a consistent interface:

- A class implementation (e.g., `SequentialThinking`, `ReactReasoning`) 
- A helper function that creates and uses the class (e.g., `process_sequential_thinking`)

## Available Methods

1. **Sequential Thinking** (`sequential_thinking.py`)
   - Step-by-step reasoning for complex problems
   - Used for breaking down questions into manageable steps

2. **ReAct Reasoning** (`react_reasoning.py`)
   - Reasoning + Acting approach for action-oriented problem solving
   - Combines thinking with tool usage for complex tasks

3. **Reflective RAG** (`reflective_rag.py`)
   - Self-reflective RAG for improved response quality
   - Evaluates and improves initial retrieval-based responses

4. **Speculative RAG** (`speculative_rag.py`)
   - Multi-query speculative approach for better retrieval
   - Uses a two-tiered system with query generation and verification

## Migration Process

These methods were originally implemented in the `bot_utilities` directory but have been modularized and migrated to this new location for better organization and maintainability.

Original files:
- `bot_utilities/sequential_thinking.py` → `features/reasoning/methods/sequential_thinking.py`
- `bot_utilities/react_utils.py` → `features/reasoning/methods/react_reasoning.py`
- `bot_utilities/reflective_rag.py` → `features/reasoning/methods/reflective_rag.py`
- `bot_utilities/speculative_rag.py` → `features/reasoning/methods/speculative_rag.py`

## Adding New Methods

To add a new reasoning method:

1. Create a new file in this directory with a class implementation and helper function
2. Update `features/reasoning/__init__.py` to include your new method
3. Register the method in `features/reasoning/reasoning_router.py`

## Integration

All methods are integrated through the `ReasoningRouter` class, which selects the appropriate method based on the query and context.

The router is then used by the `ReasoningIntegration` class to provide a seamless experience where methods can be combined and triggered naturally through conversation. 