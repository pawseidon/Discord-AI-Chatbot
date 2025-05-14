# Services Migration Guide

## Overview

This document outlines how to migrate from the old utility-based implementation to the new service-oriented architecture. The service architecture centralizes functionality, improves error handling, and provides a more consistent API.

## Service Directory Structure

The services are organized in the `bot_utilities/services/` directory:

```
bot_utilities/
  ├── services/
  │   ├── __init__.py       # Service registry and getters
  │   ├── agent_service.py  # Agent and reasoning management
  │   ├── intent_detection.py # Intent detection and pattern matching
  │   ├── memory_service.py # Conversation history and user preferences
  │   ├── message_service.py # Message formatting and sending
  │   ├── workflow_service.py # Multi-agent workflow orchestration
  │   └── symbolic_reasoning_service.py # Mathematical and logical reasoning
  └── ...
```

## Migration Guide

### 1. Memory Functions

**Old approach**:
```python
from bot_utilities.memory_utils import get_user_preferences, save_user_preferences

preferences = get_user_preferences(user_id)
preferences['theme'] = 'dark'
save_user_preferences(user_id, preferences)
```

**New approach**:
```python
from bot_utilities.services.memory_service import memory_service

preferences = await memory_service.get_user_preferences(user_id)
await memory_service.set_user_preference(user_id, 'theme', 'dark')
```

### 2. Message Formatting

**Old approach**:
```python
from bot_utilities.formatting_utils import chunk_message, format_response_for_discord

chunks = chunk_message(long_message)
formatted = format_response_for_discord(message, 'creative')
```

**New approach**:
```python
from bot_utilities.services.message_service import message_service

chunks = await message_service.split_message(long_message)
formatted = await message_service.format_with_agent_emoji(message, 'creative')
```

### 3. Intent Detection

**Old approach**:
```python
# Pattern matching in on_message.py or other handlers
if re.search(r'generate .* image', message.content.lower()):
    # Handle image generation
```

**New approach**:
```python
from bot_utilities.services.intent_detection import intent_service

intent = await intent_service.detect_intent(message.content)
if intent.type == 'image_generation':
    # Handle image generation
```

### 4. Agent Orchestration

**Old approach**:
```python
from bot_utilities.agent_orchestrator import agent_orchestrator

response = await agent_orchestrator.process_query(query, conversation_id, user_id)
```

**New approach**:
```python
from bot_utilities.services.agent_service import agent_service

response = await agent_service.process_query(
    query=query,
    user_id=user_id,
    channel_id=conversation_id
)
```

### 5. Multi-Agent Workflows

**Old approach**:
```python
from bot_utilities.agent_workflow_manager import agent_workflow_manager

response = await agent_workflow_manager.create_and_run_default_workflow(
    query, user_id, conversation_id
)
```

**New approach**:
```python
from bot_utilities.services.workflow_service import workflow_service

response = await workflow_service.create_and_run_default_workflow(
    user_query=query,
    user_id=user_id,
    conversation_id=conversation_id
)
```

### 6. Symbolic Reasoning

**Old approach**:
```python
from bot_utilities.symbolic_reasoning_registry import symbolic_reasoning_registry

reasoner = symbolic_reasoning_registry.get_reasoner("math")
result = await reasoner.solve_math_problem("(15 * 7) + (22 / 2)")
```

**New approach**:
```python
from bot_utilities.services.symbolic_reasoning_service import symbolic_reasoning_service

result = await symbolic_reasoning_service.solve_math_problem("(15 * 7) + (22 / 2)")
# The steps are included in the result
steps = result["steps"]
```

## Using the Services in Event Handlers

```python
import discord
from discord.ext import commands
from bot_utilities.services import memory_service, message_service, intent_service, agent_service

class EventHandler(commands.Cog):
    @commands.Cog.listener()
    async def on_message(self, message):
        # Skip bot messages
        if message.author.bot:
            return
        
        # Detect intent
        intent = await intent_service.detect_intent(message.content)
        
        # Process based on intent
        if intent.type == 'image_generation':
            # Handle image generation
            pass
        elif intent.type == 'reasoning_request':
            # Process with agent
            response = await agent_service.process_query(
                query=message.content,
                user_id=str(message.author.id),
                channel_id=str(message.channel.id)
            )
            
            # Format and send response
            chunks = await message_service.split_message(response)
            for chunk in chunks:
                await message.channel.send(chunk)
                
        # Store message in history
        await memory_service.add_to_history(
            user_id=str(message.author.id),
            channel_id=str(message.channel.id),
            content=message.content,
            role='user'
        )
```

## Math Problem Handling

For handling mathematical problems with step-by-step solutions:

```python
import discord
from discord.ext import commands
from bot_utilities.services import symbolic_reasoning_service

class MathCog(commands.Cog):
    @commands.command(name="solve")
    async def solve_math(self, ctx, *, expression):
        """Solve a mathematical problem with steps"""
        
        # Use the symbolic reasoning service
        result = await symbolic_reasoning_service.solve_math_problem(expression)
        
        if result["success"]:
            # Create a nice formatted response with steps
            response = f"**Solution for `{expression}`**\n\n"
            
            # Add steps
            for step in result["steps"]:
                response += f"• {step}\n"
                
            # Add final result
            response += f"\n**Result**: `{result['result']}`"
            
            await ctx.send(response)
        else:
            await ctx.send(f"Error solving the problem: {result['error']}")
```

## Error Handling

All services include centralized error handling. Errors are logged and can be handled gracefully:

```python
try:
    await memory_service.add_to_history(user_id, channel_id, message, 'user')
except Exception as e:
    logger.error(f"Error adding to history: {str(e)}")
    # Implement fallback behavior
```

## Backward Compatibility

For a transitional period, the old utility functions are maintained but marked as deprecated. They now delegate to the service implementations internally. 