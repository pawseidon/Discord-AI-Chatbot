"""
Services Module

This package contains various service modules that provide centralized functionality
for different aspects of the bot's operation.

The services are loaded lazily to avoid circular imports.
"""

# Import the service modules but not the instances directly
# to avoid circular imports
from . import agent_service
from . import intent_detection
from . import memory_service
from . import message_service
from . import workflow_service
from . import symbolic_reasoning_service
from . import sequential_thinking_service

# Define getter functions for lazy loading
def get_agent_service():
    """Get the agent service instance"""
    return agent_service.agent_service

def get_intent_service():
    """Get the intent detection service instance"""
    return intent_detection.intent_service

def get_memory_service():
    """Get the memory service instance"""
    return memory_service.memory_service

def get_message_service():
    """Get the message service instance"""
    return message_service.message_service

def get_workflow_service():
    """Get the workflow service instance"""
    return workflow_service.workflow_service

def get_symbolic_reasoning_service():
    """Get the symbolic reasoning service instance"""
    return symbolic_reasoning_service.symbolic_reasoning_service

def get_sequential_thinking_service():
    """Get the sequential thinking service instance"""
    return sequential_thinking_service.sequential_thinking_service

__all__ = [
    'agent_service',
    'intent_detection',
    'memory_service',
    'message_service',
    'workflow_service',
    'symbolic_reasoning_service',
    'sequential_thinking_service',
    'get_agent_service',
    'get_intent_service',
    'get_memory_service',
    'get_message_service',
    'get_workflow_service',
    'get_symbolic_reasoning_service',
    'get_sequential_thinking_service'
] 