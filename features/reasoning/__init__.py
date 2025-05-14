"""
Reasoning module for Discord AI Chatbot.

This module provides different reasoning methods and integration
for advanced AI reasoning capabilities.
"""

from .reasoning_router import ReasoningRouter, ReasoningMethod
from .reasoning_integration import IntegratedReasoning, create_integrated_reasoning
from .methods.sequential_thinking import SequentialThinking, process_sequential_thinking
from .methods.react_reasoning import ReactReasoning, process_react_reasoning
from .methods.reflective_rag import ReflectiveRAG, process_reflective_rag
from .methods.speculative_rag import SpeculativeRAG, process_speculative_rag
from .methods.chain_of_verification import (
    ChainOfVerificationReasoning, 
    process_chain_of_verification,
    ChainOfVerification
)

# Dictionary of available reasoning methods for easy access
REASONING_METHODS = {
    "sequential": process_sequential_thinking,
    "react": process_react_reasoning,
    "reflective_rag": process_reflective_rag,
    "speculative_rag": process_speculative_rag,
    "chain_of_verification": process_chain_of_verification
}

__all__ = [
    'ReasoningRouter',
    'ReasoningMethod',
    'IntegratedReasoning',
    'create_integrated_reasoning',
    'SequentialThinking',
    'ReactReasoning',
    'ReflectiveRAG',
    'SpeculativeRAG',
    'ChainOfVerificationReasoning',
    'ChainOfVerification',
    'process_sequential_thinking',
    'process_react_reasoning',
    'process_reflective_rag',
    'process_speculative_rag',
    'process_chain_of_verification',
    'REASONING_METHODS',
]
