"""
Reasoning methods module for Discord AI Chatbot.

This module contains various reasoning methods for the bot to use
based on query characteristics and context.
"""

from .sequential_thinking import SequentialThinking, process_sequential_thinking
from .react_reasoning import ReactReasoning, process_react_reasoning
from .reflective_rag import ReflectiveRAG, process_reflective_rag
from .speculative_rag import SpeculativeRAG, process_speculative_rag
from .chain_of_verification import (
    ChainOfVerificationReasoning, 
    process_chain_of_verification,
    ChainOfVerification,
    VerificationStep
)

__all__ = [
    'SequentialThinking',
    'process_sequential_thinking',
    'ReactReasoning',
    'process_react_reasoning',
    'ReflectiveRAG',
    'process_reflective_rag',
    'SpeculativeRAG',
    'process_speculative_rag',
    'ChainOfVerificationReasoning',
    'process_chain_of_verification',
    'ChainOfVerification',
    'VerificationStep'
] 