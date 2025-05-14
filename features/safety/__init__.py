"""
Safety module for Discord AI Chatbot.

This module provides safety features such as hallucination detection,
content moderation, and verification mechanisms.
"""

from .hallucination_handler import (
    HallucinationHandler,
    create_hallucination_handler
)

__all__ = [
    'HallucinationHandler',
    'create_hallucination_handler'
]
