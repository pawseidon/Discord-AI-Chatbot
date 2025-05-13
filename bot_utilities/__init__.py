"""Bot Utilities Module

This package contains utility functions and classes for the Discord AI bot.
"""

# Import modules to make them available when importing the package
from bot_utilities.ai_utils import *
from bot_utilities.sequential_thinking import create_sequential_thinking, SequentialThinking
from bot_utilities.reasoning_router import create_reasoning_router, ReasoningRouter

# New modules
from bot_utilities.state_manager import StateManager
from bot_utilities.response_cache import ResponseCache
from bot_utilities.router_with_state import create_sum2act_router, Sum2ActRouter
from bot_utilities.context_manager import get_context_manager, start_context_manager_tasks
from bot_utilities.router_compatibility import create_router_adapter

# Import commonly used utilities for easier access
from bot_utilities.ai_utils import generate_response
from bot_utilities.config_loader import config, load_instructions 