#!/usr/bin/env python3
"""
Multi-Agent Architecture Test Script

This script demonstrates the enhanced multi-agent architecture with service-oriented design,
advanced memory management, and different reasoning modes working together.
"""

import asyncio
import os
import argparse
import sys
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our service components - use the new service architecture
from bot_utilities.services.agent_service import agent_service
from bot_utilities.services.memory_service import memory_service
from bot_utilities.services.message_service import message_service
from bot_utilities.services.workflow_service import workflow_service
from bot_utilities.services.symbolic_reasoning_service import symbolic_reasoning_service
from bot_utilities.ai_utils import get_ai_provider

# Global variables for coloring terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

async def setup_components():
    """Set up all the components needed for the test"""
    print(f"{YELLOW}Setting up components...{RESET}")
    
    # Initialize services
    llm_provider = await get_ai_provider()
    print(f"{GREEN}✓ Connected to LLM provider{RESET}")
    
    # Initialize the services in the correct order
    await symbolic_reasoning_service.ensure_initialized()
    print(f"{GREEN}✓ Initialized symbolic reasoning service{RESET}")
    
    await memory_service.ensure_initialized()
    print(f"{GREEN}✓ Initialized memory service{RESET}")
    
    await message_service.ensure_initialized()
    print(f"{GREEN}✓ Initialized message service{RESET}")
    
    await agent_service.initialize(llm_provider)
    print(f"{GREEN}✓ Initialized agent service{RESET}")
    
    await workflow_service.initialize(llm_provider)
    print(f"{GREEN}✓ Initialized workflow service{RESET}")
    
    return {
        "llm_provider": llm_provider,
        "agent_service": agent_service,
        "memory_service": memory_service,
        "message_service": message_service,
        "workflow_service": workflow_service,
        "symbolic_reasoning_service": symbolic_reasoning_service
    }

async def test_orchestrator(components: Dict[str, Any], query: str):
    """Test the agent service"""
    print(f"\n{BLUE}{BOLD}Testing Agent Service{RESET}")
    print(f"{YELLOW}Query: {query}{RESET}")
    
    # Process the query using agent_service
    conversation_id = "test_conversation"
    print(f"{YELLOW}Processing with agent_service...{RESET}")
    
    response = await agent_service.process_query(
        query=query,
        user_id="test_user",
        conversation_id=conversation_id,
        reasoning_type="sequential",
        context={"test": True},
        enable_tools=True
    )
    
    print(f"\n{GREEN}{BOLD}Agent Service Response:{RESET}")
    print(f"{GREEN}{response}{RESET}")
    
    return response

async def test_workflow(components: Dict[str, Any], query: str):
    """Test the workflow service with LangGraph"""
    print(f"\n{BLUE}{BOLD}Testing Workflow Service{RESET}")
    print(f"{YELLOW}Query: {query}{RESET}")
    
    try:
        # Check if LangGraph is available
        import langgraph
        print(f"{YELLOW}LangGraph is available. Creating workflow...{RESET}")
        
        # Run workflow using workflow_service
        response = await workflow_service.run_workflow(
            workflow_type="default",
            user_query=query,
            user_id="test_user",
            conversation_id="test_workflow"
        )
        
        print(f"\n{GREEN}{BOLD}Workflow Response:{RESET}")
        print(f"{GREEN}{response}{RESET}")
        
        return response
    except ImportError:
        print(f"{RED}LangGraph is not installed. Skipping workflow test.{RESET}")
        print(f"{YELLOW}Install with: pip install langgraph{RESET}")
        return None

async def test_memory(components: Dict[str, Any], query: str):
    """Test the memory service"""
    print(f"\n{BLUE}{BOLD}Testing Memory Service{RESET}")
    
    # Add some test memories and preferences
    print(f"{YELLOW}Adding user preferences...{RESET}")
    await memory_service.set_user_preferences(
        user_id="test_user",
        preferences={
            "reasoning_mode": "sequential",
            "stream_enabled": True,
            "language": "en",
            "max_response_length": 2000
        }
    )
    
    # Add conversation messages
    print(f"{YELLOW}Adding conversation history...{RESET}")
    await memory_service.add_message(
        conversation_id="test_conversation",
        user_id="test_user",
        role="user",
        content="How does photosynthesis work?"
    )
    
    await memory_service.add_message(
        conversation_id="test_conversation",
        user_id="system",
        role="assistant",
        content="Photosynthesis is the process by which plants convert light energy into chemical energy..."
    )
    
    # Get user preferences
    print(f"{YELLOW}Retrieving user preferences...{RESET}")
    preferences = await memory_service.get_user_preferences("test_user")
    
    # Get conversation history
    print(f"{YELLOW}Retrieving conversation history...{RESET}")
    history = await memory_service.get_conversation_history("test_conversation")
    
    print(f"\n{GREEN}{BOLD}User Preferences:{RESET}")
    print(f"{GREEN}{preferences}{RESET}")
    
    print(f"\n{GREEN}{BOLD}Conversation History:{RESET}")
    for message in history:
        print(f"{GREEN}• {message['role']}: {message['content'][:50]}...{RESET}")
    
    return {
        "preferences": preferences,
        "history": history
    }

async def test_symbolic_reasoning():
    """Test the symbolic reasoning service"""
    print(f"\n{BLUE}{BOLD}Testing Symbolic Reasoning Service{RESET}")
    
    # Test basic calculation
    math_query = "Calculate the area of a circle with radius 5 cm"
    print(f"{YELLOW}Testing math calculation: {math_query}{RESET}")
    
    result = await symbolic_reasoning_service.process_math_expression(
        "pi * 5^2",
        show_working=True
    )
    
    print(f"\n{GREEN}{BOLD}Math Calculation Result:{RESET}")
    print(f"{GREEN}{result}{RESET}")
    
    # Test logical reasoning
    logic_query = "If all humans are mortal and Socrates is human, then is Socrates mortal?"
    print(f"{YELLOW}Testing logical reasoning: {logic_query}{RESET}")
    
    # Set up basic premises for symbolic reasoning
    premises = [
        "All humans are mortal",
        "Socrates is human"
    ]
    
    conclusion = "Socrates is mortal"
    
    logic_result = await symbolic_reasoning_service.verify_logical_argument(
        premises=premises,
        conclusion=conclusion
    )
    
    print(f"\n{GREEN}{BOLD}Logical Reasoning Result:{RESET}")
    print(f"{GREEN}{logic_result}{RESET}")
    
    return {
        "math_result": result,
        "logic_result": logic_result
    }

async def main():
    """Main function to run the tests"""
    parser = argparse.ArgumentParser(description="Test the multi-agent architecture")
    parser.add_argument("--test", choices=["all", "orchestrator", "workflow", "memory", "symbolic"], 
                        default="all", help="Which test to run")
    parser.add_argument("--query", type=str, default="Explain how photosynthesis works in detail",
                        help="Query to test with")
    args = parser.parse_args()
    
    print(f"{BLUE}{BOLD}Running Multi-Agent Architecture Tests{RESET}")
    print(f"{YELLOW}Initializing services...{RESET}")
    
    try:
        # Set up all components
        components = await setup_components()
        
        # Run the specified test
        if args.test in ["all", "orchestrator"]:
            await test_orchestrator(components, args.query)
            
        if args.test in ["all", "workflow"]:
            await test_workflow(components, args.query)
            
        if args.test in ["all", "memory"]:
            await test_memory(components, args.query)
            
        if args.test in ["all", "symbolic"]:
            await test_symbolic_reasoning()
        
        print(f"\n{GREEN}{BOLD}All tests completed successfully!{RESET}")
        
    except Exception as e:
        print(f"{RED}Error running tests: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 