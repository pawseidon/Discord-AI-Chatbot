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
import logging
import json

# Load environment variables
load_dotenv()

# Import our service components - use the new service architecture
from bot_utilities.services.agent_service import agent_service
from bot_utilities.services.memory_service import memory_service
from bot_utilities.services.message_service import message_service
from bot_utilities.services.workflow_service import workflow_service
from bot_utilities.services.symbolic_reasoning_service import symbolic_reasoning_service
from bot_utilities.ai_utils import get_ai_provider
from bot_utilities.services.sequential_thinking_service import sequential_thinking_service

# Global variables for coloring terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("service_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('test_multi_agent')

class MockLLM:
    """Mock LLM provider for testing."""
    
    def __init__(self, response_map=None):
        self.response_map = response_map or {}
        self.default_response = "This is a test response."
        self.last_system_prompt = None
        self.last_user_prompt = None
        
    async def generate_text(self, system_prompt=None, user_prompt=None, temperature=0.7):
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        
        # Return a response based on keywords in the prompt
        if user_prompt:
            for keyword, response in self.response_map.items():
                if keyword.lower() in user_prompt.lower():
                    logger.info(f"Mock LLM returning response for keyword: {keyword}")
                    return response
        
        # Return default response
        return self.default_response

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

async def setup_mock_provider():
    """Create and configure a mock LLM provider for testing."""
    
    # Create response mapping for different query types
    mock_responses = {
        # Sequential + RAG workflow responses
        "how does nuclear": "Nuclear reactors work by harnessing the energy released during nuclear fission. In this process, uranium atoms split, releasing energy that heats water to create steam, which then drives turbines to generate electricity. Key components include fuel rods, control rods, coolant systems, and containment structures.",
        
        # Verification + RAG workflow responses
        "verify if": "After cross-referencing multiple sources, I can confirm that the Earth is approximately 4.54 billion years old. This age has been determined through radiometric dating of meteorites and is consistent across several independent studies. The evidence is considered highly reliable in the scientific community.",
        
        # Calculation + Sequential workflow responses
        "calculate": "The result of the calculation is 42. This was determined by following these steps: Step 1: Identify the variables, Step 2: Apply the formula, Step 3: Compute the intermediate values, Step 4: Add the final components.",
        
        # Creative + Sequential workflow responses
        "write a story": "The Ancient Forest\n\nOnce upon a time, deep within a forgotten valley, there stood an ancient forest where trees whispered secrets of bygone eras. The forest was home to mystical creatures, each with their own stories and wisdom accumulated over centuries.\n\nThe protagonist, a young explorer named Elara, discovered this hidden realm after following an old map she found in her grandmother's attic.",
        
        # Graph + RAG workflow responses
        "relationship between": "The relationship analysis shows three primary connections:\n\n1. Company A → Company B: Parent company relationship (100% ownership)\n2. Company B ↔ Company C: Strategic partnership with shared R&D initiatives\n3. Company C → Company D: Minority investment stake (35% ownership)\n\nThis creates a hierarchical structure with Company A at the top of the influence chain.",
        
        # Multi-agent workflow response
        "complex problem": "This problem requires multiple perspectives. From a sequential reasoning standpoint, we can break this down into steps: 1) Identify the core issue, 2) Analyze contributing factors, 3) Evaluate potential solutions. From a creative perspective, we might consider unconventional approaches like biomimicry or reverse engineering. The research indicates several promising directions based on recent studies in this field."
    }
    
    # Create mock provider with responses
    mock_provider = MockLLM(response_map=mock_responses)
    return mock_provider

async def test_workflow_detection():
    """Test the workflow detection functionality."""
    
    logger.info("Testing workflow detection...")
    
    # Test queries and expected workflow types
    test_cases = [
        {"query": "How does nuclear fusion work?", "expected": "sequential_rag"},
        {"query": "Verify if the Earth is 4.5 billion years old", "expected": "verification_rag"},
        {"query": "Calculate the area of a circle with radius 5cm and explain the steps", "expected": "calculation_sequential"},
        {"query": "Write a story about an enchanted forest with a clear beginning, middle, and end", "expected": "creative_sequential"},
        {"query": "Map the relationship between tech companies in Silicon Valley", "expected": "graph_rag_verification"},
        {"query": "What's your favorite color?", "expected": "multi_agent"}  # Default for simple queries
    ]
    
    # Set up mock provider
    mock_provider = await setup_mock_provider()
    
    # Initialize workflow service
    await workflow_service.initialize(mock_provider)
    
    # Test each case
    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        expected = test_case["expected"]
        
        # Detect workflow type
        detected = await workflow_service.detect_workflow_type(query)
        
        # Log results
        logger.info(f"Test {i+1}: Query: '{query}'")
        logger.info(f"Expected: {expected}, Detected: {detected}")
        logger.info(f"Result: {'✅ PASS' if detected == expected else '❌ FAIL'}")
        logger.info("-" * 50)

async def test_reasoning_detection():
    """Test the reasoning type detection functionality."""
    
    logger.info("Testing reasoning type detection...")
    
    # Test queries and expected reasoning types
    test_cases = [
        {"query": "Explain how photosynthesis works step by step", "expected": ["sequential"]},
        {"query": "Find the latest information about climate change", "expected": ["rag"]},
        {"query": "Verify the accuracy of this statement: 'The Great Wall of China is visible from space'", "expected": ["verification"]},
        {"query": "Calculate 25% of 240", "expected": ["calculation"]},
        {"query": "Write a poem about the moon", "expected": ["creative"]},
        {"query": "Map the relationships between characters in Romeo and Juliet", "expected": ["graph"]},
        {"query": "Explain the process of photosynthesis and find the latest research on it", "expected": ["sequential", "rag"]}
    ]
    
    # Set up mock provider
    mock_provider = await setup_mock_provider()
    
    # Initialize agent service
    await agent_service.initialize(mock_provider)
    
    # Test each case
    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        expected = test_case["expected"]
        
        # Detect reasoning types
        detected = await agent_service.detect_multiple_reasoning_types(query, max_types=len(expected))
        
        # Check if all expected types are in detected types (order may vary)
        match = all(exp in detected for exp in expected) and len(detected) >= len(expected)
        
        # Log results
        logger.info(f"Test {i+1}: Query: '{query}'")
        logger.info(f"Expected: {expected}, Detected: {detected}")
        logger.info(f"Result: {'✅ PASS' if match else '❌ FAIL'}")
        logger.info("-" * 50)

async def test_workflow_processing():
    """Test processing queries with different workflows."""
    
    logger.info("Testing workflow processing...")
    
    # Test queries for different workflows
    test_cases = [
        {"query": "How does nuclear fusion work?", "workflow": "sequential_rag"},
        {"query": "Verify if the Earth is 4.5 billion years old", "workflow": "verification_rag"},
        {"query": "Calculate the area of a circle with radius 5cm and explain the steps", "workflow": "calculation_sequential"},
        {"query": "Write a story about an enchanted forest", "workflow": "creative_sequential"},
        {"query": "Map the relationship between tech companies", "workflow": "graph_rag_verification"},
        {"query": "Solve this complex problem with multiple aspects", "workflow": "multi_agent"}
    ]
    
    # Set up mock provider
    mock_provider = await setup_mock_provider()
    
    # Initialize services
    await workflow_service.initialize(mock_provider)
    await sequential_thinking_service.initialize(mock_provider)
    await agent_service.initialize(mock_provider)
    
    # Create update callback for testing
    async def update_callback(status, metadata):
        logger.info(f"Update: {status} - {json.dumps(metadata, default=str)}")
    
    # Test each workflow
    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        workflow = test_case["workflow"]
        
        logger.info(f"Testing workflow: {workflow}")
        logger.info(f"Query: '{query}'")
        
        # Process with workflow
        response = await workflow_service.process_with_workflow(
            query=query,
            user_id="test_user",
            conversation_id="test_conversation",
            workflow_type=workflow,
            update_callback=update_callback
        )
        
        # Log response
        logger.info(f"Response: {response[:100]}...")
        logger.info("-" * 50)

async def test_combined_reasoning():
    """Test detection and processing of combined reasoning types."""
    
    logger.info("Testing combined reasoning detection...")
    
    # Test queries that should use combined reasoning
    test_cases = [
        "Explain how nuclear fusion works and find the latest research on it",
        "Verify this claim and explain the process step by step: 'Photosynthesis releases oxygen'",
        "Calculate the compound interest on $1000 at a 5% annual rate for 5 years and explain the formula",
        "Write a story about space exploration with a clear narrative structure and plot development",
        "Map the relationship between major tech companies and verify their market positions"
    ]
    
    # Set up mock provider
    mock_provider = await setup_mock_provider()
    
    # Initialize services
    await agent_service.initialize(mock_provider)
    
    # Test each case
    for i, query in enumerate(test_cases):
        # Check if reasoning should be combined
        should_combine = await agent_service.should_combine_reasoning(query)
        
        # Get detected reasoning types
        reasoning_types = await agent_service.detect_multiple_reasoning_types(query, max_types=3)
        
        # Log results
        logger.info(f"Test {i+1}: Query: '{query}'")
        logger.info(f"Should combine: {should_combine}")
        logger.info(f"Detected reasoning types: {reasoning_types}")
        logger.info("-" * 50)

async def main():
    """Run all tests."""
    
    logger.info("Starting multi-agent system tests...")
    
    # Run tests
    await test_workflow_detection()
    await test_reasoning_detection()
    await test_combined_reasoning()
    await test_workflow_processing()
    
    logger.info("All tests completed.")

if __name__ == "__main__":
    asyncio.run(main()) 