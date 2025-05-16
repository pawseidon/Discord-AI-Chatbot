"""
Test sequential thinking formatting

This script tests if the sequential thinking service properly formats all thoughts.
"""

import asyncio
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the sequential thinking service
from bot_utilities.services.sequential_thinking_service import sequential_thinking_service
from bot_utilities.ai_utils import get_ai_provider

async def test_sequential_thinking_formatting():
    """Test if sequential thinking formatting works correctly for all thoughts"""
    print("Testing sequential thinking formatting...")
    
    # Initialize the service
    llm_provider = await get_ai_provider()
    await sequential_thinking_service.initialize(llm_provider)
    
    # Sample raw response with multiple thoughts
    raw_response = """
    Thought 1: This is the first thought.
    It has multiple paragraphs.
    
    It even has some line breaks.
    
    Thought 2: This is the second thought.
    It continues with more analysis.
    
    Thought 3: This is the third thought.
    With more detailed explanation.
    
    Thought 4: This is the fourth thought.
    The analysis continues here.
    
    Thought 5: This is the fifth thought.
    With the final part of analysis.
    
    Final Answer: This is the conclusion based on all the thoughts above.
    """
    
    # Process the response
    processed = sequential_thinking_service._process_thinking_response(raw_response, "sequential")
    
    # Check if all thoughts are properly formatted
    print("\nProcessed Response:")
    print(processed)
    
    # Count the number of thought headers in the processed response
    thought_count = processed.count("## 🔄 Thought")
    expected_count = 5
    
    print(f"\nFound {thought_count} thought headers in processed response (expected {expected_count})")
    if thought_count == expected_count:
        print("✅ All thoughts were properly formatted!")
    else:
        print("❌ Some thoughts are missing in the formatted output!")
    
    # Check for conclusion formatting
    if "## ✅ Conclusion" in processed:
        print("✅ Conclusion was properly formatted!")
    else:
        print("❌ Conclusion is missing or improperly formatted!")

    # Test with a mix of "Thought" and "Step" tokens
    mixed_response = """
    Thought 1: This is the first thought.
    
    Step 2: This is actually the second thought but labeled as step.
    
    Thought 3: This is the third thought.
    
    Step 4: This is the fourth thought labeled as step.
    
    Thought 5: This is the fifth thought.
    
    Final Answer: This is the conclusion.
    """
    
    # Process the mixed response
    mixed_processed = sequential_thinking_service._process_thinking_response(mixed_response, "sequential")
    
    # Check if all thoughts/steps are properly formatted
    print("\nProcessed Mixed Response:")
    print(mixed_processed)
    
    # Count the number of thought headers in the processed response
    mixed_thought_count = mixed_processed.count("## 🔄 Thought")
    
    print(f"\nFound {mixed_thought_count} thought headers in mixed response (expected {expected_count})")
    if mixed_thought_count == expected_count:
        print("✅ All thoughts/steps were properly formatted!")
    else:
        print("❌ Some thoughts/steps are missing in the formatted output!")

async def main():
    await test_sequential_thinking_formatting()

if __name__ == "__main__":
    asyncio.run(main()) 