"""
Test sequential thinking revision formatting

This script tests if the sequential thinking service properly formats revisions.
"""

import asyncio
import sys
import os
import re

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the sequential thinking service
from bot_utilities.services.sequential_thinking_service import sequential_thinking_service

async def test_revision_formatting():
    """Test if revisions in sequential thinking are properly formatted"""
    print("Testing sequential thinking revision formatting...")
    
    # Initialize the service
    from bot_utilities.ai_utils import get_ai_provider
    llm_provider = await get_ai_provider()
    await sequential_thinking_service.initialize(llm_provider)
    
    # Sample original thoughts
    original_thoughts = """# Sequential Reasoning

## 🔄 Thought 1
This is the first thought with some analysis.

_ _ _

## 🔄 Thought 2
This is the second thought with more details.

_ _ _

## 🔄 Thought 3
This is the third thought with even more analysis.

_ _ _

## ✅ Conclusion

This is the final conclusion based on all thoughts.
"""

    # Sample detailed revisions
    detailed_revisions = """
Revision for Thought 2:
- Original thought: This is the second thought with more details.
- Issue: The thought didn't consider an important factor.
- Improved thought: This is the improved second thought that addresses the missing factor.

Revision for Thought 3:
- Original thought: This is the third thought with even more analysis.
- Issue: The analysis was flawed.
- Improved thought: This is the corrected third thought with proper analysis.
"""

    # Sample unstructured revisions
    unstructured_revisions = """
After further consideration, I've realized a few important points were missed in the analysis.

For the second thought, we should consider additional perspectives.

For the third thought, the logic could be strengthened by including more examples.
"""

    # Test detailed revision formatting
    print("\nTesting detailed revision formatting...")
    detailed_result = sequential_thinking_service._combine_thoughts_with_revisions(original_thoughts, detailed_revisions)
    
    print("\nDetailed revision result:")
    print(detailed_result)
    
    # Check for proper revision formatting
    has_revision_for_thought = "### 🔁 Revision for Thought" in detailed_result
    no_revision_for_step = "Revision for Step" not in detailed_result
    
    print(f"\nRevision uses 'Thought' instead of 'Step': {'✅' if has_revision_for_thought and no_revision_for_step else '❌'}")
    
    # Test unstructured revision formatting
    print("\nTesting unstructured revision formatting...")
    unstructured_result = sequential_thinking_service._combine_thoughts_with_revisions(original_thoughts, unstructured_revisions)
    
    print("\nUnstructured revision result:")
    print(unstructured_result)
    
    # Check for proper heading
    has_thought_revisions_heading = "## 🔁 Thought Revisions" in unstructured_result
    
    print(f"\nUnstructured revisions use 'Thought Revisions' heading: {'✅' if has_thought_revisions_heading else '❌'}")

async def main():
    await test_revision_formatting()

if __name__ == "__main__":
    asyncio.run(main()) 