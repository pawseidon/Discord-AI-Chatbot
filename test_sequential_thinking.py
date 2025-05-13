import asyncio
import logging
from bot_utilities.sequential_thinking import create_sequential_thinking
import json
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)

# Simple mock LLM provider for testing
class MockLLM:
    def __init__(self, response=None):
        self.response = response or """
Thought 1: First I need to understand what the problem is asking. The user wants to calculate the sum of 5+7+3.

Thought 2: To solve this, I'll add the numbers sequentially. 5 + 7 = 12.

Thought 3: Now I need to add the final number to my current sum. 12 + 3 = 15.

Thought 4: I should double-check my work. 5 + 7 = 12, and 12 + 3 = 15. This is correct.

Answer: The sum of 5+7+3 is 15.
"""

    async def async_call(self, prompt, temperature=0.0, max_tokens=1000):
        print(f"Mock LLM received prompt: {prompt[:100]}...")
        return self.response

# Complex mock LLM for testing sequential thinking on complex problems
class ComplexMockLLM:
    async def async_call(self, prompt, temperature=0.0, max_tokens=1000):
        print(f"Complex Mock LLM received prompt: {prompt[:100]}...")
        
        # Return a more complex response that demonstrates sequential thinking
        return """
Thought 1: I need to understand what problem is being asked. The user wants to create a networking plan for connecting 3 buildings with efficient, redundant connections.

Thought 2: Let me identify the key constraints:
- Need to connect 3 buildings (A, B, C)
- Need redundancy (no single point of failure)
- Need to be cost-efficient
- Need to consider future expansion

Thought 3: Let's consider the topology options:
- Full mesh (each building connects to every other building): Most redundant but requires the most cables
- Star topology (central hub): Simple but creates a single point of failure
- Ring topology: Good redundancy with fewer connections than full mesh

Thought 4: Based on the constraints, I think a hybrid approach makes the most sense. Let me design the solution:
1. Primary connections: A↔B, B↔C, C↔A (forming a ring)
2. Each building should have a redundant network equipment setup
3. Use fiber optic for future-proofing and distance support
4. Implement OSPF or similar routing protocol to handle automatic failover

Thought 5: Actually, I should revise Thought 4. For just 3 buildings, a full mesh topology isn't much more expensive than a ring and provides better redundancy:
1. Connections: A↔B, B↔C, C↔A (forming a full mesh)
2. Redundant equipment at each site
3. Multiple fiber pairs between buildings
4. Layer 3 routing with BGP or OSPF

Thought 6: Let me consider scalability for future expansion:
- If we add a building D later, full mesh becomes significantly more complex
- Ring topology with strategic additional links might be better for growth
- Should implement network management system from the start

Answer: The most effective networking plan for connecting the three buildings would be:

1. Implement a full mesh fiber optic network topology (each building connects directly to every other building)
2. Use redundant network equipment at each site
3. Deploy dual fiber connections between buildings
4. Implement dynamic routing protocols (OSPF)
5. Include network monitoring and management systems
6. Document future expansion plan that may evolve from full mesh to partial mesh as more buildings are added

This provides maximum redundancy for the current 3-building setup while allowing for controlled growth.
"""

class MockAIProvider:
    """Mock AI provider for testing"""
    def __init__(self, response_map=None):
        self.response_map = response_map or {}
        
    async def async_call(self, prompt, temperature=0.7, max_tokens=2000):
        # Simplified matching - just look for keywords in the prompt
        for keyword, response in self.response_map.items():
            if keyword in prompt:
                return response
        # Default mock response
        return "This is a test response from the mock AI provider."

async def test_sequential_thinking():
    """Test sequential thinking with a mock LLM provider"""
    
    # Create a mock AI provider
    mock_responses = {
        "sequential": """
Thought 1: This is the first thought in the sequence.
I'll work through this problem step by step.

Thought 2: This is the second thought.
Building on the first idea, I need to analyze further.

Thought 3: This is the third thought.
I'm making progress in understanding the problem.

Thought 4: This is the fourth thought.
Almost there with the solution.

Thought 5: This is the fifth thought.
Now I understand the complete picture.

Conclusion: This is my final answer based on the sequential thinking process.
""",
        "chain-of-verification": """
Step 1: First, I need to understand the problem.
This is a test of my chain-of-verification capabilities.

Verification 1: Let me verify my understanding. 
The problem asks me to demonstrate CoV reasoning, which includes verification steps.

Step 2: Now I'll analyze the key components.
Chain-of-Verification involves careful checking of facts.

Verification 2: Checking my analysis.
I've correctly identified the key aspects of CoV reasoning.

Final Answer: Chain-of-Verification is a powerful technique that reduces hallucinations by adding explicit verification steps.
""",
        "Graph-of-Thought": """
Branch A: Exploring the first approach to the problem.
This branch considers solution from perspective X.

Branch B: Considering an alternative approach.
This branch explores the problem from perspective Y.

Connection: Branches A and B relate through concept Z.
This connection shows how the different approaches interact.

Key Insight: The most important discovery from this exploration.
This insight emerges from considering multiple perspectives.

Convergence: How branches A and B come together.
The convergence leads to a more robust solution.

Conclusion: By exploring multiple connected thought paths, I've arrived at a comprehensive solution that considers various factors and their relationships.
"""
    }
    
    mock_provider = MockAIProvider(response_map=mock_responses)
    
    # Create sequential thinking with the mock provider
    seq_thinking = create_sequential_thinking(llm_provider=mock_provider)
    
    # Test standard sequential thinking
    print("\n\nTesting Sequential Thinking:")
    success, response = await seq_thinking.run(
        problem="Test problem for sequential thinking",
        prompt_style="sequential"
    )
    print(f"Success: {success}")
    print(response)
    
    # Test Chain of Verification
    print("\n\nTesting Chain-of-Verification:")
    success, response = await seq_thinking.run(
        problem="Test problem for chain-of-verification",
        prompt_style="cov"
    )
    print(f"Success: {success}")
    print(response)
    
    # Test Graph of Thought
    print("\n\nTesting Graph-of-Thought:")
    success, response = await seq_thinking.run(
        problem="Test problem that requires exploring multiple perspectives and their relationships",
        prompt_style="got"
    )
    print(f"Success: {success}")
    print(response)

if __name__ == "__main__":
    asyncio.run(test_sequential_thinking()) 