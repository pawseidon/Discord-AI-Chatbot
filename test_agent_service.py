import asyncio
from bot_utilities.services.agent_service import agent_service
from bot_utilities.services.symbolic_reasoning_service import symbolic_reasoning_service

async def test_agent_service():
    """Test agent service enhancements"""
    print("Initializing agent service...")
    await agent_service.initialize()
    
    # Test detection of reasoning types
    test_query = "Can you search for information about quantum computing and verify its accuracy?"
    types = await agent_service.detect_multiple_reasoning_types(test_query)
    print(f"Detected reasoning types: {types}")
    
    # Test should_combine_reasoning
    combined = await agent_service.should_combine_reasoning(test_query)
    print(f"Should combine reasoning: {combined}")
    
    # Test pattern detection
    pattern_query = "First find research about AI alignment, then verify if the claims are true"
    pattern_types = await agent_service.detect_multiple_reasoning_types(pattern_query)
    print(f"Pattern detected reasoning types: {pattern_types}")
    
    # Test symbolic reasoning service
    print("\nTesting symbolic reasoning service...")
    await symbolic_reasoning_service.ensure_initialized()
    
    # Test math expression
    math_expr = "2 + 2 * 3"
    math_result = await symbolic_reasoning_service.evaluate_expression(math_expr)
    print(f"Math result for '{math_expr}': {math_result.get('result', 'Error')}")
    
    # Test equation solving
    equation = "x^2 - 4 = 0"
    eq_result = await symbolic_reasoning_service.evaluate_expression(equation)
    print(f"Equation result for '{equation}': {eq_result.get('result', 'Error')}")
    
    # Test logical statement
    logical_stmt = "A and B implies C"
    logic_result = await symbolic_reasoning_service.evaluate_expression(logical_stmt)
    print(f"Logic result for '{logical_stmt}': {logic_result.get('result', 'Error')}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_agent_service()) 