#!/usr/bin/env python3
import asyncio
from bot_utilities.mcp_utils import MCPToolsManager

async def test_mcp():
    manager = MCPToolsManager()
    
    # Test the simple_read_func
    print("Testing simple_read_func...")
    result = await manager.simple_read_func()
    print(f"Result type: {type(result)}")
    print(f"Has message attr: {hasattr(result, 'message')}")
    
    if hasattr(result, 'message'):
        print(f"Message type: {type(result.message)}")
        print(f"Message has root: {hasattr(result.message, 'root')}")

    # Test the run_simple_mcp_agent with a simple query
    # This won't have full functionality without API keys but should test our fixes
    try:
        print("\nTesting simple agent execution...")
        response = await manager.run_simple_mcp_agent(
            "Use sequential thinking to analyze: What is 2+2?",
            system_message="You are a helpful agent that uses sequential thinking."
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp()) 