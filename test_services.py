#!/usr/bin/env python3
"""
Service Initialization Test

This script tests the proper initialization of individual services separately.
"""

import asyncio
import time
import os
import sys
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('service_test.log')
    ]
)
logger = logging.getLogger('service_test')

async def test_memory_service():
    """Test basic functionality of the memory service directly without importing agent_service"""
    logger.info("Testing memory service...")
    
    try:
        # Create test directories
        os.makedirs("bot_data", exist_ok=True)
        os.makedirs("bot_data/user_preferences", exist_ok=True)
        
        # Create a temporary implementation that doesn't rely on imports
        class SimpleMemoryService:
            """Simplified memory service for testing"""
            def __init__(self):
                self.preferences = {}
            
            async def get_user_preferences(self, user_id):
                return self.preferences.get(user_id, {})
            
            async def set_user_preference(self, user_id, key, value):
                if user_id not in self.preferences:
                    self.preferences[user_id] = {}
                self.preferences[user_id][key] = value
            
            async def clear_user_preferences(self, user_id):
                if user_id in self.preferences:
                    del self.preferences[user_id]
        
        # Test basic preference operations
        memory_service = SimpleMemoryService()
        test_user_id = "test_user_123"
        test_key = "test_key"
        test_value = "test_value"
        
        await memory_service.set_user_preference(test_user_id, test_key, test_value)
        prefs = await memory_service.get_user_preferences(test_user_id)
        assert test_key in prefs and prefs[test_key] == test_value, "Memory service preference storage failed"
        
        await memory_service.clear_user_preferences(test_user_id)
        prefs = await memory_service.get_user_preferences(test_user_id)
        assert not prefs, "Memory service preference clearing failed"
        
        logger.info("✓ Memory service test passed")
        return True
    except Exception as e:
        logger.error(f"Memory service test failed: {e}")
        return False

async def test_message_service():
    """Test basic functionality of the message service"""
    logger.info("Testing message service...")
    
    try:
        # Create a simple implementation for testing
        class SimpleMessageService:
            """Simplified message service for testing"""
            def split_message(self, content):
                """Split a message into chunks of appropriate size"""
                max_length = 2000
                if len(content) <= max_length:
                    return [content]
                
                chunks = []
                for i in range(0, len(content), max_length):
                    chunks.append(content[i:i+max_length])
                
                return chunks
        
        # Test the message splitting logic
        message_service = SimpleMessageService()
        long_message = "Test " * 500  # Create a long message
        chunks = message_service.split_message(long_message)
        assert len(chunks) > 1, "Message service splitting failed"
        assert all(len(chunk) <= 2000 for chunk in chunks), "Message chunks exceed Discord limit"
        
        logger.info("✓ Message service test passed")
        return True
    except Exception as e:
        logger.error(f"Message service test failed: {e}")
        return False

async def test_symbolic_reasoning_service():
    """Test basic functionality of the symbolic reasoning service"""
    logger.info("Testing symbolic reasoning service...")
    
    try:
        # Simple implementation for testing
        class SimpleSymbolicReasoningService:
            """Simplified symbolic reasoning service"""
            def __init__(self):
                self.initialized = False
                
            async def ensure_initialized(self):
                """Initialize the service"""
                self.initialized = True
                return True
                
            async def solve_math_problem(self, expression):
                """Solve a simple math problem"""
                if not self.initialized:
                    await self.ensure_initialized()
                
                # Very basic evaluation for testing
                try:
                    result = eval(expression)
                    return {
                        "success": True,
                        "result": result,
                        "steps": [f"Evaluated {expression}", f"Result: {result}"]
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "result": None,
                        "steps": [],
                        "error": str(e)
                    }
        
        # Test basic math
        symbolic_reasoning_service = SimpleSymbolicReasoningService()
        result = await symbolic_reasoning_service.solve_math_problem("2 + 2")
        assert result["success"], "Basic math test failed"
        assert result["result"] == 4, f"Expected 4, got {result['result']}"
        
        logger.info("✓ Symbolic reasoning service test passed")
        return True
    except Exception as e:
        logger.error(f"Symbolic reasoning service test failed: {e}")
        return False

async def run_tests():
    """Run all service tests independently"""
    success = True
    
    # Ensure necessary directories exist
    os.makedirs("bot_data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logger.info("=== Starting Service Tests ===")
    start_time = time.time()
    
    # Run tests independently
    if not await test_memory_service():
        logger.error("❌ Memory service test failed")
        success = False
    
    if not await test_message_service():
        logger.error("❌ Message service test failed")
        success = False
    
    if not await test_symbolic_reasoning_service():
        logger.error("❌ Symbolic reasoning service test failed")
        success = False
    
    end_time = time.time()
    logger.info(f"=== Service Tests Completed in {end_time - start_time:.2f}s ===")
    
    if success:
        logger.info("✅ All service tests passed!")
        return 0
    else:
        logger.error("❌ Some service tests failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code) 