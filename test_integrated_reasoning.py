"""
Test for integrated reasoning system to verify it works correctly.
"""
import asyncio
import json
import logging
import sys
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

logger = logging.getLogger("test_integrated_reasoning")

# Import core components
from core.ai_provider import create_ai_provider
from caching import create_advanced_cache_system
from utils.hallucination_handler import create_hallucination_handler
from features.reasoning.reasoning_router import create_reasoning_router, ReasoningMethod
from features.reasoning import create_reasoning_integration

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file"""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except (FileNotFoundError, json.JSONDecodeError):
        logger.error(f"Error loading config from {config_path}")
        return {}

async def setup_components(config: Dict[str, Any]):
    """Set up necessary components for testing"""
    # Initialize caching system
    cache_config = {
        "cache_type": config.get("cache_type", "memory"),
        "cache_config": config.get("cache_config", {}),
        "context_cache_config": config.get("context_cache_config", {}),
        "semantic_cache_config": config.get("semantic_cache_config", {})
    }
    
    base_cache, context_cache, semantic_cache = create_advanced_cache_system(cache_config)
    
    # Initialize AI provider
    ai_type = config.get("ai_type", "base")
    ai_config = config.get("ai_config", {})
    ai_provider = create_ai_provider(
        provider_type=ai_type,
        **ai_config
    )
    
    # Initialize hallucination handler
    hallucination_handler = create_hallucination_handler(
        response_cache=base_cache,
        llm_provider=ai_provider,
        **config.get("hallucination_config", {})
    )
    
    # Initialize reasoning router
    reasoning_router = create_reasoning_router(
        ai_provider=ai_provider,
        response_cache=base_cache,
        hallucination_handler=hallucination_handler,
        **config.get("reasoning_config", {})
    )
    
    # Initialize integrated reasoning
    reasoning_integration = create_reasoning_integration(
        reasoning_router=reasoning_router,
        ai_provider=ai_provider,
        context_cache=context_cache
    )
    
    return {
        "base_cache": base_cache,
        "context_cache": context_cache,
        "semantic_cache": semantic_cache,
        "ai_provider": ai_provider,
        "hallucination_handler": hallucination_handler,
        "reasoning_router": reasoning_router,
        "reasoning_integration": reasoning_integration
    }

async def test_natural_language_triggers(reasoning_integration):
    """Test natural language triggers for different reasoning methods"""
    # Test queries with natural language triggers
    test_queries = [
        # Sequential thinking triggers
        "Can you explain step by step how to bake a cake?",
        "Walk me through the process of setting up a home network.",
        "What are the instructions for changing a car tire?",
        
        # RAG triggers
        "What is the capital of France?",
        "Tell me about quantum computing.",
        "Who was Marie Curie?",
        
        # CRAG triggers
        "As I mentioned earlier, can you elaborate on that topic?",
        "Building on what we discussed about climate change, what can individuals do?",
        "Regarding our previous conversation about machine learning, what's new?",
        
        # ReAct triggers
        "Find the latest news about artificial intelligence.",
        "Calculate the compound interest on $1000 at 5% for 5 years.",
        "Can you search for the best restaurants in New York?",
        
        # Graph triggers
        "How are the different components of an ecosystem related?",
        "Compare and contrast renewable and non-renewable energy sources.",
        "What's the relationship between inflation and unemployment?",
        
        # Hybrid/complex triggers
        "Analyze the complex interactions between climate change, economic policy, and social justice.",
        "What are the multifaceted implications of artificial intelligence on various industries?",
        "Examine the intricate relationship between global supply chains and geopolitical tensions."
    ]
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        user_id = "test_user_123"
        channel_id = "test_channel_456"
        
        # Process query with integrated reasoning
        result = await reasoning_integration.process_query(
            query=query,
            user_id=user_id,
            channel_id=channel_id,
            context={}
        )
        
        # Log results
        method_used = result.get("method", "unknown")
        method_emoji = result.get("method_emoji", "")
        logger.info(f"Method used: {method_used} {method_emoji}")
        
        # Check if it's a hybrid method
        if "thinking_process" in result and isinstance(result["thinking_process"], dict):
            if "sequential" in result["thinking_process"] and "graph" in result["thinking_process"]:
                logger.info("Used hybrid method: Sequential + Graph")
            elif "retrieved_documents" in result["thinking_process"] and "sequential" in result["thinking_process"]:
                logger.info("Used hybrid method: RAG + Sequential")
            elif "planning" in result["thinking_process"] and "actions" in result["thinking_process"]:
                logger.info("Used hybrid method: ReAct + Sequential")
        
        print("\n" + "=" * 80 + "\n")

async def test_conversation_context(reasoning_integration):
    """Test conversation context awareness"""
    user_id = "test_user_789"
    channel_id = "test_channel_101"
    
    # Start a conversation
    logger.info("Starting conversation context test...")
    
    # First message
    query1 = "What is machine learning?"
    result1 = await reasoning_integration.process_query(
        query=query1,
        user_id=user_id,
        channel_id=channel_id,
        context={}
    )
    logger.info(f"First query: {query1}")
    logger.info(f"Method used: {result1.get('method', 'unknown')} {result1.get('method_emoji', '')}")
    
    # Second message referring to the first
    query2 = "What are the different types of it?"
    result2 = await reasoning_integration.process_query(
        query=query2,
        user_id=user_id,
        channel_id=channel_id,
        context={}
    )
    logger.info(f"Second query: {query2}")
    logger.info(f"Method used: {result2.get('method', 'unknown')} {result2.get('method_emoji', '')}")
    
    # Third message with increased complexity
    query3 = "Can you explain how neural networks relate to these different types?"
    result3 = await reasoning_integration.process_query(
        query=query3,
        user_id=user_id,
        channel_id=channel_id,
        context={}
    )
    logger.info(f"Third query: {query3}")
    logger.info(f"Method used: {result3.get('method', 'unknown')} {result3.get('method_emoji', '')}")
    
    print("\n" + "=" * 80 + "\n")

async def main():
    """Main test function"""
    logger.info("Starting integrated reasoning system test")
    
    # Load config
    config = load_config()
    
    # Set up components
    components = await setup_components(config)
    reasoning_integration = components["reasoning_integration"]
    
    # Run tests
    await test_natural_language_triggers(reasoning_integration)
    await test_conversation_context(reasoning_integration)
    
    logger.info("Tests completed")

if __name__ == "__main__":
    asyncio.run(main()) 