"""
Testing Utilities

This module provides tools for testing different reasoning types and comparing their performance.
"""

import json
import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('testing_utils')

# Directory for storing test results
TEST_RESULTS_DIR = os.path.join("bot_data", "testing")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

class ReasoningTester:
    """Utility for testing and comparing different reasoning types"""
    
    def __init__(self, agent_service):
        """
        Initialize the reasoning tester
        
        Args:
            agent_service: The agent service to use for testing
        """
        self.agent_service = agent_service
        self.test_queries = {
            "factual": [
                "What is the capital of France?",
                "When was the first moon landing?",
                "Who wrote the novel 1984?",
                "What's the distance from Earth to the Moon?",
                "Explain how photosynthesis works."
            ],
            "analytical": [
                "Compare and contrast democracy and autocracy.",
                "Analyze the impact of social media on society.",
                "What are the pros and cons of renewable energy?",
                "Evaluate the effectiveness of current climate change policies.",
                "Break down the causes of the 2008 financial crisis."
            ],
            "creative": [
                "Write a short poem about autumn.",
                "Create a brief story about a time traveler.",
                "Imagine the future of transportation in 100 years.",
                "Design a superhero with unique powers.",
                "Describe an alien civilization unlike any in popular media."
            ],
            "mathematical": [
                "Calculate 15% of 237.",
                "Solve the equation 3x + 7 = 22.",
                "If a rectangle has a width of 5 meters and a length of 8 meters, what is its area?",
                "What is the square root of 144?",
                "If I invest $1000 at 5% annual interest compounded yearly, how much will I have after 3 years?"
            ],
            "planning": [
                "How should I prepare for a job interview?",
                "Outline steps for learning a new language.",
                "What's a good workout routine for beginners?",
                "How would you plan a small dinner party?",
                "What steps should I take to start a small business?"
            ],
            "current_events": [
                "What's the current situation in Ukraine?",
                "Tell me about recent developments in AI technology.",
                "What are the latest climate change policies?",
                "What's the current state of the global economy?",
                "What are the most recent space exploration achievements?"
            ]
        }
        self.reasoning_types = [
            "sequential", "rag", "conversational", "knowledge", "verification", 
            "creative", "calculation", "planning", "graph", "multi_agent",
            "step_back", "cot", "react"
        ]
    
    async def test_reasoning_type(self, reasoning_type: str, query: str, user_id: str = "test_user", conversation_id: str = "test_conversation") -> Tuple[str, float, Dict[str, Any]]:
        """
        Test a specific reasoning type on a query
        
        Args:
            reasoning_type: The reasoning type to test
            query: The query to test with
            user_id: The user ID to use for the test
            conversation_id: The conversation ID to use for the test
            
        Returns:
            The response, time taken, and any additional metadata
        """
        start_time = datetime.now()
        metadata = {}
        
        try:
            # Process with the specified reasoning type
            response = await self.agent_service.process_query(
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                reasoning_type=reasoning_type
            )
            
            # Calculate time taken
            end_time = datetime.now()
            time_taken = (end_time - start_time).total_seconds()
            
            return response, time_taken, metadata
        except Exception as e:
            logger.error(f"Error testing reasoning type {reasoning_type} on query '{query}': {str(e)}")
            return f"ERROR: {str(e)}", -1, {"error": str(e)}
    
    async def compare_reasoning_types(self, query: str, reasoning_types: List[str] = None, user_id: str = "test_user") -> Dict[str, Any]:
        """
        Compare multiple reasoning types on the same query
        
        Args:
            query: The query to test
            reasoning_types: List of reasoning types to compare (default: all)
            user_id: The user ID to use for the test
            
        Returns:
            A dictionary with results for each reasoning type
        """
        if reasoning_types is None:
            reasoning_types = self.reasoning_types
            
        results = {}
        conversation_id = f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        for reasoning_type in reasoning_types:
            # Test this reasoning type
            response, time_taken, metadata = await self.test_reasoning_type(
                reasoning_type=reasoning_type,
                query=query,
                user_id=user_id,
                conversation_id=conversation_id
            )
            
            # Store results
            results[reasoning_type] = {
                "response": response,
                "time_taken": time_taken,
                "metadata": metadata
            }
            
            # Reset conversation state between tests
            await self.agent_service.reset_conversation(conversation_id)
            
            # Sleep briefly to prevent rate limiting
            await asyncio.sleep(1)
        
        return results
    
    async def run_test_battery(self, category: str = None, num_queries: int = 3, reasoning_types: List[str] = None) -> Dict[str, Any]:
        """
        Run a battery of tests on different query types
        
        Args:
            category: The category of queries to test (default: all categories)
            num_queries: The number of queries to test per category
            reasoning_types: List of reasoning types to test (default: all)
            
        Returns:
            A dictionary with test results
        """
        battery_results = {}
        categories = [category] if category else list(self.test_queries.keys())
        
        for cat in categories:
            if cat not in self.test_queries:
                logger.warning(f"Unknown category: {cat}")
                continue
                
            # Select queries for this category
            queries = self.test_queries[cat][:num_queries]
            cat_results = {}
            
            for query in queries:
                # Compare all reasoning types on this query
                query_results = await self.compare_reasoning_types(
                    query=query,
                    reasoning_types=reasoning_types
                )
                
                cat_results[query] = query_results
            
            battery_results[cat] = cat_results
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"reasoning_test_{timestamp}.json"
        file_path = os.path.join(TEST_RESULTS_DIR, filename)
        
        with open(file_path, 'w') as f:
            json.dump(battery_results, f, indent=2)
        
        logger.info(f"Test battery results saved to {file_path}")
        return battery_results
    
    @staticmethod
    async def analyze_results(results_file: str = None) -> Dict[str, Any]:
        """
        Analyze test results to identify patterns and redundancies
        
        Args:
            results_file: Path to a results file (default: latest file)
            
        Returns:
            Analysis of the test results
        """
        # Find the latest results file if none specified
        if results_file is None:
            files = os.listdir(TEST_RESULTS_DIR)
            test_files = [f for f in files if f.startswith("reasoning_test_")]
            
            if not test_files:
                return {"error": "No test results found"}
                
            test_files.sort(reverse=True)
            results_file = os.path.join(TEST_RESULTS_DIR, test_files[0])
        
        # Load the results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Process results
        analysis = {
            "best_performers": {},
            "time_comparison": {},
            "similar_reasoning_types": [],
            "recommendations": []
        }
        
        # Identify best performers for each category
        for category, queries in results.items():
            category_best = {}
            
            for query, query_results in queries.items():
                # Determine best reasoning type for this query based on heuristics
                best_type = None
                min_errors = float('inf')
                
                for reasoning_type, data in query_results.items():
                    response = data["response"]
                    time_taken = data["time_taken"]
                    
                    # Count error indicators in response
                    error_count = response.lower().count("error") + response.lower().count("i don't know") + response.lower().count("i cannot")
                    
                    if error_count < min_errors and time_taken > 0:  # Exclude failed tests
                        min_errors = error_count
                        best_type = reasoning_type
                
                if best_type:
                    if best_type not in category_best:
                        category_best[best_type] = 0
                    category_best[best_type] += 1
            
            # Find the most successful reasoning type for this category
            if category_best:
                best_type = max(category_best.items(), key=lambda x: x[1])[0]
                analysis["best_performers"][category] = best_type
        
        # Identify redundant reasoning types (those with very similar responses)
        similar_pairs = []
        all_reasoning_types = set()
        
        for category, queries in results.items():
            for query, query_results in queries.items():
                for type1 in query_results:
                    all_reasoning_types.add(type1)
                    resp1 = query_results[type1]["response"]
                    
                    for type2 in query_results:
                        if type1 >= type2:  # Skip duplicate pairs
                            continue
                            
                        resp2 = query_results[type2]["response"]
                        
                        # Calculate similarity (very basic version)
                        similarity = len(set(resp1.split()) & set(resp2.split())) / max(len(set(resp1.split())), len(set(resp2.split())))
                        
                        if similarity > 0.8:  # Arbitrary threshold
                            similar_pairs.append((type1, type2, similarity))
        
        # Count frequencies of similar pairs
        pair_counts = {}
        for type1, type2, similarity in similar_pairs:
            pair = tuple(sorted([type1, type2]))
            if pair not in pair_counts:
                pair_counts[pair] = 0
            pair_counts[pair] += 1
        
        # Find consistently similar reasoning types
        for pair, count in pair_counts.items():
            if count > len(results) * 2:  # If similar in multiple categories
                analysis["similar_reasoning_types"].append({
                    "types": pair,
                    "similarity_count": count
                })
        
        # Generate recommendations
        if analysis["similar_reasoning_types"]:
            for similar in analysis["similar_reasoning_types"]:
                type1, type2 = similar["types"]
                analysis["recommendations"].append(
                    f"Consider using only one of these similar reasoning types: {type1} and {type2}"
                )
        
        # Recommend best reasoning types for each category
        for category, best_type in analysis["best_performers"].items():
            analysis["recommendations"].append(
                f"Use {best_type} reasoning for {category} queries"
            )
        
        return analysis


# Example usage in a command:
"""
@bot.command()
async def test_reasoning(ctx, category=None, num_queries=3):
    \"\"\"Test the performance of different reasoning types\"\"\"
    # Initialize tester
    from bot_utilities.services.agent_service import agent_service
    tester = ReasoningTester(agent_service)
    
    await ctx.send(f"Running reasoning tests on {num_queries} queries per category...")
    
    # Run tests
    results = await tester.run_test_battery(category, int(num_queries))
    
    # Analyze results
    analysis = await ReasoningTester.analyze_results()
    
    # Send key findings
    message = "**Test Results Analysis**\n\n"
    
    message += "**Best Performing Reasoning Types:**\n"
    for category, reasoning_type in analysis["best_performers"].items():
        message += f"- {category.title()}: {reasoning_type}\n"
    
    message += "\n**Similar Reasoning Types:**\n"
    for similar in analysis["similar_reasoning_types"]:
        message += f"- {' and '.join(similar['types'])}\n"
    
    message += "\n**Recommendations:**\n"
    for recommendation in analysis["recommendations"]:
        message += f"- {recommendation}\n"
    
    await ctx.send(message)
""" 