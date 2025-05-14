"""
Reasoning utilities for Discord AI Chatbot.

This module provides common utilities for reasoning methods
including method selection, complexity detection, and reasoning integration.
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Set up logging
logger = logging.getLogger("reasoning_utils")

# Keywords to help identify complex topics
COMPLEX_TOPIC_KEYWORDS = [
    "complex", "complicated", "intricate", "nuanced", "multifaceted",
    "analyze", "investigate", "explain", "compare", "contrast",
    "perspective", "viewpoint", "reasoning", "rationale", "logic",
    "steps", "process", "procedure", "method", "approach",
    "opinion", "view", "stance", "position", "argument", "debate",
    "ethical", "moral", "philosophical", "cultural", "technical",
    "pros and cons", "advantages", "disadvantages", "benefits", "risks",
    "implications", "consequences", "impact", "effect", "influence",
    "strategy", "plan", "solution", "alternative", "option",
    "history", "development", "evolution", "origin", "future",
    "politics", "economics", "science", "technology", "psychology",
    "society", "government", "system", "structure", "organization",
    "environment", "climate", "ecology", "sustainability", "conservation",
    "relationship", "correlation", "connection", "link", "association"
]

# Keywords that suggest factual queries
FACTUAL_QUERY_KEYWORDS = [
    "what is", "who is", "where is", "when was", "when did", "how many", 
    "what are", "who are", "which", "define", "explain",
    "describe", "tell me about", "information on", "details about",
    "fact", "factual", "objective", "actual", "accurate", "precise",
    "correct", "true", "real", "genuine", "authentic", "verifiable",
    "data", "statistics", "numbers", "figures", "percentage", "rate",
    "history", "historical", "event", "discovery", "invention",
    "definition", "concept", "term", "meaning", "significance",
    "population", "area", "distance", "size", "length", "height",
    "example", "instance", "case", "illustration", "demonstration"
]

# Keywords that suggest a need for action
ACTION_KEYWORDS = [
    "action", "step", "do", "perform", "execute", "conduct", "carry out",
    "implement", "apply", "operate", "practice", "employ", "utilize",
    "handle", "manage", "administer", "direct", "control", "run",
    "organize", "coordinate", "arrange", "prepare", "plan", "setup",
    "configure", "install", "build", "develop", "create", "generate",
    "produce", "make", "compose", "construct", "assemble", "fabricate",
    "search", "find", "locate", "retrieve", "obtain", "acquire", "get",
    "calculate", "compute", "solve", "figure out", "determine", "work out",
    "research", "investigate", "explore", "examine", "inspect", "study"
]

def detect_query_complexity(query: str) -> float:
    """
    Detect the complexity of a query based on various factors
    
    Args:
        query: The user's query string
        
    Returns:
        Complexity score between 0.0 and 1.0
    """
    # Initialize complexity score
    complexity = 0.0
    
    # Check length
    length = len(query)
    if length > 200:
        complexity += 0.3
    elif length > 100:
        complexity += 0.2
    elif length > 50:
        complexity += 0.1
    
    # Check for complex question words
    if re.search(r'\b(why|how|explain|analyze|compare|contrast|evaluate|synthesize|critique)\b', 
                query.lower()):
        complexity += 0.2
    
    # Check for multiple question marks or complex punctuation
    if len(re.findall(r'\?', query)) > 1 or len(re.findall(r'[;:]', query)) > 1:
        complexity += 0.1
    
    # Check for complex topic keywords
    keyword_count = 0
    for keyword in COMPLEX_TOPIC_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', query.lower()):
            keyword_count += 1
    
    # Add complexity based on keyword density
    keyword_density = min(1.0, keyword_count / 5)  # Cap at 1.0
    complexity += 0.2 * keyword_density
    
    # Check for multiple questions in one query
    sentences = re.split(r'[.!?]+', query)
    question_count = sum(1 for s in sentences if '?' in s)
    if question_count > 1:
        complexity += 0.1 * min(3, question_count - 1)  # Add 0.1 per additional question, max 0.3
    
    # Normalize to ensure we're within 0.0 to 1.0 range
    complexity = min(1.0, complexity)
    
    return complexity

def detect_factual_nature(query: str) -> float:
    """
    Detect if a query is primarily factual
    
    Args:
        query: The user's query string
        
    Returns:
        Factual score between 0.0 and 1.0
    """
    query_lower = query.lower()
    
    # Check for factual question patterns
    factual_score = 0.0
    
    # Explicit factual question starters
    for starter in ["what is", "who is", "where is", "when was", "how many"]:
        if query_lower.startswith(starter):
            factual_score += 0.4
            break
    
    # Count factual keywords
    keyword_count = 0
    for keyword in FACTUAL_QUERY_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
            keyword_count += 1
    
    # Add score based on keyword density
    keyword_density = min(1.0, keyword_count / 3)  # Cap at 1.0
    factual_score += 0.3 * keyword_density
    
    # Check for opinion indicators (negative signal)
    opinion_indicators = ["opinion", "think", "feel", "believe", "view", "stance"]
    for indicator in opinion_indicators:
        if re.search(r'\b' + re.escape(indicator) + r'\b', query_lower):
            factual_score -= 0.1
    
    # Normalize
    factual_score = max(0.0, min(1.0, factual_score))
    
    return factual_score

def detect_action_needs(query: str) -> float:
    """
    Detect if a query requires action taking
    
    Args:
        query: The user's query string
        
    Returns:
        Action score between 0.0 and 1.0
    """
    query_lower = query.lower()
    
    # Check for action-oriented language
    action_score = 0.0
    
    # Explicit action requests
    action_requests = ["can you", "please", "help me", "i need", "show me", "find", "search"]
    for request in action_requests:
        if request in query_lower:
            action_score += 0.2
            break
    
    # Count action keywords
    keyword_count = 0
    for keyword in ACTION_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
            keyword_count += 1
    
    # Add score based on keyword density
    keyword_density = min(1.0, keyword_count / 3)  # Cap at 1.0
    action_score += 0.3 * keyword_density
    
    # Check for "how to" patterns
    if re.search(r'\bhow to\b', query_lower):
        action_score += 0.3
    
    # Normalize
    action_score = max(0.0, min(1.0, action_score))
    
    return action_score

def select_reasoning_method(query: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Select the most appropriate reasoning method based on query analysis
    
    Args:
        query: The user's query
        context: Optional context information
        
    Returns:
        Name of the reasoning method to use
    """
    # Analyze query
    complexity = detect_query_complexity(query)
    factual_nature = detect_factual_nature(query)
    action_needs = detect_action_needs(query)
    
    # Log the scores for debugging
    logger.debug(f"Query analysis - Complexity: {complexity:.2f}, Factual: {factual_nature:.2f}, Action: {action_needs:.2f}")
    
    # Make decision based on analysis
    if action_needs > 0.6:
        # High action needs suggest ReAct
        return "react"
    elif factual_nature > 0.7:
        # Highly factual queries benefit from RAG
        if complexity > 0.6:
            # Complex factual query - use reflective RAG
            return "reflective_rag"
        else:
            # Simple factual query - use speculative RAG
            return "speculative_rag"
    elif complexity > 0.7:
        # Very complex query - use graph of thought
        return "graph_of_thought"
    elif complexity > 0.4:
        # Moderately complex - use sequential thinking
        return "sequential"
    else:
        # Simple query - use basic approach
        return "default"

def format_reasoning_process(method: str, steps: List[Dict[str, Any]]) -> str:
    """
    Format reasoning process steps for display
    
    Args:
        method: Name of the reasoning method
        steps: List of reasoning steps with content
        
    Returns:
        Formatted reasoning process
    """
    if not steps:
        return ""
    
    method_emoji = {
        "sequential": "🧠",
        "graph_of_thought": "🌐",
        "react": "🔍",
        "reflective_rag": "📚",
        "speculative_rag": "💡",
        "default": "✨"
    }
    
    emoji = method_emoji.get(method, "✨")
    
    # Format header based on method
    header = f"{emoji} **Reasoning Process ({method.replace('_', ' ').title()})** {emoji}\n\n"
    
    # Format each step
    formatted_steps = []
    for i, step in enumerate(steps, 1):
        step_content = step.get("content", "")
        step_type = step.get("type", "thought")
        
        # Format based on step type
        if step_type == "thought":
            formatted_step = f"**Step {i} (Thought)**: {step_content}"
        elif step_type == "action":
            formatted_step = f"**Step {i} (Action)**: {step_content}"
        elif step_type == "observation":
            formatted_step = f"**Step {i} (Observation)**: {step_content}"
        elif step_type == "reflection":
            formatted_step = f"**Step {i} (Reflection)**: {step_content}"
        else:
            formatted_step = f"**Step {i}**: {step_content}"
            
        formatted_steps.append(formatted_step)
    
    # Join steps with separators
    process = "\n\n".join(formatted_steps)
    
    # Add footer
    footer = f"\n\n{emoji} **End of Reasoning Process** {emoji}"
    
    return header + process + footer

def measure_reasoning_time(func: Callable) -> Callable:
    """
    Decorator to measure reasoning execution time
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        # If result is a tuple and has a dict as first element, add time_taken
        if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], dict):
            result[0]["time_taken"] = end_time - start_time
            
        # If result is a dict, add time_taken
        elif isinstance(result, dict):
            result["time_taken"] = end_time - start_time
            
        return result
    
    return wrapper 