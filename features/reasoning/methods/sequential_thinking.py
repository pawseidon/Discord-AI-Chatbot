import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger("sequential_thinking")

class SequentialThinking:
    """
    Sequential thinking approach with step-by-step reasoning for complex problems
    """
    
    def __init__(self, ai_provider=None, response_cache=None):
        """
        Initialize sequential thinking
        
        Args:
            ai_provider: AI provider for reasoning
            response_cache: Optional cache for responses
        """
        self.ai_provider = ai_provider
        self.response_cache = response_cache
        
    async def process(self, 
                    query: str, 
                    user_id: str,
                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query using sequential thinking
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            Dict containing the answer and thinking steps
        """
        context = context or {}
        conversation_history = context.get("conversation_history", [])
        
        # Prepare variables for the thinking process
        thoughts = []
        thinking_completed = False
        max_steps = 10
        current_step = 1
        
        # Start sequential thinking process
        while not thinking_completed and current_step <= max_steps:
            # Build prompt for current thought
            prompt = self._build_thinking_prompt(query, thoughts, context)
            
            # Generate next thought
            thought_result = await self.ai_provider.generate_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract thought components
            thought_data = self._parse_thought(thought_result, current_step)
            thoughts.append(thought_data)
            
            # Check if thinking is complete
            thinking_completed = not thought_data.get('next_thought_needed', True)
            current_step += 1
        
        # Generate final answer based on thinking process
        answer = await self._generate_final_answer(query, thoughts, context)
        
        # Prepare result
        result = {
            "answer": answer,
            "thinking_steps": thoughts,
            "method": "sequential",
            "method_emoji": "🧠"
        }
        
        return result
    
    def _build_thinking_prompt(self, 
                             query: str, 
                             current_thoughts: List[Dict[str, Any]],
                             context: Dict[str, Any] = None) -> str:
        """
        Build prompt for the next thought in the sequential thinking process
        
        Args:
            query: User query
            current_thoughts: List of thoughts generated so far
            context: Additional context
            
        Returns:
            Prompt for the next thinking step
        """
        # Extract relevant context
        conversation_history = context.get("conversation_history", [])
        retrieved_information = context.get("retrieved_information", [])
        
        # Build base prompt
        prompt = (
            "You are a careful thinker who solves problems step by step. "
            "Let's break down this question into sequential steps.\n\n"
        )
        
        # Add conversation context if available
        if conversation_history:
            prompt += "Previous conversation:\n"
            for i, message in enumerate(conversation_history[-3:]):  # Last 3 messages
                if "user" in message:
                    prompt += f"User: {message['user']}\n"
                if "bot" in message:
                    prompt += f"Assistant: {message['bot']}\n"
            prompt += "\n"
        
        # Add retrieved information if available
        if retrieved_information:
            prompt += "Relevant information:\n"
            for i, info in enumerate(retrieved_information):
                prompt += f"- {info}\n"
            prompt += "\n"
        
        # Add query
        prompt += f"Question: {query}\n\n"
        
        # Add previous thoughts
        if current_thoughts:
            prompt += "Thinking process so far:\n"
            for i, thought in enumerate(current_thoughts):
                prompt += f"Thought {i+1}: {thought.get('thought', '')}\n"
            prompt += "\n"
        
        # Add instructions for next thought
        if not current_thoughts:
            prompt += (
                "Think step by step. First understand what the question is asking, "
                "then work through it methodically.\n"
                "Format your response as:\n"
                "{\n"
                "  \"thought\": \"Your step-by-step reasoning for this step\",\n"
                "  \"next_thought_needed\": true/false,\n"
                "  \"thought_number\": 1\n"
                "}\n"
            )
        else:
            prompt += (
                f"Continue the thinking process with step {len(current_thoughts) + 1}.\n"
                "Format your response as:\n"
                "{\n"
                "  \"thought\": \"Your step-by-step reasoning for this step\",\n"
                "  \"next_thought_needed\": true/false,\n"
                f"  \"thought_number\": {len(current_thoughts) + 1}\n"
                "}\n"
            )
        
        return prompt
    
    def _parse_thought(self, thought_text: str, current_step: int) -> Dict[str, Any]:
        """
        Parse the raw thought text into structured format
        
        Args:
            thought_text: Raw thought from AI provider
            current_step: Current step number
            
        Returns:
            Structured thought data
        """
        # Default thought structure
        thought_data = {
            "thought": thought_text,
            "next_thought_needed": True,
            "thought_number": current_step
        }
        
        # Try to parse JSON format if available
        try:
            # Look for JSON-like structure
            json_match = re.search(r'{.*}', thought_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_data = eval(json_str)  # Simple eval for dict-like strings
                
                # Update thought data with parsed values
                for key, value in parsed_data.items():
                    thought_data[key] = value
        except Exception as e:
            logger.error(f"Error parsing thought: {e}")
            # Fall back to using the whole text as the thought
        
        return thought_data
    
    async def _generate_final_answer(self, 
                                  query: str, 
                                  thoughts: List[Dict[str, Any]],
                                  context: Dict[str, Any] = None) -> str:
        """
        Generate a final answer based on the thinking process
        
        Args:
            query: User query
            thoughts: List of thoughts from the thinking process
            context: Additional context
            
        Returns:
            Final answer to the query
        """
        # Build prompt for final answer
        prompt = (
            "Based on your step-by-step thinking process, provide a clear, comprehensive "
            "answer to the original question. Your answer should be well-structured, "
            "accurate, and directly address what was asked.\n\n"
        )
        
        # Add query
        prompt += f"Question: {query}\n\n"
        
        # Add thoughts
        prompt += "Your thinking process:\n"
        for i, thought in enumerate(thoughts):
            prompt += f"Thought {i+1}: {thought.get('thought', '')}\n"
        prompt += "\n"
        
        # Add instructions for final answer
        prompt += (
            "Now, synthesize your thoughts into a clear, concise, and comprehensive answer. "
            "Your answer should be well-structured, easy to understand, and directly address "
            "the question. Include relevant details from your thinking process."
        )
        
        # Generate final answer
        answer = await self.ai_provider.generate_response(
            prompt=prompt,
            temperature=0.5,
            max_tokens=1500
        )
        
        return answer

async def process_sequential_thinking(
    query: str,
    user_id: str,
    ai_provider=None,
    response_cache=None,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a query using sequential thinking
    
    Args:
        query: User query
        user_id: User ID
        ai_provider: AI provider for reasoning
        response_cache: Optional cache for responses
        context: Additional context
        
    Returns:
        Dict containing the answer and thinking steps
    """
    sequential_thinking = SequentialThinking(
        ai_provider=ai_provider,
        response_cache=response_cache
    )
    
    return await sequential_thinking.process(query, user_id, context) 