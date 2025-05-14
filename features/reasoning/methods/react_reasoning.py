import logging
import re
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger("react_reasoning")

class ReactReasoning:
    """
    Reasoning + Acting (ReAct) approach for action-oriented problem solving
    """
    
    def __init__(self, ai_provider=None, response_cache=None, tool_providers=None):
        """
        Initialize ReAct reasoning
        
        Args:
            ai_provider: AI provider for reasoning
            response_cache: Optional cache for responses
            tool_providers: Dict of available tools by name
        """
        self.ai_provider = ai_provider
        self.response_cache = response_cache
        self.tool_providers = tool_providers or {}
        
    async def process(self, 
                    query: str, 
                    user_id: str,
                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query using ReAct reasoning
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            Dict containing the answer and reasoning steps
        """
        context = context or {}
        conversation_history = context.get("conversation_history", [])
        
        # Initialize reasoning state
        reasoning_state = {
            "thought": "",
            "action": "",
            "action_input": "",
            "observation": "",
            "steps": [],
            "final_answer": ""
        }
        
        # Set up reasoning loop
        max_steps = 8
        current_step = 0
        reasoning_complete = False
        
        # Start ReAct reasoning loop
        while not reasoning_complete and current_step < max_steps:
            # Build prompt for current reasoning step
            prompt = self._build_reasoning_prompt(query, reasoning_state, context)
            
            # Generate next reasoning step
            reasoning_result = await self.ai_provider.generate_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse reasoning result
            parsed_result = self._parse_reasoning_step(reasoning_result)
            current_step += 1
            
            # Update reasoning state
            reasoning_state["thought"] = parsed_result.get("thought", "")
            reasoning_state["action"] = parsed_result.get("action", "")
            reasoning_state["action_input"] = parsed_result.get("action_input", "")
            
            # Check if reasoning is complete with a final answer
            if parsed_result.get("final_answer", ""):
                reasoning_state["final_answer"] = parsed_result["final_answer"]
                reasoning_complete = True
                break
            
            # Execute action if specified
            if reasoning_state["action"] and reasoning_state["action"] != "Final Answer":
                observation = await self._execute_action(
                    reasoning_state["action"], 
                    reasoning_state["action_input"],
                    user_id,
                    context
                )
                reasoning_state["observation"] = observation
            else:
                reasoning_state["observation"] = "No action specified. Please provide a valid action or Final Answer."
            
            # Add step to reasoning history
            reasoning_state["steps"].append({
                "thought": reasoning_state["thought"],
                "action": reasoning_state["action"],
                "action_input": reasoning_state["action_input"],
                "observation": reasoning_state["observation"]
            })
        
        # If no final answer provided within max steps, generate one
        if not reasoning_state["final_answer"]:
            reasoning_state["final_answer"] = await self._generate_final_answer(query, reasoning_state, context)
        
        # Prepare result
        result = {
            "answer": reasoning_state["final_answer"],
            "reasoning_steps": reasoning_state["steps"],
            "method": "react",
            "method_emoji": "⚙️"
        }
        
        return result
    
    def _build_reasoning_prompt(self, 
                              query: str, 
                              reasoning_state: Dict[str, Any],
                              context: Dict[str, Any] = None) -> str:
        """
        Build prompt for the next reasoning step
        
        Args:
            query: User query
            reasoning_state: Current reasoning state
            context: Additional context
            
        Returns:
            Prompt for the next reasoning step
        """
        # Extract relevant context
        conversation_history = context.get("conversation_history", [])
        
        # Build base prompt
        prompt = (
            "You are a helpful assistant that uses a Reasoning and Acting (ReAct) approach to solve problems. "
            "You think about what to do, take actions, observe the results, and repeat until you reach a solution.\n\n"
        )
        
        # Add tool descriptions
        prompt += "You have access to the following tools:\n"
        for tool_name, tool in self.tool_providers.items():
            tool_desc = getattr(tool, "description", f"Tool for {tool_name}")
            prompt += f"- {tool_name}: {tool_desc}\n"
        prompt += "\n"
        
        # Add conversation context if available
        if conversation_history:
            prompt += "Previous conversation:\n"
            for i, message in enumerate(conversation_history[-2:]):  # Last 2 messages
                if "user" in message:
                    prompt += f"User: {message['user']}\n"
                if "bot" in message:
                    prompt += f"Assistant: {message['bot']}\n"
            prompt += "\n"
        
        # Add query
        prompt += f"Question: {query}\n\n"
        
        # Add reasoning history
        if reasoning_state["steps"]:
            prompt += "Reasoning steps so far:\n"
            for i, step in enumerate(reasoning_state["steps"]):
                prompt += f"Step {i+1}:\n"
                prompt += f"Thought: {step['thought']}\n"
                prompt += f"Action: {step['action']}\n"
                prompt += f"Action Input: {step['action_input']}\n"
                prompt += f"Observation: {step['observation']}\n\n"
        
        # Add current reasoning state
        if reasoning_state["thought"] or reasoning_state["observation"]:
            prompt += f"Current step:\n"
            if reasoning_state["thought"]:
                prompt += f"Thought: {reasoning_state['thought']}\n"
            if reasoning_state["action"]:
                prompt += f"Action: {reasoning_state['action']}\n"
            if reasoning_state["action_input"]:
                prompt += f"Action Input: {reasoning_state['action_input']}\n"
            if reasoning_state["observation"]:
                prompt += f"Observation: {reasoning_state['observation']}\n"
        
        # Add instructions for next step
        prompt += (
            "\nYou should now continue with the next step. Follow this format:\n"
            "Thought: Think about what to do next\n"
            "Action: Choose one of the available tools or 'Final Answer' to give an answer\n"
            "Action Input: The input to the tool\n"
            "Final Answer: Your final answer to the original question when you're ready\n"
        )
        
        return prompt
    
    def _parse_reasoning_step(self, step_text: str) -> Dict[str, str]:
        """
        Parse the reasoning step text into components
        
        Args:
            step_text: Raw reasoning step text
            
        Returns:
            Parsed components (thought, action, action_input, final_answer)
        """
        result = {
            "thought": "",
            "action": "",
            "action_input": "",
            "final_answer": ""
        }
        
        # Extract thought
        thought_match = re.search(r'Thought:\s*(.*?)(?:$|Action:|Final Answer:)', step_text, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Check for final answer
        final_answer_match = re.search(r'Final Answer:\s*(.*?)(?:$|Thought:)', step_text, re.DOTALL)
        if final_answer_match:
            result["final_answer"] = final_answer_match.group(1).strip()
            return result
        
        # Extract action and action_input
        action_match = re.search(r'Action:\s*(.*?)(?:$|Action Input:|Thought:|Final Answer:)', step_text, re.DOTALL)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        action_input_match = re.search(r'Action Input:\s*(.*?)(?:$|Thought:|Action:|Final Answer:)', step_text, re.DOTALL)
        if action_input_match:
            result["action_input"] = action_input_match.group(1).strip()
        
        return result
    
    async def _execute_action(self, 
                           action: str, 
                           action_input: str,
                           user_id: str,
                           context: Dict[str, Any] = None) -> str:
        """
        Execute an action using the appropriate tool
        
        Args:
            action: The action to execute
            action_input: Input for the action
            user_id: User ID
            context: Additional context
            
        Returns:
            Observation from executing the action
        """
        # Check if action is available
        if action not in self.tool_providers:
            return f"Error: Action '{action}' is not available. Please choose from: {', '.join(self.tool_providers.keys())}"
        
        # Get tool
        tool = self.tool_providers[action]
        
        # Execute action
        try:
            if hasattr(tool, "execute"):
                result = await tool.execute(action_input, user_id, context)
            elif hasattr(tool, "run"):
                result = await tool.run(action_input, user_id, context)
            elif callable(tool):
                result = await tool(action_input, user_id, context)
            else:
                return f"Error: Tool '{action}' is not executable"
            
            # Format result
            if isinstance(result, dict):
                # Format dict as string
                formatted_result = json.dumps(result, indent=2)
            elif isinstance(result, list):
                # Format list as string
                formatted_result = "\n".join([str(item) for item in result])
            else:
                formatted_result = str(result)
            
            return formatted_result
        except Exception as e:
            logger.error(f"Error executing action '{action}': {e}")
            return f"Error executing action '{action}': {str(e)}"
    
    async def _generate_final_answer(self, 
                                  query: str, 
                                  reasoning_state: Dict[str, Any],
                                  context: Dict[str, Any] = None) -> str:
        """
        Generate a final answer based on the reasoning process
        
        Args:
            query: User query
            reasoning_state: Current reasoning state
            context: Additional context
            
        Returns:
            Final answer to the query
        """
        # Build prompt for final answer
        prompt = (
            "Based on your reasoning process and the observations from the actions you've taken, "
            "provide a clear, comprehensive answer to the original question. Your answer should "
            "be well-structured, accurate, and directly address what was asked.\n\n"
        )
        
        # Add query
        prompt += f"Question: {query}\n\n"
        
        # Add reasoning steps
        prompt += "Your reasoning process:\n"
        for i, step in enumerate(reasoning_state["steps"]):
            prompt += f"Step {i+1}:\n"
            prompt += f"Thought: {step['thought']}\n"
            prompt += f"Action: {step['action']}\n"
            prompt += f"Action Input: {step['action_input']}\n"
            prompt += f"Observation: {step['observation']}\n\n"
        
        # Add instructions for final answer
        prompt += (
            "Now, synthesize your reasoning into a clear, concise, and comprehensive answer. "
            "Your answer should be well-structured, easy to understand, and directly address "
            "the question. Include relevant details from your reasoning and observations."
        )
        
        # Generate final answer
        answer = await self.ai_provider.generate_response(
            prompt=prompt,
            temperature=0.5,
            max_tokens=1000
        )
        
        return answer

async def process_react_reasoning(
    query: str,
    user_id: str,
    ai_provider=None,
    response_cache=None,
    tool_providers=None,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a query using ReAct reasoning
    
    Args:
        query: User query
        user_id: User ID
        ai_provider: AI provider for reasoning
        response_cache: Optional cache for responses
        tool_providers: Dict of available tools by name
        context: Additional context
        
    Returns:
        Dict containing the answer and reasoning steps
    """
    import json  # Import here to avoid circular imports
    
    react_reasoning = ReactReasoning(
        ai_provider=ai_provider,
        response_cache=response_cache,
        tool_providers=tool_providers
    )
    
    return await react_reasoning.process(query, user_id, context) 