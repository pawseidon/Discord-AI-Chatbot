import os
import re
import json
import asyncio
from typing import List, Dict, Any, Tuple, Optional, Callable, Awaitable
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from bot_utilities.ai_utils import get_ai_provider, search_internet
from bot_utilities.token_utils import token_optimizer

class ReActStep:
    """Represents a single step in the ReAct reasoning process"""
    def __init__(self, thought: str = "", action: str = "", action_input: str = "", observation: str = ""):
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation
        
    def __str__(self) -> str:
        result = ""
        if self.thought:
            result += f"Thought: {self.thought}\n"
        if self.action:
            result += f"Action: {self.action}\n"
        if self.action_input:
            result += f"Action Input: {self.action_input}\n"
        if self.observation:
            result += f"Observation: {self.observation}\n"
        return result
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation
        }

class ReActAgent:
    """
    Implementation of the ReAct pattern (Reasoning-Action-Observation cycle)
    Based on "ReAct: Synergizing Reasoning and Acting in Language Models"
    (https://arxiv.org/abs/2210.03629)
    """
    
    def __init__(self, api_key: str = None, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        """Initialize the ReAct agent"""
        self.api_key = api_key or os.environ.get("API_KEY")
        self.model_name = model_name
        
        # Set up the LLM
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name
        )
        
        # Available tools/actions
        self.tools = {
            "WebSearch": self.web_search,
            "Calculate": self.calculate,
            "GetDatetime": self.get_datetime,
            "FinalAnswer": self.final_answer
        }
        
        # ReAct prompt template
        self.react_prompt = PromptTemplate.from_template(
            """You are an AI assistant that follows the ReAct (Reasoning-Action-Observation) framework to solve problems. 
            Your goal is to answer the user's question by carefully reasoning about what information you need and taking actions to gather that information.
            
            You have these tools at your disposal:
            - WebSearch: Search the web for information using this action when you need factual or current information
            - Calculate: Perform arithmetic calculations (addition, subtraction, multiplication, division)
            - GetDatetime: Get the current date and time when needed
            - FinalAnswer: When you've gathered enough information to answer the question, use this to provide your final response to the user
            
            For each step in your reasoning:
            1. Thought: Reflect on what you know, what you need to know, and how to get that information
            2. Action: Choose one of the available tools to use
            3. Action Input: Provide the necessary input for the chosen tool
            4. Observation: Review the result of the action
            
            Continue this cycle until you have enough information to provide a final answer.
            
            User's question: {question}
            {context}
            
            Begin your reasoning now:
            """
        )
    
    async def web_search(self, query: str) -> str:
        """Perform a web search using the available search utility"""
        try:
            result = await search_internet(query)
            
            # Optimize result to reduce tokens
            if result:
                result = token_optimizer.clean_text(result)
                result = token_optimizer.truncate_text(result, max_tokens=800)
                
            return result or "No relevant search results found."
        except Exception as e:
            return f"Error performing search: {str(e)}"
    
    async def calculate(self, expression: str) -> str:
        """Perform basic arithmetic calculations"""
        # Clean the expression of any unsafe characters
        clean_expr = re.sub(r'[^0-9+\-*/().%\s]', '', expression)
        
        try:
            # Use Python's eval with only math operations
            # This is relatively safe since we've sanitized the input
            result = eval(clean_expr, {"__builtins__": {}})
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}. Make sure to use only basic arithmetic operations."
    
    async def get_datetime(self, _: str = "") -> str:
        """Get the current date and time"""
        from datetime import datetime
        now = datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    
    async def final_answer(self, answer: str) -> str:
        """Indicate that this is the final answer"""
        return answer
    
    async def parse_step(self, text: str) -> ReActStep:
        """Parse a ReAct step from the model's output"""
        step = ReActStep()
        
        # Extract thought
        thought_match = re.search(r"Thought:\s*(.*?)(?:Action:|Action Input:|Observation:|$)", text, re.DOTALL)
        if thought_match:
            step.thought = thought_match.group(1).strip()
        
        # Extract action
        action_match = re.search(r"Action:\s*(.*?)(?:Action Input:|Observation:|$)", text, re.DOTALL)
        if action_match:
            step.action = action_match.group(1).strip()
        
        # Extract action input
        action_input_match = re.search(r"Action Input:\s*(.*?)(?:Observation:|$)", text, re.DOTALL)
        if action_input_match:
            step.action_input = action_input_match.group(1).strip()
        
        # Extract observation
        observation_match = re.search(r"Observation:\s*(.*?)$", text, re.DOTALL)
        if observation_match:
            step.observation = observation_match.group(1).strip()
        
        return step
    
    async def execute_action(self, action: str, action_input: str) -> str:
        """Execute the specified action with the given input"""
        # Find the corresponding tool
        action = action.strip()
        
        # Check for close matches (case-insensitive)
        for tool_name, tool_func in self.tools.items():
            if action.lower() == tool_name.lower():
                return await tool_func(action_input)
        
        # If no match found
        return f"Error: Unknown action '{action}'. Available actions are: {', '.join(self.tools.keys())}"
    
    async def generate_next_step(self, question: str, steps: List[ReActStep], context: str = "") -> ReActStep:
        """Generate the next step in the ReAct process"""
        # Format the current conversation history
        history = ""
        for step in steps:
            history += str(step) + "\n"
        
        # Format the prompt
        full_prompt = self.react_prompt.format(
            question=question,
            context=f"Context: {context}" if context else ""
        ) + "\n" + history
        
        # Get the next step from the model
        response = await self.llm.ainvoke(full_prompt)
        response_text = response.content
        
        # Parse the response into a ReActStep
        return await self.parse_step(response_text)
    
    async def solve(
        self, 
        question: str, 
        context: str = "", 
        max_steps: int = 8,
        timeout: int = 60
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Solve a problem using the ReAct framework
        
        Args:
            question: The user's question
            context: Optional additional context
            max_steps: Maximum number of reasoning steps
            timeout: Timeout in seconds
            
        Returns:
            Tuple: (final_answer, reasoning_steps)
        """
        steps: List[ReActStep] = []
        final_answer = ""
        
        # Start a task with timeout
        try:
            task = asyncio.create_task(self._solve_with_steps(question, steps, context, max_steps))
            final_answer = await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            # If we hit the timeout, construct a reasonable response from what we have
            final_answer = self._construct_partial_answer(question, steps)
        except Exception as e:
            final_answer = f"I encountered an error while solving this problem: {str(e)}"
            
        # Convert steps to dictionary format for serialization
        step_dicts = [step.to_dict() for step in steps]
        
        return final_answer, step_dicts
    
    async def _solve_with_steps(
        self, 
        question: str, 
        steps: List[ReActStep],
        context: str = "",
        max_steps: int = 8
    ) -> str:
        """Internal method to solve a problem with step tracking"""
        for i in range(max_steps):
            # Generate the next step
            next_step = await self.generate_next_step(question, steps, context)
            
            # If the action is FinalAnswer, we're done
            if next_step.action == "FinalAnswer":
                final_answer = next_step.action_input
                steps.append(next_step)  # Add the final step to the history
                return final_answer
                
            # If there's an action, execute it
            if next_step.action:
                observation = await self.execute_action(next_step.action, next_step.action_input)
                next_step.observation = observation
                
            # Add this step to the history
            steps.append(next_step)
            
            # Check if we have a final answer pattern in the thought
            if "final answer" in next_step.thought.lower():
                final_answer_match = re.search(r"final answer:?\s*(.*?)$", next_step.thought, re.IGNORECASE | re.DOTALL)
                if final_answer_match:
                    return final_answer_match.group(1).strip()
        
        # If we've reached the maximum number of steps without a final answer,
        # construct a reasonable response from what we have
        return self._construct_partial_answer(question, steps)
    
    def _construct_partial_answer(self, question: str, steps: List[ReActStep]) -> str:
        """Construct a partial answer from incomplete reasoning steps"""
        if not steps:
            return "I couldn't solve this problem within the time limit. Please try asking a more specific question."
            
        # Try to extract useful information from the steps
        observations = []
        thoughts = []
        
        for step in steps:
            if step.observation:
                observations.append(step.observation)
            if step.thought:
                thoughts.append(step.thought)
                
        # If we have observations, use them to construct an answer
        answer = f"I was working on your question about {question.split()[:5]}... but couldn't complete all my reasoning in time.\n\n"
        
        if observations:
            answer += "Here's what I found:\n"
            for i, obs in enumerate(observations[-3:]):  # Use last 3 observations
                answer += f"{i+1}. {obs.split('.')[:2]}...\n"
                
        answer += "\nBased on this partial information, "
        
        if thoughts:
            # Use the last thought as a basis for the answer
            answer += thoughts[-1]
        else:
            answer += "I would need more time to give you a complete answer. Please try asking a more specific question."
            
        return answer
            
async def run_react_agent(
    question: str, 
    context: str = "", 
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    max_steps: int = 8,
    timeout: int = 60
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Run a ReAct agent to solve a problem
    
    Args:
        question: The user's question
        context: Optional additional context
        model: Model to use
        max_steps: Maximum number of reasoning steps
        timeout: Timeout in seconds
        
    Returns:
        Tuple: (final_answer, reasoning_steps)
    """
    api_key = os.environ.get("API_KEY")
    agent = ReActAgent(api_key=api_key, model_name=model)
    return await agent.solve(question, context, max_steps, timeout) 