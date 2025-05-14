"""
Sequential Thinking Service

This module provides a service for sequential thinking reasoning,
with enhanced capabilities for thought revision, self-reflection,
and context-aware reasoning.
"""

import asyncio
import json
import os
import re
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union

# Import AI provider for accessing the language model
from bot_utilities.ai_utils import get_ai_provider

class SequentialThinkingService:
    """
    A service for implementing sequential thinking reasoning, with enhanced
    capabilities for thought revision, self-reflection, and context-aware reasoning.
    """
    
    def __init__(self):
        """
        Initialize the sequential thinking service
        """
        self.llm_provider = None
        self.chain_result = ""
        self.thought_history = {}  # Store thought history by session_id
        self.reflection_history = {}  # Store reflection history by session_id
        self.revision_history = {}  # Store revision history by session_id
        self._initialized = False
        
    async def initialize(self, llm_provider=None):
        """
        Initialize the service with an LLM provider
        
        Args:
            llm_provider: Optional language model provider to use
        """
        if not self._initialized:
            if llm_provider is None:
                # Import required modules lazily to avoid circular dependencies
                from bot_utilities.ai_utils import get_ai_provider
                llm_provider = await get_ai_provider()
                
            self.llm_provider = llm_provider
            self._initialized = True
            
    async def ensure_initialized(self):
        """Ensure the service is initialized with required components"""
        if not self._initialized:
            await self.initialize()
            
    async def process_sequential_thinking(self, 
                  problem: str, 
                  context: Dict[str, Any] = None, 
                  prompt_style: str = "sequential", 
                  num_thoughts: int = 5, 
                  temperature: float = 0.3,
                  enable_revision: bool = False,
                  enable_reflection: bool = False,
                  session_id: str = None) -> Tuple[bool, str]:
        """
        Process a problem using sequential thinking
        
        Args:
            problem: The problem to solve
            context: Additional context to provide
            prompt_style: The style of prompt to use (sequential, cot, got, cov, step_back)
            num_thoughts: Number of thoughts to generate
            temperature: Temperature for generation
            enable_revision: Whether to enable thought revision
            enable_reflection: Whether to enable self-reflection after thoughts
            session_id: Optional session ID for tracking thought history
            
        Returns:
            Tuple of (success, response)
        """
        await self.ensure_initialized()
                
        try:
            # Create a session ID if none provided
            if not session_id:
                session_id = f"session_{hash(problem)}"
                
            # Initialize thought history for this session if needed
            if session_id not in self.thought_history:
                self.thought_history[session_id] = []
            
            # Select the appropriate prompt based on style
            if prompt_style == "sequential":
                prompt = self._create_sequential_thinking_prompt(problem, context, num_thoughts, enable_revision)
            elif prompt_style == "cot":
                prompt = self._create_chain_of_thought_prompt(problem, context)
            elif prompt_style == "got":
                prompt = self._create_graph_of_thought_prompt(problem, context, num_thoughts)
            elif prompt_style == "cov":
                prompt = self._create_chain_of_verification_prompt(problem, context)
            elif prompt_style == "step_back":
                prompt = self._create_step_back_prompt(problem, context)
            else:
                # Default to sequential thinking
                prompt = self._create_sequential_thinking_prompt(problem, context, num_thoughts, enable_revision)
            
            # Call the LLM
            model_response = await self.llm_provider.async_call(
                prompt=prompt,
                temperature=temperature
            )
            
            # Process the response based on the prompt style
            processed_response = self._process_thinking_response(model_response, prompt_style)
            
            # Store the thought history
            self.thought_history[session_id].append({
                'problem': problem,
                'style': prompt_style,
                'thoughts': processed_response
            })
            
            # Add revision phase if enabled
            if enable_revision and prompt_style in ["sequential", "got"]:
                # Take the processed response and add a revision prompt
                revision_prompt = self._create_revision_prompt(problem, processed_response, context)
                revision_response = await self.llm_provider.async_call(
                    prompt=revision_prompt,
                    temperature=max(0.1, temperature-0.1)
                )
                
                # Track the revision
                if session_id not in self.revision_history:
                    self.revision_history[session_id] = []
                
                self.revision_history[session_id].append({
                    'original_thoughts': processed_response,
                    'revision': revision_response
                })
                
                # Combine original thoughts with revisions
                processed_response = self._combine_thoughts_with_revisions(processed_response, revision_response)
            
            # Add reflection phase if enabled
            if enable_reflection:
                reflection_prompt = self._create_reflection_prompt(problem, processed_response, context)
                reflection_response = await self.llm_provider.async_call(
                    prompt=reflection_prompt,
                    temperature=temperature
                )
                
                # Track the reflection
                if session_id not in self.reflection_history:
                    self.reflection_history[session_id] = []
                
                self.reflection_history[session_id].append({
                    'thoughts': processed_response,
                    'reflection': reflection_response
                })
                
                # Add the reflection to the response
                processed_response += f"\n\n**Reflection:**\n{reflection_response}"
            
            return True, processed_response
            
        except Exception as e:
            stack_trace = traceback.format_exc()
            print(f"Error in sequential thinking: {e}\n{stack_trace}")
            
            # Attempt a fallback simplified approach
            try:
                # Create a simpler prompt
                fallback_prompt = f"""
                Please think through this problem carefully:
                {problem}
                
                Provide a detailed, step-by-step response.
                """
                fallback_response = await self.llm_provider.async_call(
                    prompt=fallback_prompt,
                    temperature=0.3
                )
                return False, fallback_response
            except:
                return False, f"Failed to process the problem using sequential thinking: {str(e)}"
                
    def _create_sequential_thinking_prompt(self, 
                                        problem: str, 
                                        context: Dict[str, Any] = None,
                                        num_thoughts: int = 5,
                                        enable_revision: bool = False) -> str:
        """
        Create a sequential thinking prompt
        
        Args:
            problem: The problem to solve
            context: Additional context
            num_thoughts: Number of thoughts to generate
            enable_revision: Whether to enable thought revision capabilities
            
        Returns:
            str: The formatted prompt
        """
        revision_instructions = ""
        if enable_revision:
            revision_instructions = """
            As you progress through your thoughts, feel free to revise earlier thoughts if you realize they were incorrect or could be improved.
            When revising a thought, clearly indicate:
            - Which thought you're revising
            - Why you're revising it
            - The updated thought
            
            For example:
            "Thought 4: I need to revise Thought 2. I initially thought X, but now I realize Y because of Z. The corrected thought is..."
            """
        
        # Base prompt for sequential thinking
        prompt = f"""
        You are using Sequential Thinking to solve a complex problem.
        
        Guidelines for Sequential Thinking:
        1. Break down the problem into clear, logical steps
        2. Think through each step carefully and explicitly
        3. Number each thought (Thought 1, Thought 2, etc.)
        4. Explore the problem space thoroughly
        5. Synthesize your thoughts into a final, coherent answer
        6. Maintain a clear chain of reasoning throughout
        {revision_instructions}
        
        Problem to solve:
        {problem}
        """
        
        # Add context if provided
        if context:
            context_str = "\nAdditional Context:\n"
            for key, value in context.items():
                if isinstance(value, dict):
                    context_str += f"- {key}:\n"
                    for k, v in value.items():
                        context_str += f"  - {k}: {v}\n"
                elif isinstance(value, list):
                    context_str += f"- {key}: {', '.join(map(str, value))}\n"
                else:
                    context_str += f"- {key}: {value}\n"
            prompt += context_str
        
        # Add closing instructions
        prompt += f"""
        Generate {num_thoughts} detailed thoughts, analyzing the problem step-by-step.
        After your thoughts, provide a final answer that synthesizes your analysis.
        
        Format your response like this:
        
        Thought 1: [Your first thought]
        Thought 2: [Your second thought]
        ...
        Thought {num_thoughts}: [Your final thought]
        
        Final Answer: [Your comprehensive answer based on the thoughts above]
        """
        
        return prompt
        
    def _create_chain_of_thought_prompt(self, problem: str, context: Dict[str, Any] = None) -> str:
        """
        Create a chain-of-thought prompt
        
        Args:
            problem: The problem to solve
            context: Additional context
            
        Returns:
            str: The formatted prompt
        """
        # Base prompt for chain-of-thought
        prompt = f"""
        You are using Chain-of-Thought reasoning to solve a complex problem.
        
        Guidelines for Chain-of-Thought:
        1. Think about the problem step-by-step
        2. Show your complete reasoning process
        3. Break down complex parts into simpler components
        4. Make explicit any assumptions or background knowledge
        5. Derive a clear answer from your reasoning chain
        
        Problem to solve:
        {problem}
        """
        
        # Add context if provided
        if context:
            context_str = "\nAdditional Context:\n"
            for key, value in context.items():
                if isinstance(value, dict):
                    context_str += f"- {key}:\n"
                    for k, v in value.items():
                        context_str += f"  - {k}: {v}\n"
                elif isinstance(value, list):
                    context_str += f"- {key}: {', '.join(map(str, value))}\n"
                else:
                    context_str += f"- {key}: {value}\n"
            prompt += context_str
        
        # Add closing instructions
        prompt += """
        When responding, please follow this format:
        
        Reasoning:
        [Show your step-by-step reasoning process here]
        
        Answer:
        [Your final answer based on the reasoning above]
        """
        
        return prompt
        
    def _create_graph_of_thought_prompt(self, 
                                      problem: str, 
                                      context: Dict[str, Any] = None,
                                      num_nodes: int = 5) -> str:
        """
        Create a graph-of-thought prompt
        
        Args:
            problem: The problem to solve
            context: Additional context
            num_nodes: Number of thought nodes to generate
            
        Returns:
            str: The formatted prompt
        """
        # Base prompt for graph-of-thought
        prompt = f"""
        You are using Graph-of-Thought reasoning to explore multiple approaches to a problem simultaneously.
        
        Guidelines for Graph-of-Thought:
        1. Generate multiple independent thought paths (nodes)
        2. Each node should explore a different perspective or approach
        3. Connect related thoughts by referencing other nodes
        4. Explore diverse solution strategies
        5. Synthesize the most promising paths into a final answer
        
        Problem to solve:
        {problem}
        """
        
        # Add context if provided
        if context:
            context_str = "\nAdditional Context:\n"
            for key, value in context.items():
                if isinstance(value, dict):
                    context_str += f"- {key}:\n"
                    for k, v in value.items():
                        context_str += f"  - {k}: {v}\n"
                elif isinstance(value, list):
                    context_str += f"- {key}: {', '.join(map(str, value))}\n"
                else:
                    context_str += f"- {key}: {value}\n"
            prompt += context_str
        
        # Add closing instructions
        prompt += f"""
        Generate {num_nodes} thought nodes, each exploring a different approach to the problem.
        
        Format your response like this:
        
        Node 1: [Brief title for this approach]
        [Detailed exploration of the first approach]
        
        Node 2: [Brief title for this approach]
        [Detailed exploration of the second approach]
        ...
        
        Node {num_nodes}: [Brief title for this approach]
        [Detailed exploration of the final approach]
        
        Synthesis:
        [Synthesize the most promising aspects of the different nodes into a comprehensive answer]
        """
        
        return prompt
        
    def _create_chain_of_verification_prompt(self, problem: str, context: Dict[str, Any] = None) -> str:
        """
        Create a chain-of-verification prompt
        
        Args:
            problem: The problem to solve
            context: Additional context
            
        Returns:
            str: The formatted prompt
        """
        # Base prompt for chain-of-verification
        prompt = f"""
        You are using Chain-of-Verification reasoning to solve a problem while minimizing errors.
        
        Guidelines for Chain-of-Verification:
        1. First, break down the problem and think about it step-by-step
        2. Propose an initial answer
        3. Identify potential errors or biases in your answer
        4. Verify each part of your reasoning
        5. Correct any errors found
        6. Provide a revised, higher-quality answer
        
        Problem to solve:
        {problem}
        """
        
        # Add context if provided
        if context:
            context_str = "\nAdditional Context:\n"
            for key, value in context.items():
                if isinstance(value, dict):
                    context_str += f"- {key}:\n"
                    for k, v in value.items():
                        context_str += f"  - {k}: {v}\n"
                elif isinstance(value, list):
                    context_str += f"- {key}: {', '.join(map(str, value))}\n"
                else:
                    context_str += f"- {key}: {value}\n"
            prompt += context_str
        
        # Add closing instructions
        prompt += """
        When responding, please follow this format:
        
        Initial Reasoning:
        [Your initial step-by-step reasoning process]
        
        Initial Answer:
        [Your first answer based on the reasoning above]
        
        Verification Steps:
        1. [First verification check]
        2. [Second verification check]
        ...
        
        Corrections:
        [Any corrections needed based on verification]
        
        Final Answer:
        [Your revised answer after verification]
        """
        
        return prompt
        
    def _create_step_back_prompt(self, problem: str, context: Dict[str, Any] = None) -> str:
        """
        Create a step-back prompt
        
        Args:
            problem: The problem to solve
            context: Additional context
            
        Returns:
            str: The formatted prompt
        """
        # Base prompt for step-back reasoning
        prompt = f"""
        You are using Step-Back reasoning to solve a complex problem.
        
        Guidelines for Step-Back Reasoning:
        1. First "step back" and analyze the problem from a higher, more abstract level
        2. Consider the general principles, theories, or frameworks that apply
        3. Then "step in" to apply these broader concepts to the specific problem
        4. Break down the application into concrete steps
        5. Synthesize a comprehensive answer using both high-level knowledge and specific application
        
        Problem to solve:
        {problem}
        """
        
        # Add context if provided
        if context:
            context_str = "\nAdditional Context:\n"
            for key, value in context.items():
                if isinstance(value, dict):
                    context_str += f"- {key}:\n"
                    for k, v in value.items():
                        context_str += f"  - {k}: {v}\n"
                elif isinstance(value, list):
                    context_str += f"- {key}: {', '.join(map(str, value))}\n"
                else:
                    context_str += f"- {key}: {value}\n"
            prompt += context_str
        
        # Add closing instructions
        prompt += """
        When responding, please follow this format:
        
        Step Back (High-Level Analysis):
        [Analyze the problem from a broader perspective, identifying the general principles or frameworks that apply]
        
        Step In (Specific Application):
        [Apply the high-level concepts to the specific problem, breaking it down into concrete steps]
        
        Final Answer:
        [Your comprehensive answer that integrates both the general principles and specific application]
        """
        
        return prompt
        
    def _create_revision_prompt(self, problem: str, thoughts: str, context: Dict[str, Any] = None) -> str:
        """
        Create a revision prompt for reviewing previous thoughts
        
        Args:
            problem: The original problem
            thoughts: The thoughts to revise
            context: Additional context
            
        Returns:
            str: The formatted prompt
        """
        prompt = f"""
        You are reviewing a set of thoughts about a problem and looking for potential improvements or corrections.
        
        Original Problem:
        {problem}
        
        Previous Thoughts:
        {thoughts}
        
        Guidelines for Revision:
        1. Carefully review each thought for errors, inconsistencies, or logical flaws
        2. Consider if any assumptions were incorrect
        3. Look for opportunities to strengthen the reasoning
        4. Identify any thoughts that should be revised or extended
        5. Be specific about which thoughts need revision and why
        
        For each thought that needs revision, please specify:
        - The thought number being revised
        - Why it needs revision
        - The corrected or improved thought
        
        Format your revisions like this:
        
        Revision for Thought X:
        - Original thought: [brief summary of original]
        - Issue: [explanation of what's wrong or could be improved]
        - Improved thought: [the corrected thought]
        
        If you find no issues with a thought, you don't need to mention it.
        """
        
        return prompt
        
    def _create_reflection_prompt(self, problem: str, thoughts: str, context: Dict[str, Any] = None) -> str:
        """
        Create a reflection prompt for meta-analysis of the thinking process
        
        Args:
            problem: The original problem
            thoughts: The thoughts to reflect on
            context: Additional context
            
        Returns:
            str: The formatted prompt
        """
        prompt = f"""
        Reflect on the following problem-solving process:
        
        Original Problem:
        {problem}
        
        Thinking Process:
        {thoughts}
        
        Please provide a brief meta-analysis of the thinking approach:
        1. What were the strengths of this reasoning process?
        2. Were there any potential blind spots or biases?
        3. Were there alternative approaches that could have been considered?
        4. How confident should we be in the final answer, and why?
        
        Provide a concise reflection that highlights the most important insights about the thinking process itself.
        """
        
        return prompt
        
    def _process_thinking_response(self, response: str, prompt_style: str) -> str:
        """
        Process the raw thinking response based on prompt style
        
        Args:
            response: The raw response from the LLM
            prompt_style: The prompt style used
            
        Returns:
            str: The processed response
        """
        # Clean up the response
        processed = response.strip()
        
        # Process based on prompt style
        if prompt_style == "sequential":
            # Extract the thoughts and final answer
            thoughts_pattern = r"(?:Thought|THOUGHT|Step|STEP)\s*(\d+)\s*:\s*(.*?)(?=(?:Thought|THOUGHT|Step|STEP)\s*\d+|Final Answer|FINAL ANSWER|$)"
            final_answer_pattern = r"(?:Final Answer|FINAL ANSWER)\s*:\s*(.*?)$"
            
            thoughts = re.findall(thoughts_pattern, processed, re.DOTALL)
            final_answer = re.search(final_answer_pattern, processed, re.DOTALL)
            
            # Format the output
            formatted_thoughts = ""
            for num, thought in thoughts:
                formatted_thoughts += f"Thought {num}: {thought.strip()}\n\n"
                
            if final_answer:
                formatted_thoughts += f"Final Answer: {final_answer.group(1).strip()}"
            else:
                # Try to extract the last part as the answer if no explicit final answer
                parts = processed.split("\n\n")
                if parts and not any(p.startswith(("Thought", "THOUGHT", "Step", "STEP")) for p in parts[-1:]):
                    formatted_thoughts += f"Final Answer: {parts[-1].strip()}"
                    
            return formatted_thoughts
            
        elif prompt_style == "cot":
            # Extract reasoning and answer
            reasoning_pattern = r"(?:Reasoning|REASONING)\s*:\s*(.*?)(?=(?:Answer|ANSWER)\s*:|$)"
            answer_pattern = r"(?:Answer|ANSWER)\s*:\s*(.*?)$"
            
            reasoning = re.search(reasoning_pattern, processed, re.DOTALL)
            answer = re.search(answer_pattern, processed, re.DOTALL)
            
            # Format the output
            formatted_output = ""
            if reasoning:
                formatted_output += f"Reasoning:\n{reasoning.group(1).strip()}\n\n"
                
            if answer:
                formatted_output += f"Answer:\n{answer.group(1).strip()}"
            else:
                # If no explicit answer, use the last paragraph
                parts = processed.split("\n\n")
                if parts and not parts[-1].startswith(("Reasoning", "REASONING")):
                    formatted_output += f"Answer:\n{parts[-1].strip()}"
                    
            return formatted_output
            
        elif prompt_style == "got":
            # Extract nodes and synthesis
            node_pattern = r"(?:Node|NODE)\s*(\d+)\s*:\s*(.*?)(?=(?:Node|NODE|Synthesis|SYNTHESIS)\s*\d*:|$)"
            synthesis_pattern = r"(?:Synthesis|SYNTHESIS)\s*:\s*(.*?)$"
            
            nodes = re.findall(node_pattern, processed, re.DOTALL)
            synthesis = re.search(synthesis_pattern, processed, re.DOTALL)
            
            # Format the output
            formatted_output = ""
            for num, content in nodes:
                node_title, *node_content = content.split("\n", 1)
                node_detail = node_content[0] if node_content else ""
                formatted_output += f"Node {num}: {node_title.strip()}\n{node_detail.strip()}\n\n"
                
            if synthesis:
                formatted_output += f"Synthesis:\n{synthesis.group(1).strip()}"
                
            return formatted_output
            
        elif prompt_style in ["cov", "step_back"]:
            # For these styles, minimal processing needed
            # Just ensure proper formatting
            return processed
            
        # Default: return as is
        return processed
        
    def _combine_thoughts_with_revisions(self, original_thoughts: str, revisions: str) -> str:
        """
        Combine original thoughts with revisions
        
        Args:
            original_thoughts: The original thoughts
            revisions: The revisions
            
        Returns:
            str: Combined thoughts with revisions
        """
        # Extract revisions for specific thoughts
        revision_pattern = r"Revision for Thought (\d+):\s*-\s*Original thought:.*?-\s*Issue:.*?-\s*Improved thought:(.*?)(?=Revision for Thought \d+:|$)"
        found_revisions = re.findall(revision_pattern, revisions, re.DOTALL)
        
        # If no structured revisions found, append the full revision text
        if not found_revisions:
            return f"{original_thoughts}\n\n**Revisions:**\n{revisions.strip()}"
            
        # Create a dictionary of thought number -> revised thought
        revision_dict = {int(num): content.strip() for num, content in found_revisions}
        
        # Process the original thoughts and insert revisions
        thought_pattern = r"(Thought (\d+):)(.*?)(?=Thought \d+:|Final Answer:|$)"
        processed_thoughts = original_thoughts
        
        # Replace each thought with original + revision if it exists
        for match in re.finditer(thought_pattern, original_thoughts, re.DOTALL):
            full_match, thought_num_str, thought_content = match.groups()
            thought_num = int(thought_num_str)
            
            if thought_num in revision_dict:
                # Format the revised thought
                revised_section = f"{full_match}{thought_content.strip()}\n\n**Revised Thought {thought_num}:** {revision_dict[thought_num]}\n\n"
                # Replace the original thought with original + revision
                processed_thoughts = processed_thoughts.replace(match.group(0), revised_section)
                
        return processed_thoughts

# Singleton instance
sequential_thinking_service = SequentialThinkingService() 