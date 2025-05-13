import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('sequential_thinking')

class SequentialThinking:
    """
    Implements sequential thinking capabilities directly using the LLM
    
    Based on research in "Fine-Tuning Large Language Models with Sequential Instructions"
    (https://arxiv.org/html/2403.07794v1)
    """
    
    def __init__(self, llm_provider=None):
        """
        Initialize the sequential thinking module.
        
        Args:
            llm_provider: A language model provider instance that has an async call method
        """
        self.llm_provider = llm_provider
        self.thinking_style_templates = {
            "sequential": {
                "prompt": """You are an AI assistant tasked with solving problems using sequential thinking.
                
I'll present you with a problem, and I'd like you to solve it by breaking it down into clear, logical steps.

For this problem: "{problem}"

Think through this step by step:
1. First, understand what's being asked
2. Break down the problem into manageable parts
3. Address each part in a logical sequence
4. Be precise and thorough in your analysis
5. Check your reasoning for errors or inconsistencies

Context information: {context}

Please solve this problem by analyzing it sequentially. Show your thinking as you work through each step.
""",
                "output_parser": self._parse_sequential_output
            },
            "list": {
                "prompt": """You are an AI assistant tasked with solving problems using list-based thinking.
                
I'll present you with a problem, and I'd like you to solve it by breaking it down into a list of clear, logical points.

For this problem: "{problem}"

Think through this by creating a clear list of points:
- First, understand what's being asked
- Break down the problem into a list of key considerations
- Address each point thoroughly
- Be precise and thorough in your analysis
- Check your list for completeness and accuracy

Context information: {context}

Please solve this problem by analyzing it as a structured list. Show your thinking as you work through each point.
""",
                "output_parser": self._parse_list_output
            },
            "cot": {
                "prompt": """You are an AI assistant tasked with solving problems using chain-of-thought reasoning.

I'll present you with a problem, and I'd like you to solve it by working through a chain of logical reasoning.

For this problem: "{problem}"

Think through this chain of thought:
1. First, understand what's being asked
2. Identify the key variables and relationships
3. Reason step by step to derive the answer
4. Be explicit about your logical connections
5. Check your reasoning carefully

Context information: {context}

Please solve this problem by tracing through a clear chain of thought. Show your reasoning explicitly as you derive the answer.
""",
                "output_parser": self._parse_cot_output
            },
            "cov": {
                "prompt": """You are an AI assistant tasked with solving problems using chain-of-verification reasoning.

I'll present you with a problem, and I'd like you to solve it by working through a chain of logical reasoning, 
but with an additional verification step to catch any errors or inconsistencies.

For this problem: "{problem}"

Think through this chain of verification process:
1. First, understand what's being asked
2. Identify the key variables and relationships
3. Reason step by step to derive an initial answer
4. VERIFICATION: Check each step of your reasoning for:
   - Factual accuracy
   - Logical coherence
   - Missing information
   - Potential biases
   - Alternative interpretations
5. Revise your answer if verification reveals issues
6. Provide your final, verified answer

Context information: {context}

Please solve this problem by using a chain-of-verification approach. Show your reasoning explicitly, 
including the verification steps where you double-check your work.
""",
                "output_parser": self._parse_cov_output
            },
            "got": {
                "prompt": """You are an AI assistant tasked with solving problems using Graph-of-Thought reasoning.

I'll present you with a problem, and I'd like you to solve it by creating a mental graph of connected thoughts 
and concepts rather than a strictly linear sequence.

For this problem: "{problem}"

Apply Graph-of-Thought reasoning as follows:
1. First, understand what's being asked
2. Identify key concepts and their relationships (nodes in your thought graph)
3. Explore multiple thought branches in parallel where relevant
4. Create connections between related thoughts
5. Identify knowledge gaps and bridge them
6. Converge diverse thought paths into a cohesive solution
7. Organize your final answer based on the most important insights

Context information: {context}

Please solve this problem by visualizing and working through a graph of interconnected thoughts. 
Show your thinking as you explore different branches of reasoning, making connections between concepts,
and converging on a solution. Label different thought branches clearly.
""",
                "output_parser": self._parse_got_output
            }
        }
        
    async def set_llm_provider(self, llm_provider):
        """Set the LLM provider to use for generating responses"""
        self.llm_provider = llm_provider
        
    async def _call_llm(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2000):
        """
        Call the language model with the given prompt.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature (lower for more deterministic outputs)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            str: The model's response
        """
        if not self.llm_provider:
            raise ValueError("LLM provider not set. Call set_llm_provider first.")
            
        try:
            # Attempt to call the model's async method
            if hasattr(self.llm_provider, 'async_call'):
                return await self.llm_provider.async_call(prompt, temperature=temperature, max_tokens=max_tokens)
            elif hasattr(self.llm_provider, 'generate'):
                return await self.llm_provider.generate(prompt, temperature=temperature, max_tokens=max_tokens)
            elif hasattr(self.llm_provider, 'call'):
                # If only a synchronous call method is available, run it in a thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        executor, 
                        lambda: self.llm_provider.call(prompt, temperature=temperature, max_tokens=max_tokens)
                    )
            else:
                # Default attempt - try treating the provider as a callable
                return await asyncio.to_thread(self.llm_provider, prompt, temperature=temperature, max_tokens=max_tokens)
        
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            # In case of error, try to use the Discord bot's built-in AI utilities
            try:
                from bot_utilities.ai_utils import generate_response
                response = await generate_response(prompt=prompt)
                return response
            except Exception as inner_e:
                logger.error(f"Error using fallback AI utils: {inner_e}")
                raise RuntimeError(f"Failed to generate response: {e}. Fallback also failed: {inner_e}")
    
    def _create_sequential_thinking_prompt(self, problem: str, num_thoughts: int = 5) -> str:
        """
        Create a prompt for sequential thinking.
        
        Args:
            problem: The problem to solve
            num_thoughts: Target number of thoughts (may be adjusted by the model)
            
        Returns:
            str: A structured prompt for sequential thinking
        """
        return f"""I need to solve this problem using sequential thinking: {problem}

To solve this problem effectively, I'll break it down into individual thoughts, which will allow me to track my thinking process and revise as needed.

IMPORTANT INSTRUCTIONS:
1. I should use approximately {num_thoughts} thoughts to solve this, but I can adjust if needed
2. For each thought, I'll include:
   - The thought number (e.g., "Thought 1:")
   - My current thinking
   - Whether I need to revise any previous thoughts
   - Whether I need more thoughts beyond my initial estimate
3. I should explicitly address when I'm revising previous thinking
4. I should be willing to branch into alternative approaches if needed
5. After working through my thoughts, I'll provide a clear final answer

Let me think through this step-by-step:

"""
    
    def _create_sequential_list_prompt(self, problem: str, num_steps: int = 5) -> str:
        """
        Create a prompt for sequential list-based thinking (simpler structure).
        
        Args:
            problem: The problem to solve
            num_steps: Target number of steps
            
        Returns:
            str: A structured prompt for list-based sequential thinking
        """
        return f"""I need to solve this problem: {problem}

To solve this effectively, I'll break down my approach into approximately {num_steps} clear, sequential steps.

For each step, I will:
1. Think carefully about what needs to be done
2. Consider how this step builds on previous steps
3. Explain my reasoning clearly
4. Revise previous steps if I realize they need adjustment

Starting with Step 1, I'll work through this problem methodically:

"""

    def _create_cot_prompt(self, problem: str) -> str:
        """
        Create a simpler chain-of-thought prompt (alternative approach).
        
        Args:
            problem: The problem to solve
            
        Returns:
            str: A structured prompt for chain-of-thought
        """
        return f"""I need to solve this problem: {problem}

Let me work through this step-by-step:

"""

    def _create_got_prompt(self, problem: str) -> str:
        """
        Create a Graph-of-Thought prompt for non-linear problem solving.
        
        Args:
            problem: The problem to solve
            
        Returns:
            str: A structured prompt for graph-of-thought reasoning
        """
        return f"""I need to solve this problem using Graph-of-Thought reasoning: {problem}

Unlike traditional linear thinking, I'll explore multiple thought branches and connect related concepts:

THOUGHT GRAPH STRUCTURE:
- I'll identify key concepts as nodes in my thought graph
- I'll explore multiple thought branches in parallel labeled as Branch A, Branch B, etc.
- I'll create connections between related thoughts with "Connection:" labels
- I'll identify where branches converge with "Convergence:" labels
- I'll identify critical insights with "Key Insight:" labels

Let me build this thought graph to solve the problem:

"""

    async def solve_with_sequential_thinking(
        self, 
        problem: str, 
        context: Optional[Dict[str, Any]] = None,
        prompt_style: str = "sequential",
        num_thoughts: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        retry_attempts: int = 2
    ) -> Tuple[bool, str]:
        """
        Generate a solution to a problem using sequential thinking.
        
        Args:
            problem: The problem to solve
            context: Additional context dictionary to include in the prompt
            prompt_style: The thinking style to use ('sequential', 'list', 'cot', 'cov', or 'got')
            num_thoughts: Target number of thoughts in the sequential thinking output
            temperature: Temperature parameter for the LLM
            max_tokens: Maximum number of tokens to generate
            retry_attempts: Number of retries if the response validation fails
            
        Returns:
            tuple: (success_flag, response_text)
        """
        if not self.llm_provider:
            from bot_utilities.ai_utils import generate_response
            
        # Format context string for inclusion in the prompt
        context_str = "No additional context provided."
        if context:
            try:
                context_str = "Context information:\n"
                for key, value in context.items():
                    if isinstance(value, dict):
                        context_str += f"- {key}:\n"
                        for sub_key, sub_value in value.items():
                            context_str += f"  - {sub_key}: {sub_value}\n"
                    else:
                        context_str += f"- {key}: {value}\n"
            except Exception as e:
                logger.error(f"Error formatting context: {e}")
                context_str = f"Error formatting context: {str(e)}"
                
        # Choose the appropriate thinking style template
        if prompt_style not in self.thinking_style_templates:
            logger.warning(f"Unknown prompt style '{prompt_style}', defaulting to 'sequential'")
            prompt_style = "sequential"
            
        # Get the template for the chosen style
        template = self.thinking_style_templates[prompt_style]
        
        # Format the prompt with the problem and context
        try:
            prompt_template = template["prompt"]
            prompt = prompt_template.format(problem=problem, context=context_str)
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            return False, f"Error formatting prompt: {e}"
            
        # Get the output parser for the chosen style
        output_parser = template.get("output_parser", self._parse_sequential_output)
        
        # Keep track of attempts
        for attempt in range(retry_attempts + 1):
            try:
                # Call the LLM with the prompt
                response = await self._call_llm(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Validate the response
                is_valid = self._validate_sequential_response(response, prompt_style)
                
                if is_valid:
                    # Parse and format the response
                    parsed_response = output_parser(response)
                    return True, parsed_response
                else:
                    # Log retry attempts
                    if attempt < retry_attempts:
                        logger.warning(f"Response validation failed, retrying ({attempt+1}/{retry_attempts})")
                        # Try with different temperature
                        temperature += 0.1
                    else:
                        logger.error("Response validation failed after all attempts, returning raw response")
                        # If all retries fail, try to clean up response as best we can
                        return False, self._parse_sequential_output(response)
            except Exception as e:
                logger.error(f"Error in solve_with_sequential_thinking: {str(e)}")
                if attempt < retry_attempts:
                    logger.info(f"Retrying after error ({attempt+1}/{retry_attempts})")
                    await asyncio.sleep(1)  # Brief pause before retry
                else:
                    return False, f"Error generating sequential thinking response after {retry_attempts} attempts: {str(e)}"
                    
        # This should not be reached, but just in case all attempts silently fail
        return False, "Failed to generate a valid response after multiple attempts."

    def _validate_sequential_response(self, response: str, prompt_style: str) -> bool:
        """
        Validate that the response follows the expected structure.
        
        Args:
            response: The response to validate
            prompt_style: The thinking style used
            
        Returns:
            bool: True if the response is valid, False otherwise
        """
        # Empty responses are not valid
        if not response or len(response.strip()) < 20:
            return False
            
        # Basic structure validation based on style
        if prompt_style == "sequential" or prompt_style == "cot":
            # Check for steps or thoughts
            step_patterns = [
                r"(?:^|\n)(?:step|thought|thinking|point)\s*\d+",
                r"(?:^|\n)\d+\s*[\.\)]",
                r"(?:^|\n)(?:first|second|third|fourth|fifth|next)",
                r"(?:^|\n)(?:initial|preliminary|following|subsequent)"
            ]
            has_steps = any(re.search(pattern, response.lower()) for pattern in step_patterns)
            
            # Should have a conclusion or answer
            conclusion_patterns = [
                r"(?:^|\n)(?:conclusion|answer|summary|result)",
                r"(?:^|\n)(?:in conclusion|to conclude|to summarize|in summary)",
                r"(?:^|\n)(?:therefore|thus|hence|so)",
                r"(?:^|\n)(?:the solution is|the answer is)"
            ]
            has_conclusion = any(re.search(pattern, response.lower()) for pattern in conclusion_patterns)
            
            return has_steps and has_conclusion
            
        elif prompt_style == "cov":
            # Check for verification steps
            verification_patterns = [
                r"(?:^|\n)(?:verification|verify|checking|check)",
                r"(?:^|\n)(?:fact.?check|validate|validation)",
                r"(?:^|\n)(?:reviewing|review|double.?check)"
            ]
            has_verification = any(re.search(pattern, response.lower()) for pattern in verification_patterns)
            
            return has_verification and self._validate_sequential_response(response, "cot")
            
        elif prompt_style == "list":
            # Check for bullet points or numbered list
            list_patterns = [
                r"(?:^|\n)\s*[\-\*\•]\s+",
                r"(?:^|\n)\s*\d+\s*[\.\)]\s+"
            ]
            has_list_items = any(re.search(pattern, response) for pattern in list_patterns)
            
            return has_list_items
            
        elif prompt_style == "got":
            # Check for branches or connections
            branch_patterns = [
                r"(?:^|\n)(?:branch|path|approach|perspective|angle)\s*\d*",
                r"(?:^|\n)(?:alternative|option|consideration)",
                r"(?:^|\n)(?:connection|link|relationship|related)"
            ]
            has_branches = any(re.search(pattern, response.lower()) for pattern in branch_patterns)
            
            return has_branches
            
        # If style not recognized, consider valid (can't validate)
        return True

    async def run(
        self, 
        problem: str, 
        context: Dict[str, Any] = None,
        prompt_style: str = "auto",
        num_thoughts: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout: int = 60
    ) -> Tuple[bool, str]:
        """
        Main entry point for sequential thinking. Automatically detects the problem type
        and chooses the appropriate reasoning strategy.
        
        Args:
            problem: The problem to solve
            context: Additional context for the problem
            prompt_style: The thinking style to use
            num_thoughts: Target number of thoughts
            temperature: Temperature parameter for the LLM
            max_tokens: Maximum number of tokens to generate
            timeout: Timeout in seconds
            
        Returns:
            tuple: (success_flag, response_text)
        """
        
        # If prompt_style is "auto", detect the best style for the problem
        if prompt_style == "auto":
            prompt_style = await self._detect_problem_type(problem)
            
        start_time = time.time()
        
        try:
            # Set up a timeout for the task
            success, response = await asyncio.wait_for(
                self.solve_with_sequential_thinking(
                    problem=problem,
                    context=context,
                    prompt_style=prompt_style,
                    num_thoughts=num_thoughts,
                    temperature=temperature,
                    max_tokens=max_tokens
                ),
                timeout=timeout
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Sequential thinking completed in {elapsed_time:.2f}s using {prompt_style} style")
            
            # If successful, return the response
            if success:
                return True, response
                
            # If not successful but we have a response, try to format it
            if response:
                formatted_response = self.format_response(response)
                return False, formatted_response
                
            # Fall back to a simpler approach if everything fails
            fallback_message = f"""
            I tried to analyze this problem using sequential thinking ({prompt_style}), but encountered difficulties.
            
            Let me try a simpler approach to the problem:
            
            **Problem**: {problem}
            
            **Analysis**:
            I'll break this down step by step:
            
            1. First, let's understand what we're being asked
            2. Next, I'll consider the key components or variables
            3. Then, I'll work through possible approaches
            4. Finally, I'll provide my best answer based on available information
            
            Sorry for any inconvenience. I'm providing my best response given the constraints.
            """
            
            return False, fallback_message
            
        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            logger.error(f"Sequential thinking timed out after {elapsed_time:.2f}s")
            
            timeout_message = f"""
            I started analyzing this problem using sequential thinking, but it's taking longer than expected.
            
            Let me provide a partial analysis based on what I've considered so far:
            
            **Problem**: {problem}
            
            **Partial Analysis**:
            1. The problem involves {problem.split()[:5]}...
            2. Key considerations include the scope, context, and specific requirements
            3. While I couldn't complete a full analysis in time, here's my current understanding
            
            I apologize for not providing a complete sequential analysis. Please let me know if you'd
            like me to focus on a specific aspect of this problem instead.
            """
            
            return False, timeout_message
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Error in sequential thinking after {elapsed_time:.2f}s: {str(e)}")
            
            error_message = f"""
            I encountered an error while attempting to analyze this problem:
            
            **Problem**: {problem}
            
            **Simple Analysis**:
            Despite the error, let me try to provide a basic response:
            
            1. This appears to be about {problem.split()[:3]}...
            2. Key points to consider would typically include...
            3. Based on general knowledge, I would suggest...
            
            I apologize for the technical difficulties. Please try rephrasing your question
            if you'd like a more detailed analysis.
            """
            
            return False, error_message

    async def _detect_problem_type(self, problem: str) -> str:
        """
        Automatically detect what type of reasoning would be best for the problem.
        
        Args:
            problem: The problem to analyze
            
        Returns:
            str: The recommended reasoning style
        """
        problem_lower = problem.lower()
        
        # Keywords that suggest factual verification would be helpful
        factual_keywords = [
            "fact", "accurate", "truth", "verify", "correct", "accuracy", 
            "precise", "exact", "valid", "check", "confirm", "evidence",
            "proof", "source", "citation", "reference", "statistic", 
            "data", "figure", "number", "percentage", "rate", "date",
            "historical", "scientific", "research", "study", "analysis",
            "report", "survey", "poll", "census", "experiment", "finding",
            "discovery", "publication", "paper", "journal", "article",
            "news", "event", "incident", "occurrence", "phenomenon",
            "development", "trend", "pattern", "movement", "shift"
        ]
        
        # Keywords that suggest complex, multifaceted thinking
        complex_keywords = [
            "complex", "complicated", "intricate", "multifaceted", "nuanced",
            "sophisticated", "advanced", "detailed", "elaborate", "involved",
            "compare", "contrast", "analyze", "different angles", "perspectives",
            "approaches", "alternatives", "options", "possibilities", "considerations",
            "factors", "variables", "parameters", "dimensions", "aspects",
            "elements", "components", "branches", "connections", "relationships",
            "links", "networks", "systems", "structures", "frameworks",
            "models", "theories", "concepts", "ideas", "principles",
            "philosophies", "ideologies", "doctrines", "schools of thought"
        ]
        
        # Check if the problem contains factual keywords
        needs_verification = any(keyword in problem_lower for keyword in factual_keywords)
        
        # Check if the problem is complex
        is_complex = any(keyword in problem_lower for keyword in complex_keywords)
        
        # Check if it's a multi-step problem
        is_multi_step = (
            "step" in problem_lower or
            "process" in problem_lower or
            "procedure" in problem_lower or
            "sequence" in problem_lower or
            "chain" in problem_lower or
            "progression" in problem_lower or
            "workflow" in problem_lower
        )
        
        # Check if it's likely a creative problem
        is_creative = (
            "create" in problem_lower or
            "design" in problem_lower or
            "develop" in problem_lower or
            "innovate" in problem_lower or
            "imagine" in problem_lower or
            "brainstorm" in problem_lower or
            "generate" in problem_lower or
            "invent" in problem_lower or
            "craft" in problem_lower or
            "compose" in problem_lower or
            "author" in problem_lower or
            "write" in problem_lower
        )
        
        # Decision logic for the most appropriate style
        if needs_verification:
            return "cov"  # Chain of Verification for factual problems
        elif is_complex and not is_multi_step:
            return "got"  # Graph of Thought for complex, interconnected problems
        elif is_multi_step and not is_creative:
            return "sequential"  # Sequential for procedural problems
        elif is_creative:
            return "list"  # List for creative ideation
        else:
            return "cot"  # Chain of Thought as the default for general reasoning

    def _parse_sequential_output(self, output: str) -> str:
        """
        Parse sequential thinking output
        
        Args:
            output: Raw LLM output
            
        Returns:
            str: Formatted output for display
        """
        # Process output and format with Markdown for Discord
        formatted_output = ""
        
        # Extract thoughts using regex
        thoughts_pattern = re.compile(r'(?:Thought|THOUGHT|Step|STEP)\s*(\d+)[:\)\.]\s*(.*?)(?=(?:Thought|THOUGHT|Step|STEP)\s*\d+[:\)\.]\s*|Conclusion|CONCLUSION|$)', re.DOTALL)
        conclusion_pattern = re.compile(r'(?:Conclusion|CONCLUSION)[:\)\.]\s*(.*)', re.DOTALL)
        
        # Extract thoughts
        thoughts = thoughts_pattern.findall(output)
        
        # If no thoughts are found with the pattern, just return the original output
        if not thoughts:
            return self.format_response(output)
            
        # Format thoughts
        for num, thought in thoughts:
            formatted_output += f"**Thought {num}**: {thought.strip()}\n\n"
        
        # Extract conclusion
        conclusion_match = conclusion_pattern.search(output)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            formatted_output += f"**Conclusion**: {conclusion}\n"
        else:
            # If no explicit conclusion, use last paragraph as conclusion
            paragraphs = output.split('\n\n')
            if paragraphs:
                last_paragraph = paragraphs[-1].strip()
                if "thought" not in last_paragraph.lower() and "step" not in last_paragraph.lower():
                    formatted_output += f"**Conclusion**: {last_paragraph}\n"
        
        return formatted_output
    
    def _parse_list_output(self, output: str) -> str:
        """
        Parse list-based thinking output
        
        Args:
            output: Raw LLM output
            
        Returns:
            str: Formatted output for display
        """
        # Format with Markdown for Discord
        formatted_output = ""
        
        # Split into lines for processing
        lines = output.split('\n')
        bullet_pattern = re.compile(r'^\s*[-•*]\s*(.*)')
        conclusion_pattern = re.compile(r'(?:Conclusion|CONCLUSION|Summary|SUMMARY)[:\)\.]\s*(.*)', re.DOTALL)
        
        # Process bullet points
        point_number = 1
        in_bullet_section = False
        processed_lines = []
        
        for line in lines:
            bullet_match = bullet_pattern.match(line)
            if bullet_match:
                in_bullet_section = True
                point_text = bullet_match.group(1).strip()
                processed_lines.append(f"**Point {point_number}**: {point_text}")
                point_number += 1
            elif in_bullet_section and line.strip() == "":
                processed_lines.append("")  # Preserve empty lines
            elif in_bullet_section and line.strip():
                # This is continuation text for the previous bullet
                processed_lines[-1] += " " + line.strip()
            else:
                processed_lines.append(line)
        
        # Combine processed lines
        formatted_output = "\n".join(processed_lines)
        
        # Extract conclusion
        conclusion_match = conclusion_pattern.search(output)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            
            # Check if conclusion is already in formatted_output
            if f"**Conclusion**: {conclusion}" not in formatted_output:
                formatted_output += f"\n\n**Conclusion**: {conclusion}"
        
        return formatted_output
    
    def _parse_cot_output(self, output: str) -> str:
        """
        Parse chain-of-thought output
        
        Args:
            output: Raw LLM output
            
        Returns:
            str: Formatted output for display
        """
        # This is similar to sequential output parsing
        return self._parse_sequential_output(output)
    
    def _parse_cov_output(self, output: str) -> str:
        """
        Parse chain-of-verification output
        
        Args:
            output: Raw LLM output
            
        Returns:
            str: Formatted output for display
        """
        # Process output and format with Markdown for Discord
        formatted_output = ""
        
        # Extract steps and verifications using regex patterns
        steps_pattern = re.compile(r'(?:Step|STEP)\s*(\d+)[:\)\.]\s*(.*?)(?=(?:Verification|VERIFICATION|Step|STEP)\s*\d+[:\)\.]\s*|Conclusion|CONCLUSION|$)', re.DOTALL)
        verifications_pattern = re.compile(r'(?:Verification|VERIFICATION)\s*(\d+)[:\)\.]\s*(.*?)(?=(?:Step|STEP|Verification|VERIFICATION)\s*\d+[:\)\.]\s*|Conclusion|CONCLUSION|$)', re.DOTALL)
        conclusion_pattern = re.compile(r'(?:Conclusion|CONCLUSION|Final Answer|FINAL ANSWER)[:\)\.]\s*(.*)', re.DOTALL)
        
        # Extract steps and verifications
        steps = steps_pattern.findall(output)
        verifications = verifications_pattern.findall(output)
        
        # If no structured content is found, fall back to sequential parsing
        if not steps and not verifications:
            return self._parse_sequential_output(output)
        
        # Create a combined ordered list of steps and verifications
        combined_elements = []
        
        # Add steps to the combined list
        for num, content in steps:
            combined_elements.append(("step", int(num), content.strip()))
            
        # Add verifications to the combined list
        for num, content in verifications:
            combined_elements.append(("verification", int(num), content.strip()))
        
        # Sort by number and then type (steps before verifications)
        combined_elements.sort(key=lambda x: (x[1], 0 if x[0] == "step" else 1))
        
        # Format combined elements
        for element_type, num, content in combined_elements:
            if element_type == "step":
                formatted_output += f"**Step {num}**: {content}\n\n"
            else:
                formatted_output += f"**Verification {num}**: {content}\n\n"
        
        # Extract conclusion
        conclusion_match = conclusion_pattern.search(output)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            formatted_output += f"**Final Verified Answer**: {conclusion}\n"
        else:
            # If no explicit conclusion, use last paragraph as conclusion
            paragraphs = output.split('\n\n')
            if paragraphs:
                last_paragraph = paragraphs[-1].strip()
                if "step" not in last_paragraph.lower() and "verification" not in last_paragraph.lower():
                    formatted_output += f"**Final Verified Answer**: {last_paragraph}\n"
        
        return formatted_output
    
    def _parse_got_output(self, output: str) -> str:
        """
        Parse Graph-of-Thought output
        
        Args:
            output: Raw LLM output
            
        Returns:
            str: Formatted output for display
        """
        # Process output and format with Markdown for Discord
        formatted_output = ""
        
        # Extract branches, connections, and key insights using regex patterns
        branch_pattern = re.compile(r'(?:Branch|BRANCH)\s*([A-Z])[:\)\.]\s*(.*?)(?=(?:Branch|BRANCH|Connection|CONNECTION|Key Insight|KEY INSIGHT|Convergence|CONVERGENCE)\s*|Conclusion|CONCLUSION|$)', re.DOTALL)
        connection_pattern = re.compile(r'(?:Connection|CONNECTION)[:\)\.]\s*(.*?)(?=(?:Branch|BRANCH|Connection|CONNECTION|Key Insight|KEY INSIGHT|Convergence|CONVERGENCE)\s*|Conclusion|CONCLUSION|$)', re.DOTALL)
        insight_pattern = re.compile(r'(?:Key Insight|KEY INSIGHT)[:\)\.]\s*(.*?)(?=(?:Branch|BRANCH|Connection|CONNECTION|Key Insight|KEY INSIGHT|Convergence|CONVERGENCE)\s*|Conclusion|CONCLUSION|$)', re.DOTALL)
        convergence_pattern = re.compile(r'(?:Convergence|CONVERGENCE)[:\)\.]\s*(.*?)(?=(?:Branch|BRANCH|Connection|CONNECTION|Key Insight|KEY INSIGHT|Convergence|CONVERGENCE)\s*|Conclusion|CONCLUSION|$)', re.DOTALL)
        conclusion_pattern = re.compile(r'(?:Conclusion|CONCLUSION|Final Answer|FINAL ANSWER)[:\)\.]\s*(.*)', re.DOTALL)
        
        # Extract all elements
        branches = branch_pattern.findall(output)
        connections = connection_pattern.findall(output)
        insights = insight_pattern.findall(output)
        convergences = convergence_pattern.findall(output)
        
        # If no structured content is found, fall back to sequential parsing
        if not branches and not connections and not insights and not convergences:
            return self._parse_sequential_output(output)
        
        # Format branches
        if branches:
            formatted_output += "**Thought Branches:**\n\n"
            for branch_id, content in branches:
                formatted_output += f"**Branch {branch_id}**: {content.strip()}\n\n"
        
        # Format connections
        if connections:
            formatted_output += "**Connections:**\n\n"
            for i, content in enumerate(connections, 1):
                formatted_output += f"**Connection {i}**: {content.strip()}\n\n"
        
        # Format key insights
        if insights:
            formatted_output += "**Key Insights:**\n\n"
            for i, content in enumerate(insights, 1):
                formatted_output += f"**Key Insight {i}**: {content.strip()}\n\n"
        
        # Format convergence points
        if convergences:
            formatted_output += "**Convergence Points:**\n\n"
            for i, content in enumerate(convergences, 1):
                formatted_output += f"**Convergence {i}**: {content.strip()}\n\n"
        
        # Extract conclusion
        conclusion_match = conclusion_pattern.search(output)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
            formatted_output += f"**Conclusion**: {conclusion}\n"
        else:
            # If no explicit conclusion, use last paragraph as conclusion
            paragraphs = output.split('\n\n')
            if paragraphs:
                last_paragraph = paragraphs[-1].strip()
                if not any(term in last_paragraph.lower() for term in ["branch", "connection", "insight", "convergence"]):
                    formatted_output += f"**Conclusion**: {last_paragraph}\n"
        
        return formatted_output
    
    def format_response(self, response: str) -> str:
        """
        Format a sequential thinking response for display
        
        Args:
            response: The raw response text
            
        Returns:
            str: Formatted response for display
        """
        # Simple formatting to make the response more readable
        formatted_response = response
        
        # Use regex to find and format thought patterns
        thought_pattern = re.compile(r'(?:^|\n)(Thought|THOUGHT|Step|STEP)\s*(\d+)[:\)\.]\s*(.*?)(?=(?:\n(?:Thought|THOUGHT|Step|STEP)\s*\d+[:\)\.]\s*)|$)', re.DOTALL)
        
        # Replace thoughts with formatted versions
        def replace_thought(match):
            prefix = match.group(1)  # Thought or THOUGHT or Step or STEP
            number = match.group(2)  # The number
            content = match.group(3).strip()  # The content
            return f"\n**{prefix} {number}**: {content}\n"
            
        formatted_response = thought_pattern.sub(replace_thought, formatted_response)
        
        # Handle conclusions
        conclusion_pattern = re.compile(r'(?:^|\n)(Conclusion|CONCLUSION)[:\)\.]\s*(.*?)(?=$)', re.DOTALL)
        
        def replace_conclusion(match):
            content = match.group(2).strip()
            return f"\n**Conclusion**: {content}\n"
            
        formatted_response = conclusion_pattern.sub(replace_conclusion, formatted_response)
        
        # Make sure there are no excessive newlines
        formatted_response = re.sub(r'\n{3,}', '\n\n', formatted_response)
        
        return formatted_response.strip()

# Factory function to create the sequential thinking instance
def create_sequential_thinking(llm_provider=None):
    """
    Create a sequential thinking instance
    
    Args:
        llm_provider: LLM provider to use
        
    Returns:
        SequentialThinking instance
    """
    return SequentialThinking(llm_provider=llm_provider)


# Example usage in a Discord bot command
"""
Example of how to use SequentialThinking in a Discord bot command:

```python
from bot_utilities.sequential_thinking import create_sequential_thinking

@bot.command()
async def think(ctx, *, problem: str):
    \"\"\"Use sequential thinking to solve complex problems\"\"\"
    
    # Display typing indicator to show the bot is working
    async with ctx.typing():
        # Create sequential thinking instance using your LLM provider
        # Replace 'your_llm_provider' with your actual LLM instance
        seq_thinking = create_sequential_thinking(llm_provider=your_llm_provider)
        
        # Run sequential thinking on the problem
        success, response = await seq_thinking.run(
            problem=problem,
            # Optional parameters:
            # context={"additional_info": "context"},
            # prompt_style="sequential",  # or "list" or "cot"
            # num_thoughts=5,
            # temperature=0.2,
            # max_tokens=2000
        )
        
        if success:
            # Split response if too long for Discord
            if len(response) > 1990:
                chunks = [response[i:i+1990] for i in range(0, len(response), 1990)]
                for chunk in chunks:
                    await ctx.send(chunk)
            else:
                await ctx.send(response)
        else:
            await ctx.send("I had trouble thinking through that problem. Could you try rephrasing it?")
```

For use with the ReAct agent pattern (LlamaIndex):

```python
from typing import List, Dict, Any
from llama_index.core.tools import BaseTool, FunctionTool

# Create SequentialThinking as a ReAct tool
def create_sequential_thinking_tool(llm_provider=None):
    \"\"\"Creates a sequential thinking tool for ReAct agents\"\"\"
    
    seq_thinking = create_sequential_thinking(llm_provider=llm_provider)
    
    async def sequential_thinking_func(problem: str, context: Dict[str, Any] = None) -> str:
        \"\"\"
        Solves complex problems using sequential thinking
        
        Args:
            problem: The problem to solve
            context: Optional additional context
            
        Returns:
            str: The step-by-step solution
        \"\"\"
        success, response = await seq_thinking.run(problem=problem, context=context)
        if success:
            return response
        else:
            return f"Failed to solve problem: {problem}"
            
    return FunctionTool.from_defaults(
        name="sequential_thinking",
        description="Solve complex problems by breaking them down into sequential steps",
        fn=sequential_thinking_func
    )
```
""" 