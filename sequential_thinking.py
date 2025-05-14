import asyncio
import json
import os
import re
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union

# Import AI provider for accessing the language model
from bot_utilities.ai_utils import get_ai_response, get_ai_provider

class SequentialThinking:
    """
    A class for implementing sequential thinking reasoning, with enhanced
    capabilities for thought revision, self-reflection, and context-aware reasoning.
    """
    
    def __init__(self, llm_provider=None):
        """
        Initialize the sequential thinking system
        
        Args:
            llm_provider: Optional language model provider to use
        """
        self.llm_provider = llm_provider
        self.chain_result = ""
        self.thought_history = {}  # Store thought history by session_id
        self.reflection_history = {}  # Store reflection history by session_id
        self.revision_history = {}  # Store revision history by session_id
        
    async def set_llm_provider(self, provider):
        """Set the language model provider"""
        self.llm_provider = provider
        
    async def run(self, 
                  problem: str, 
                  context: Dict[str, Any] = None, 
                  prompt_style: str = "sequential", 
                  num_thoughts: int = 5, 
                  temperature: float = 0.3,
                  enable_revision: bool = False,
                  enable_reflection: bool = False,
                  session_id: str = None) -> Tuple[bool, str]:
        """
        Run the sequential thinking process on a problem
        
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
        if not self.llm_provider:
            try:
                self.llm_provider = await get_ai_provider()
            except Exception as e:
                print(f"Error getting AI provider: {e}")
                return False, f"Failed to initialize AI provider: {str(e)}"
                
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
            model_response = await self.llm_provider.async_call(prompt, temperature=temperature)
            
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
                revision_response = await self.llm_provider.async_call(revision_prompt, temperature=temperature-0.1)
                
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
                reflection_response = await self.llm_provider.async_call(reflection_prompt, temperature=temperature)
                
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
                fallback_response = await self.llm_provider.async_call(fallback_prompt, temperature=0.3)
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
            context_str = "\n\nAdditional context:\n"
            for key, value in context.items():
                context_str += f"{key}: {value}\n"
            prompt += context_str
        
        # Add format instructions
        prompt += f"""
        Please generate exactly {num_thoughts} sequential thoughts, followed by a final answer.
        
        Format your response as:
        
        Thought 1: [Your first thought]
        Thought 2: [Your second thought]
        ...
        Thought {num_thoughts}: [Your final thought]
        
        Final Answer: [Your comprehensive answer based on the sequential thinking process]
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
        prompt = f"""
        You are using Chain-of-Thought reasoning to solve a problem.
        
        Guidelines for Chain-of-Thought reasoning:
        1. Break down your reasoning process into logical steps
        2. Each step should build on previous steps
        3. Link your thoughts in a clear causal chain
        4. Identify connections between ideas
        5. Arrive at a well-reasoned conclusion
        
        Problem to solve:
        {problem}
        """
        
        # Add context if provided
        if context:
            context_str = "\n\nAdditional context:\n"
            for key, value in context.items():
                context_str += f"{key}: {value}\n"
            prompt += context_str
        
        # Add format instructions
        prompt += """
        Format your response as:
        
        Reasoning:
        [Your step-by-step chain of thought reasoning]
        
        Conclusion:
        [Your final answer based on the chain of thought]
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
            num_nodes: Number of conceptual nodes to generate
            
        Returns:
            str: The formatted prompt
        """
        prompt = f"""
        You are using Graph-of-Thought reasoning to explore a complex problem with interconnected concepts.
        
        Guidelines for Graph-of-Thought reasoning:
        1. Identify key concepts related to the problem (nodes)
        2. Explore relationships between these concepts (edges)
        3. Map out a conceptual network around the problem
        4. Consider non-linear connections between ideas
        5. Synthesize insights from the conceptual graph
        
        Problem to explore:
        {problem}
        """
        
        # Add context if provided
        if context:
            context_str = "\n\nAdditional context:\n"
            for key, value in context.items():
                context_str += f"{key}: {value}\n"
            prompt += context_str
        
        # Add format instructions
        prompt += f"""
        Format your response as:
        
        Node 1 (Concept): [First key concept]
        Related to: [Which other nodes connect to this one, e.g., "Node 2, Node 3"]
        Analysis: [Exploration of this concept]
        
        Node 2 (Concept): [Second key concept]
        Related to: [Which other nodes connect to this one]
        Analysis: [Exploration of this concept]
        
        ... (Continue for {num_nodes} nodes)
        
        Synthesis:
        [Synthesize insights from the graph of connected concepts into a comprehensive answer]
        """
        
        return prompt
        
    def _create_chain_of_verification_prompt(self, problem: str, context: Dict[str, Any] = None) -> str:
        """
        Create a chain-of-verification prompt to reduce hallucinations
        
        Args:
            problem: The problem to verify
            context: Additional context
            
        Returns:
            str: The formatted prompt
        """
        prompt = f"""
        You are using Chain-of-Verification reasoning to analyze a problem and ensure factual accuracy.
        
        Guidelines for Chain-of-Verification:
        1. Break down the problem into verifiable components
        2. For each component, assess your confidence in the facts
        3. Identify any assumptions or uncertainties
        4. Verify or correct potential misunderstandings
        5. Provide a final verified response
        
        Problem to verify:
        {problem}
        """
        
        # Add context if provided
        if context:
            context_str = "\n\nAdditional context:\n"
            for key, value in context.items():
                context_str += f"{key}: {value}\n"
            prompt += context_str
        
        # Add format instructions
        prompt += """
        Format your response as:
        
        Claim 1: [First factual claim related to the problem]
        Verification: [Verify this claim's accuracy, note uncertainties]
        Confidence: [High/Medium/Low]
        
        Claim 2: [Second factual claim]
        Verification: [Verify this claim's accuracy, note uncertainties]
        Confidence: [High/Medium/Low]
        
        (Continue for all relevant claims)
        
        Verified Response:
        [Provide a response to the original problem that incorporates only verified information]
        
        Uncertainty Note:
        [Note any areas where information is incomplete or uncertain]
        """
        
        return prompt
        
    def _create_step_back_prompt(self, problem: str, context: Dict[str, Any] = None) -> str:
        """
        Create a step-back prompt for higher-level reasoning
        
        Args:
            problem: The problem to analyze
            context: Additional context
            
        Returns:
            str: The formatted prompt
        """
        prompt = f"""
        You are using Step-Back reasoning to analyze a problem from a broader perspective before diving into details.
        
        Guidelines for Step-Back reasoning:
        1. First, take a step back and consider the problem from a high level
        2. Identify the broader context and underlying principles
        3. Consider what domain knowledge is relevant
        4. Reframe the problem if needed
        5. Then approach the specific problem with this broader perspective
        
        Problem to analyze:
        {problem}
        """
        
        # Add context if provided
        if context:
            context_str = "\n\nAdditional context:\n"
            for key, value in context.items():
                context_str += f"{key}: {value}\n"
            prompt += context_str
        
        # Add format instructions
        prompt += """
        Format your response as:
        
        Broader Perspective:
        [Analysis of the higher-level context, principles, and domain]
        
        Reframed Problem:
        [The problem reframed with the broader perspective in mind]
        
        Detailed Analysis:
        [Analysis of the specific problem with the broader context in mind]
        
        Conclusion:
        [Final answer informed by the step-back analysis]
        """
        
        return prompt
        
    def _create_revision_prompt(self, problem: str, thoughts: str, context: Dict[str, Any] = None) -> str:
        """
        Create a prompt for revising thoughts
        
        Args:
            problem: The original problem
            thoughts: The original thoughts to revise
            context: Additional context
            
        Returns:
            str: The formatted revision prompt
        """
        prompt = f"""
        You will now review and potentially revise a set of sequential thoughts about a problem.
        
        Original problem:
        {problem}
        
        Original thoughts:
        {thoughts}
        
        Guidelines for thought revision:
        1. Carefully review each thought for accuracy, clarity, and logical consistency
        2. Identify any errors, flawed assumptions, or logical gaps
        3. Suggest improved versions of thoughts that need revision
        4. Explain why each revision is necessary
        5. Leave correct and well-reasoned thoughts unchanged
        
        Format your response as:
        
        Thought 1 Review:
        [If revision needed]: Original thought had issues with [specific issues]. Revised thought: [improved thought]
        [If no revision needed]: This thought is sound and does not need revision.
        
        Thought 2 Review:
        (etc. for each thought)
        
        Final Assessment:
        [Brief overview of the revisions and the improved reasoning]
        """
        
        # Add context if provided
        if context:
            context_str = "\n\nAdditional context to consider:\n"
            for key, value in context.items():
                context_str += f"{key}: {value}\n"
            prompt += context_str
            
        return prompt
        
    def _create_reflection_prompt(self, problem: str, thoughts: str, context: Dict[str, Any] = None) -> str:
        """
        Create a prompt for self-reflection on the thinking process
        
        Args:
            problem: The original problem
            thoughts: The thoughts to reflect on
            context: Additional context
            
        Returns:
            str: The formatted reflection prompt
        """
        prompt = f"""
        Now that you've completed your thinking process about the problem, please reflect on your reasoning approach.
        
        Original problem:
        {problem}
        
        Your thinking process:
        {thoughts}
        
        Guidelines for reflection:
        1. Identify strengths in your reasoning approach
        2. Identify any weaknesses, limitations, or biases
        3. Consider alternative approaches that might have been valuable
        4. Assess confidence in the final answer
        5. Suggest ways to improve the reasoning process
        
        Provide a concise reflection on your thinking process.
        """
        
        return prompt
        
    def _process_thinking_response(self, response: str, prompt_style: str) -> str:
        """
        Process the raw thinking response based on the prompt style
        
        Args:
            response: The raw response from the LLM
            prompt_style: The style of prompt used
            
        Returns:
            str: The processed response
        """
        # Clean up the response (remove extra whitespace, etc.)
        response = response.strip()
        
        # Process based on prompt style
        if prompt_style == "sequential":
            # Format sequential thinking - already well formatted
            return response
            
        elif prompt_style == "cot":
            # Format Chain of Thought response
            # Already well formatted
            return response
            
        elif prompt_style == "got":
            # Format Graph of Thought response
            # Add some markdown formatting to make the node structure clearer
            nodes = re.split(r'Node \d+ \(Concept\):', response)
            if len(nodes) > 1:
                formatted_response = ""
                for i, node in enumerate(nodes[1:], 1):
                    formatted_response += f"**Node {i} (Concept):**{node}\n\n"
                    
                # Extract and format synthesis separately
                synthesis_match = re.search(r'Synthesis:(.*?)($|$)', response, re.DOTALL)
                if synthesis_match:
                    synthesis = synthesis_match.group(1).strip()
                    formatted_response += f"**Synthesis:**\n{synthesis}"
                    
                return formatted_response
            
            return response
            
        elif prompt_style == "cov":
            # Format Chain of Verification response
            # Add some formatting to highlight confidence levels
            claims = re.split(r'Claim \d+:', response)
            if len(claims) > 1:
                formatted_response = ""
                for i, claim in enumerate(claims[1:], 1):
                    # Highlight confidence levels
                    claim = re.sub(r'Confidence: (High|Medium|Low)', r'Confidence: **\1**', claim)
                    formatted_response += f"**Claim {i}:**{claim}\n\n"
                    
                return formatted_response
            
            return response
            
        elif prompt_style == "step_back":
            # Format Step Back response
            # Add some markdown formatting
            sections = [
                ("Broader Perspective:", "**Broader Perspective:**"),
                ("Reframed Problem:", "**Reframed Problem:**"),
                ("Detailed Analysis:", "**Detailed Analysis:**"),
                ("Conclusion:", "**Conclusion:**")
            ]
            
            formatted_response = response
            for original, formatted in sections:
                formatted_response = formatted_response.replace(original, formatted)
                
            return formatted_response
            
        else:
            # Default formatting
            return response
            
    def _combine_thoughts_with_revisions(self, original_thoughts: str, revisions: str) -> str:
        """
        Combine original thoughts with revisions
        
        Args:
            original_thoughts: The original thoughts
            revisions: The revisions
            
        Returns:
            str: Combined response with revisions highlighted
        """
        # Extract thought review sections
        review_pattern = r'Thought (\d+) Review:(.*?)(?=Thought \d+ Review:|Final Assessment:|$)'
        review_matches = re.findall(review_pattern, revisions, re.DOTALL)
        
        # If no matches found, return combined with explanation
        if not review_matches:
            return f"{original_thoughts}\n\n**Revisions:**\n{revisions}"
        
        # Get the original thoughts
        thought_pattern = r'Thought (\d+): (.*?)(?=Thought \d+:|Final Answer:|$)'
        thought_matches = re.findall(thought_pattern, original_thoughts, re.DOTALL)
        
        # Create a dictionary of original thoughts
        original_thought_dict = {num.strip(): content.strip() for num, content in thought_matches}
        
        # Create a dictionary of revised thoughts
        revised_thought_dict = {}
        for num, review in review_matches:
            num = num.strip()
            review = review.strip()
            
            # Check if a revision was suggested
            if "Revised thought:" in review:
                # Extract the revised thought
                revised_match = re.search(r'Revised thought: (.*?)$', review, re.DOTALL)
                if revised_match:
                    revised_thought = revised_match.group(1).strip()
                    revised_thought_dict[num] = revised_thought
        
        # Combine original thoughts with revisions
        result = ""
        for num, content in thought_matches:
            num = num.strip()
            result += f"Thought {num}: {original_thought_dict[num]}\n\n"
            
            # Add revision if there is one
            if num in revised_thought_dict:
                result += f"**Revised Thought {num}**: {revised_thought_dict[num]}\n\n"
        
        # Add the final answer
        final_answer_match = re.search(r'Final Answer: (.*?)$', original_thoughts, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            result += f"Final Answer: {final_answer}\n"
        
        return result

# Create a singleton instance
def create_sequential_thinking(llm_provider=None):
    return SequentialThinking(llm_provider) 