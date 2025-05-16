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
import time

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
        self.session_thoughts = {}  # Store thoughts for each session

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

    async def process_sequential_thinking(
        self,
        problem: str,
        context: Dict[str, Any] = None,
        prompt_style: str = "sequential",
        enable_revision: bool = True,
        session_id: str = None,
        max_steps: int = 7,
    ) -> Tuple[bool, str]:
        """
        Process a problem using sequential thinking

        Args:
            problem: The problem to solve
            context: Additional context for the problem
            prompt_style: Style of prompt to use (sequential, step_by_step, etc.)
            enable_revision: Whether to enable thought revision
            session_id: Unique identifier for this thinking session
            max_steps: Maximum number of thinking steps

        Returns:
            Tuple[bool, str]: Success flag and formatted thinking result
        """
        if not session_id:
            session_id = f"seq_{int(time.time())}"

        if not context:
            context = {}
            
        # Add interleaved_format flag to context to ensure revisions appear after each thought
        context["interleaved_format"] = True

        # Track thoughts for this session
        if session_id not in self.session_thoughts:
            self.session_thoughts[session_id] = []

        # Create system instruction
        system_instruction = self._create_system_instruction(
            prompt_style, enable_revision
        )

        # Create user prompt
        user_prompt = f"""Problem to solve: {problem}
        
Additional context: {json.dumps(context) if context else "None"}

Think through this step-by-step, with each thought building on the previous ones.
You MUST actively look for opportunities to revise earlier thoughts when you gain new insights.
For each sequential thinking session, include at least 1-2 revisions of earlier thoughts.

Include working memory to track key information from your thoughts.
Evaluate different options or approaches before reaching a conclusion.
Provide a final conclusion based on your step-by-step analysis.

Important instructions for thought revision:
1. After every 2-3 thoughts, look back at your earlier thinking
2. Identify at least one thought that could be improved, refined, or corrected
3. Create a revision by clearly stating "Revision of Thought X" (where X is the thought number)
4. Explain what you're changing and why
5. Provide the improved thought

Remember: Thought revision is REQUIRED, not optional. It demonstrates how your understanding evolves.
"""

        try:
            # Get AI provider
            from ..ai_utils import get_ai_provider

            ai_provider = await get_ai_provider()

            # Prepare messages for the AI
            messages = [{"role": "system", "content": system_instruction}]

            # Add existing thoughts as context if continuing a session
            if self.session_thoughts.get(session_id):
                thoughts_context = "\n\n".join(
                    [
                        f"Thought {t.get('thoughtNumber')}: {t.get('thought')}"
                        for t in self.session_thoughts[session_id]
                    ]
                )
                messages.append({"role": "assistant", "content": thoughts_context})

            # Add the problem as user message
            messages.append({"role": "user", "content": user_prompt})

            # Generate thinking steps from the AI
            raw_response = await ai_provider.generate_text(
                messages=messages, temperature=0.5, max_tokens=3000
            )

            # Process the raw response into structured thoughts
            thoughts = self._parse_thinking_response(raw_response, max_steps)

            # Store thoughts in session
            if thoughts:
                for thought in thoughts:
                    if thought not in self.session_thoughts[session_id]:
                        self.session_thoughts[session_id].append(thought)

            # Format the full thinking process
            formatted_output = self.format_sequential_thinking(
                thoughts=self.session_thoughts[session_id],
                problem=problem,
                include_prompt=True,
                format_style="markdown",
            )

            return True, formatted_output

        except Exception as e:
            print(f"Error in sequential thinking: {str(e)}")
            traceback.print_exc()
            return False, f"Error in sequential thinking process: {str(e)}"

    def _parse_thinking_response(
        self, response: str, max_steps: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Parse the LLM response into structured thinking steps

        Args:
            response: Raw LLM response
            max_steps: Maximum number of steps to parse

        Returns:
            List of thought step dictionaries
        """
        thoughts = []
        current_thought = {}
        thought_number = 1

        # Simple regex parsing for thought structure
        thought_pattern = r"(## 🔄 \*\*Thought (\d+)\*\*|Thought (\d+):|🔄 \*\*Thought (\d+)\*\*|Step (\d+):|Thought (\d+)[\.:])"
        revision_pattern = r"(## 🔄 \*\*Revision of Thought (\d+)\*\*|Revision of Thought (\d+):|🔄 \*\*Revision of Thought (\d+)\*\*|REVISION OF THOUGHT (\d+))"
        review_pattern = r"(## 🔍 \*\*Review of Thought (\d+)\*\*|Review of Thought (\d+):|🔍 \*\*Review of Thought (\d+)\*\*)"
        conclusion_pattern = r"(## ✅ \*\*Conclusion\*\*|✅ Conclusion:|Final Answer:)"

        # Split the response by lines for processing
        lines = response.split("\n")
        current_section = None
        section_content = []
        revises_thought_num = None

        for line in lines:
            # Check for thought markers
            thought_match = re.search(thought_pattern, line, re.IGNORECASE)
            revision_match = re.search(revision_pattern, line, re.IGNORECASE)
            review_match = re.search(review_pattern, line, re.IGNORECASE)
            conclusion_match = re.search(conclusion_pattern, line, re.IGNORECASE)

            # Process section transitions
            if thought_match:
                # Save previous section if exists
                if current_section and section_content:
                    content = "\n".join(section_content).strip()
                    if current_section == "thought":
                        thoughts.append(
                            {
                                "thought": content,
                                "thoughtNumber": thought_number,
                                "nextThoughtNeeded": thought_number < max_steps,
                                "totalThoughts": max_steps,
                                "isRevision": False,
                                "revisesThought": None,
                            }
                        )
                        thought_number += 1
                    elif current_section == "revision":
                        thoughts.append(
                            {
                                "thought": content,
                                "thoughtNumber": thought_number,
                                "nextThoughtNeeded": thought_number < max_steps,
                                "totalThoughts": max_steps,
                                "isRevision": True,
                                "revisesThought": revises_thought_num,
                            }
                        )
                        thought_number += 1
                    elif current_section == "review":
                        # Add review to the most recent thought
                        if thoughts:
                            thoughts[-1]["review"] = content
                    elif current_section == "conclusion":
                        thoughts.append(
                            {
                                "thought": f"✅ Conclusion: {content}",
                                "thoughtNumber": thought_number,
                                "nextThoughtNeeded": False,
                                "totalThoughts": max_steps,
                                "isRevision": False,
                                "revisesThought": None,
                            }
                        )
                    
                # Reset for new section
                current_section = "thought"
                section_content = []
                
                # Try to extract thought number
                for group_idx in range(2, 7):
                    if thought_match.group(group_idx) and thought_match.group(group_idx).isdigit():
                        thought_number = int(thought_match.group(group_idx))
                        break
                
                # Remove the marker from the current line
                line = re.sub(thought_pattern, "", line).strip()
                
            elif revision_match:
                # Save previous section if exists
                if current_section and section_content:
                    content = "\n".join(section_content).strip()
                    if current_section == "thought":
                        thoughts.append(
                            {
                                "thought": content,
                                "thoughtNumber": thought_number,
                                "nextThoughtNeeded": thought_number < max_steps,
                                "totalThoughts": max_steps,
                                "isRevision": False,
                                "revisesThought": None,
                            }
                        )
                        thought_number += 1
                    
                # Reset for new section
                current_section = "revision"
                section_content = []
                
                # Try to extract which thought is being revised
                for group_idx in range(2, 6):
                    if revision_match.group(group_idx) and revision_match.group(group_idx).isdigit():
                        revises_thought_num = int(revision_match.group(group_idx))
                        break
                
                # Remove the marker from the current line
                line = re.sub(revision_pattern, "", line).strip()
                
            elif review_match:
                # Save previous section if exists
                if current_section and section_content:
                    content = "\n".join(section_content).strip()
                    if current_section == "thought":
                        thoughts.append(
                            {
                                "thought": content,
                                "thoughtNumber": thought_number,
                                "nextThoughtNeeded": thought_number < max_steps,
                                "totalThoughts": max_steps,
                                "isRevision": False,
                                "revisesThought": None,
                            }
                        )
                        thought_number += 1
                    
                # Reset for new section
                current_section = "review"
                section_content = []
                
                # Remove the marker from the current line
                line = re.sub(review_pattern, "", line).strip()
                
            elif conclusion_match:
                # Save previous section if exists
                if current_section and section_content:
                    content = "\n".join(section_content).strip()
                    if current_section == "thought":
                        thoughts.append(
                            {
                                "thought": content,
                                "thoughtNumber": thought_number,
                                "nextThoughtNeeded": thought_number < max_steps,
                                "totalThoughts": max_steps,
                                "isRevision": False,
                                "revisesThought": None,
                            }
                        )
                        thought_number += 1
                    elif current_section == "revision":
                        thoughts.append(
                            {
                                "thought": content,
                                "thoughtNumber": thought_number,
                                "nextThoughtNeeded": thought_number < max_steps,
                                "totalThoughts": max_steps,
                                "isRevision": True,
                                "revisesThought": revises_thought_num,
                            }
                        )
                        thought_number += 1
                
                # Reset for new section
                current_section = "conclusion"
                section_content = []
                
                # Remove the marker from the current line
                line = re.sub(conclusion_pattern, "", line).strip()
            
            # Add the processed line to the current section content
            if line.strip():
                section_content.append(line)
        
        # Handle the last section
        if current_section and section_content:
            content = "\n".join(section_content).strip()
            if current_section == "thought":
                thoughts.append(
                    {
                        "thought": content,
                        "thoughtNumber": thought_number,
                        "nextThoughtNeeded": thought_number < max_steps,
                        "totalThoughts": max_steps,
                        "isRevision": False,
                        "revisesThought": None,
                    }
                )
            elif current_section == "revision":
                thoughts.append(
                    {
                        "thought": content,
                        "thoughtNumber": thought_number,
                        "nextThoughtNeeded": thought_number < max_steps,
                        "totalThoughts": max_steps,
                        "isRevision": True,
                        "revisesThought": revises_thought_num,
                    }
                )
            elif current_section == "review":
                # Add review to the most recent thought
                if thoughts:
                    thoughts[-1]["review"] = content
            elif current_section == "conclusion":
                thoughts.append(
                    {
                        "thought": f"✅ Conclusion: {content}",
                        "thoughtNumber": thought_number,
                        "nextThoughtNeeded": False,
                        "totalThoughts": max_steps,
                        "isRevision": False,
                        "revisesThought": None,
                    }
                )
        
        # Now check the thoughts for any revisions embedded within them
        for i, thought in enumerate(thoughts):
            if not thought.get("isRevision", False):
                thought_content = thought.get("thought", "")
                
                # Look for "Revision of Thought X" patterns within the content
                embedded_revision_match = re.search(r"Revision of Thought (\d+):", thought_content, re.IGNORECASE)
                if embedded_revision_match:
                    # Extract the revision target
                    try:
                        revises_num = int(embedded_revision_match.group(1))
                        
                        # Mark this thought as a revision
                        thought["isRevision"] = True
                        thought["revisesThought"] = revises_num
                        
                        # Explicitly add a tag to the beginning for clarity
                        if not thought_content.startswith("🔄 REVISION"):
                            thought["thought"] = f"🔄 REVISION OF THOUGHT {revises_num}: {thought_content}"
                    except:
                        pass
        
        return thoughts

    def _create_system_instruction(
        self, prompt_style: str, enable_revision: bool
    ) -> str:
        """
        Create system instruction for sequential thinking

        Args:
            prompt_style: Style of prompt to use
            enable_revision: Whether to enable thought revision

        Returns:
            System instruction string
        """
        base_instruction = """You are an expert in sequential thinking that breaks down problems step-by-step.
Follow these guidelines:
1. Analyze the problem thoroughly, breaking it into logical steps
2. Each thought should be clear, detailed, and focused on one aspect
3. Keep track of key information in working memory
4. Look for opportunities to improve or revise earlier thoughts
5. Evaluate options and alternatives before concluding
6. Finish with a well-reasoned conclusion

IMPORTANT: You MUST explicitly revise earlier thoughts when you gain new insights.
This is a REQUIRED part of the process, not optional.
For EACH sequential thinking session, include at least 1-2 revisions of earlier thoughts.

When revising a thought:
1. Clearly label it as "Revision of Thought X" (where X is the thought number)
2. Explicitly state what you're changing about the original thought
3. Explain why this revision is necessary based on new insights
4. Provide the corrected or enhanced thought in full

For example:
Thought 1: Initial understanding of the problem
Thought 2: Analysis of key factors
Thought 3: Exploring potential solutions
**Revision of Thought 1**: Now I realize my initial understanding was incorrect because...
Thought 4: Additional considerations
Working Memory: Key points to remember from all thoughts
Evaluation: Assessment of different options
Conclusion: Final answer based on the complete analysis

The revision process is CRITICAL - it shows how your understanding evolves. Always revise when:
- You discover contradictory information
- You realize you missed something important 
- You need to refine an earlier assumption
- New information changes your perspective
- You develop a deeper understanding that affects earlier thoughts

Format your response with clear section headers:
- 🔄 **Thought X**: (for each sequential thought)
- 🔄 **Revision of Thought X**: (for revisions - make these STAND OUT)
- 📝 **Working Memory**: (for tracking key information)
- ⚖️ **Evaluation**: (for assessing options)
- ✅ **Conclusion**: (for your final answer)
"""

        if enable_revision:
            revision_instructions = """You MUST actively look for opportunities to revise your earlier thoughts. 
This is not optional - each thinking session should include at least 1-2 thought revisions.
When you gain new insights or information, go back and explicitly revise previous thoughts.
Make your revisions stand out by clearly labeling them as "Revision of Thought X".
This reflection and revision is a critical part of high-quality thinking.

DO NOT simply move on to new thoughts without ever revising earlier ones.
If you realize something from an earlier thought needs refinement, explicitly revise it."""
            base_instruction += "\n\n" + revision_instructions

        return base_instruction

    def _create_thinking_prompt(
        self,
        problem: str,
        thoughts: list,
        step: int,
        enable_revision: bool,
        context: dict = None,
    ) -> str:
        """
        Create prompt for the next thinking step

        Args:
            problem: Problem to solve
            thoughts: Previous thoughts
            step: Current step number
            enable_revision: Whether revision is enabled
            context: Additional context

        Returns:
            Prompt string
        """
        prompt = f"Problem: {problem}\n\n"

        # Add context if provided
        if context:
            ctx_str = "\n".join(
                [
                    f"{k}: {v}"
                    for k, v in context.items()
                    if k != "user_id" and k != "conversation_id"
                ]
            )
            if ctx_str:
                prompt += f"Context:\n{ctx_str}\n\n"

        # Add previous thoughts
        if thoughts:
            prompt += "Previous thoughts:\n"
            for t in thoughts:
                if t.get("isRevision", False):
                    revises = t.get("revisesThought", 0)
                    prompt += f"Thought {t.get('thoughtNumber', '?')} (REVISION of Thought {revises}): {t.get('thought', '')}\n\n"
                else:
                    prompt += f"Thought {t.get('thoughtNumber', '?')}: {t.get('thought', '')}\n\n"

        # Add current step instruction
        if step == 1:
            prompt += (
                "Begin by understanding the problem and breaking it down into parts."
            )
        elif enable_revision and step > 2 and step % 3 == 0:
            prompt += "Consider if any previous thoughts need revision based on what you know now."
        elif step == len(thoughts) + 1:
            prompt += "Continue your sequential thinking process. Consider adding more depth or exploring a new aspect."
        else:
            prompt += (
                "Continue your sequential thinking process with the next logical step."
            )

        # Format as JSON request
        prompt += "\n\nRespond with your next thought in JSON format with fields: thought, thoughtNumber, nextThoughtNeeded, isRevision, and revisesThought (if applicable)."

        return prompt

    def _create_working_memory_prompt(
        self, problem: str, thoughts: list, context: dict = None
    ) -> str:
        """
        Create prompt for working memory synthesis

        Args:
            problem: Problem to solve
            thoughts: Previous thoughts
            context: Additional context

        Returns:
            Prompt string
        """
        prompt = f"Problem: {problem}\n\n"

        # Add context if provided
        if context:
            ctx_str = "\n".join(
                [
                    f"{k}: {v}"
                    for k, v in context.items()
                    if k != "user_id" and k != "conversation_id"
                ]
            )
            if ctx_str:
                prompt += f"Context:\n{ctx_str}\n\n"

        # Add previous thoughts in summary form
        if thoughts:
            prompt += "Summary of thoughts so far:\n"
            for t in thoughts[-3:]:  # Just summarize the most recent thoughts
                if not t.get("isRevision", False) and "Working Memory" not in t.get(
                    "thought", ""
                ):
                    prompt += f"- {t.get('thought', '')[:100]}...\n"

        prompt += "\n\nCreate a concise working memory that captures the key points from your thinking so far. What are the crucial elements to remember? What insights will be useful for upcoming thinking steps?"

        return prompt

    def _create_revision_prompt(
        self, problem: str, thoughts: list, context: dict = None
    ) -> str:
        """
        Create prompt for checking if revisions are needed

        Args:
            problem: Problem to solve
            thoughts: Previous thoughts
            context: Additional context

        Returns:
            Prompt string
        """
        prompt = f"Problem: {problem}\n\n"

        # Add previous thoughts with emphasis on reviewing them
        if thoughts:
            prompt += "Review your previous thoughts carefully:\n"
            for i, t in enumerate(thoughts):
                if not t.get("isRevision", False) and "Working Memory" not in t.get(
                    "thought", ""
                ):
                    prompt += f"Thought {t.get('thoughtNumber', i+1)}: {t.get('thought', '')}\n\n"

        prompt += (
            "\n\nDo any of your previous thoughts need revision based on what you know now? Look for:"
            "\n- Logical errors or inconsistencies"
            "\n- Incomplete analysis that can now be expanded"
            "\n- Incorrect assumptions revealed by later thinking"
            "\n- Missed connections between ideas"
            "\n\nIf a revision is needed, respond with a JSON object including 'isRevision: true' and 'revisesThought: N' where N is the thought number being revised. Otherwise, respond with 'isRevision: false'."
        )

        return prompt

    def _create_evaluation_prompt(
        self, problem: str, thoughts: list, context: dict = None
    ) -> str:
        """
        Create prompt for evaluation step

        Args:
            problem: Problem to solve
            thoughts: Previous thoughts
            context: Additional context

        Returns:
            Prompt string
        """
        prompt = f"Problem: {problem}\n\n"

        # Add a summary of key thoughts
        relevant_thoughts = [
            t
            for t in thoughts
            if not t.get("isRevision", False)
            and "Working Memory" not in t.get("thought", "")
        ]
        if relevant_thoughts:
            prompt += "Key points from your thinking:\n"
            for i, t in enumerate(relevant_thoughts[-min(len(relevant_thoughts), 3) :]):
                prompt += f"- {t.get('thought', '')[:150]}...\n"

        prompt += "\n\nNow evaluate the different options, approaches, or solutions you've considered. Compare their strengths and weaknesses. What criteria should be used to assess them? Which approach seems most promising and why?"

        return prompt

    def _create_conclusion_prompt(
        self, problem: str, thoughts: list, context: dict = None
    ) -> str:
        """
        Create prompt for conclusion step

        Args:
            problem: Problem to solve
            thoughts: Previous thoughts
            context: Additional context

        Returns:
            Prompt string
        """
        prompt = f"Problem: {problem}\n\n"

        # Add a summary of key thoughts including working memory and evaluation if available
        working_memory = next(
            (
                t.get("thought", "")
                for t in thoughts
                if "Working Memory" in t.get("thought", "")
            ),
            None,
        )
        evaluation = next(
            (
                t.get("thought", "")
                for t in thoughts
                if "Evaluation" in t.get("thought", "")
            ),
            None,
        )

        if working_memory:
            prompt += f"Working Memory:\n{working_memory}\n\n"

        if evaluation:
            prompt += f"Evaluation:\n{evaluation}\n\n"

        prompt += "Based on your step-by-step thinking process, provide a comprehensive conclusion that addresses the original problem. Make sure your conclusion is well-reasoned, clear, and draws from the insights in your sequential thinking process."

        return prompt

    def _create_sequential_thinking_prompt(
        self,
        problem: str,
        context: Dict[str, Any] = None,
        num_thoughts: int = 5,
        enable_revision: bool = False,
    ) -> str:
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
        # Check for interleaved_revisions flag in context
        interleaved_format = True  # Default to true for better revision visibility
        if context and "interleaved_format" in context:
            interleaved_format = context["interleaved_format"]

        revision_instructions = ""
        if enable_revision:
            if interleaved_format:
                revision_instructions = """
                CRITICAL INSTRUCTION: After each thought, you MUST immediately review and consider revising that thought before moving to the next one.
                
                For each thought, follow this exact process:
                1. First, express the thought clearly and thoroughly
                2. Then explicitly review that thought, looking for errors, inconsistencies, or improvements
                3. If needed (and you should need to revise at least 1-2 thoughts), provide a clear revision
                
                Format EXACTLY as follows:
                🔄 **Thought 1**: [Your first thought in detail]
                
                🔍 **Review of Thought 1**:
                [Critically analyze your thought for weaknesses, assumptions, or areas of improvement]
                
                🔄 **Revision of Thought 1**:
                [If revision is needed, provide the improved version with explanations of what changed and why]
                
                Only after completing this full cycle should you proceed to Thought 2.
                
                Remember:
                - At least 1-2 thoughts MUST be revised during your thinking process
                - Make each revision explicit and clear about what changed
                - Always explain WHY the revision was necessary
                - Revisions should improve clarity, accuracy, or completeness
                """
            else:
                revision_instructions = """
                CRITICAL INSTRUCTION: As you progress through your thoughts, you MUST revise earlier thoughts if you realize they were incorrect or could be improved.
                
                When revising a thought:
                - Clearly indicate which specific thought you're revising (by number)
                - Explain in detail why the revision is necessary 
                - Provide the complete revised thought, not just the changes
                
                For example:
                "🔄 **Revision of Thought 2**: I need to revise my earlier thinking. Initially, I thought X, but now I realize Y because of Z. The corrected thought is..."
                
                At least 1-2 thoughts MUST be revised in your thinking process. This demonstrates how your understanding evolves and improves over time.
                """

        # Base prompt for sequential thinking
        prompt = f"""
        You are using Sequential Thinking to solve a complex problem.
        
        Guidelines for Sequential Thinking:
        1. Break down the problem into clear, logical steps
        2. Think through each step carefully and explicitly
        3. Number each thought (Thought 1, Thought 2, etc.)
        4. Explore the problem space thoroughly
        5. CRITICALLY IMPORTANT: Revise at least 1-2 of your thoughts as your understanding deepens
        6. Synthesize your thoughts into a final, coherent answer
        {revision_instructions}
        
        Problem to solve:
        {problem}
        """

        # Add context if provided
        if context:
            context_str = "\nAdditional Context:\n"
            for key, value in context.items():
                if key == "interleaved_revisions":
                    # Skip the special flag
                    continue
                elif key == "information" and value:
                    # Special handling for retrieved information
                    context_str += f"Retrieved Information:\n{value}\n\n"
                elif isinstance(value, dict):
                    context_str += f"- {key}:\n"
                    for k, v in value.items():
                        context_str += f"  - {k}: {v}\n"
                elif isinstance(value, list):
                    context_str += f"- {key}: {', '.join(map(str, value))}\n"
                else:
                    context_str += f"- {key}: {value}\n"
            prompt += context_str

        # Add closing instructions based on format
        if interleaved_format:
            prompt += f"""
            Generate {num_thoughts} detailed thoughts with reviews and revisions if needed after each thought.
            After all thoughts, provide a final answer that synthesizes your analysis.
            
            Format your response like this:
            
            🔄 **Thought 1**: [Your first thought]
            
            🔍 **Review**: [Your review of thought 1]
            
            🔄 **Revision** (if needed): [Your revision of thought 1]
            
            🔄 **Thought 2**: [Your second thought]
            
            [Continue for all thoughts]
            
            ✅ **Conclusion**: [Your comprehensive answer based on the thoughts above]
            """
        else:
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

    def _create_chain_of_thought_prompt(
        self, problem: str, context: Dict[str, Any] = None
    ) -> str:
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

    def _create_graph_of_thought_prompt(
        self, problem: str, context: Dict[str, Any] = None, num_nodes: int = 5
    ) -> str:
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

    def _create_chain_of_verification_prompt(
        self, problem: str, context: Dict[str, Any] = None
    ) -> str:
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

    def _create_step_back_prompt(
        self, problem: str, context: Dict[str, Any] = None
    ) -> str:
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

    def _combine_thoughts_with_revisions(
        self, original_thoughts: str, revisions: str
    ) -> str:
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
        revision_detail_pattern = r"Revision for Thought (\d+):\s*-\s*Original thought:(.*?)-\s*Issue:(.*?)-\s*Improved thought:(.*?)(?=Revision for Thought \d+:|$)"

        # Try to find detailed revisions first
        detailed_revisions = re.findall(revision_detail_pattern, revisions, re.DOTALL)
        if detailed_revisions:
            # Create a dictionary of thought number -> (original, issue, improved)
            revision_dict = {
                int(num): (orig.strip(), issue.strip(), improved.strip())
                for num, orig, issue, improved in detailed_revisions
            }

            # Process the original thoughts and insert revisions
            thought_pattern = r"(#+\s*🔄\s*(?:Thought|Step)\s*(\d+).*?)(?=#+\s*🔄\s*(?:Thought|Step)|#+\s*✅\s*(?:Final Answer|Conclusion)|$)"
            processed_thoughts = original_thoughts

            # Replace each thought with original + detailed revision if it exists
            for match in re.finditer(thought_pattern, original_thoughts, re.DOTALL):
                full_match, thought_num_str = match.groups()
                thought_num = int(thought_num_str)

                if thought_num in revision_dict:
                    # Get the revision details
                    orig, issue, improved = revision_dict[thought_num]

                    # Format the revised thought with clearer visual distinction for Discord
                    revised_section = (
                        f"{full_match}\n\n### 🔁 Revision for Thought {thought_num}\n\n"
                    )
                    revised_section += f"**Original interpretation:** {orig}\n\n"
                    revised_section += f"**Issue identified:** {issue}\n\n"
                    revised_section += f"**Improved understanding:** {improved}\n\n"
                    revised_section += "_ _ _\n\n"  # Discord-friendly separator

                    # Replace the original thought with original + revision
                    processed_thoughts = processed_thoughts.replace(
                        match.group(0), revised_section
                    )

            return processed_thoughts

        # Fall back to simpler revision formatting if detailed parsing fails
        found_revisions = re.findall(revision_pattern, revisions, re.DOTALL)

        # If no structured revisions found, append the full revision text
        if not found_revisions:
            # Format unstructured revisions more consistently with the rest of the output
            formatted_revisions = "\n\n## 🔁 Thought Revisions\n\n"

            # Split the revisions into paragraphs for better readability
            paragraphs = revisions.strip().split("\n\n")
            for paragraph in paragraphs:
                if paragraph.strip():
                    formatted_revisions += f"{paragraph.strip()}\n\n"

            # Add a separator before the unstructured revisions
            formatted_revisions = f"\n\n_ _ _\n{formatted_revisions}"

            return f"{original_thoughts}{formatted_revisions}"

        # Create a dictionary of thought number -> revised thought
        revision_dict = {int(num): content.strip() for num, content in found_revisions}

        # Process the original thoughts and insert revisions
        thought_pattern = r"(#+\s*🔄\s*(?:Thought|Step)\s*(\d+).*?)(?=#+\s*🔄\s*(?:Thought|Step)|#+\s*✅\s*(?:Final Answer|Conclusion)|$)"
        processed_thoughts = original_thoughts

        # Replace each thought with original + revision if it exists
        for match in re.finditer(thought_pattern, original_thoughts, re.DOTALL):
            full_match, thought_num_str = match.groups()
            thought_num = int(thought_num_str)

            if thought_num in revision_dict:
                # Format the revised thought with visual distinction for Discord
                revised_section = f"{full_match}\n\n### 🔁 Revised Understanding (Thought {thought_num})\n\n{revision_dict[thought_num]}\n\n"
                # Add a separator for clarity in Discord
                revised_section += "_ _ _\n\n"
                # Replace the original thought with original + revision
                processed_thoughts = processed_thoughts.replace(
                    match.group(0), revised_section
                )

        return processed_thoughts

    def extract_steps(self, sequential_output: str) -> list:
        """
        Extract steps from sequential thinking output

        Args:
            sequential_output: The sequential thinking output

        Returns:
            list: Extracted steps
        """
        steps = []

        # Find step markers using regex
        import re

        step_pattern = r"(?:\*\*Step \d+\*\*|\*\*Step:\*\* \d+|\d+\. |\*\*Step \d+:\*\*)(.*?)(?=\*\*Step \d+\*\*|\*\*Step:\*\* \d+|\d+\. |\*\*Step \d+:\*\*|$)"
        step_matches = re.finditer(step_pattern, sequential_output, re.DOTALL)

        for match in step_matches:
            step_text = match.group(1).strip()
            if step_text:
                steps.append(step_text)

        # If no steps found with standard format, try emoji markers
        if not steps:
            emoji_step_pattern = r"(?:🔍|🧩|📋|🔎|🧠) \*\*(?:Step \d+|[^:]+)(?:\**|:)\*\*(.*?)(?=(?:🔍|🧩|📋|🔎|🧠) \*\*|$)"
            emoji_matches = re.finditer(
                emoji_step_pattern, sequential_output, re.DOTALL
            )

            for match in emoji_matches:
                step_text = match.group(1).strip()
                if step_text:
                    steps.append(step_text)

        # Fallback to paragraph splitting if no structured steps found
        if not steps and sequential_output:
            paragraphs = sequential_output.split("\n\n")
            # Filter out very short paragraphs and headers
            steps = [p for p in paragraphs if len(p) > 60 and not p.startswith("#")]

        return steps

    def extract_conclusion(self, sequential_output: str) -> str:
        """
        Extract conclusion from sequential thinking output

        Args:
            sequential_output: The sequential thinking output

        Returns:
            str: Extracted conclusion
        """
        import re

        # Try to find conclusion with various markers
        conclusion_patterns = [
            r"\*\*Conclusion:\*\*(.*?)(?=$|(?:\n\n))",
            r"(?:✅|📝) \*\*Conclusion\*\*(.*?)(?=$|(?:\n\n))",
            r"(?:In conclusion|To conclude|In summary|To summarize)(.*?)(?=$|(?:\n\n))",
        ]

        for pattern in conclusion_patterns:
            matches = re.search(pattern, sequential_output, re.DOTALL | re.IGNORECASE)
            if matches:
                conclusion = matches.group(1).strip()
                if conclusion:
                    return conclusion

        # If no explicit conclusion found, use the last paragraph as conclusion
        paragraphs = sequential_output.split("\n\n")
        if paragraphs:
            last_meaningful_paragraph = None
            # Find the last paragraph that's not a footnote or reference
            for p in reversed(paragraphs):
                if len(p) > 60 and not any(
                    marker in p.lower() for marker in ["reference", "footnote", "note:"]
                ):
                    last_meaningful_paragraph = p
                    break

            if last_meaningful_paragraph:
                return last_meaningful_paragraph

        # If all else fails, just return empty string
        return ""

    def _format_sequential_thinking_output(
        self,
        thoughts: List[Dict[str, str]],
        revisions: Optional[List[Dict[str, str]]] = None,
        conclusion: Optional[str] = None,
    ) -> str:
        """
        Format sequential thinking output in a clear, structured way

        Args:
            thoughts: List of thought dictionaries
            revisions: Optional list of revision dictionaries
            conclusion: Optional conclusion string

        Returns:
            Formatted sequential thinking output
        """
        output = "# 🔄 Sequential Reasoning Process\n\n"

        # Check if we're using the interleaved format or separate revisions
        interleaved_format = len(thoughts) > 0 and "revision" in thoughts[0]

        if interleaved_format:
            # For interleaved format, thoughts already contain their revisions
            for i, thought in enumerate(thoughts):
                thought_num = i + 1

                # Original thought with thinking emoji
                output += f"## 🔄 **Thought {thought_num}**\n"
                output += f"{thought['content']}\n\n"

                # Review with magnifying glass emoji
                if "review" in thought and thought["review"]:
                    output += f"## 🔍 **Review of Thought {thought_num}**\n"
                    output += f"{thought['review']}\n\n"

                # Revision with refresh emoji - make it stand out
                if "revision" in thought and thought["revision"]:
                    output += f"## 🔄 **REVISION OF THOUGHT {thought_num}** 🔄\n"
                    output += f"```diff\n+ {thought['revision']}\n```\n\n"
        else:
            # Traditional format with separate revisions
            # Original thoughts
            for i, thought in enumerate(thoughts):
                thought_num = i + 1
                output += f"## 🔄 **Thought {thought_num}**\n"
                output += f"{thought['content']}\n\n"

            # Add revisions if available
            if revisions:
                output += "# 🔁 **THOUGHT REVISIONS**\n\n"
                for i, revision in enumerate(revisions):
                    thought_num = revision.get("revisesThought", i + 1)

                    if "review" in revision and revision["review"]:
                        output += f"## 🔍 **Review of Thought {thought_num}**\n"
                        output += f"{revision['review']}\n\n"

                    if "revision" in revision and revision["revision"]:
                        output += f"## 🔄 **REVISION OF THOUGHT {thought_num}** 🔄\n"
                        output += f"```diff\n+ {revision['revision']}\n```\n\n"

        # Add conclusion if available
        if conclusion:
            output += "# ✅ **Conclusion**\n\n"
            output += f"{conclusion}\n\n"

        return output

    def _parse_sequential_thinking_response(
        self, response: str
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
        """
        Parse the LLM response to extract thoughts, revisions, and conclusion

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (thoughts, revisions, conclusion)
        """
        thoughts = []
        revisions = []
        conclusion = ""

        # Check if we're dealing with the new interleaved format
        if "Thought1:" in response and "Review:" in response:
            # This appears to be the interleaved format with reviews and revisions inline
            thought_counter = 1
            current_thought = {}

            for line in response.split("\n"):
                line = line.strip()

                # Start of a new thought
                thought_pattern = re.search(r"Thought(\d+):\s*(.*)", line)
                if thought_pattern:
                    # Save previous thought if it exists
                    if current_thought and "content" in current_thought:
                        thoughts.append(current_thought)
                        current_thought = {}

                    # Start new thought
                    thought_num = int(thought_pattern.group(1))
                    thought_content = thought_pattern.group(2)
                    current_thought = {"content": thought_content}
                    continue

                # Review section
                if line.startswith("Review:"):
                    review_content = line[len("Review:") :].strip()
                    current_thought["review"] = review_content
                    continue

                # Add to review content if we're in a review section
                if "review" in current_thought and not line.startswith("Revision:"):
                    current_thought["review"] += " " + line
                    continue

                # Revision section
                if line.startswith("Revision:"):
                    revision_content = line[len("Revision:") :].strip()
                    current_thought["revision"] = revision_content
                    continue

                # Add to revision content if we're in a revision section
                if "revision" in current_thought:
                    current_thought["revision"] += " " + line
                    continue

                # Conclusion
                if line.startswith("Conclusion:"):
                    conclusion = line[len("Conclusion:") :].strip()
                    # Make sure to save the last thought
                    if current_thought and "content" in current_thought:
                        thoughts.append(current_thought)
                        current_thought = {}
                    break

                # Add to conclusion if we've started it
                if conclusion and line:
                    conclusion += " " + line

            # Make sure to save the last thought
            if current_thought and "content" in current_thought:
                thoughts.append(current_thought)

            # Return interleaved thoughts with empty revisions list
            return thoughts, [], conclusion

        # Original format with separate thoughts and revisions sections
        sections = response.split("\n\n\n")
        current_section = ""

        for section in sections:
            if "Thought" in section and not "Revision" in section:
                current_section = "thoughts"
                matches = re.findall(
                    r"Thought\s*(\d+):\s*(.*?)(?=\n*Thought\s*\d+:|$)",
                    section,
                    re.DOTALL,
                )
                for match in matches:
                    thought_num, content = match
                    thoughts.append({"content": content.strip()})
            elif "Revision" in section:
                current_section = "revisions"
                matches = re.findall(
                    r"Revision\s*(\d+):\s*(.*?)(?=\n*Revision\s*\d+:|$)",
                    section,
                    re.DOTALL,
                )
                for match in matches:
                    revision_num, content = match
                    revisions.append({"revision": content.strip()})
            elif "Conclusion" in section:
                current_section = "conclusion"
                conclusion_match = re.search(r"Conclusion:(.*)$", section, re.DOTALL)
                if conclusion_match:
                    conclusion = conclusion_match.group(1).strip()
            elif current_section == "thoughts" and section.strip():
                # Additional thought content
                if thoughts:
                    thoughts[-1]["content"] += "\n\n" + section.strip()
            elif current_section == "revisions" and section.strip():
                # Additional revision content
                if revisions:
                    revisions[-1]["revision"] += "\n\n" + section.strip()
            elif current_section == "conclusion" and section.strip():
                # Additional conclusion content
                conclusion += "\n\n" + section.strip()

        return thoughts, revisions, conclusion

    def _cache_thinking_session(
        self,
        session_id: str,
        problem: str,
        thoughts: List[Dict[str, str]],
        revisions: List[Dict[str, str]],
        conclusion: str,
    ):
        """
        Cache the thinking session data

        Args:
            session_id: The session ID
            problem: The problem
            thoughts: List of thought dictionaries
            revisions: List of revision dictionaries
            conclusion: The conclusion
        """
        self.thought_history[session_id] = thoughts
        self.revision_history[session_id] = revisions
        self.chain_result = conclusion

    def format_sequential_thinking(
        self, thoughts, problem=None, include_prompt=False, format_style="markdown"
    ):
        """
        Format sequential thinking results into a readable output

        Args:
            thoughts: List of thought steps
            problem: Original problem to solve
            include_prompt: Whether to include the original prompt
            format_style: Formatting style (markdown, discord, etc.)

        Returns:
            Formatted thinking output
        """
        if not thoughts:
            return "No thinking steps available."

        # Begin with the problem if requested
        parts = []
        if include_prompt and problem:
            parts.append(f"**Problem:** {problem}")
            parts.append("")  # Empty line

        # Add title
        parts.append("# 🔄 Sequential Reasoning Process\n")

        # Track original thoughts and revisions
        original_thoughts = {}
        revisions = {}
        working_memory = []
        evaluations = []
        insights = []

        # First pass: categorize thoughts
        for thought in thoughts:
            thought_num = thought.get("thoughtNumber", 0)
            content = thought.get("thought", "")
            is_revision = thought.get("isRevision", False)
            revises_thought = thought.get("revisesThought", None)

            if content.lower().startswith("working memory:"):
                working_memory.append(content)
            elif content.lower().startswith("evaluation:"):
                evaluations.append(content)
            elif content.lower().startswith("insight:") or "💡" in content[:10]:
                insights.append(content)
            elif is_revision and revises_thought:
                if revises_thought not in revisions:
                    revisions[revises_thought] = []
                revisions[revises_thought].append((thought_num, content))
            else:
                original_thoughts[thought_num] = content

        # Second pass: format in order
        for thought_num in sorted(original_thoughts.keys()):
            # Add the original thought
            thought_content = original_thoughts[thought_num]
            parts.append(f"## 🔄 **Thought {thought_num}**")
            parts.append(thought_content)
            parts.append("")  # Empty line

            # Add any revisions to this thought
            if thought_num in revisions:
                for rev_num, rev_content in sorted(revisions[thought_num]):
                    parts.append(
                        f"## 🔄 **Revision of Thought {thought_num}** (at step {rev_num})"
                    )
                    parts.append(rev_content)
                    parts.append("")  # Empty line

        # Add insights if any
        if insights:
            parts.append("## 💡 **Key Insights**")
            for insight in insights:
                parts.append(insight)
            parts.append("")  # Empty line

        # Add working memory if any
        if working_memory:
            parts.append("## 📝 **Working Memory**")
            for memory in working_memory:
                # Remove "Working Memory:" prefix if present
                cleaned_memory = memory
                if memory.lower().startswith("working memory:"):
                    cleaned_memory = memory[len("working memory:") :].strip()
                parts.append(cleaned_memory)
            parts.append("")  # Empty line

        # Add evaluations if any
        if evaluations:
            parts.append("## ⚖️ **Evaluation**")
            for eval_text in evaluations:
                # Remove "Evaluation:" prefix if present
                cleaned_eval = eval_text
                if eval_text.lower().startswith("evaluation:"):
                    cleaned_eval = eval_text[len("evaluation:") :].strip()
                parts.append(cleaned_eval)
            parts.append("")  # Empty line

        # Find final answer if present
        final_answer = None
        for thought in reversed(thoughts):
            content = thought.get("thought", "")
            if (
                content.lower().startswith("final answer:")
                or "✅ conclusion:" in content.lower()
            ):
                final_answer = content
                break

        if final_answer:
            parts.append("## ✅ **Conclusion**")

            # Remove "Final Answer:" prefix if present
            if final_answer.lower().startswith("final answer:"):
                final_answer = final_answer[len("final answer:") :].strip()
            elif "✅ conclusion:" in final_answer.lower():
                try:
                    final_answer = final_answer.split("✅ conclusion:", 1)[1].strip()
                except IndexError:
                    # If splitting fails, use the original
                    pass

            parts.append(final_answer)

        # Join everything together
        return "\n".join(parts)


# Singleton instance
sequential_thinking_service = SequentialThinkingService()
