"""
Workflow Service

This module provides a service for managing complex multi-agent workflows,
with support for different agent combinations, transitions, and state management.
"""

import asyncio
import json
import os
import re
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

from . import agent_service as agent_service_module
from . import symbolic_reasoning_service 
from . import sequential_thinking_service
from . import memory_service

# Get the agent_service instance
from .agent_service import agent_service

# Import AI provider for accessing the language model
from bot_utilities.ai_utils import get_ai_provider

class WorkflowService:
    """
    A service for implementing multi-agent workflows with different reasoning combinations,
    state management, and orchestration between different reasoning types.
    """
    
    def __init__(self):
        """
        Initialize the workflow service
        """
        self.llm_provider = None
        self.workflows = {
            "sequential_rag": self.sequential_rag_workflow,
            "verification_rag": self.verification_rag_workflow,
            "calculation_sequential": self.calculation_sequential_workflow,
            "creative_sequential": self.creative_sequential_workflow,
            "graph_rag_verification": self.graph_rag_verification_workflow,
            "multi_agent": self.multi_agent_workflow
        }
        self._initialized = False
        self._conversation_contexts = {}  # Local storage for conversation context
        
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
    
    def is_workflow_available(self) -> bool:
        """Check if workflow service is initialized and available for use"""
        return self._initialized
            
    async def detect_workflow_type(self, query: str, conversation_id: str = None) -> str:
        """
        Detect the most appropriate workflow type for a given query
        
        Args:
            query: The query to analyze
            conversation_id: Optional conversation ID for context
            
        Returns:
            str: The recommended workflow type
        """
        await self.ensure_initialized()
        
        # First check for calculation + sequential workflow
        if re.search(r'(calculate|compute|solve|equation|formula|math problem)', query, re.IGNORECASE):
            has_explanation_request = re.search(r'(explain|show steps|show work|why|how)', query, re.IGNORECASE)
            if has_explanation_request:
                return "calculation_sequential"
        
        # Check for verification + RAG workflow
        factual_verification_pattern = r'(verify|fact check|is it true|confirm|evidence for|sources for|research on|current events)'
        if re.search(factual_verification_pattern, query, re.IGNORECASE):
            return "verification_rag"
            
        # Check for creative + sequential workflow
        creative_pattern = r'(write|create|generate|story|poem|creative|imagine|fiction)'
        explanation_pattern = r'(explain|analyze|outline|structure|organize|plan)'
        if re.search(creative_pattern, query, re.IGNORECASE) and re.search(explanation_pattern, query, re.IGNORECASE):
            return "creative_sequential"
            
        # Check for graph + RAG + verification workflow
        graph_pattern = r'(relationship|network|connect|graph|diagram|map the|connections between)'
        if re.search(graph_pattern, query, re.IGNORECASE):
            return "graph_rag_verification"
            
        # Default to sequential + RAG for educational/explanatory content
        educational_pattern = r'(what is|how does|explain|describe|why does|how do|what are|definition of)'
        if re.search(educational_pattern, query, re.IGNORECASE):
            return "sequential_rag"
            
        # Multi-agent is the most general workflow
        return "multi_agent"
    
    async def process_with_workflow(self, 
                                   query: str,
                                   user_id: str,
                                   conversation_id: str = None,
                                   workflow_type: str = None,
                                   update_callback: Callable = None) -> str:
        """
        Process a query using a specific workflow
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID 
            workflow_type: Optional workflow type to use (otherwise autodetected)
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The response from the workflow
        """
        await self.ensure_initialized()
        
        try:
            # Auto-detect workflow type if not specified
            if not workflow_type:
                workflow_type = await self.detect_workflow_type(query, conversation_id)
                
            # Store the workflow type safely
            await self._safe_store_context(conversation_id, {"last_workflow": workflow_type})
                
            # Extract user_id from conversation_id if possible
            conversation_parts = conversation_id.split(":")
            target_user_id = user_id  # Default to the passed user_id
            
            if len(conversation_parts) == 2:
                guild_id, channel_id = conversation_parts
                # If it's a DM, the channel_id might be the user_id
                if guild_id == "DM":
                    target_user_id = channel_id
            
            # Call the appropriate workflow handler
            if workflow_type in self.workflows:
                workflow_handler = self.workflows[workflow_type]
                return await workflow_handler(
                    query=query,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    update_callback=update_callback
                )
            else:
                # Default to multi-agent workflow
                return await self.multi_agent_workflow(
                    query=query,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    update_callback=update_callback
                )
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"Error in workflow processing: {e}\n{error_traceback}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def sequential_rag_workflow(self,
                                     query: str,
                                     user_id: str,
                                     conversation_id: str = None,
                                     update_callback: Callable = None) -> str:
        """
        Process a query using the Sequential + RAG workflow
        This workflow is ideal for educational content and detailed explanations
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The final response
        """
        # Notify about switching to RAG mode
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["rag", "sequential"],
                "is_combined": True
            })
            
        # Step 1: Retrieve relevant information using RAG
        if update_callback:
            await update_callback("thinking", {"thinking": "Retrieving information about topics and connections..."})
            
        rag_results = await agent_service.search_web(query)
        
        # Create a context object with the RAG results
        context = {
            "retrieved_information": rag_results,
            "user_id": user_id,
            "conversation_id": conversation_id
        }
        
        # Notify about switching to sequential thinking mode
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["sequential"],
                "is_combined": False
            })
            
        # Step 2: Process with sequential thinking to organize and explain information
        if update_callback:
            await update_callback("thinking", {"thinking": "Organizing information into a logical explanation..."})
            
        success, response = await sequential_thinking_service.sequential_thinking_service.process_sequential_thinking(
            problem=query,
            context=context,
            prompt_style="sequential",
            num_thoughts=5,
            temperature=0.3,
            enable_revision=True,
            enable_reflection=False,
            session_id=f"session_{user_id}_{conversation_id}"
        )
        
        # Format as a combined RAG + Sequential response
        formatted_response = f"🌐📚 **Sequential Explanation Based on Retrieved Information**\n\n{response}"
        
        return formatted_response
    
    async def verification_rag_workflow(self,
                                       query: str,
                                       user_id: str,
                                       conversation_id: str = None,
                                       update_callback: Callable = None) -> str:
        """
        Process a query using the Verification + RAG + Sequential workflow
        This workflow is ideal for fact-checking and verifying information
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The final response
        """
        # Notify about verification workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["rag", "verification"],
                "is_combined": True
            })
            
        # Step 1: Retrieve relevant information using RAG searching
        if update_callback:
            await update_callback("thinking", {"thinking": "Retrieving information from multiple sources..."})
            
        rag_results = await agent_service.search_web(query, num_results=5)
        
        # Step 2: Verify the information with fact checking
        if update_callback:
            await update_callback("thinking", {"thinking": "Verifying information from different sources..."})
            
        # Create verification prompt
        verification_prompt = """
        You are a verification agent with expertise in fact-checking and evaluating information reliability.
        
        Verify the facts in the following information.
        
        Query: """ + query + """
        
        Retrieved Information:
        """ + rag_results + """
        
        For each major factual claim:
        1. Identify the claim
        2. Cross-reference it with other sources in the information
        3. Rate the claim's reliability (High, Medium, Low)
        4. Note any contradictions or uncertainties
        
        Provide a summary of verified facts and identify any potential misinformation.
        """
        
        # Call the LLM for verification
        verification_response = await self.llm_provider.async_call(
            prompt=verification_prompt,
            temperature=0.2
        )
        
        # Notify about switching to sequential mode
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["sequential", "verification"],
                "is_combined": True
            })
            
        # Step 3: Organize the verified information with sequential thinking
        context = {
            "retrieved_information": rag_results,
            "verification_results": verification_response,
            "user_id": user_id,
            "conversation_id": conversation_id
        }
        
        # Process with sequential thinking
        if update_callback:
            await update_callback("thinking", {"thinking": "Organizing verified information into a coherent response..."})
            
        success, sequential_response = await sequential_thinking_service.sequential_thinking_service.process_sequential_thinking(
            problem=f"Create a well-organized response to the query: {query}",
            context=context,
            prompt_style="cov",  # Chain of verification
            num_thoughts=4,
            temperature=0.3,
            enable_revision=True,
            session_id=f"session_{user_id}_{conversation_id}"
        )
        
        # Format the final response
        formatted_response = f"🧪📚 **Verified Information**\n\n{sequential_response}"
        
        return formatted_response
    
    async def calculation_sequential_workflow(self,
                                            query: str,
                                            user_id: str,
                                            conversation_id: str = None,
                                            update_callback: Callable = None) -> str:
        """
        Process a query using the Calculation + Sequential workflow
        This workflow combines precise mathematical operations with step-by-step explanations
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The final response
        """
        # Notify about calculation workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["calculation", "sequential"],
                "is_combined": True
            })
            
        # Step 1: Extract the mathematical expression from the query
        if update_callback:
            await update_callback("thinking", {"thinking": "Extracting mathematical expression..."})
            
        # Create extraction prompt
        extraction_prompt = """
        You are a specialized agent for extracting mathematical expressions from text.
        
        Extract a mathematical or logical expression that represents the following problem. 
        Use standard mathematical notation that could be evaluated by a computer.
        
        Problem: """ + query + """
        
        Expression:
        """
        
        # Call the LLM to extract the expression
        expression = await self.llm_provider.async_call(
            prompt=extraction_prompt,
            temperature=0.1
        )
        
        # Step 2: Calculate the result using symbolic reasoning
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["calculation"],
                "is_combined": False
            })
            
        if update_callback:
            await update_callback("thinking", {"thinking": "Calculating result..."})
            
        calculation_result = await symbolic_reasoning_service.symbolic_reasoning_service.evaluate_expression(expression)
        
        # Step 3: Provide a sequential explanation of the calculation
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["sequential"],
                "is_combined": False
            })
            
        if update_callback:
            await update_callback("thinking", {"thinking": "Creating step-by-step explanation..."})
            
        # Create context for sequential thinking
        context = {
            "expression": expression,
            "calculation_result": calculation_result,
            "user_id": user_id,
            "conversation_id": conversation_id
        }
        
        # Process with sequential thinking for explanation
        success, explanation = await sequential_thinking_service.sequential_thinking_service.process_sequential_thinking(
            problem=f"Explain how to solve this mathematical problem step by step: {expression}",
            context=context,
            prompt_style="sequential",
            num_thoughts=4,
            temperature=0.3,
            session_id=f"session_{user_id}_{conversation_id}"
        )
        
        # Format the final response
        result_value = calculation_result.get("result", "Unable to calculate")
        steps = "\n".join([f"• {step}" for step in calculation_result.get("steps", [])])
        
        formatted_response = f"""🧮📚 **Mathematical Solution with Explanation**

**Expression:** `{expression}`
**Result:** `{result_value}`

**Step-by-step solution:**
{explanation}"""
        
        return formatted_response
    
    async def creative_sequential_workflow(self,
                                         query: str,
                                         user_id: str,
                                         conversation_id: str = None,
                                         update_callback: Callable = None) -> str:
        """
        Process a query using the Creative + Sequential workflow
        This workflow combines creative content generation with structured organization
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The final response
        """
        # Notify about creative workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["creative", "sequential"],
                "is_combined": True
            })
            
        # Step 1: Generate creative content ideas
        if update_callback:
            await update_callback("thinking", {"thinking": "Generating creative ideas..."})
            
        # Create creative prompt
        creative_prompt = """
        You are a highly creative assistant with expertise in storytelling, writing, and imaginative content.
        
        Generate creative ideas, metaphors, and imaginative content for the following request:
        
        """ + query + """
        
        Be original, engaging, and creative in your response.
        """
        
        # Call the LLM for creative generation
        creative_ideas = await self.llm_provider.async_call(
            prompt=creative_prompt,
            temperature=0.8
        )
        
        # Step 2: Structure and organize the creative content
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["sequential"],
                "is_combined": False
            })
            
        if update_callback:
            await update_callback("thinking", {"thinking": "Organizing creative ideas into a structured format..."})
            
        # Create context for sequential thinking
        context = {
            "creative_ideas": creative_ideas,
            "user_id": user_id,
            "conversation_id": conversation_id
        }
        
        # Process with sequential thinking for structured organization
        success, structured_content = await sequential_thinking_service.sequential_thinking_service.process_sequential_thinking(
            problem=f"Organize and structure these creative ideas into a cohesive response: {query}",
            context=context,
            prompt_style="sequential",
            num_thoughts=4,
            temperature=0.4,
            session_id=f"session_{user_id}_{conversation_id}"
        )
        
        # Format the final response
        formatted_response = f"✨📝 **Creative Structured Content**\n\n{structured_content}"
        
        return formatted_response
    
    async def graph_rag_verification_workflow(self,
                                            query: str,
                                            user_id: str,
                                            conversation_id: str = None,
                                            update_callback: Callable = None) -> str:
        """
        Process a query using the Graph + RAG + Verification workflow
        This workflow maps relationships between entities with verified information
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The final response
        """
        # Notify about graph workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["graph", "rag"],
                "is_combined": True
            })
            
        # Step 1: Retrieve relevant information using RAG
        if update_callback:
            await update_callback("thinking", {"thinking": "Retrieving information about topics and connections..."})
            
        rag_results = await agent_service.search_web(query)
        
        # Extract key entities and relationships from RAG results
        # ... existing code ...
        
        # Step 2: Build a graph model of entities and relationships
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["graph"],
                "is_combined": False
            })
            
        if update_callback:
            await update_callback("thinking", {"thinking": "Building a graph of relationships between entities..."})
            
        # Create graph construction prompt
        graph_prompt = """
        You are a graph mapping assistant with expertise in analyzing complex relationships and networks.
        
        Create a graph model to analyze the following query:
        
        """ + query + """
        
        For this query:
        1. Identify the key entities, concepts, and relationships
        2. Describe how these elements connect and relate to each other
        3. Analyze the importance and influence of each node in the network
        4. Explain what insights can be drawn from this graph representation
        """
        
        # Call the LLM for graph construction
        graph_model = await self.llm_provider.async_call(
            prompt=graph_prompt,
            temperature=0.3
        )
        
        # Step 3: Verify the relationships and connections
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["verification"],
                "is_combined": False
            })
            
        if update_callback:
            await update_callback("thinking", {"thinking": "Verifying the accuracy of relationships..."})
            
        # Verification prompt
        verification_prompt = f"""
        Verify the accuracy of the following relationship graph:
        
        {graph_model}
        
        Based on this information:
        {rag_results}
        
        For each relationship:
        1. Check if it's supported by the information
        2. Note any contradictions or uncertainties
        3. Rate the confidence level (High, Medium, Low)
        
        Provide a verified version of the relationship graph.
        """
        
        # Call the LLM for verification
        verified_graph = await self.llm_provider.async_call(
            prompt=verification_prompt,
            temperature=0.2
        )
        
        # Format the final response
        formatted_response = f"""🔄🌐 **Relationship Graph Analysis**

{verified_graph}

This graph represents the verified relationships between entities based on the information retrieved.
"""
        
        return formatted_response
    
    async def multi_agent_workflow(self,
                                  query: str,
                                  user_id: str,
                                  conversation_id: str = None,
                                  update_callback: Callable = None) -> str:
        """
        Process a query using the flexible Multi-Agent workflow
        
        This workflow dynamically selects and combines multiple reasoning types based on the query.
        It provides a coordinated approach with clear transitions between reasoning phases.
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Optional conversation ID
            update_callback: Optional callback for streaming updates
            
        Returns:
            str: The final response
        """
        # Detect multiple reasoning types to use
        reasoning_types = await agent_service.detect_multiple_reasoning_types(
            query=query, 
            conversation_id=conversation_id,
            max_types=3
        )
        
        # Ensure we have at least two reasoning types
        if len(reasoning_types) < 2:
            # Add a complementary reasoning type
            if reasoning_types[0] == "rag":
                reasoning_types.append("sequential")
            elif reasoning_types[0] == "sequential":
                reasoning_types.append("verification")
            elif reasoning_types[0] == "creative":
                reasoning_types.append("sequential")
            elif reasoning_types[0] == "verification":
                reasoning_types.append("rag")
            else:
                reasoning_types.append("sequential")
        
        # Notify about the reasoning types being used
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": reasoning_types[:2],
                "is_combined": True
            })
            
        # Create a structured workflow plan based on reasoning types
        primary_type = reasoning_types[0]
        secondary_type = reasoning_types[1]
        workflow_plan = self._create_workflow_plan(primary_type, secondary_type, query)
        
        # Notify about the workflow plan
        if update_callback:
            thinking_text = f"Multi-Agent Workflow Plan:\n\n"
            thinking_text += f"1. Primary reasoning: {primary_type.capitalize()}\n"
            thinking_text += f"2. Secondary reasoning: {secondary_type.capitalize()}\n\n"
            thinking_text += f"Approach:\n{workflow_plan}"
            await update_callback("thinking", {"thinking": thinking_text})
            
        # Step 1: Execute primary reasoning
        if update_callback:
            await update_callback("agent_switch", {"agent_type": primary_type})
            
        primary_response = await self._execute_reasoning_step(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            reasoning_type=primary_type,
            update_callback=update_callback
        )
        
        # Step 2: Execute secondary reasoning with context from primary
        if update_callback:
            transition_text = f"Transitioning from {primary_type.capitalize()} to {secondary_type.capitalize()} reasoning to refine and enhance the analysis."
            await update_callback("thinking", {"thinking": transition_text})
            await update_callback("agent_switch", {"agent_type": secondary_type})
            
        # Create a context object with the primary response
        enriched_context = {
            "primary_reasoning_result": primary_response,
            "primary_reasoning_type": primary_type,
            "reasoning_types": reasoning_types
        }
        
        # Use a meta-prompt to guide the secondary reasoning based on the primary results
        meta_prompt = self._create_meta_prompt(primary_type, secondary_type, query, primary_response)
        
        # Execute secondary reasoning
        secondary_response = await self._execute_reasoning_step(
            query=meta_prompt,
            user_id=user_id,
            conversation_id=conversation_id,
            reasoning_type=secondary_type,
            update_callback=update_callback,
            context=enriched_context
        )
        
        # Notify of synthesis
        if update_callback:
            await update_callback("thinking", {"thinking": "Integrating insights from multiple reasoning approaches to provide a comprehensive response."})
            
        # Prepare synthesis inputs
        synthesis_inputs = f"""
        1. {primary_type.capitalize()} reasoning produced: 
        {primary_response}
        
        2. {secondary_type.capitalize()} reasoning produced:
        {secondary_response}
        """
        
        # Create synthesis prompt with all responses
        synthesis_prompt = """
        You are an expert synthesizer that combines multiple reasoning approaches into cohesive responses that highlight different perspectives and insights.
        
        Synthesize the following different reasoning approaches into a cohesive response to the user's question.
        
        User Question: """ + query + """
        
        """ + synthesis_inputs + """
        
        Your response should:
        1. Integrate the insights from all reasoning approaches
        2. Highlight where different approaches agree and where they offer unique perspectives
        3. Present a coherent, comprehensive answer that benefits from the multiple viewpoints
        4. Be engaging, clear, and well-structured
        """
        
        # Generate synthesis response
        response = await self.llm_provider.async_call(
            prompt=synthesis_prompt,
            temperature=0.4
        )
        
        # Format with multi-agent header and prepend reasoning types used
        response_with_header = f"**Multi-Agent Analysis** (using {primary_type.capitalize()} and {secondary_type.capitalize()} reasoning)\n\n{response}"
        
        return response_with_header
        
    def _create_workflow_plan(self, primary_type: str, secondary_type: str, query: str) -> str:
        """
        Create a workflow plan description based on the reasoning types
        
        Args:
            primary_type: Primary reasoning type
            secondary_type: Secondary reasoning type
            query: The original query
            
        Returns:
            str: Human-readable workflow plan
        """
        workflow_descriptions = {
            "rag": "retrieve factual information and external knowledge",
            "sequential": "break down complex problems step-by-step",
            "creative": "generate creative and imaginative content",
            "verification": "verify facts and check accuracy of information",
            "calculation": "perform precise calculations and mathematical analysis",
            "graph": "analyze relationships and connections between entities",
            "conversational": "engage in natural, context-aware dialogue"
        }
        
        primary_desc = workflow_descriptions.get(primary_type, f"use {primary_type} reasoning")
        secondary_desc = workflow_descriptions.get(secondary_type, f"use {secondary_type} reasoning")
        
        return f"First, I'll {primary_desc} to address the core aspects of your query. Then, I'll {secondary_desc} to refine and enhance the analysis. Finally, I'll synthesize insights from both approaches into a comprehensive response."
        
    def _create_meta_prompt(self, primary_type: str, secondary_type: str, query: str, primary_response: str) -> str:
        """
        Create a meta-prompt to guide secondary reasoning based on primary results
        
        Args:
            primary_type: Primary reasoning type
            secondary_type: Secondary reasoning type
            query: The original query
            primary_response: Response from primary reasoning
            
        Returns:
            str: Meta-prompt for secondary reasoning
        """
        # Base prompt template
        meta_prompt = f"""
        Original query: "{query}"
        
        The {primary_type} reasoning approach has produced this response:
        {primary_response}
        
        Now I want you to analyze this same query using {secondary_type} reasoning. 
        """
        
        # Add specific instructions based on secondary type
        if secondary_type == "verification":
            meta_prompt += "Verify the accuracy of the information provided, identify any factual errors or inconsistencies, and provide corrections if needed."
        elif secondary_type == "sequential":
            meta_prompt += "Break down the problem systematically, showing clear step-by-step reasoning. Expand on any complex aspects that require more detailed explanation."
        elif secondary_type == "rag":
            meta_prompt += "Supplement with relevant factual information, research findings, or external knowledge that would enhance understanding."
        elif secondary_type == "creative":
            meta_prompt += "Add creative examples, analogies, or alternative perspectives that help illustrate the concepts."
        elif secondary_type == "calculation":
            meta_prompt += "Add precise calculations, numerical analysis, or quantitative support for any claims or concepts."
        elif secondary_type == "graph":
            meta_prompt += "Analyze relationships, connections, and structural patterns between entities or concepts mentioned."
        
        meta_prompt += "\nProvide your analysis, focusing on aspects that complement or enhance the original response."
        
        return meta_prompt
    
    async def _execute_reasoning_step(self,
                                      query: str,
                                      user_id: str,
                                      conversation_id: str,
                                      reasoning_type: str,
                                      update_callback: Optional[Callable] = None,
                                      context: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a single reasoning step with a specific reasoning type
        
        Args:
            query: The query to process
            user_id: User ID for memory and context
            conversation_id: Conversation ID
            reasoning_type: Reasoning type to use
            update_callback: Optional callback for streaming updates
            context: Optional additional context
            
        Returns:
            str: The response from the reasoning step
        """
        # Process using agent service with specific reasoning type
        response = await agent_service.process_query(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            reasoning_type=reasoning_type,
            update_callback=update_callback
        )
        
        return response

    async def _safe_store_context(self, conversation_id: str, context_data: dict):
        """
        Safely store context data for conversations - works even if memory_service lacks store_conversation_context
        
        Args:
            conversation_id: The conversation ID
            context_data: The context data to store
        """
        # Try to store in memory service first
        try:
            # Check if the memory_service has the store_conversation_context method
            if hasattr(memory_service, "store_conversation_context"):
                await memory_service.store_conversation_context(conversation_id, context_data)
                return
        except Exception as e:
            print(f"Error using memory_service.store_conversation_context: {e}")
            
        # Fallback to local storage
        if conversation_id not in self._conversation_contexts:
            self._conversation_contexts[conversation_id] = {}
            
        self._conversation_contexts[conversation_id].update(context_data)
        
    async def _safe_get_context(self, conversation_id: str, key: str = None):
        """
        Safely get context data - works even if memory_service lacks get_conversation_context
        
        Args:
            conversation_id: The conversation ID
            key: Optional specific key to get
            
        Returns:
            The context data
        """
        # Try to get from memory service first
        try:
            if hasattr(memory_service, "get_conversation_context"):
                result = await memory_service.get_conversation_context(conversation_id, key)
                if result is not None:
                    return result
        except Exception as e:
            print(f"Error using memory_service.get_conversation_context: {e}")
            
        # Fallback to local storage
        if conversation_id not in self._conversation_contexts:
            return None if key else {}
            
        if key:
            return self._conversation_contexts[conversation_id].get(key)
        else:
            return self._conversation_contexts[conversation_id]

# Instantiate the workflow service
workflow_service = WorkflowService()