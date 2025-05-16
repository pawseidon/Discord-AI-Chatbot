"""
Controversial Topics Workflow Implementation

This module implements the Controversial Topics workflow, which combines
multi-agent analysis, graph-based relationship mapping, and verification
for balanced analysis of topics with multiple viewpoints.
"""

import logging
import time
import traceback
import uuid
from typing import Dict, Any, Optional, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('controversial_topics_workflow')

async def execute_controversial_topics_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Controversial Topics workflow combining multi-agent, graph, and verification methods
    
    Args:
        query: The user query to process
        user_id: The user ID for context and memory
        conversation_id: Optional conversation ID
        update_callback: Optional callback for status updates
        
    Returns:
        str: The formatted workflow response
    """
    # Import services (lazy imports to avoid circular dependencies)
    from bot_utilities.services.agent_service import agent_service
    from bot_utilities.ai_utils import get_ai_provider
    from bot_utilities.memory_utils import get_user_preferences
    from bot_utilities.services.memory_service import memory_service
    
    workflow_start_time = time.time()
    
    try:
        # Notify about starting the controversial topics workflow
        if update_callback:
            await update_callback("workflow_start", {
                "workflow_type": "controversial_topics",
                "emoji_sequence": "👥→📊→✅",
                "message": "Starting Controversial Topic Analysis workflow..."
            })
        
        # Initialize the conversation state if needed
        conversation_id = conversation_id or f"workflow_{uuid.uuid4()}"
        
        # Get the AI provider
        ai = await get_ai_provider()
        
        # Step 1: Multi-Agent Analysis - Get different perspectives (👥)
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["multi_agent"],
                "current_type": "multi_agent",
                "message": "Analyzing from multiple perspectives..."
            })
        
        # Check if clarification is needed before proceeding
        clarification_system = """You are an AI assistant that evaluates if a query about a controversial topic needs clarification.
Determine if the query is ambiguous, vague, or could be interpreted in multiple ways.
Only request clarification if absolutely necessary to provide a balanced analysis.
Returns a JSON response with:
- needs_clarification: boolean
- clarification_query: string (only if needs_clarification is true)
"""
        
        clarification_prompt = [
            {"role": "system", "content": clarification_system},
            {"role": "user", "content": f"Query about potentially controversial topic: {query}\n\nDoes this query need clarification before proceeding with a balanced analysis?"}
        ]
        
        clarification_response = await ai.generate_text(
            messages=clarification_prompt,
            temperature=0.2,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        # Parse clarification response
        import json
        try:
            clarification_data = json.loads(clarification_response)
            needs_clarification = clarification_data.get("needs_clarification", False)
            clarification_query = clarification_data.get("clarification_query", "")
        except Exception as e:
            logger.error(f"Error parsing clarification JSON: {str(e)}")
            needs_clarification = False
            clarification_query = ""
        
        if needs_clarification:
            if update_callback:
                await update_callback("clarification_needed", {
                    "clarification_query": clarification_query,
                    "original_query": query,
                    "message": "Clarification needed before proceeding."
                })
            return f"**Clarification Needed**: {clarification_query}"
        
        # Multi-agent system prompt
        multi_agent_system = """You are an AI assistant that provides balanced analysis from multiple perspectives.

For the given query on a controversial or multi-faceted topic, identify and present 3-5 distinct perspectives.
Each perspective should:
- Represent a significant viewpoint on the topic
- Be presented fairly and charitably
- Include key arguments and considerations
- Acknowledge limitations of that perspective

Format your response with clear headings for each perspective (e.g., "## Economic Perspective").
For each perspective, provide 2-3 paragraphs of analysis.
Don't include an introduction or conclusion - focus only on presenting the perspectives."""
        
        multi_agent_prompt = [
            {"role": "system", "content": multi_agent_system},
            {"role": "user", "content": f"Query on controversial/multi-faceted topic: {query}\n\nPlease provide 3-5 distinct perspectives."}
        ]
        
        # Get the multi-agent response
        multi_agent_response = await ai.generate_text(
            messages=multi_agent_prompt,
            temperature=0.5,
            max_tokens=1500
        )
        
        # Store intermediate result
        multi_agent_data = {
            "perspectives": multi_agent_response,
            "query": query
        }
        
        # Step 2: Graph Analysis - Map relationships between perspectives (📊)
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["multi_agent", "graph"],
                "current_type": "graph",
                "message": "Mapping relationships between perspectives..."
            })
        
        # Graph system prompt
        graph_system = """You are an AI assistant that maps relationships between different perspectives on controversial topics.

Analyze the different perspectives provided and:
1. Identify key points of agreement and disagreement
2. Map the conceptual relationships between these perspectives
3. Highlight common underlying values or assumptions
4. Note where perspectives might be reconciled or are fundamentally at odds

Create a conceptual map of these relationships using text formatting.
Use headings, bullet points, and clear organization to show the connections.
Focus on insightful analysis of how these viewpoints relate to each other."""
        
        # Create an enhanced query for the graph reasoning step
        graph_prompt = [
            {"role": "system", "content": graph_system},
            {"role": "user", "content": f"Query: {query}\n\nPerspectives to map:\n\n{multi_agent_response}\n\nPlease map the relationships between these perspectives."}
        ]
        
        # Get the graph response
        graph_response = await ai.generate_text(
            messages=graph_prompt,
            temperature=0.4,
            max_tokens=1200
        )
        
        # Extract relationship data to pass to the verification step
        graph_data = {
            "relationships": graph_response,
            "perspectives": multi_agent_response,
            "query": query
        }
        
        # Step 3: Verification - Validate key claims from different perspectives (✅)
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["multi_agent", "graph", "verification"],
                "current_type": "verification",
                "message": "Verifying key claims from different perspectives..."
            })
        
        # Verification system prompt
        verification_system = """You are an AI assistant that verifies claims from multiple perspectives on controversial topics.

Your task is to:
1. Identify key claims made by each perspective
2. Assess the factual accuracy of each claim (where applicable)
3. Note where claims may be opinion rather than fact
4. Highlight areas where additional context or nuance is needed
5. Provide a balanced assessment of the overall reliability

Focus on factual verification rather than taking a position.
Your goal is to ensure factual accuracy while respecting the legitimacy of different value systems.
Be fair to all perspectives while maintaining commitment to factual truth."""
        
        verification_prompt = [
            {"role": "system", "content": verification_system},
            {"role": "user", "content": f"Query: {query}\n\nPerspectives to verify:\n\n{multi_agent_response}\n\nRelationship mapping:\n\n{graph_response}\n\nPlease verify key claims and provide a balanced assessment."}
        ]
        
        # Get the verification response
        verification_response = await ai.generate_text(
            messages=verification_prompt,
            temperature=0.3,
            max_tokens=1200
        )
        
        # Step 4: Synthesize final response
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["multi_agent", "graph", "verification", "synthesis"],
                "current_type": "synthesis",
                "message": "Synthesizing final balanced response..."
            })
        
        # Synthesis system prompt
        synthesis_system = """You are an AI assistant that synthesizes balanced analysis of controversial topics.

Create a comprehensive response that:
1. Summarizes the key perspectives on this topic (briefly)
2. Highlights areas of agreement and disagreement
3. Presents the verified factual context
4. Acknowledges value differences that may not be resolvable by facts alone
5. Provides a balanced conclusion that respects multiple viewpoints

Your goal is to help the user understand the topic from multiple angles without imposing a particular conclusion.
Present information in a way that respects the legitimacy of different values while maintaining factual accuracy."""
        
        synthesis_prompt = [
            {"role": "system", "content": synthesis_system},
            {"role": "user", "content": f"Query: {query}\n\nMultiple perspectives:\n\n{multi_agent_response}\n\nRelationship mapping:\n\n{graph_response}\n\nVerification notes:\n\n{verification_response}\n\nPlease synthesize a final balanced response."}
        ]
        
        # Get the final response
        final_response = await ai.generate_text(
            messages=synthesis_prompt,
            temperature=0.4,
            max_tokens=1800
        )
        
        # Format with workflow indicators
        formatted_response = f"""# 👥→📊→✅ Controversial Topic Analysis

{final_response}
"""
        
        # Add to conversation history if available
        if conversation_id:
            try:
                # Add user message
                await memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id.split(':')[-1] if ':' in conversation_id else None,
                    entry={"role": "user", "content": query}
                )
                
                # Add assistant response
                await memory_service.add_to_history(
                    user_id=user_id,
                    channel_id=conversation_id.split(':')[-1] if ':' in conversation_id else None,
                    entry={"role": "assistant", "content": formatted_response}
                )
            except Exception as e:
                logger.error(f"Error adding to history: {str(e)}")
        
        # Calculate workflow timing
        workflow_time = time.time() - workflow_start_time
        logger.info(f"Controversial Topics workflow completed in {workflow_time:.2f} seconds")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in controversial_topics_workflow: {str(e)}")
        traceback.print_exc()
        
        # Fall back to multi-agent reasoning on error
        try:
            from bot_utilities.services.agent_service import agent_service
            return await agent_service.execute_reasoning(
                reasoning_type="multi_agent",
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                update_callback=update_callback
            )
        except Exception as fallback_error:
            return f"I encountered an error processing your request about this topic: {str(e)}" 