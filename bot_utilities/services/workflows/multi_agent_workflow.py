"""
Multi-Agent Workflow Implementation

This module implements the Multi-Agent workflow, which simulates
multiple perspectives on a topic for balanced viewpoints and analysis.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('multi_agent_workflow')

async def execute_multi_agent_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Multi-Agent workflow that simulates multiple perspectives on a topic
    
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
        # Step 1: Initial analysis to determine perspectives
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "analysis",
                "emoji": "🔍",
                "message": "Analyzing query to identify perspectives..."
            })
            
        # Get AI provider
        ai = await get_ai_provider()
        
        # Determine what perspectives would be useful for this query
        perspective_system = """You are an analytical AI that identifies different perspectives for analyzing a topic.
For the given query, identify 3-5 distinct perspectives or viewpoints that would provide a balanced and comprehensive analysis.
Each perspective should:
- Represent a unique angle or stakeholder viewpoint
- Be relevant to the query topic
- Help provide balanced coverage
- Be described in 2-3 sentences with a clear focus

Format your response as a JSON array with each perspective having:
- name: A short, descriptive name for this perspective (e.g., "Economic Impact", "Environmental Concerns")
- description: A brief 2-3 sentence description of this perspective
- focus: Main points this perspective would focus on (1-2 sentences)
"""
        
        perspective_messages = [
            {"role": "system", "content": perspective_system},
            {"role": "user", "content": f"Query: {query}\n\nIdentify balanced perspectives for analyzing this topic."}
        ]
        
        perspectives_response = await ai.generate_text(
            messages=perspective_messages,
            temperature=0.7,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        try:
            import json
            perspectives_data = json.loads(perspectives_response)
            perspectives = perspectives_data.get("perspectives", [])
            
            if not perspectives or len(perspectives) == 0:
                # If we couldn't parse perspectives, create default ones
                perspectives = [
                    {"name": "Practical Perspective", "description": "Focuses on practical implications and real-world applications.", "focus": "Concrete applications and practical implications."},
                    {"name": "Critical Perspective", "description": "Examines potential issues, challenges, and limitations.", "focus": "Challenges, limitations, and potential problems."},
                    {"name": "Analytical Perspective", "description": "Provides objective analysis based on data and evidence.", "focus": "Data-driven analysis and logical evaluation."}
                ]
        except Exception as e:
            logger.error(f"Error parsing perspectives JSON: {str(e)}")
            # Create default perspectives
            perspectives = [
                {"name": "Practical Perspective", "description": "Focuses on practical implications and real-world applications.", "focus": "Concrete applications and practical implications."},
                {"name": "Critical Perspective", "description": "Examines potential issues, challenges, and limitations.", "focus": "Challenges, limitations, and potential problems."},
                {"name": "Analytical Perspective", "description": "Provides objective analysis based on data and evidence.", "focus": "Data-driven analysis and logical evaluation."}
            ]
        
        # Step 2: Generate responses from each perspective
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "multi_perspective",
                "emoji": "👥",
                "message": "Generating multiple perspectives..."
            })
        
        perspective_responses = []
        
        for i, perspective in enumerate(perspectives):
            if update_callback:
                await update_callback("workflow_stage", {
                    "stage": f"perspective_{i+1}",
                    "emoji": "🔄",
                    "message": f"Processing {perspective['name']}..."
                })
                
            # Create system prompt for this perspective
            perspective_prompt = f"""You are an AI assistant providing analysis from the perspective of "{perspective['name']}".

Perspective Description: {perspective['description']}
Focus Areas: {perspective['focus']}

Analyze the query from this specific perspective. Your response should:
1. Stay true to this perspective's viewpoint and focus areas
2. Provide 2-3 paragraphs of clear, insightful analysis
3. Include relevant supporting evidence or examples
4. Acknowledge the perspective's limitations where appropriate

Remember that you are representing just one viewpoint. Other perspectives will be considered separately.
"""
            
            # Generate response for this perspective
            perspective_messages = [
                {"role": "system", "content": perspective_prompt},
                {"role": "user", "content": f"Query: {query}\n\nProvide analysis from the {perspective['name']} perspective."}
            ]
            
            perspective_response = await ai.generate_text(
                messages=perspective_messages,
                temperature=0.5,
                max_tokens=800
            )
            
            perspective_responses.append({
                "name": perspective['name'],
                "description": perspective['description'],
                "analysis": perspective_response
            })
        
        # Step 3: Synthesize the perspectives
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "synthesis",
                "emoji": "⚖️",
                "message": "Synthesizing perspectives..."
            })
            
        # Format the perspective responses
        formatted_perspectives = "\n\n".join([
            f"## {resp['name']}\n{resp['analysis']}"
            for resp in perspective_responses
        ])
        
        # Create system prompt for synthesis
        synthesis_prompt = """You are an AI assistant tasked with synthesizing multiple perspectives on a topic.
Your goal is to create a balanced, comprehensive response that:

1. Acknowledges the validity of different viewpoints
2. Identifies areas of agreement and disagreement
3. Provides a nuanced understanding of the topic
4. Helps the user form their own informed opinion

Create a response with these sections:
- Summary: A brief overview of the topic and key perspectives (1 paragraph)
- Areas of Agreement: Points where the perspectives align (1-2 paragraphs)
- Key Differences: Important distinctions between perspectives (1-2 paragraphs)
- Balanced Conclusion: A fair synthesis that respects each viewpoint (1 paragraph)

Make your response engaging, balanced, and informative.
"""
        
        # Generate the synthesis
        synthesis_messages = [
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": f"Query: {query}\n\nPerspectives to synthesize:\n\n{formatted_perspectives}"}
        ]
        
        final_response = await ai.generate_text(
            messages=synthesis_messages,
            temperature=0.5,
            max_tokens=1500
        )
        
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
                    entry={"role": "assistant", "content": final_response}
                )
            except Exception as e:
                logger.error(f"Error adding to history: {str(e)}")
        
        # Calculate workflow timing
        workflow_time = time.time() - workflow_start_time
        logger.info(f"Multi-agent workflow completed in {workflow_time:.2f} seconds")
        
        return final_response
        
    except Exception as e:
        logger.error(f"Error in multi_agent_workflow: {str(e)}")
        traceback.print_exc()
        
        # Fall back to conversational reasoning on error
        try:
            from bot_utilities.services.agent_service import agent_service
            return await agent_service.execute_reasoning(
                reasoning_type="conversational",
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                update_callback=update_callback
            )
        except Exception as fallback_error:
            return f"I encountered an error processing your request: {str(e)}" 