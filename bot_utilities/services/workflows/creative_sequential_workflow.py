"""
Creative Sequential Workflow Implementation

This module implements the Creative Sequential workflow, which combines
creative content generation with structured organization for
creative content that is both original and well-structured.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('creative_sequential_workflow')

async def execute_creative_sequential_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Creative Sequential workflow combining creative content generation with structured organization
    
    Args:
        query: The user query to process
        user_id: The user ID for context and memory
        conversation_id: Optional conversation ID
        update_callback: Optional callback for status updates
        
    Returns:
        str: The formatted workflow response
    """
    # Import services (lazy imports to avoid circular dependencies)
    from bot_utilities.services.sequential_thinking_service import sequential_thinking_service
    from bot_utilities.ai_utils import get_ai_provider
    
    workflow_start_time = time.time()
    
    try:
        # Notify about creative workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["creative", "sequential"],
                "is_combined": True
            })
            
        # Step 1: Generate creative content ideas
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "creative",
                "emoji": "🎨",
                "message": "Generating creative ideas..."
            })
            
        # Get AI provider for prompting
        ai = await get_ai_provider()
        
        # Create creative prompt
        creative_prompt = [
            {"role": "system", "content": "You are a highly creative assistant with expertise in storytelling, writing, and imaginative content. Generate creative ideas, metaphors, and imaginative content for the user's request. Be original, engaging, and creative in your response."},
            {"role": "user", "content": query}
        ]
        
        # Call the AI for creative generation
        creative_ideas = await ai.generate_text(
            messages=creative_prompt,
            temperature=0.8,
            max_tokens=1000
        )
        
        # Step 2: Structure and organize the creative content
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "organization",
                "emoji": "📝",
                "message": "Organizing creative ideas into a structured format..."
            })
            
        # Create context for sequential thinking
        context = {
            "creative_ideas": creative_ideas,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "interleaved_format": True  # Use the interleaved format for revisions
        }
        
        # Process with sequential thinking for structured organization
        success, structured_content = await sequential_thinking_service.process_sequential_thinking(
            problem=f"Organize and structure these creative ideas into a cohesive response: {query}\n\nCreative Ideas:\n{creative_ideas}",
            context=context,
            prompt_style="sequential",
            max_steps=5,
            enable_revision=True,
            session_id=f"creative_seq_{user_id}_{int(time.time())}"
        )
        
        if not success:
            logger.warning(f"Sequential thinking for organization had issues: {structured_content[:100]}...")
        
        # Format the final response
        formatted_response = f"✨📝 **Creative Structured Content**\n\n{structured_content}"
        
        # Calculate workflow timing
        workflow_time = time.time() - workflow_start_time
        logger.info(f"Creative Sequential workflow completed in {workflow_time:.2f} seconds")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in creative_sequential_workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to conversational reasoning
        try:
            return f"I encountered an error while processing your creative request: {str(e)}"
        except Exception as fallback_error:
            return f"Error processing creative content: {str(e)}" 