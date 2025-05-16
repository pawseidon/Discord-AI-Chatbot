"""
Creative Development Workflow Implementation

This module implements the Creative Development workflow, which combines
creative content generation with step-back analysis and sequential thinking
for structured creative content with purpose and coherence.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('creative_development_workflow')

async def execute_creative_development_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Creative Development workflow combining creative generation, step-back analysis,
    and sequential thinking for structured creative content
    
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
        # Notify about starting the creative development workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["creative", "step_back", "sequential"],
                "is_combined": True,
                "workflow": "creative_development"
            })
        
        # Get AI provider
        ai = await get_ai_provider()
        
        # Step 1: Generate creative content
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "creative_generation",
                "emoji": "🎨",
                "message": "Generating creative content..."
            })
        
        # Create creative system message
        creative_system = """You are a highly creative content creator, specialized in generating imaginative and original content.

For the given request, create high-quality creative content that is:
1. Original and unique
2. Vivid and emotionally engaging
3. Rich in imagery and sensory details
4. Conceptually interesting
5. Authentic to the requested style/genre

Focus on creative quality rather than structure at this stage. Let your imagination explore interesting directions and unexpected connections.
Be bold and take creative risks rather than following predictable patterns.
"""
        
        # Create creative prompt
        creative_prompt = [
            {"role": "system", "content": creative_system},
            {"role": "user", "content": f"Creative request: {query}\n\nPlease generate creative content for this request."}
        ]
        
        # Call the AI for creative generation
        creative_response = await ai.generate_text(
            messages=creative_prompt,
            temperature=0.9,
            max_tokens=1500
        )
        
        # Step 2: Step back and consider the broader purpose
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "step_back_analysis",
                "emoji": "🔍",
                "message": "Analyzing purpose and themes..."
            })
        
        # Create step-back system message
        step_back_system = """You are a big-picture analyst who excels at identifying deeper meanings, themes, and purposes.

For the creative content provided, analyze:
1. Core themes and underlying messages
2. Broader purpose and value
3. Target audience and intended impact
4. Stylistic elements and aesthetic qualities
5. Potential improvements to enhance meaning and impact

Step back from the details and consider:
- What is this piece ultimately trying to achieve?
- What emotions or thoughts should it evoke?
- How does it connect to universal human experiences?
- What would make it more meaningful or impactful?

Provide a thoughtful analysis that focuses on meaning and purpose rather than technical details.
"""
        
        # Create step-back prompt
        step_back_prompt = [
            {"role": "system", "content": step_back_system},
            {"role": "user", "content": f"Original request: {query}\n\nCreative content to analyze:\n{creative_response}\n\nPlease provide a big-picture analysis."}
        ]
        
        # Call the AI for step-back analysis
        step_back_response = await ai.generate_text(
            messages=step_back_prompt,
            temperature=0.4,
            max_tokens=1000
        )
        
        # Step 3: Apply sequential thinking to structure the content
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "sequential_structuring",
                "emoji": "🔄",
                "message": "Creating structured organization..."
            })
        
        # Create sequential system message
        sequential_system = """You are a master of structure and organization who excels at creating logical progression and coherence.

Based on the creative content and big-picture analysis provided, create a well-structured version that:
1. Has a clear beginning, middle, and end
2. Follows a logical progression of ideas or events
3. Maintains consistent themes and character development (if applicable)
4. Uses appropriate transitions between sections
5. Builds toward a meaningful conclusion or resolution

Your goal is to preserve the creative essence and purpose while adding:
- Stronger structural scaffolding
- More coherent organization
- Better pacing and flow
- Enhanced development of ideas
- Clearer relationships between elements

Present a refined version that maintains the original creativity but adds logical structure and coherence.
"""
        
        # Create sequential prompt
        sequential_prompt = [
            {"role": "system", "content": sequential_system},
            {"role": "user", "content": f"Original request: {query}\n\nCreative content:\n{creative_response}\n\nPurpose analysis:\n{step_back_response}\n\nPlease create a well-structured version."}
        ]
        
        # Call the AI for sequential structuring
        sequential_response = await ai.generate_text(
            messages=sequential_prompt,
            temperature=0.4,
            max_tokens=2000
        )
        
        # Format the final response with all components
        formatted_response = f"""# 🎨→🔍→🔄 Creative Development

## 🎨 Creative Generation
{creative_response}

## 🔍 Purpose Analysis
{step_back_response}

## 🔄 Structured Organization
{sequential_response}
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
        logger.info(f"Creative Development workflow completed in {workflow_time:.2f} seconds")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in creative_development_workflow: {str(e)}")
        traceback.print_exc()
        
        # Fall back to sequential reasoning on error
        try:
            from bot_utilities.services.agent_service import agent_service
            return await agent_service.execute_reasoning(
                reasoning_type="sequential",
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                update_callback=update_callback
            )
        except Exception as fallback_error:
            return f"I encountered an error processing your creative request: {str(e)}" 