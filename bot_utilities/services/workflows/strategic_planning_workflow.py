"""
Strategic Planning Workflow Implementation

This module implements the Strategic Planning workflow, which combines
graph-based relationship mapping, sequential organization, and step-back analysis
for project planning, roadmaps, and strategy development.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('strategic_planning_workflow')

async def execute_strategic_planning_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Strategic Planning workflow combining graph mapping, sequential organization,
    and step-back analysis for comprehensive planning and strategy development
    
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
        # Notify about starting the strategic planning workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["graph", "sequential", "step_back"],
                "is_combined": True,
                "workflow": "strategic_planning"
            })
        
        # Get AI provider
        ai = await get_ai_provider()
        
        # Step 1: Map relationships and dependencies with graph reasoning
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "relationship_mapping",
                "emoji": "📊",
                "message": "Mapping strategic relationships and dependencies..."
            })
        
        # Create graph system message
        graph_system = """You are a strategic relationship mapper who specializes in identifying connections and dependencies.

For the given strategic planning request, create a comprehensive map that shows:
1. Key stakeholders and their relationships
2. Critical components and dependencies
3. Resource flows and constraints
4. Timeline dependencies and milestones
5. Potential bottlenecks or critical paths

Your analysis should:
- Clearly identify all important elements in the strategic context
- Show how different components affect each other
- Highlight dependencies that could impact timelines
- Identify potential risks or challenges
- Consider both internal and external factors

Create a structured map of relationships that provides a foundation for detailed planning.
"""
        
        # Create graph prompt
        graph_prompt = [
            {"role": "system", "content": graph_system},
            {"role": "user", "content": f"Strategic planning request: {query}\n\nPlease create a comprehensive relationship map for this strategy."}
        ]
        
        # Call the AI for graph mapping
        graph_response = await ai.generate_text(
            messages=graph_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        
        # Step 2: Organize into a sequential plan
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "sequential_planning",
                "emoji": "🔄",
                "message": "Creating structured action plan..."
            })
        
        # Create sequential system message
        sequential_system = """You are a strategic planning expert who excels at organizing actions in logical sequences.

Based on the relationship map provided, create a detailed sequential plan that:
1. Organizes actions into clear phases or stages
2. Establishes a logical progression from start to completion
3. Accounts for dependencies identified in the relationship map
4. Specifies timelines or sequencing constraints
5. Designates ownership or responsibility for key actions

Your sequential plan should:
- Have a clear beginning, middle, and end
- Break down complex processes into manageable steps
- Specify prerequisites for each major step
- Include milestones for tracking progress
- Consider resource allocation across the timeline

Provide a well-structured plan that can be followed step-by-step to achieve the strategic goals.
"""
        
        # Create sequential prompt
        sequential_prompt = [
            {"role": "system", "content": sequential_system},
            {"role": "user", "content": f"Strategic planning request: {query}\n\nRelationship map:\n{graph_response}\n\nPlease create a detailed sequential plan."}
        ]
        
        # Call the AI for sequential planning
        sequential_response = await ai.generate_text(
            messages=sequential_prompt,
            temperature=0.3,
            max_tokens=1500
        )
        
        # Step 3: Take a step back for strategic alignment and big picture analysis
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "step_back_analysis",
                "emoji": "🔍",
                "message": "Analyzing strategic alignment and big picture..."
            })
        
        # Create step-back system message
        step_back_system = """You are a strategic advisor who excels at seeing the big picture and ensuring alignment with broader objectives.

For the strategic plan and relationship map provided, perform a high-level analysis that:
1. Evaluates alignment with overarching goals and mission
2. Considers long-term implications and strategic positioning
3. Identifies potential blind spots or unexamined assumptions
4. Assesses adaptability to changing conditions
5. Examines alignment with organizational values and culture

Your analysis should:
- Step back from tactical details to see the bigger picture
- Question underlying assumptions
- Consider alternative approaches or scenarios
- Identify potential unintended consequences
- Ensure the strategy is coherent and internally consistent

Provide strategic insights that strengthen the plan by connecting it to broader context and purpose.
"""
        
        # Create step-back prompt
        step_back_prompt = [
            {"role": "system", "content": step_back_system},
            {"role": "user", "content": f"Strategic planning request: {query}\n\nRelationship map:\n{graph_response}\n\nSequential plan:\n{sequential_response}\n\nPlease provide a big-picture strategic analysis."}
        ]
        
        # Call the AI for step-back analysis
        step_back_response = await ai.generate_text(
            messages=step_back_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        
        # Format the final response with all components
        formatted_response = f"""# 📊→🔄→🔍 Strategic Planning

## 📊 Strategic Relationship Map
{graph_response}

## 🔄 Sequential Action Plan
{sequential_response}

## 🔍 Strategic Alignment Analysis
{step_back_response}
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
        logger.info(f"Strategic Planning workflow completed in {workflow_time:.2f} seconds")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in strategic_planning_workflow: {str(e)}")
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
            return f"I encountered an error processing your strategic planning request: {str(e)}"
