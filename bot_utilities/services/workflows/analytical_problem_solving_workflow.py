"""
Analytical Problem Solving Workflow Implementation

This module implements the Analytical Problem Solving workflow, which combines
detailed analysis, component breakdown, and symbolic calculation
for complex problem analysis, debugging, and technical solutions.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('analytical_problem_solving_workflow')

async def execute_analytical_problem_solving_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Analytical Problem Solving workflow combining detailed analysis, component breakdown, and technical solutions
    
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
        # Notify about starting the analytical problem solving workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["detail_analysis", "component_breakdown", "symbolic_calculation"],
                "is_combined": True,
                "workflow": "analytical_problem_solving"
            })
        
        # Get AI provider
        ai = await get_ai_provider()
        
        # Step 1: Detailed examination of the problem from all angles
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "detailed_analysis",
                "emoji": "🔎",
                "message": "Examining the problem in detail..."
            })
        
        # Create detailed analysis system message
        analysis_system = """You are an analytical problem-solving expert who excels at detailed analysis.

Examine the given problem in comprehensive detail by:
1. Defining the problem scope and boundaries precisely
2. Identifying key symptoms, errors, or inefficiencies
3. Listing potential causes and contributing factors
4. Noting patterns or unusual characteristics
5. Considering contextual factors that may be relevant

Be thorough in your analysis, looking for:
- Subtle details that might be overlooked
- Edge cases and exception conditions
- Environmental factors affecting the system
- Performance constraints and bottlenecks
- Historical issues or recurring patterns

Your goal is to create a comprehensive problem analysis that doesn't jump to solutions yet,
but instead thoroughly explores the problem space from multiple angles."""
        
        # Create detailed analysis prompt
        analysis_prompt = [
            {"role": "system", "content": analysis_system},
            {"role": "user", "content": f"Problem to analyze in detail: {query}\n\nPlease provide a comprehensive analysis of this problem."}
        ]
        
        # Call the AI for detailed analysis
        analysis_response = await ai.generate_text(
            messages=analysis_prompt,
            temperature=0.4,
            max_tokens=1200
        )
        
        # Step 2: Break down the problem into individual components
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "component_breakdown",
                "emoji": "🧩",
                "message": "Breaking the problem into components..."
            })
        
        # Create component breakdown system message
        component_system = """You are a systems analyst who specializes in breaking complex problems into modular components.

For the problem and analysis provided, create a systematic breakdown of components by:
1. Identifying distinct modular components or subsystems
2. Analyzing each component's role, function, and purpose
3. Examining interactions and dependencies between components
4. Prioritizing components by their impact on the overall problem
5. Identifying which components are likely causing issues or bottlenecks

Your component breakdown should:
- Use clear hierarchical organization
- Show parent-child relationships between components
- Indicate critical paths and dependencies
- Highlight key components that warrant further investigation
- Estimate the relative complexity or importance of each component

Focus on creating a clear, structured breakdown that would help someone understand the system architecture."""
        
        # Create component breakdown prompt
        component_prompt = [
            {"role": "system", "content": component_system},
            {"role": "user", "content": f"Problem: {query}\n\nDetailed analysis:\n{analysis_response}\n\nPlease break this problem down into its component parts."}
        ]
        
        # Call the AI for component breakdown
        component_response = await ai.generate_text(
            messages=component_prompt,
            temperature=0.4,
            max_tokens=1200
        )
        
        # Step 3: Determine precise solutions with calculations if needed
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "solution_development",
                "emoji": "🧮",
                "message": "Calculating precise solutions..."
            })
        
        # Create solution system message
        solution_system = """You are a technical problem solver who develops precise, actionable solutions.

Based on the detailed analysis and component breakdown provided, develop comprehensive solutions that:
1. Address each problematic component systematically
2. Provide specific, actionable remedies with clear steps
3. Include calculations, formulas, or algorithms where appropriate
4. Specify optimal parameters or configurations
5. Support solutions with clear reasoning and evidence

Your solutions should be:
- Precise and detailed enough to implement directly
- Prioritized by importance and impact
- Supported with technical justifications
- Accompanied by expected outcomes
- Presented with any necessary calculations or formulas

If calculations are required, show your work clearly and explain the reasoning behind the calculations.
Focus on creating solutions that are both technically sound and practical to implement."""
        
        # Create solution prompt
        solution_prompt = [
            {"role": "system", "content": solution_system},
            {"role": "user", "content": f"Problem: {query}\n\nDetailed analysis:\n{analysis_response}\n\nComponent breakdown:\n{component_response}\n\nPlease develop precise technical solutions for this problem."}
        ]
        
        # Call the AI for solution development
        solution_response = await ai.generate_text(
            messages=solution_prompt,
            temperature=0.3,
            max_tokens=1500
        )
        
        # Format the final response with all components
        formatted_response = f"""# 🔎→🧩→🧮 Analytical Problem Solving

## 🔎 Detailed Problem Analysis
{analysis_response}

## 🧩 Component Breakdown
{component_response}

## 🧮 Precise Solutions
{solution_response}
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
        logger.info(f"Analytical Problem Solving workflow completed in {workflow_time:.2f} seconds")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in analytical_problem_solving_workflow: {str(e)}")
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
            return f"I encountered an error processing your problem solving request: {str(e)}"