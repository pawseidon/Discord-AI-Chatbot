"""
Technical Problem Workflow Implementation

This module implements the Technical Problem workflow, which combines
symbolic calculation with graph reasoning and verification
for solving complex technical and mathematical problems.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('technical_problem_workflow')

async def execute_technical_problem_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Technical Problem workflow combining symbolic calculation, graph reasoning, and verification
    for solving technical and mathematical problems
    
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
    from bot_utilities.services.symbolic_reasoning_service import symbolic_reasoning_service
    
    workflow_start_time = time.time()
    
    try:
        # Notify about starting the technical problem workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["symbolic", "graph", "verification"],
                "is_combined": True,
                "workflow": "technical_problem"
            })
        
        # Get AI provider
        ai = await get_ai_provider()
        
        # Step 1: Perform symbolic calculations and operations
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "symbolic_calculation",
                "emoji": "🧮",
                "message": "Performing mathematical calculations..."
            })
        
        # Check if there are mathematical expressions to calculate
        try:
            # Try to perform symbolic calculation on the query directly
            symbolic_result = await symbolic_reasoning_service.process_symbolic_reasoning(query)
            
            if symbolic_result and "result" in symbolic_result:
                symbolic_output = f"Symbolic Calculation:\n{symbolic_result.get('thinking', '')}\n\nResult: {symbolic_result.get('result', '')}"
            else:
                # If no direct calculation, perform a more general mathematical analysis
                symbolic_system = """You are a mathematical and computational expert who excels at solving technical problems.

For the given technical problem, provide a precise mathematical analysis:
1. Identify the key variables and parameters
2. Set up relevant equations or formulas
3. Perform step-by-step calculations
4. Apply appropriate mathematical or computational techniques
5. Provide exact numerical results with appropriate units

Show all your work clearly, explaining each step in a mathematically rigorous way.
Focus on precision and accuracy in your calculations and reasoning.
"""
                
                symbolic_prompt = [
                    {"role": "system", "content": symbolic_system},
                    {"role": "user", "content": f"Technical problem to solve: {query}\n\nPlease provide a detailed mathematical analysis with calculations."}
                ]
                
                symbolic_output = await ai.generate_text(
                    messages=symbolic_prompt,
                    temperature=0.2,
                    max_tokens=1200
                )
                
        except Exception as symbolic_error:
            logger.error(f"Error in symbolic calculation: {str(symbolic_error)}")
            symbolic_output = f"Symbolic calculation attempted but encountered an error: {str(symbolic_error)}\n\nProceeding with alternative analysis."
        
        # Step 2: Map relationships and dependencies with graph reasoning
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "relationship_mapping",
                "emoji": "📊",
                "message": "Mapping technical relationships and dependencies..."
            })
        
        # Create graph system message
        graph_system = """You are a systems analyst who specializes in mapping complex technical relationships and dependencies.

For the given technical problem and mathematical analysis, create a detailed relationship map that shows:
1. Key components and their interconnections
2. Cause-effect relationships between variables
3. Dependencies and constraints in the system
4. Critical paths and bottlenecks
5. System boundaries and interfaces

Your analysis should:
- Clearly identify all important elements in the system
- Show how changes in one component affect others
- Highlight feedback loops and non-linear relationships
- Identify potential points of failure or optimization
- Provide a holistic view of the entire technical system

Use clear structural organization and visual descriptions to convey the relationships.
"""
        
        # Create graph prompt
        graph_prompt = [
            {"role": "system", "content": graph_system},
            {"role": "user", "content": f"Technical problem: {query}\n\nMathematical analysis:\n{symbolic_output}\n\nPlease map the relationships and dependencies."}
        ]
        
        # Call the AI for graph mapping
        graph_response = await ai.generate_text(
            messages=graph_prompt,
            temperature=0.4,
            max_tokens=1200
        )
        
        # Step 3: Verify solution accuracy and completeness
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "solution_verification",
                "emoji": "✅",
                "message": "Verifying solution accuracy..."
            })
        
        # Create verification system message
        verification_system = """You are a technical validator who specializes in ensuring the correctness and completeness of solutions.

For the given technical problem, mathematical analysis, and relationship mapping, verify:
1. The correctness of all calculations and mathematical operations
2. The validity of all assumptions and constraints
3. The completeness of the solution with respect to all requirements
4. The robustness of the approach to edge cases and exceptions
5. The efficiency and optimality of the solution

Your verification should:
- Identify any errors or inaccuracies in the calculations
- Check for missing considerations or requirements
- Validate the approach against best practices and standards
- Test the solution with sample inputs or edge cases
- Assess overall solution quality and reliability

Provide a thorough verification that builds confidence in the final solution.
"""
        
        # Create verification prompt
        verification_prompt = [
            {"role": "system", "content": verification_system},
            {"role": "user", "content": f"Technical problem: {query}\n\nMathematical analysis:\n{symbolic_output}\n\nRelationship mapping:\n{graph_response}\n\nPlease verify the accuracy and completeness of this solution."}
        ]
        
        # Call the AI for verification
        verification_response = await ai.generate_text(
            messages=verification_prompt,
            temperature=0.3,
            max_tokens=1200
        )
        
        # Format the final response with all components
        formatted_response = f"""# 🧮→📊→✅ Technical Problem Solving

## 🧮 Mathematical Analysis
{symbolic_output}

## 📊 Relationship Mapping
{graph_response}

## ✅ Solution Verification
{verification_response}
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
        logger.info(f"Technical Problem workflow completed in {workflow_time:.2f} seconds")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in technical_problem_workflow: {str(e)}")
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
            return f"I encountered an error processing your technical problem: {str(e)}" 