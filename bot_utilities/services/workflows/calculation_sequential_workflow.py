"""
Calculation Sequential Workflow Implementation

This module implements the Calculation Sequential workflow, which combines
symbolic calculation with sequential thinking for precise mathematical
operations with step-by-step explanations.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('calculation_sequential_workflow')

async def execute_calculation_sequential_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Calculation Sequential workflow combining precise mathematical operations with step-by-step explanations
    
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
    from bot_utilities.services.sequential_thinking_service import sequential_thinking_service
    from bot_utilities.services.symbolic_reasoning_service import symbolic_reasoning_service
    from bot_utilities.ai_utils import get_ai_provider
    
    workflow_start_time = time.time()
    
    try:
        # Notify about calculation workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["calculation", "sequential"],
                "is_combined": True
            })
            
        # Step 1: Extract the mathematical expression from the query
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "extraction",
                "emoji": "🔍",
                "message": "Extracting mathematical expression..."
            })
            
        # Get AI provider for prompting
        ai = await get_ai_provider()
        
        # Create extraction prompt
        extraction_prompt = [
            {"role": "system", "content": "You are a specialized agent for extracting mathematical expressions from text. Extract a mathematical or logical expression that represents the problem. Use standard mathematical notation that could be evaluated by a computer."},
            {"role": "user", "content": f"Problem: {query}\n\nExpression:"}
        ]
        
        # Call the AI to extract the expression
        expression = await ai.generate_text(
            messages=extraction_prompt,
            temperature=0.1,
            max_tokens=200
        )
        
        # Clean up expression
        expression = expression.strip()
        logger.info(f"Extracted expression: {expression}")
        
        # Step 2: Calculate the result using symbolic reasoning
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "calculation",
                "emoji": "🧮",
                "message": "Calculating result..."
            })
            
        calculation_result = await symbolic_reasoning_service.evaluate_expression(expression)
        
        # Step 3: Provide a sequential explanation of the calculation
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "explanation",
                "emoji": "🔄",
                "message": "Creating step-by-step explanation..."
            })
            
        # Create context for sequential thinking
        context = {
            "expression": expression,
            "calculation_result": calculation_result,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "interleaved_format": True  # Use the interleaved format for revisions
        }
        
        # Process with sequential thinking for explanation
        success, explanation = await sequential_thinking_service.process_sequential_thinking(
            problem=f"Explain how to solve this mathematical problem step by step: {expression}",
            context=context,
            prompt_style="sequential",
            max_steps=5,
            enable_revision=True,
            session_id=f"calc_seq_{user_id}_{int(time.time())}"
        )
        
        if not success:
            logger.warning(f"Sequential thinking for explanation had issues: {explanation[:100]}...")
        
        # Format the final response
        result_value = calculation_result.get("result", "Unable to calculate")
        steps = "\n".join([f"• {step}" for step in calculation_result.get("steps", [])])
        
        formatted_response = f"""🧮📚 **Mathematical Solution with Explanation**

**Expression:** `{expression}`
**Result:** `{result_value}`

**Step-by-step solution:**
{explanation}"""
        
        # Calculate workflow timing
        workflow_time = time.time() - workflow_start_time
        logger.info(f"Calculation Sequential workflow completed in {workflow_time:.2f} seconds")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in calculation_sequential_workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to conversational reasoning
        try:
            return f"I encountered an error while processing your mathematical request: {str(e)}"
        except Exception as fallback_error:
            return f"Error processing mathematical calculation: {str(e)}" 