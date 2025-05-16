"""
Verification RAG Workflow Implementation

This module implements the Verification RAG workflow, which combines
RAG (Retrieval-Augmented Generation) with verification reasoning for
fact-checking and accuracy validation.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List, Union

# Import workflow helper
from bot_utilities.services.workflow_helper import (
    WorkflowResult, 
    add_to_conversation_history,
    parse_conversation_id,
    standardize_workflow_output
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('verification_rag_workflow')

async def execute_verification_rag_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None,
    search_results: Optional[str] = None
) -> Union[str, Dict[str, Any]]:
    """
    Execute the Verification RAG workflow combining search and verification
    
    Args:
        query: The user query to process
        user_id: The user ID for context and memory
        conversation_id: Optional conversation ID
        update_callback: Optional callback for status updates
        search_results: Optional pre-fetched search results
        
    Returns:
        Union[str, Dict[str, Any]]: The formatted workflow response
    """
    # Import services (lazy imports to avoid circular dependencies)
    from bot_utilities.services.agent_service import agent_service
    from bot_utilities.ai_utils import search_internet
    
    workflow_start_time = time.time()
    
    try:
        # Step 1: RAG - Retrieval stage
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "retrieval",
                "emoji": "📚",
                "message": "Retrieving information..."
            })
        
        # Use provided search results or fetch new ones
        if not search_results:
            search_results = await search_internet(query)
            
        if not search_results or search_results.strip() == "":
            search_results = "No search results found. Using existing knowledge to answer."
            
        # Format search results for the AI
        formatted_results = f"### Search Results:\n{search_results}\n\n"
        
        # Log for debugging
        logger.info(f"Retrieved search results for '{query[:30]}...' ({len(search_results)} chars)")
        
        # Step 2: Extract claim(s) to verify
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "claim_extraction",
                "emoji": "🔍",
                "message": "Identifying key claims to verify..."
            })
            
        # Prepare claim extraction prompt
        extraction_prompt = f"""
        Based on the following query, identify the main factual claims that need verification:
        
        Query: {query}
        
        Please list each distinct claim that needs to be verified. Focus only on claims that 
        can be fact-checked against external information.
        """
        
        # Execute the claim extraction step
        claims_response = await agent_service.execute_reasoning(
            reasoning_type="analytical",
            query=extraction_prompt,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Standardize the response
        claims_result = standardize_workflow_output(claims_response)
        claims_text = claims_result["response"]
        
        # Step 3: Verification stage
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "verification",
                "emoji": "✅",
                "message": "Verifying claims against information sources..."
            })
            
        # Prepare verification prompt
        verification_prompt = f"""
        Verify the following claims based on the search results:
        
        Claims to verify:
        {claims_text}
        
        Search Results:
        {formatted_results}
        
        For each claim:
        1. State whether it is True, False, Partially True, or Unverified
        2. Provide evidence from the search results that supports your assessment
        3. Note any conflicting information or uncertainties
        4. Rate your confidence in the verification (High, Medium, Low)
        
        After verifying each claim, provide an overall assessment of the query's accuracy.
        """
        
        # Execute the verification step
        verification_response = await agent_service.execute_reasoning(
            reasoning_type="verification",
            query=verification_prompt,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Standardize the response
        verification_result = standardize_workflow_output(verification_response)
        verification_text = verification_result["response"]
        
        # Format the final response
        final_response = f"""# Verification Results

## Original Query
{query}

## Verification Analysis
{verification_text}

"""
        
        # Parse conversation_id to get channel_id
        _, channel_id = parse_conversation_id(conversation_id)
        
        # Store conversation in memory
        if channel_id:
            try:
                # Add conversation to history
                await add_to_conversation_history(
                    user_id=user_id,
                    channel_id=channel_id,
                    query=query,
                    response=final_response
                )
            except Exception as e:
                logger.error(f"Error storing conversation: {str(e)}")
                traceback.print_exc()
        
        # Calculate workflow timing
        workflow_time = time.time() - workflow_start_time
        logger.info(f"Verification RAG workflow completed in {workflow_time:.2f} seconds")
        
        # Create result object
        result = WorkflowResult(
            response=final_response,
            confidence=0.9,  # Verification workflows typically have high confidence
            metadata={
                "workflow": "verification_rag",
                "execution_time": workflow_time,
                "search_results_length": len(search_results) if search_results else 0,
                "claims_analyzed": len(claims_text.split("\n")) if claims_text else 0
            },
            thinking=f"Claims extracted:\n{claims_text}\n\nVerification process:\n{verification_text}"
        )
        
        # Return standardized result
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error in verification_rag_workflow: {str(e)}")
        traceback.print_exc()
        error_result = WorkflowResult.error(str(e))
        return error_result.to_dict() 