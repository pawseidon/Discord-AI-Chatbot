"""
RAG Alone Workflow Implementation

This module implements the pure RAG (Retrieval-Augmented Generation) workflow 
for simple factual queries and information retrieval.
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
logger = logging.getLogger('rag_alone_workflow')

async def execute_rag_alone_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None,
    search_results: Optional[str] = None
) -> Union[str, Dict[str, Any]]:
    """
    Execute the pure RAG workflow for information retrieval
    
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
            
        # Log for debugging
        if search_results:
            logger.info(f"Retrieved search results for '{query[:30]}...' ({len(search_results)} chars)")
        else:
            logger.info("No search results found")
            
        if not search_results or search_results.strip() == "":
            search_results = "No search results found. Using existing knowledge to answer."
            
        # Format search results for the AI
        formatted_results = f"### Search Results:\n{search_results}\n\n"
        
        # Step 2: Information Synthesis
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "synthesis",
                "emoji": "📝",
                "message": "Synthesizing information into a response..."
            })
        
        # Prepare RAG prompt
        rag_prompt = f"""
        Based on the search results below, please provide a comprehensive and accurate answer 
        to the following query:
        
        Query: {query}
        
        {formatted_results}
        
        Your answer should:
        1. Be accurate and directly address the query
        2. Be well-structured and easy to read
        3. Synthesize information from multiple sources when relevant
        4. Acknowledge when information is limited or uncertain
        """
        
        # Execute the RAG reasoning step
        rag_response = await agent_service.execute_reasoning(
            reasoning_type="rag",
            query=rag_prompt,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Standardize the response
        rag_result = standardize_workflow_output(rag_response)
        response_text = rag_result["response"]
        
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
                    response=response_text
                )
            except Exception as e:
                logger.error(f"Error storing conversation: {str(e)}")
                traceback.print_exc()
        
        # Calculate workflow timing
        workflow_time = time.time() - workflow_start_time
        logger.info(f"RAG Alone workflow completed in {workflow_time:.2f} seconds")
        
        # Create result object
        result = WorkflowResult(
            response=response_text,
            confidence=0.8,
            metadata={
                "workflow": "rag_alone",
                "execution_time": workflow_time,
                "search_results_length": len(search_results) if search_results else 0
            }
        )
        
        # Return standardized result
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error in rag_alone_workflow: {str(e)}")
        traceback.print_exc()
        error_result = WorkflowResult.error(str(e))
        return error_result.to_dict() 