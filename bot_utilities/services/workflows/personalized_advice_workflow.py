"""
Personalized Advice Workflow Implementation

This module implements the Personalized Advice workflow, which combines
Contextual → RAG → Verification (👤→📚→✅) reasoning for personalized 
recommendations and adaptive responses.
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
logger = logging.getLogger('personalized_advice_workflow')

async def execute_personalized_advice_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> Union[str, Dict[str, Any]]:
    """
    Execute the Personalized Advice workflow for tailored recommendations
    
    Args:
        query: The user query to process
        user_id: The user ID for context and memory
        conversation_id: Optional conversation ID
        update_callback: Optional callback for status updates
        
    Returns:
        Union[str, Dict[str, Any]]: The formatted workflow response
    """
    # Import services (lazy imports to avoid circular dependencies)
    from bot_utilities.services.agent_service import agent_service
    from bot_utilities.services.memory_service import memory_service
    from bot_utilities.ai_utils import search_internet
    
    workflow_start_time = time.time()
    
    try:
        # Step 1: Contextual - Analyze user context and preferences
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "contextual",
                "emoji": "👤",
                "message": "Analyzing user context and preferences..."
            })
        
        # Retrieve user memory and preferences
        user_memory = await memory_service.get_conversation_history(user_id, limit=10)
        user_preferences = await memory_service.get_user_preferences(user_id)
        
        # Create context analysis prompt
        context_prompt = f"""
        Based on this user's history and preferences, analyze the context for personalizing this request:
        
        Current request: {query}
        
        User conversation history (most recent 10 exchanges):
        {user_memory if user_memory else "No previous conversation history available."}
        
        User preferences:
        {user_preferences if user_preferences else "No specific preferences stored."}
        
        Analyze:
        1. What is this user's communication style and preferences?
        2. What topics has this user shown interest in previously?
        3. What level of expertise does this user appear to have?
        4. How should the response be tailored specifically for this user?
        """
        
        # Execute the contextual analysis step
        context_response = await agent_service.execute_reasoning(
            reasoning_type="contextual",
            query=context_prompt,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Standardize the response
        context_result = standardize_workflow_output(context_response)
        context_analysis = context_result["response"]
        
        # Step 2: RAG - Retrieve personalized information
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "rag",
                "emoji": "📚",
                "message": "Retrieving relevant personalized information..."
            })
            
        # Create personalized search query
        enhanced_query = f"""
        {query}
        
        Context: {context_analysis[:300]}...
        """
        
        # Perform the information retrieval
        search_results = await search_internet(enhanced_query)
            
        if not search_results or search_results.strip() == "":
            search_results = "No specific external information found. Using existing knowledge to respond."
            
        # Format search results for the AI
        formatted_results = f"### Search Results:\n{search_results}\n\n"
        
        # Create personalized information prompt
        info_prompt = f"""
        Based on the following search results and user context, provide personalized information
        tailored specifically to this user's needs and preferences:
        
        User query: {query}
        
        User context: {context_analysis}
        
        Search results: {formatted_results}
        
        Provide information that is:
        1. Directly relevant to the user's specific query
        2. Tailored to their apparent level of expertise and interests
        3. Presented in a style that matches their communication preferences
        4. Comprehensive enough to be helpful without overwhelming
        """
        
        # Execute the personalized information retrieval
        info_response = await agent_service.execute_reasoning(
            reasoning_type="rag",
            query=info_prompt,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Standardize the response
        info_result = standardize_workflow_output(info_response)
        personalized_info = info_result["response"]
        
        # Step 3: Verification - Ensure recommendations are accurate and relevant
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "verification",
                "emoji": "✅",
                "message": "Verifying personalized recommendations for relevance and accuracy..."
            })
            
        # Create verification prompt
        verification_prompt = f"""
        Verify the accuracy and personal relevance of these recommendations:
        
        User query: {query}
        
        User context: {context_analysis}
        
        Proposed recommendations: {personalized_info}
        
        Verification checks:
        1. Factual accuracy: Are all statements and claims accurate?
        2. Personal relevance: How well do these recommendations align with the user's context?
        3. Completeness: Is any key information missing based on the query?
        4. Potential improvements: What could make these recommendations more personally relevant?
        
        Provide a verification assessment including any necessary corrections or enhancements.
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
        verification = verification_result["response"]
        
        # Create final synthesis prompt
        synthesis_prompt = f"""
        Create personalized advice that combines:
        
        1. User context: {context_analysis}
        2. Retrieved information: {personalized_info}
        3. Verification insights: {verification}
        
        Original query: {query}
        
        Format the response as personalized advice specifically tailored to this user's context, 
        preferences, and needs. Make sure the tone is appropriate and the recommendations are 
        practical and actionable.
        """
        
        # Execute the final synthesis
        from bot_utilities.ai_utils import get_ai_provider
        ai = await get_ai_provider()
        
        final_response = await ai.generate_text(
            prompt=synthesis_prompt,
            max_tokens=2000,
            temperature=0.5
        )
        
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
                
        # Update user preferences with inferred preferences
        try:
            # Extract any user preferences from the context analysis
            preference_prompt = f"""
            Based on this context analysis, identify specific user preferences that could be saved
            for future personalization. Format as a JSON dictionary with preference keys and values.
            
            Context analysis:
            {context_analysis}
            
            Example format:
            {{
              "communication_style": "technical",
              "expertise_level": "advanced",
              "interests": ["technology", "finance"]
            }}
            
            Only include preferences that are clearly indicated in the context.
            """
            
            pref_response = await ai.generate_text(
                prompt=preference_prompt,
                max_tokens=500,
                temperature=0.2
            )
            
            # Extract JSON dict from the response if possible
            import json
            import re
            
            # Look for JSON-like structure in the response
            json_match = re.search(r'\{.*\}', pref_response, re.DOTALL)
            
            if json_match:
                try:
                    pref_dict = json.loads(json_match.group(0))
                    if pref_dict and isinstance(pref_dict, dict):
                        # Update user preferences
                        current_prefs = await memory_service.get_user_preferences(user_id) or {}
                        
                        # Merge with existing preferences, giving priority to new ones
                        updated_prefs = {**current_prefs, **pref_dict}
                        
                        # Save updated preferences
                        await memory_service.set_user_preferences(user_id, updated_prefs)
                        logger.info(f"Updated user preferences for {user_id} with inferred data")
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Couldn't parse inferred preferences: {e}")
        except Exception as e:
            logger.warning(f"Error updating user preferences: {e}")
        
        # Calculate workflow timing
        workflow_time = time.time() - workflow_start_time
        logger.info(f"Personalized Advice workflow completed in {workflow_time:.2f} seconds")
        
        # Create result object
        result = WorkflowResult(
            response=final_response,
            confidence=0.85,
            metadata={
                "workflow": "personalized_advice",
                "execution_time": workflow_time,
                "reasoning_chain": "contextual→rag→verification"
            },
            thinking=f"User context:\n{context_analysis}\n\nPersonalized information:\n{personalized_info}\n\nVerification:\n{verification}"
        )
        
        # Return standardized result
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error in personalized_advice_workflow: {str(e)}")
        traceback.print_exc()
        error_result = WorkflowResult.error(str(e))
        return error_result.to_dict() 