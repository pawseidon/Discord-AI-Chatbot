"""
Educational Explanations Workflow Implementation

This module implements the Educational Explanations workflow, which combines
RAG (Retrieval-Augmented Generation) with sequential thinking and verification
for accurate and structured educational content.
"""

import logging
import json
import time
import traceback
import datetime
from typing import Dict, Any, Optional, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('educational_explanations_workflow')

async def execute_educational_explanations_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Educational Explanations workflow combining RAG, sequential thinking, and verification
    
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
    from bot_utilities.web_search import search_web
    from bot_utilities.ai_utils import get_ai_provider
    from bot_utilities.memory_utils import get_user_preferences
    from bot_utilities.services.memory_service import memory_service
    
    workflow_start_time = time.time()
    
    try:
        # Step 1: RAG - Retrieval stage
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "retrieval",
                "emoji": "📚",
                "message": "Searching for educational information..."
            })
        
        # Search for information
        search_results = await search_web(query)
            
        if not search_results:
            # Fall back to regular sequential reasoning if no search results
            return await agent_service.execute_reasoning(
                reasoning_type="sequential",
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                update_callback=update_callback
            )
        
        # Format search results
        formatted_results = "\n\n".join(
            f"Source {i+1}: {result['title']}\n{result['url']}\n{result['snippet']}"
            for i, result in enumerate(search_results[:5])
        )
        
        # Get AI provider
        ai = await get_ai_provider()
        
        # Create RAG context message
        rag_system = """You are an AI assistant that provides educational explanations.
Extract the most relevant information from the search results to answer the educational query.
Organize this information in a way that will be helpful for creating a step-by-step explanation."""
        
        rag_prompt = [
            {"role": "system", "content": rag_system},
            {"role": "user", "content": f"Educational Query: {query}\n\nSearch Results:\n{formatted_results}"}
        ]
        
        # Get the RAG response
        rag_response = await ai.generate_text(
            messages=rag_prompt,
            temperature=0.2,
            max_tokens=1000
        )
        
        # Store RAG context
        rag_context = {
            "search_results": search_results,
            "extracted_information": rag_response,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Step 2: Sequential Thinking - Create step-by-step explanation
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "sequential",
                "emoji": "🔄",
                "message": "Creating step-by-step explanation..."
            })
        
        # Create sequential prompt
        sequential_system = """You are an educational expert using step-by-step sequential thinking.
        
Create a clear, structured explanation for the educational query using the information provided.

Your explanation should:
1. Start with a brief overview of the topic
2. Break down complex concepts into clear sequential steps
3. Use examples to illustrate key points
4. Include helpful analogies where appropriate
5. End with a summarizing conclusion

Format your response with clear headings, numbered steps, and well-organized paragraphs.
Focus on accuracy and educational value."""
        
        sequential_prompt = [
            {"role": "system", "content": sequential_system},
            {"role": "user", "content": f"Educational Query: {query}\n\nRelevant Information:\n{rag_response}\n\nPlease provide a step-by-step educational explanation."}
        ]
        
        # Get the sequential response
        sequential_response = await ai.generate_text(
            messages=sequential_prompt,
            temperature=0.3,
            max_tokens=1500
        )
        
        # Store sequential context
        sequential_context = {
            "structured_explanation": sequential_response,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Standardize sequential output for verification
        standardized_sequential_data = {
            "claims_to_verify": sequential_response,
            "query": query
        }
        
        # Step 3: Verify the information for accuracy
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "verification",
                "emoji": "✅",
                "message": "Verifying accuracy of information..."
            })
            
        # Create verification prompt
        verification_system = """You are an expert fact-checker verifying educational content.
        
Carefully review the educational explanation for accuracy, looking for:
1. Factual errors or misconceptions
2. Oversimplifications that might be misleading
3. Areas where more nuance is needed
4. Claims that require additional evidence

Provide a concise verification summary that highlights:
- Points that are well-supported and accurate
- Any corrections or clarifications needed
- Suggestions for improving accuracy
- Overall assessment of reliability

Focus on the most important accuracy issues rather than minor details."""
        
        verification_prompt = [
            {"role": "system", "content": verification_system},
            {"role": "user", "content": f"Original query: {query}\n\nEducational explanation to verify:\n{sequential_response}"}
        ]
        
        # Get the verification response
        verification_response = await ai.generate_text(
            messages=verification_prompt,
            temperature=0.2,
            max_tokens=800
        )
        
        # Create verification context
        verification_context = {
            "is_verified": True,  # Default to true unless explicitly marked otherwise
            "evidence": verification_response,
            "confidence": 0.9,  # Default high confidence
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Check for corrections or low confidence indicators
        low_confidence_indicators = ["uncertain", "unclear", "not confident", "low confidence", "insufficient evidence"]
        correction_indicators = ["correction", "incorrect", "inaccurate", "should be", "actually", "in fact"]
        
        for indicator in low_confidence_indicators:
            if indicator in verification_response.lower():
                verification_context["confidence"] = 0.6
        
        for indicator in correction_indicators:
            if indicator in verification_response.lower():
                verification_context["is_verified"] = False
                break
        
        # Format the final response with all three reasoning components
        formatted_response = f"""# 📚→🔄→✅ Educational Explanation

{sequential_response}

## ✅ Verification Notes:
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
        
        # Store the complete workflow result in context for future reference
        workflow_result = {
            "rag_data": rag_context,
            "sequential_data": sequential_context,
            "verification_data": verification_context,
            "final_response": formatted_response
        }
        
        # Calculate workflow timing
        workflow_time = time.time() - workflow_start_time
        logger.info(f"Educational Explanations workflow completed in {workflow_time:.2f} seconds")
        
        # Create context for sequential thinking
        context = {
            "retrieved_information": search_results,
            "verification_results": verification_response,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "interleaved_format": True  # Enable interleaved format for clear revision visibility
        }
        
        # Process with sequential thinking
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "organization",
                "emoji": "🔄",
                "message": "Organizing verified information with thought revision and critical analysis..."
            })
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in educational_explanations_workflow: {str(e)}")
        traceback.print_exc()
        
        # Fall back to sequential reasoning on error
        try:
            return await agent_service.execute_reasoning(
                reasoning_type="sequential",
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                update_callback=update_callback
            )
        except Exception as fallback_error:
            return f"I encountered an error processing your educational request: {str(e)}" 