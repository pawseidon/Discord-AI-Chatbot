"""
Fact Checking Workflow Implementation

This module implements the Fact Checking workflow, which combines
RAG (Retrieval-Augmented Generation), verification, and multi-agent perspectives
for validating claims and analyzing information accuracy.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fact_checking_workflow')

async def execute_fact_checking_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Fact Checking workflow combining RAG, verification, and multiple perspectives
    for validating claims and analyzing information accuracy
    
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
        # Notify about starting the fact checking workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["rag", "verification", "multi_agent"],
                "is_combined": True,
                "workflow": "fact_checking"
            })
        
        # Extract potential claims from the query if not already clear
        claims_to_check = query
        if not any(keyword in query.lower() for keyword in ["verify", "fact check", "check if", "is it true", "accurate"]):
            # If query doesn't explicitly ask for verification, try to extract claims
            from bot_utilities.ai_utils import get_ai_provider
            ai = await get_ai_provider()
            
            # Create claim extraction system message
            extraction_system = """You are a claim extraction specialist. Your task is to identify specific factual claims in a query that can be verified.
            
For the given query, extract the key factual claims that need verification. Focus on statements presented as facts rather than opinions or preferences.
            
Format each claim as a clear, verifiable statement."""
            
            # Create extraction prompt
            extraction_prompt = [
                {"role": "system", "content": extraction_system},
                {"role": "user", "content": f"Query: {query}\n\nPlease extract the key factual claims that need verification."}
            ]
            
            # Get potential claims
            extraction_response = await ai.generate_text(
                messages=extraction_prompt,
                temperature=0.2,
                max_tokens=500
            )
            
            claims_to_check = extraction_response
        
        # Step 1: Gather information through RAG
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "information_gathering",
                "emoji": "📚",
                "message": "Gathering relevant information..."
            })
        
        # Get web search results
        search_results = await agent_service.search_web(claims_to_check)
        
        # Get AI provider
        ai = await get_ai_provider()
        
        # Create RAG system message
        rag_system = """You are a fact-finding specialist tasked with gathering comprehensive information on specific claims.

For each claim, collect relevant information from multiple sources that can help determine its accuracy. Your goal is to:
1. Gather factual information related to each claim
2. Include details from reliable sources
3. Provide context and background information
4. Note different perspectives or interpretations
5. Include relevant statistics, dates, or measurements if available

Focus on collecting information objectively without making judgments about the claim's validity at this stage.
Organize your information clearly, separating facts related to different aspects of the claim.
"""
        
        # Create RAG prompt
        rag_prompt = [
            {"role": "system", "content": rag_system},
            {"role": "user", "content": f"Claims to investigate: {claims_to_check}\n\nInformation from sources:\n{search_results}\n\nPlease gather relevant information about these claims."}
        ]
        
        # Call the AI for information gathering
        rag_response = await ai.generate_text(
            messages=rag_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        
        # Step 2: Verify the claims based on evidence
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "claim_verification",
                "emoji": "✅",
                "message": "Verifying claims against evidence..."
            })
        
        # Create verification system message
        verification_system = """You are a fact-checking expert who rigorously verifies claims against evidence.

For each claim, carefully analyze the evidence to determine accuracy. Your verification should:
1. Evaluate each claim against the available evidence
2. Check for consistency with established facts
3. Identify potential contradictions or inconsistencies
4. Consider source reliability and potential biases
5. Rate claim accuracy on a scale (Accurate, Mostly Accurate, Partially Accurate, Mostly Inaccurate, Inaccurate)

Be careful to:
- Distinguish between proven facts, likely probabilities, and speculation
- Note where evidence is strong vs. where it is limited
- Identify potential errors or misunderstandings in the claims
- Provide context that might explain apparent contradictions
- Remain neutral and evidence-focused in your assessment

For each claim, provide a clear verdict with justification based strictly on the evidence.
"""
        
        # Create verification prompt
        verification_prompt = [
            {"role": "system", "content": verification_system},
            {"role": "user", "content": f"Claims to verify: {claims_to_check}\n\nEvidence and information:\n{rag_response}\n\nPlease verify these claims based on the evidence."}
        ]
        
        # Call the AI for verification
        verification_response = await ai.generate_text(
            messages=verification_prompt,
            temperature=0.3,
            max_tokens=1500
        )
        
        # Step 3: Analyze multiple perspectives on the claims
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "perspective_analysis",
                "emoji": "👥",
                "message": "Analyzing different perspectives..."
            })
        
        # Create multi-agent system message
        multi_agent_system = """You are an analyst who examines claims from multiple expert perspectives.

For the claims and verification provided, analyze how different experts or stakeholders might view these conclusions. Consider:
1. How different domain experts might interpret the evidence
2. Potential critiques or alternative interpretations of the verification
3. Areas where additional context might change understanding
4. How consensus vs. controversy in the field affects interpretation
5. Limitations of available evidence that might influence conclusions

Present at least 3 different expert perspectives that highlight:
- Areas of potential disagreement or differing interpretations
- Additional context that might be relevant
- Alternative frameworks for evaluating the claims
- Nuances that might be missed in a binary true/false assessment
- Implications of the claims being true, partially true, or false

The goal is to provide a more nuanced understanding that goes beyond simple verification.
"""
        
        # Create multi-agent prompt
        multi_agent_prompt = [
            {"role": "system", "content": multi_agent_system},
            {"role": "user", "content": f"Claims: {claims_to_check}\n\nEvidence gathered:\n{rag_response}\n\nVerification results:\n{verification_response}\n\nPlease analyze these from multiple expert perspectives."}
        ]
        
        # Call the AI for multi-perspective analysis
        multi_agent_response = await ai.generate_text(
            messages=multi_agent_prompt,
            temperature=0.5,
            max_tokens=1500
        )
        
        # Format the final response with all components
        formatted_response = f"""# 📚→✅→👥 Fact-Checking Analysis

## 📚 Evidence Collection
{rag_response}

## ✅ Claim Verification
{verification_response}

## 👥 Multiple Perspectives
{multi_agent_response}
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
        logger.info(f"Fact Checking workflow completed in {workflow_time:.2f} seconds")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in fact_checking_workflow: {str(e)}")
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
            return f"I encountered an error while fact-checking your request: {str(e)}" 