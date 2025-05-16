"""
Knowledge Synthesis Workflow Implementation

This module implements the Knowledge Synthesis workflow, which combines
RAG (Retrieval-Augmented Generation), graph-based relationship mapping, and
chain-of-thought reasoning for comprehensive learning and knowledge integration.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('knowledge_synthesis_workflow')

async def execute_knowledge_synthesis_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Knowledge Synthesis workflow combining RAG, graph mapping, and chain-of-thought reasoning
    
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
        # Notify about starting the knowledge synthesis workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["rag", "graph", "chain_of_thought"],
                "is_combined": True,
                "workflow": "knowledge_synthesis"
            })
        
        # Step 1: Gather comprehensive information from diverse sources
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "knowledge_gathering",
                "emoji": "📚",
                "message": "Gathering comprehensive information..."
            })
        
        # Get AI provider
        ai = await get_ai_provider()
        
        # Get web search results with enhanced detail
        rag_results = await agent_service.search_web(query)
        
        # Create knowledge gathering system message
        knowledge_system = """You are a knowledge synthesis expert tasked with creating a comprehensive knowledge base on a specific topic.

Your goal is to organize and structure information into a clear, hierarchical framework that spans from foundational to advanced concepts.

For the given topic and information, synthesize it into these sections:
1. Core concepts and fundamentals - key definitions and basic principles
2. Key principles and mechanisms - how the system/topic works
3. Important relationships and dependencies - how different components interact
4. Historical context and development - evolution and key milestones
5. Current applications and relevance - real-world usage and significance

Use clear headings and bullet points to organize information logically.
Focus on accuracy, comprehensiveness, and educational value.
"""
        
        # Create knowledge gathering prompt
        knowledge_prompt = [
            {"role": "system", "content": knowledge_system},
            {"role": "user", "content": f"Topic for knowledge synthesis: {query}\n\nInformation from various sources:\n{rag_results}\n\nPlease create a comprehensive knowledge synthesis."}
        ]
        
        # Call the AI for knowledge synthesis
        knowledge_response = await ai.generate_text(
            messages=knowledge_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        
        # Step 2: Create a relationship graph of knowledge connections
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "knowledge_mapping",
                "emoji": "📊",
                "message": "Mapping knowledge connections and relationships..."
            })
        
        # Create knowledge graph system message
        graph_system = """You are a knowledge mapping specialist who creates conceptual graphs of relationships between ideas.

Create a comprehensive knowledge map that shows relationships between concepts for the given topic.
Focus on these relationship types:

1. Hierarchical relationships (is-a, part-of)
2. Causal relationships (leads-to, causes)
3. Functional relationships (used-for, enables)
4. Comparative relationships (similar-to, different-from)
5. Temporal relationships (precedes, follows)

Format the map using clear headings, bullet points, and indentation to show structure.
Use special characters like →, ↔, ⟹, etc. to indicate relationship types.
Create a clear visual hierarchy using text formatting like sections and subsections.
"""
        
        # Create graph prompt
        graph_prompt = [
            {"role": "system", "content": graph_system},
            {"role": "user", "content": f"Topic for knowledge mapping: {query}\n\nKnowledge synthesis to map:\n{knowledge_response}\n\nPlease create a detailed knowledge relationship map."}
        ]
        
        # Call the AI for graph creation
        graph_response = await ai.generate_text(
            messages=graph_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        
        # Step 3: Integrate knowledge through chain-of-thought reasoning
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "knowledge_integration",
                "emoji": "⛓️",
                "message": "Integrating knowledge through logical reasoning..."
            })
        
        # Create integration system message
        integration_system = """You are an integration specialist who uses Chain-of-Thought reasoning to connect ideas into coherent understanding.

Create an integrated understanding of the topic that builds knowledge progressively and logically. Your response should:

1. Build from basic to advanced concepts in a clear progression
2. Make explicit logical connections between ideas using "because," "therefore," "which means," etc.
3. Identify dependencies and prerequisites for understanding complex ideas
4. Connect theoretical principles to practical applications
5. Synthesize information into a unified mental model

Use a step-by-step reasoning approach that shows how each concept builds upon previous ones.
Include "thinking steps" that reveal the logical bridges connecting different concepts.
Create an explanation that would help someone develop true understanding, not just memorize facts.
"""
        
        # Create integration prompt
        integration_prompt = [
            {"role": "system", "content": integration_system},
            {"role": "user", "content": f"Topic for knowledge integration: {query}\n\nKnowledge synthesis:\n{knowledge_response}\n\nKnowledge relationships:\n{graph_response}\n\nPlease create an integrated understanding using chain-of-thought reasoning."}
        ]
        
        # Call the AI for integration
        integration_response = await ai.generate_text(
            messages=integration_prompt,
            temperature=0.3,
            max_tokens=1800
        )
        
        # Format the final response with all components
        formatted_response = f"""# 📚→📊→⛓️ Knowledge Synthesis

## 📚 Foundational Knowledge
{knowledge_response}

## 📊 Conceptual Relationships
{graph_response}

## ⛓️ Integrated Understanding
{integration_response}
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
        logger.info(f"Knowledge Synthesis workflow completed in {workflow_time:.2f} seconds")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in knowledge_synthesis_workflow: {str(e)}")
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
            return f"I encountered an error processing your knowledge request: {str(e)}" 