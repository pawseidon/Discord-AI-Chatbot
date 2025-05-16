"""
Relationship Analysis Workflow Implementation

This module implements the Relationship Analysis workflow, which combines
graph-based relationship mapping, RAG for information retrieval, and multi-agent analysis
for understanding complex networks, systems, and interconnections.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('relationship_analysis_workflow')

async def execute_relationship_analysis_workflow(
    query: str,
    user_id: str,
    conversation_id: str = None,
    update_callback: Optional[Callable] = None
) -> str:
    """
    Execute the Relationship Analysis workflow combining graph mapping, information retrieval,
    and multiple perspectives for analyzing complex networks and interconnections
    
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
        # Notify about starting the relationship analysis workflow
        if update_callback:
            await update_callback("reasoning_switch", {
                "reasoning_types": ["graph", "rag", "multi_agent"],
                "is_combined": True,
                "workflow": "relationship_analysis"
            })
        
        # Get AI provider
        ai = await get_ai_provider()
        
        # Step 1: Create a graph-based relationship map
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "relationship_mapping",
                "emoji": "📊",
                "message": "Creating relationship map..."
            })
        
        # Create graph system message
        graph_system = """You are a network analyst who specializes in mapping complex relationships and interconnections.

For the given relationship analysis request, create a comprehensive network map that shows:
1. Key entities or nodes in the network
2. Connections and relationships between these entities
3. Relationship types, directions, and strengths
4. Clusters, hubs, or central nodes
5. Structural patterns in the network

Your analysis should:
- Clearly identify each important entity or component
- Precisely describe the nature of relationships between entities
- Highlight central entities that have many connections
- Identify isolated clusters or subgroups
- Note any unusual or significant patterns in the network structure

Create a detailed map of the relationship network that helps visualize the complex interconnections.
"""
        
        # Create graph prompt
        graph_prompt = [
            {"role": "system", "content": graph_system},
            {"role": "user", "content": f"Relationship analysis request: {query}\n\nPlease create a comprehensive relationship map."}
        ]
        
        # Call the AI for graph mapping
        graph_response = await ai.generate_text(
            messages=graph_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        
        # Step 2: Gather additional information about the relationships using RAG
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "information_retrieval",
                "emoji": "📚",
                "message": "Gathering additional relationship information..."
            })
        
        # Get web search results if helpful
        try:
            # Extract key entities to search for additional information
            from bot_utilities.ai_utils import get_ai_provider
            ai = await get_ai_provider()
            
            # Create entity extraction system message
            extraction_system = """You are an entity extraction specialist. Your task is to identify key entities in a relationship network that need more information.
            
For the given relationship map, extract 3-5 key entities or relationships that would benefit from additional information. Focus on the most central or important elements in the network.
            
Format each entity as a clear, searchable term."""
            
            # Create extraction prompt
            extraction_prompt = [
                {"role": "system", "content": extraction_system},
                {"role": "user", "content": f"Relationship analysis request: {query}\n\nRelationship map:\n{graph_response}\n\nPlease extract key entities for information gathering."}
            ]
            
            # Get key entities
            extraction_response = await ai.generate_text(
                messages=extraction_prompt,
                temperature=0.2,
                max_tokens=300
            )
            
            # Search for additional information
            search_query = f"{query} {extraction_response}"
            search_results = await agent_service.search_web(search_query)
        except Exception as search_error:
            logger.error(f"Error in search: {str(search_error)}")
            search_results = "Unable to gather additional information from web search."
        
        # Create RAG system message
        rag_system = """You are a relationship information specialist who enhances network maps with additional details and context.

Using the additional information provided, enhance the relationship map by:
1. Adding missing relationships or entities that are relevant
2. Providing additional context about key relationships
3. Explaining historical developments or changes in relationships
4. Clarifying the nature, strength, or quality of connections
5. Adding quantitative data or qualitative assessments about relationships

Your goal is to:
- Integrate new information with the existing relationship map
- Resolve any contradictions or ambiguities in the relationship structure
- Add depth and nuance to the understanding of key connections
- Highlight especially significant or influential relationships
- Provide a more complete picture of the network dynamics

Create an enhanced relationship map that incorporates this additional information.
"""
        
        # Create RAG prompt
        rag_prompt = [
            {"role": "system", "content": rag_system},
            {"role": "user", "content": f"Relationship analysis request: {query}\n\nInitial relationship map:\n{graph_response}\n\nAdditional information:\n{search_results}\n\nPlease enhance the relationship map with this information."}
        ]
        
        # Call the AI for information enhancement
        rag_response = await ai.generate_text(
            messages=rag_prompt,
            temperature=0.4,
            max_tokens=1500
        )
        
        # Step 3: Analyze the relationships from multiple perspectives
        if update_callback:
            await update_callback("workflow_stage", {
                "stage": "multi_perspective_analysis",
                "emoji": "👥",
                "message": "Analyzing from multiple perspectives..."
            })
        
        # Create multi-agent system message
        multi_agent_system = """You are a multi-perspective analyst who examines relationships from diverse viewpoints.

For the given relationship network, analyze it from at least 3 different expert perspectives, such as:
1. A network theorist analyzing structural patterns and emergent properties
2. A domain expert interpreting the significance of specific relationships
3. A historian examining the evolution or development of these connections
4. A predictive analyst forecasting how these relationships might change
5. A critical theorist questioning underlying assumptions or power dynamics

For each perspective, provide:
- The specific lens or framework being applied
- Key insights from this particular viewpoint
- Areas this perspective highlights that others might miss
- Potential implications or conclusions from this perspective
- Limitations of this perspective in understanding the full network

Offer a rich, multi-dimensional analysis that illuminates different aspects of the relationship network.
"""
        
        # Create multi-agent prompt
        multi_agent_prompt = [
            {"role": "system", "content": multi_agent_system},
            {"role": "user", "content": f"Relationship analysis request: {query}\n\nEnhanced relationship map:\n{rag_response}\n\nPlease analyze this relationship network from multiple perspectives."}
        ]
        
        # Call the AI for multi-perspective analysis
        multi_agent_response = await ai.generate_text(
            messages=multi_agent_prompt,
            temperature=0.5,
            max_tokens=1500
        )
        
        # Format the final response with all components
        formatted_response = f"""# 📊→📚→👥 Relationship Analysis

## 📊 Relationship Network Map
{graph_response}

## 📚 Enhanced Relationship Information
{rag_response}

## 👥 Multi-Perspective Analysis
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
        logger.info(f"Relationship Analysis workflow completed in {workflow_time:.2f} seconds")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error in relationship_analysis_workflow: {str(e)}")
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
            return f"I encountered an error analyzing these relationships: {str(e)}" 