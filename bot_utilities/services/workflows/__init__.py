"""
Workflows Package

This package contains various workflow implementations for different reasoning types.
Each workflow is designed to handle specific types of queries using a sequence of reasoning steps.
"""

# Import all workflow modules
from importlib import import_module
import os
import sys

# List all workflow modules
__all__ = [
    "analytical_problem_solving_workflow",
    "calculation_sequential_workflow",
    "controversial_topics_workflow",
    "creative_development_workflow",
    "creative_sequential_workflow",
    "cross_domain_innovation_workflow",
    "educational_explanations_workflow",
    "fact_checking_workflow",
    "graph_rag_verification_workflow",
    "knowledge_synthesis_workflow",
    "multi_agent_workflow",
    "personalized_advice_workflow",
    "predictive_scenarios_workflow",
    "rag_alone_workflow",
    "relationship_analysis_workflow",
    "sequential_rag_workflow",
    "strategic_planning_workflow",
    "technical_problem_workflow",
    "verification_rag_workflow"
]

# Function to get workflow by name
def get_workflow(workflow_name):
    """Get a workflow module by name"""
    try:
        # Convert from workflow_type naming to module naming
        module_name = f"{workflow_name}_workflow" if not workflow_name.endswith("_workflow") else workflow_name
        return import_module(f".{module_name}", package="bot_utilities.services.workflows")
    except ImportError as e:
        print(f"Error importing workflow {workflow_name}: {e}")
        return None 