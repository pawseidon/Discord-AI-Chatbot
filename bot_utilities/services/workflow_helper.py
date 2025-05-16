"""
Workflow Helper Module

This module provides helper functions, utilities, and standardized interfaces
for the multi-agent workflow system, ensuring consistent behavior across workflows.
"""

import logging
import traceback
import time
import os
import json
import re
from typing import Dict, Any, Optional, Callable, Tuple, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('workflow_helper')

class WorkflowResult:
    """
    Standardized workflow result container to ensure consistent return values
    """
    def __init__(
        self, 
        response: str, 
        confidence: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        thinking: Optional[str] = None
    ):
        """
        Initialize a workflow result
        
        Args:
            response: The formatted response text
            confidence: Confidence score (0.0-1.0) for the result
            metadata: Optional metadata about the workflow execution
            error: Optional error message if the workflow failed
            thinking: Optional thinking/reasoning process
        """
        self.response = response
        self.confidence = min(max(confidence, 0.0), 1.0)  # Clamp to 0.0-1.0
        self.metadata = metadata or {}
        self.error = error
        self.thinking = thinking
        self.success = error is None
        
    @classmethod
    def error(cls, error_message: str) -> 'WorkflowResult':
        """
        Create an error result
        
        Args:
            error_message: The error message
            
        Returns:
            WorkflowResult: An error result object
        """
        return cls(
            response=f"Error: {error_message}",
            confidence=0.0,
            error=error_message,
            metadata={"error": True}
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary
        
        Returns:
            Dict[str, Any]: The result as a dictionary
        """
        return {
            "response": self.response,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "error": self.error,
            "thinking": self.thinking,
            "success": self.success
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowResult':
        """
        Create a result from a dictionary
        
        Args:
            data: The dictionary data
            
        Returns:
            WorkflowResult: A result object
        """
        return cls(
            response=data.get("response", ""),
            confidence=data.get("confidence", 0.8),
            metadata=data.get("metadata", {}),
            error=data.get("error"),
            thinking=data.get("thinking")
        )

def standardize_workflow_output(output: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Standardize the output format from a workflow to ensure consistent interface
    
    Args:
        output: The output from a workflow execution (string or dict)
        
    Returns:
        Dict[str, Any]: Standardized output dictionary
    """
    try:
        # If output is already a string, wrap it in a minimal dict
        if isinstance(output, str):
            return {
                "response": output,
                "confidence": 0.8,
                "metadata": {},
                "success": True
            }
            
        # If output is a WorkflowResult, convert to dict
        if isinstance(output, WorkflowResult):
            return output.to_dict()
            
        # If output is already a dict, ensure it has required fields
        if isinstance(output, dict):
            # If it appears to be in the expected format, return as is
            if "response" in output:
                # Ensure all required fields exist
                output.setdefault("confidence", 0.8)
                output.setdefault("metadata", {})
                output.setdefault("success", output.get("error") is None)
                return output
                
            # If it has "result" instead of "response", standardize it
            if "result" in output:
                return {
                    "response": output["result"],
                    "confidence": output.get("confidence", 0.8),
                    "metadata": output.get("metadata", {}),
                    "error": output.get("error"),
                    "thinking": output.get("thinking"),
                    "success": output.get("error") is None
                }
                
            # If it has neither, try to extract content or use the whole dict
            if "content" in output:
                response = output["content"]
            elif "text" in output:
                response = output["text"]
            else:
                # Convert the whole dict to a string
                response = str(output)
                
            return {
                "response": response,
                "confidence": output.get("confidence", 0.8),
                "metadata": {},
                "success": True
            }
            
        # For any other type, convert to string
        return {
            "response": str(output),
            "confidence": 0.5,
            "metadata": {"warning": "Non-standard output format"},
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error standardizing workflow output: {str(e)}")
        traceback.print_exc()
        
        # Return error result
        return {
            "response": f"Error processing result: {str(e)}",
            "confidence": 0.0,
            "metadata": {"error": True},
            "error": str(e),
            "success": False
        }

def parse_conversation_id(conversation_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a conversation ID to extract guild_id and channel_id
    
    Args:
        conversation_id: The conversation ID string (format: "guild_id:channel_id")
        
    Returns:
        Tuple[Optional[str], Optional[str]]: The guild_id and channel_id
    """
    if not conversation_id:
        return None, None
        
    parts = conversation_id.split(":", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return None, conversation_id

async def add_to_conversation_history(
    user_id: str,
    channel_id: str,
    query: str,
    response: str
) -> bool:
    """
    Add an exchange to the conversation history
    
    Args:
        user_id: The user ID
        channel_id: The channel ID
        query: The user query
        response: The bot response
        
    Returns:
        bool: Whether the operation was successful
    """
    try:
        # Import here to avoid circular imports
        from bot_utilities.services.memory_service import memory_service
        
        # Add to history
        await memory_service.add_to_history(
            user_id=user_id,
            channel_id=channel_id,
            message=query,
            response=response
        )
        return True
    except Exception as e:
        logger.error(f"Error adding to conversation history: {str(e)}")
        traceback.print_exc()
        return False

async def record_workflow_usage(
    workflow_name: str,
    query: str,
    user_id: str,
    execution_time: float,
    success: bool = True,
    error: Optional[str] = None
) -> bool:
    """
    Record workflow usage for analytics
    
    Args:
        workflow_name: The workflow name/type
        query: The query that was processed
        user_id: The user ID
        execution_time: The execution time in seconds
        success: Whether the workflow completed successfully
        error: Optional error message if the workflow failed
        
    Returns:
        bool: True if successfully recorded
    """
    try:
        # Store raw usage data to avoid circular imports with workflow_service
        # This prevents the recursive call pattern that was causing stack overflow
        
        # Print basic stats (without calling workflow_service)
        print(f"📊 Workflow recorded: {workflow_name}, execution time: {execution_time:.2f}s")
        
        # Log any errors
        if not success and error:
            print(f"⚠️ Workflow '{workflow_name}' failed with error: {error}")
                
        return True
    except Exception as e:
        print(f"Error recording workflow usage: {str(e)}")
        traceback.print_exc()
        return False

# Optional type definitions for workflow interfaces
WorkflowExecutor = Callable[[str, str, Optional[str], Optional[Callable]], Union[str, Dict[str, Any]]]
UpdateCallback = Callable[[str, Dict[str, Any]], None]

# Helper functions for string formatting
def extract_error_message(exception: Exception) -> str:
    """
    Extract a clean error message from an exception
    
    Args:
        exception: The exception object
        
    Returns:
        str: Clean error message
    """
    error_msg = str(exception)
    # Remove common error prefixes
    error_msg = re.sub(r'^Error: ', '', error_msg)
    return error_msg

def format_workflow_error(
    workflow_name: str,
    error: Exception,
    query: str = ""
) -> str:
    """
    Format a user-friendly error message for workflow failures
    
    Args:
        workflow_name: The name of the workflow
        error: The exception that occurred
        query: The original query
        
    Returns:
        str: Formatted error message
    """
    error_msg = extract_error_message(error)
    workflow_display_name = workflow_name.replace('_', ' ').title()
    
    return f"""I encountered an issue while processing your request using the {workflow_display_name} workflow.

Error details: {error_msg}

Please try rephrasing your query or using a different approach.""" 