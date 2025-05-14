"""
Intent Detection Service

This module provides a centralized service for detecting user intent from message content.
It extracts patterns and intent detection logic from multiple places in the codebase.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Union
import discord

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('intent_detection_service')

class IntentDetectionService:
    """Service for detecting user intent from message content"""
    
    def __init__(self):
        # Removed image generation, image analysis, voice transcription, and sentiment analysis patterns
        
        # Web search intent patterns
        self.search_patterns = [
            r"(?:search|find|look up|google|research) (.+)",
            r"what is the latest(?: news| information) (?:about|on) (.+)",
            r"find information (?:about|on) (.+)"
        ]
        
        # Sequential thinking intent patterns
        self.sequential_thinking_patterns = [
            r"(?:use |try |apply |do )?sequential thinking(?: for| on| about)? (.+)",
            r"(?:solve |think through |break down |analyze |approach )(?:this |the |my )?(?:problem|question|task)(?: step by step| sequentially| step-by-step)? (.+)",
            r"(?:step by step|step-by-step|sequentially) (?:solve|approach|analyze|think about) (.+)",
            r"(?:help me|can you|please) (?:solve|approach|analyze|work through) (?:this|the following) (?:step by step|systematically|sequentially) (.+)"
        ]
        
        # MCP Agent intent patterns
        self.mcp_agent_patterns = [
            r"(?:use |try |apply |with )?(?:mcp|agent)(?: for| on| about)? (.+)",
            r"(?:agent|assistant)(?: help me| can you help| please help)(?: with)? (.+)"
        ]
        
        # Multi-agent intent patterns
        self.multi_agent_patterns = [
            r"(?:use |try |apply |with )?(?:multi-agent|multi agent|multiple agents)(?: for| on| about)? (.+)",
            r"(?:use |try |complex)(?: reasoning| solution| approach)(?: for| on| about)? (.+)",
            r"(?:solve with|use) specialized agents(?: for| on| about)? (.+)",
            r"(?:delegate|collaborate|team)(?: this| on this| with this)? (.+)",
            r"(?:orchestrate|coordinate)(?: a solution to| an answer for)? (.+)"
        ]
        
        # Privacy command patterns
        self.privacy_patterns = [
            r"clear my data",
            r"forget me",
            r"delete my (?:data|information|history)",
            r"reset my (?:data|information|history)",
            r"forget our conversation",
            r"remove my data",
            r"privacy"
        ]

    async def detect_intent(self, message_content: str, message: discord.Message) -> Dict[str, Any]:
        """
        Detect user intent from message content and attachments
        
        Args:
            message_content: The text content of the message
            message: The full Discord message object with attachments
            
        Returns:
            Dict containing the detected intent and relevant parameters
        """
        message_lower = message_content.lower()
        
        # Check for privacy commands first - highest priority
        for pattern in self.privacy_patterns:
            if re.search(pattern, message_lower):
                return {"intent": "privacy", "action": "clear_data"}
        
        # Removed image and voice detection code
        
        # Check for multi-agent intent
        for pattern in self.multi_agent_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    query = groups[0].strip()
                    return {"intent": "multi_agent", "query": query}
        
        # Check for sequential thinking intent
        for pattern in self.sequential_thinking_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    problem = groups[0].strip()
                    return {"intent": "sequential_thinking", "problem": problem}
        
        # Check for MCP agent intent
        for pattern in self.mcp_agent_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    query = groups[0].strip()
                    return {"intent": "mcp_agent", "query": query}
        
        # Check for web search intent
        for pattern in self.search_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    query = groups[0].strip()
                    return {"intent": "web_search", "query": query}
        
        # Check for symbolic reasoning intent (math, logic)
        if re.search(r"(solve|calculate|compute|evaluate|what is|simplify|factor)", message_lower):
            # Detect mathematical expressions using a basic regex
            math_match = re.search(r"[0-9+\-*/()^=<>]+", message_content)
            if math_match:
                expression = math_match.group(0).strip()
                return {"intent": "symbolic_reasoning", "expression": expression}
        
        # Default to chat intent if no specific intent detected
        return {"intent": "chat", "content": message_content}

intent_service = IntentDetectionService() 