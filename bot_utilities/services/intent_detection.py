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
            r"(?:search|find|look up|google|research|browse|fetch) (?:for |about |information on |details on |info on |data on )?(.+)",
            r"what is the latest(?: news| information| update| development) (?:about|on|regarding|concerning|for) (.+)",
            r"find (?:me )?(?:some |recent |more |any )?information (?:about|on|regarding|concerning|for) (.+)",
            r"can you (?:search|google|find|look up|research) (?:for |about )?(.+)"
        ]
        
        # Sequential thinking intent patterns
        self.sequential_thinking_patterns = [
            r"(?:use |try |apply |do )?sequential (?:thinking|reasoning|approach|method)(?: for| on| about| to solve)? (.+)",
            r"(?:solve |think through |break down |analyze |approach |tackle |understand |explain |evaluate |examine )(?:this |the |my |following )?(?:problem|question|task|query|issue|challenge)(?: step by step| sequentially| step-by-step| methodically| systematically)? (.+)",
            r"(?:step by step|step-by-step|sequentially|systematically|methodically) (?:solve|approach|analyze|think about|break down|explain|evaluate) (.+)",
            r"(?:help me|can you|please|could you) (?:solve|approach|analyze|work through|understand|explain) (?:this|the following|the) (?:step by step|systematically|sequentially|methodically) (.+)",
            r"(?:walk|guide) me through (?:solving|understanding|tackling|approaching) (.+)"
        ]
        
        # MCP Agent intent patterns
        self.mcp_agent_patterns = [
            r"(?:use |try |apply |with )?(?:mcp|agent)(?: for| on| about| to)? (.+)",
            r"(?:agent|assistant)(?: help me| can you help| please help| could you help)(?: with)? (.+)",
            r"(?:let |have )(?:the )?(?:agent|assistant) (?:tackle|solve|handle|address|manage|work on) (.+)"
        ]
        
        # Multi-agent intent patterns
        self.multi_agent_patterns = [
            r"(?:use |try |apply |with )?(?:multi-agent|multi agent|multiple agents|team of agents)(?: for| on| about| to)? (.+)",
            r"(?:use |try |apply )?complex(?: reasoning| solution| approach| analysis)(?: for| on| about| to)? (.+)",
            r"(?:solve with|use) (?:specialized|multiple|various|different) agents(?: for| on| about| to)? (.+)",
            r"(?:delegate|collaborate|team)(?: this| on this| with this| up for)? (.+)",
            r"(?:orchestrate|coordinate|collaborate on)(?: a solution to| an answer for| a response to| an approach for)? (.+)",
            r"(?:have |make |let )(?:agents|multiple agents) (?:work |collaborate |cooperate )(?:on|together on|to solve) (.+)"
        ]
        
        # Creative reasoning patterns
        self.creative_patterns = [
            r"(?:create|write|compose|draft|generate) (?:a |an |some )?(?:story|poem|song|essay|article|blog post|creative text|narrative|fiction) (?:about|on|regarding|concerning|for) (.+)",
            r"(?:make|be|get) (?:creative|imaginative|artistic) (?:with|about|on) (.+)",
            r"(?:imagine|envision|visualize) (?:a |an |some )?(?:scenario|situation|world|setting|story) (?:where|in which|about) (.+)",
            r"(?:use |try |apply )?creative (?:thinking|reasoning|approach|method)(?: for| on| about| to)? (.+)"
        ]
        
        # Knowledge/RAG reasoning patterns
        self.knowledge_patterns = [
            r"(?:explain|describe|what is|what are|tell me about|define|elaborate on) (.+)",
            r"(?:how does|how do|how can|how did|who is|who was|where is|when was|why is|why does) (.+)",
            r"(?:I want to learn|teach me|educate me|inform me) (?:about|on|regarding) (.+)",
            r"(?:what|who|where|when|why|how) (?:.{1,30})? (.+)\?",
            r"(?:use |try |apply )?(?:knowledge|rag|information|facts)(?: for| on| about| to)? (.+)"
        ]
        
        # Calculation/math patterns
        self.calculation_patterns = [
            r"(?:calculate|compute|evaluate|what is|solve|find) (?:the )?(?:value of |answer to |result of |solution to )?([0-9+\-*/()^=<>√∛∜π√{}\[\]%±×÷∑∏∫∬∭∮∯∰∱∲∳⊕⊖⊗⊘⊙⌈⌉⌊⌋⌢⌣⟨⟩∀∃∄∈∉∋∌∧∨].{0,100})",
            r"(?:simplify|factor|expand|differentiate|integrate|derive) (?:the )?(?:expression|equation|formula|function)? ([0-9+\-*/()^=<>√∛∜π√{}\[\]%±×÷∑∏∫∬∭∮∯∰∱∲∳⊕⊖⊗⊘⊙⌈⌉⌊⌋⌢⌣⟨⟩∀∃∄∈∉∋∌∧∨].{0,100})",
            r"(?:convert|transform) ([0-9+\-*/()^=<>√∛∜π√{}\[\]%±×÷].{0,50}) to (?:decimal|fraction|percent|binary|hexadecimal|octal)",
            r"(?:use |try |apply )?(?:math|calculation|numerical|arithmetic)(?: reasoning| approach)(?: for| on| about| to)? (.+)"
        ]
        
        # Verification patterns
        self.verification_patterns = [
            r"(?:verify|check|confirm|validate|is it true that|is this correct:) (.+)",
            r"(?:fact[- ]check|truth value of|accuracy of) (.+)",
            r"(?:is|are) (?:.{1,50}) (?:true|correct|accurate|valid|right)(?:\?|$)",
            r"(?:use |try |apply )?verification(?: reasoning| approach)(?: for| on| about| to)? (.+)"
        ]
        
        # Privacy command patterns
        self.privacy_patterns = [
            r"clear my data",
            r"forget me",
            r"delete my (?:data|information|history|conversation|messages)",
            r"reset my (?:data|information|history|conversation|messages)",
            r"forget (?:our|this) conversation",
            r"remove my data",
            r"privacy",
            r"clear (?:all|my) history"
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
        
        # Check for creative reasoning intent
        for pattern in self.creative_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    query = groups[0].strip()
                    return {"intent": "creative_reasoning", "query": query}
        
        # Check for knowledge/RAG intent
        for pattern in self.knowledge_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    query = groups[0].strip()
                    return {"intent": "knowledge", "query": query}
        
        # Check for calculation/math intent
        for pattern in self.calculation_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    expression = groups[0].strip()
                    return {"intent": "calculation", "expression": expression}
        
        # Check for verification intent
        for pattern in self.verification_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    query = groups[0].strip()
                    return {"intent": "verification", "query": query}
        
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
            # Detect mathematical expressions using a broader regex
            math_match = re.search(r"[0-9+\-*/()^=<>√∛∜π√{}\[\]%±×÷∑∏∫∬∭∮∯∰∱∲∳⊕⊖⊗⊘⊙⌈⌉⌊⌋⌢⌣⟨⟩∀∃∄∈∉∋∌∧∨]{2,}", message_content)
            if math_match:
                expression = math_match.group(0).strip()
                return {"intent": "symbolic_reasoning", "expression": expression}
        
        # Default to chat intent if no specific intent detected
        return {"intent": "chat", "content": message_content}

intent_service = IntentDetectionService() 