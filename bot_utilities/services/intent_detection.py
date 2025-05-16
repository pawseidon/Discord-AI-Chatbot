"""
Intent Detection Service

This module provides a centralized service for detecting user intent from message content.
It extracts patterns and intent detection logic from multiple places in the codebase.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Union
import discord
from bot_utilities.ai_utils import get_bot_names_and_triggers

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
            r"can you (?:search|google|find|look up|research) (?:for |about )?(.+)",
            r"(?:recent|latest|current|ongoing|today's|this week's|new) (?:events|news|developments|situation|conflict|crisis|updates) (?:about|on|regarding|concerning|in|with|between) (.+)",
            r"(?:break down|explain|analyze|summarize) (?:the |)(?:recent|latest|current|ongoing) (.+)"
        ]
        
        # Social relationship intent patterns
        self.social_relationship_patterns = [
            r"how (?:do|can|should) (?:i|you|one|we) (?:find|get|meet|attract|date|approach|talk to) (women|men|girls|guys|dates|partners)",
            r"where (?:do|can|should) (?:i|you|one|we) (?:find|get|meet|attract|date) (women|men|girls|guys|dates|partners)",
            r"ways to (?:find|get|meet|attract|date|approach|talk to) (women|men|girls|guys|dates|partners)",
            r"tips for (?:finding|getting|meeting|attracting|dating|approaching) (women|men|girls|guys|dates|partners)"
        ]
        
        # Crypto price intent patterns
        self.crypto_price_patterns = [
            r"(?:what(?:'s| is) (?:the )?(?:current |latest |present |today(?:'s)? )?(?:price|value) (?:of |for )?)(bitcoin|btc|ethereum|eth|solana|sol|dogecoin|doge|cardano|ada|bnb|xrp|polkadot|dot|avalanche|avax|polygon|matic).*",
            r"(?:how much (?:is |does )(?:a |one )?)(bitcoin|btc|ethereum|eth|solana|sol|dogecoin|doge|cardano|ada|bnb|xrp|polkadot|dot|avalanche|avax|polygon|matic) (?:cost|worth|going for).*",
            r"(bitcoin|btc|ethereum|eth|solana|sol|dogecoin|doge|cardano|ada|bnb|xrp|polkadot|dot|avalanche|avax|polygon|matic) (?:price|value).*"
        ]
        
        # Sequential thinking intent patterns
        self.sequential_thinking_patterns = [
            r"(?:use |try |apply |do )?sequential (?:thinking|reasoning|approach|method)(?: for| on| about| to solve)? (.+)",
            r"(?:solve |think through |break down |analyze |approach |tackle |understand |explain |evaluate |examine )(?:this |the |my |following )?(?:problem|question|task|query|issue|challenge)(?: step by step| sequentially| step-by-step| methodically| systematically)? (.+)",
            r"(?:step by step|step-by-step|sequentially|systematically|methodically) (?:solve|approach|analyze|think about|break down|explain|evaluate) (.+)",
            r"(?:help me|can you|please|could you) (?:solve|approach|analyze|work through|understand|explain) (?:this|the following|the) (?:step by step|systematically|sequentially|methodically) (.+)",
            r"(?:walk|guide) me through (?:solving|understanding|tackling|approaching) (.+)",
            r"(?:break down|analyze|explain) (?!recent|current|ongoing|latest)(.+) (?:step by step|systematically|sequentially|one by one)"
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
            r"(?:use |try |apply )?(?:complex|multiple perspective|varied) (?:reasoning|solution|approach|analysis)(?: for| on| about| to)? (.+)",
            r"(?:solve with|use) (?:specialized|multiple|various|different) agents(?: for| on| about| to)? (.+)",
            r"(?:analyze|explore) (.+) (?:from|with) (?:multiple|different|various) (?:perspectives|viewpoints|angles|lenses)",
            r"show me (?:different|multiple|various|contrasting) (?:perspectives|viewpoints|opinions|takes) on (.+)",
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
        
        # Check for social relationship patterns which need specialized handling
        for pattern in self.social_relationship_patterns:
            match = re.search(pattern, message_lower)
            if match:
                return {"intent": "social_relationship", "query": message_content.strip()}
        
        # Check for direct search intent with "search for" patterns
        search_intent_prefixes = [
            "search", "look up", "find information", "search for", "google", 
            "find", "look up", "research", "break down", "check", "info on",
            "information about", "recent news", "tell me about", "latest on"
        ]
        
        # Check for country or conflict names that often require recent information
        news_entities = [
            "ukraine", "russia", "israel", "gaza", "palestine", "china", "taiwan", 
            "india", "pakistan", "conflict", "war", "crisis", "election", "climate",
            "economy", "pandemic", "protest", "attack", "summit", "agreement", "treaty"
        ]
        
        # First, detect if the message is asking for a current news or search query
        is_asking_for_search = any(prefix in message_lower for prefix in search_intent_prefixes)
        is_asking_about_news = any(entity in message_lower for entity in news_entities)
        contains_current_markers = any(marker in message_lower for marker in ["recent", "latest", "current", "ongoing", "today", "now"])
        
        # If it looks like a search query for recent information
        if (is_asking_for_search or is_asking_about_news or contains_current_markers):
            # First try to extract query using regex patterns
            for pattern in self.search_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    groups = match.groups()
                    if groups:
                        query = groups[0].strip() if groups[0] else message_content
                        return {"intent": "web_search", "query": query}
            
            # If regex didn't work but it's clearly a search intent, extract query intelligently
            if is_asking_for_search or contains_current_markers or is_asking_about_news:
                # Look for @Bot mention and extract the query
                bot_mention_pattern = r'@\w+\s+(.*)'
                bot_mention_match = re.search(bot_mention_pattern, message_content)
                
                if bot_mention_match:
                    # Extract everything after the bot mention
                    query = bot_mention_match.group(1).strip()
                else:
                    # Try to extract query by removing common prefixes
                    query = message_content
                    for prefix in ["search for", "search", "find", "tell me about", "check"]:
                        if message_lower.startswith(prefix):
                            query = message_content[len(prefix):].strip()
                            break
                
                # Clean up the query by removing bot names and persona mentions
                bot_info = get_bot_names_and_triggers()
                bot_names = bot_info["names"]
                trigger_words = bot_info["triggers"]
                
                # Remove bot names from query
                for name in bot_names:
                    query = re.sub(f"\\b{re.escape(name)}\\b", "", query, flags=re.IGNORECASE).strip()
                    
                # Remove common command prefixes that might be in the query
                prefixes = ["hey", "hi", "hello", "ok", "okay", "please", "can you", "could you", "help me", "i need"]
                for prefix in prefixes:
                    query = re.sub(f"^{re.escape(prefix)}\\s+", "", query, flags=re.IGNORECASE).strip()
                
                # Remove trailing question mark(s) and punctuation marks
                query = re.sub(r'[\?\!\.\,]+$', '', query).strip()
                
                # Clean up the query if needed (remove common prompt phrases)
                for phrase in ["give me your honest opinion", "and give me your honest opinion", 
                              "what do you think", "and tell me what you think", "what's your take"]:
                    if phrase in query.lower():
                        query = query.lower().replace(phrase, "").strip()
                
                return {"intent": "web_search", "query": query}
        
        # Check for crypto price intent
        for pattern in self.crypto_price_patterns:
            match = re.search(pattern, message_lower)
            if match:
                crypto_name = match.groups()[0]
                return {"intent": "crypto_price", "crypto": crypto_name}
        
        # Removed image and voice detection code
        
        # Check for sequential thinking intent - prioritize over multi-agent for step-by-step analysis
        if "step by step" in message_lower or "step-by-step" in message_lower:
            for pattern in self.sequential_thinking_patterns:
                match = re.search(pattern, message_lower)
                if match:
                    groups = match.groups()
                    if groups:
                        problem = groups[0].strip() if groups[0] else message_content
                        return {"intent": "sequential_thinking", "problem": problem}
            
            # If contains "step by step" but didn't match a specific pattern, default to sequential
            return {"intent": "sequential_thinking", "problem": message_content}
        
        # Check for multi-agent intent
        for pattern in self.multi_agent_patterns:
            match = re.search(pattern, message_lower)
            if match:
                groups = match.groups()
                if groups:
                    query = groups[0].strip()
                    return {"intent": "multi_agent", "query": query}
        
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