import re
import asyncio
from typing import Dict, Any, List, Tuple, Optional
import logging
from .reasoning_cache import ReasoningCache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('reasoning_utils')

class ReasoningDetector:
    """
    Detects the most appropriate reasoning mode based on user query content.
    Integrates emoji indicators and handles transitions between reasoning types.
    """
    
    # Emoji indicators for different reasoning types
    REASONING_EMOJIS = {
        'sequential': '🧠',  # Sequential thinking
        'rag': '🔍',         # RAG-based information retrieval
        'conversational': '💬',  # General conversation
        'knowledge': '📚',   # Knowledge-based reasoning
        'verification': '✅', # Verification and fact-checking
        'creative': '🎨',    # Creative tasks
        'calculation': '🔢',  # Math and calculations
        'planning': '📋',    # Planning and organization
        'graph': '🕸️',       # Graph-of-thought
        'multi_agent': '👥',  # Multi-agent reasoning
        'step_back': '🔎',    # Step-back thinking
        'cot': '⛓️',          # Chain-of-thought
        'react': '🔄',        # ReAct reasoning
    }
    
    # Keywords that indicate specific reasoning types
    REASONING_KEYWORDS = {
        'sequential': [
            'step by step', 'analyze', 'break down', 'complex', 'reasoning',
            'think through', 'logical steps', 'detailed analysis', 'in sequence',
            'methodically', 'systematically', 'thorough examination', 'deeper analysis',
            'carefully analyze', 'sequential thinking', 'step-wise', 'one by one',
            'in order', 'sequential process', 'walkthrough', 'detailed steps',
            'work out', 'solve methodically', 'procedural analysis', 'sequential approach'
        ],
        'rag': [
            'search', 'find', 'lookup', 'information about', 'tell me about',
            'research', 'what is', 'who is', 'when did', 'where is', 
            'facts about', 'details on', 'retrieve', 'source', 'look up',
            'search for', 'find information', 'search the web', 'get data on',
            'retrieve information', 'latest information', 'recent news about',
            'search online', 'look online', 'find out', 'search results',
            'fetch information', 'get me information', 'find me details'
        ],
        'conversational': [
            'chat', 'talk', 'discuss', 'conversation', 'opinion',
            'thoughts on', 'what do you think', 'how are you', 'tell me',
            'share your', 'let\'s talk', 'just chatting', 'casual discussion',
            'your perspective', 'your view', 'what\'s your take', 'hello',
            'hi there', 'good morning', 'good evening', 'nice to meet you',
            'how\'s it going', 'how have you been', 'what\'s up', 'hey there'
        ],
        'knowledge': [
            'explain', 'define', 'describe', 'summarize', 'elaborate on',
            'concept of', 'theory of', 'principle', 'meaning of', 'overview',
            'knowledge about', 'understanding', 'comprehension', 'summary',
            'teach me', 'learn about', 'educational', 'academic', 'scholarly',
            'in-depth explanation', 'detailed description', 'expound on',
            'clarify', 'elucidate', 'exposition', 'breakdown of', 'teach about'
        ],
        'verification': [
            'verify', 'fact-check', 'confirm', 'is it true', 'validate',
            'accuracy', 'correct', 'trustworthy', 'reliable', 'legitimate',
            'authenticate', 'check if', 'prove', 'evidence', 'citation',
            'is this correct', 'verify this claim', 'confirm accuracy',
            'substantiate', 'cross-check', 'double-check', 'corroborate',
            'debunk', 'refute', 'support this claim', 'back this up',
            'verify accuracy', 'is this verified', 'fact or fiction'
        ],
        'creative': [
            'create', 'generate', 'design', 'imagine', 'write',
            'story', 'poem', 'song', 'art', 'novel', 'script',
            'creative', 'innovative', 'original', 'invent', 'compose',
            'craft', 'artistic', 'fiction', 'narrative', 'fantasy',
            'imaginative', 'dream up', 'come up with', 'ideate',
            'brainstorm', 'conceptualize', 'envision', 'make up'
        ],
        'calculation': [
            'calculate', 'compute', 'solve', 'equation', 'formula',
            'math', 'arithmetic', 'numerical', 'quantitative', 'algebra',
            'calculus', 'computation', 'mathematical', 'number', 'quantify',
            'figure out', 'work out', 'add up', 'multiply', 'divide',
            'subtract', 'calculate the', 'find the value', 'what is the result',
            'compute the answer', 'solve for', 'evaluate', 'tally'
        ],
        'planning': [
            'plan', 'schedule', 'organize', 'strategy', 'roadmap',
            'steps to', 'how to', 'process for', 'approach to', 'method for',
            'outline', 'blueprint', 'framework', 'structure', 'agenda',
            'timeline', 'project plan', 'action plan', 'game plan', 'strategic plan',
            'planning for', 'planning steps', 'systematic approach', 'best way to',
            'create a plan', 'develop a strategy', 'organize a', 'layout'
        ],
        'graph': [
            'relationships', 'connections', 'network', 'linked concepts',
            'interconnected', 'graph', 'map the', 'conceptual map',
            'complex system', 'non-linear', 'interconnections', 'visualize connections',
            'concept map', 'mind map', 'network diagram', 'relationship map',
            'connection graph', 'relationship web', 'interlinked', 'nodal connections',
            'concept web', 'idea connections', 'linked themes', 'connection analysis',
            'map out connections', 'visualize relationships', 'explore connections'
        ],
        'multi_agent': [
            'different perspectives', 'multiple viewpoints', 'diverse opinions',
            'debate', 'pros and cons', 'arguments for and against',
            'collaborative analysis', 'team approach', 'expert panel',
            'multiple angles', 'various standpoints', 'differing views',
            'from multiple perspectives', 'conflicting opinions', 'balanced view',
            'from all sides', 'comprehensive perspectives', 'diverse viewpoints',
            'devil\'s advocate', 'consider all angles', 'balance perspectives'
        ],
        'step_back': [
            'broader perspective', 'big picture', 'holistic view', 'step back',
            'zoom out', 'higher level', 'overall context', 'broader implications',
            'wider context', 'meta-analysis', 'contextual analysis', 'bird\'s eye view',
            'helicopter view', 'from a distance', 'seeing the whole', 'forest for the trees',
            'broader context', 'overall landscape', 'big-picture thinking', 'macro view',
            'strategic perspective', 'contextual understanding', 'wider implications'
        ],
        'cot': [
            'chain of thought', 'logical sequence', 'one step at a time',
            'logical progression', 'if-then', 'causal chain', 
            'sequential logic', 'logical flow', 'step-by-step reasoning',
            'logical chain', 'consequential thinking', 'causal reasoning',
            'if this, then that', 'logical consequence', 'cause and effect',
            'logical path', 'reason through', 'logical steps', 'thought chain',
            'reasoning chain', 'follow the logic', 'logical approach'
        ],
        'react': [
            'act on', 'implement', 'execute', 'perform', 'carry out',
            'take action', 'do something', 'respond to', 'react to',
            'intervene', 'operation', 'action plan', 'execution',
            'put into practice', 'enact', 'undertake', 'conduct',
            'pursue a course of action', 'put to work', 'application',
            'actionable steps', 'take steps to', 'active response',
            'practical application', 'implement solution', 'executable plan'
        ]
    }
    
    # Query complexity indicators
    COMPLEXITY_INDICATORS = [
        'complex', 'complicated', 'intricate', 'elaborate', 'sophisticated',
        'multifaceted', 'nuanced', 'layered', 'detailed', 'in-depth',
        'thorough', 'comprehensive', 'exhaustive', 'advanced', 'high-level',
        'expert', 'difficult', 'challenging', 'hard', 'non-trivial',
        'specialized', 'technical', 'professional', 'recursive', 'nested',
        'hierarchical', 'interconnected', 'interdependent', 'multilayered',
        'sophisticated', 'deep dive', 'rigorous', 'scholarly', 'academic',
        'complicated matter', 'complex issue', 'hard problem', 'complex question'
    ]
    
    def __init__(self, enable_cache: bool = True, cache_ttl: int = 3600):
        """Initialize the reasoning detector"""
        # Store user preference for reasoning types (user_id -> reasoning_type)
        self.user_preferences = {}
        # Store conversation context (conversation_id -> context)
        self.conversation_contexts = {}
        # Store reasoning history for smooth transitions
        self.reasoning_history = {}
        # Store domain-specific reasoning preferences
        self.domain_preferences = {
            'math': 'calculation',
            'science': 'sequential',
            'history': 'knowledge',
            'politics': 'multi_agent',
            'art': 'creative',
            'programming': 'sequential',
            'philosophy': 'step_back',
            'news': 'rag',
            'business': 'planning',
            'technology': 'graph'
        }
        
        # Initialize caching system
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache = ReasoningCache(ttl=cache_ttl)
            logger.info("Reasoning cache initialized")
        
    def detect_reasoning_type(self, query: str, 
                             conversation_history: List[str] = None,
                             user_id: Optional[str] = None,
                             current_reasoning: Optional[str] = None,
                             conversation_id: Optional[str] = None) -> Tuple[str, float]:
        """
        Detects the most appropriate reasoning type for a query.
        
        Args:
            query: The user's query text
            conversation_history: Optional list of previous messages in the conversation
            user_id: Optional user ID for personalized detection
            current_reasoning: Optional current reasoning type for considering transitions
            conversation_id: Optional conversation/thread ID for caching
            
        Returns:
            tuple: (reasoning_type, confidence_score)
        """
        # Check cache first if enabled
        if self.enable_cache and conversation_id:
            cached_result = self.cache.get_cached_reasoning(
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                conversation_history=conversation_history
            )
            if cached_result:
                logger.debug(f"Cache hit for query: {query[:30]}...")
                return cached_result
                
        # Check for explicit reasoning mode requests
        explicit_mode = self._check_explicit_mode_request(query)
        if explicit_mode:
            result = (explicit_mode, 1.0)
            # Cache the result
            if self.enable_cache and conversation_id:
                self.cache.cache_reasoning(
                    query=query,
                    reasoning_result=result,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    conversation_history=conversation_history
                )
            return result
            
        # Check user preferences if user_id provided
        if user_id:
            # Try to get from cache first if enabled
            if self.enable_cache:
                preferred_type = self.cache.get_user_preference(user_id)
                if preferred_type is None and user_id in self.user_preferences:
                    # Cache miss but we have it in memory
                    preferred_type = self.user_preferences[user_id]
                    # Update cache
                    self.cache.set_user_preference(user_id, preferred_type)
            else:
                # Use in-memory preferences
                preferred_type = self.user_preferences.get(user_id)
                
            # Increase likelihood of using preferred type, but don't guarantee it
            preferred_bonus = 0.2 if preferred_type else 0.0
        else:
            preferred_type = None
            preferred_bonus = 0.0
            
        # Initialize scores for each reasoning type
        scores = {reasoning_type: 0.0 for reasoning_type in self.REASONING_EMOJIS.keys()}
        
        # Score based on keywords
        for reasoning_type, keywords in self.REASONING_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query:
                    scores[reasoning_type] += 0.3
                    # Increase score based on exact match vs partial match
                    if f" {keyword} " in f" {query} ":
                        scores[reasoning_type] += 0.2
                    
        # Apply complexity analysis
        complexity_score = self._assess_complexity(query)
        if complexity_score > 0.6:
            # Complex queries benefit from certain reasoning types
            scores['sequential'] += 0.3
            scores['graph'] += 0.3
            scores['step_back'] += 0.2
            scores['multi_agent'] += 0.2
        elif complexity_score < 0.3:
            # Simple queries are often better handled conversationally
            scores['conversational'] += 0.2
            
        # Apply context from conversation history
        if conversation_history:
            history_type, history_score = self._analyze_conversation_history(conversation_history)
            scores[history_type] += history_score * 0.3
            
        # Apply user preference bonus
        if preferred_type:
            scores[preferred_type] += preferred_bonus
            
        # Consider domain-specific preferences
        domain_type = self._detect_domain(query)
        if domain_type:
            scores[domain_type] += 0.15
            
        # Consider likelihood of continuation for the current reasoning type
        if current_reasoning:
            continuation_score = self._assess_continuation_likelihood(query, current_reasoning)
            scores[current_reasoning] += continuation_score * 0.4
            
        # For certain pairs of reasoning types, favor transitions
        if current_reasoning == 'rag' and scores['knowledge'] > 0.3:
            # Prefer transitioning from search to knowledge application
            scores['knowledge'] += 0.2
        elif current_reasoning == 'step_back' and scores['sequential'] > 0.3:
            # Prefer transitioning from big picture to detailed steps
            scores['sequential'] += 0.2
            
        # Special case - if the query contains a math expression
        if self._contains_math_expression(query):
            scores['calculation'] += 0.5
            
        # Special case - if the query is a creative request
        if self._is_creative_request(query):
            scores['creative'] += 0.4
            
        # Find the highest scoring reasoning type
        highest_score = 0.0
        highest_type = 'conversational'  # Default fallback
        
        for reasoning_type, score in scores.items():
            if score > highest_score:
                highest_score = score
                highest_type = reasoning_type
                
        # Normalize confidence to 0-1 range
        confidence = min(1.0, max(0.3, highest_score))
        
        # Default to conversational if confidence is too low
        if confidence < 0.3:
            highest_type = 'conversational'
            confidence = 0.3
            
        result = (highest_type, confidence)
        
        # Cache the result
        if self.enable_cache and conversation_id:
            self.cache.cache_reasoning(
                query=query,
                reasoning_result=result,
                user_id=user_id,
                conversation_id=conversation_id,
                conversation_history=conversation_history
            )
            
        return result
        
    def _check_explicit_mode_request(self, query: str) -> Optional[str]:
        """Check if the user explicitly requested a specific reasoning mode"""
        # Check for explicit emoji usage
        for reasoning_type, emoji in self.REASONING_EMOJIS.items():
            if emoji in query:
                return reasoning_type
                
        # Check for explicit mode keywords - expanded list
        explicit_requests = {
            'sequential thinking': 'sequential',
            'sequential reasoning': 'sequential',
            'think step by step': 'sequential',
            'step by step thinking': 'sequential',
            'step by step approach': 'sequential',
            'use rag': 'rag',
            'search for': 'rag',
            'find information': 'rag',
            'search the web': 'rag',
            'look up information': 'rag',
            'fact check': 'verification',
            'verify this': 'verification',
            'check if this is true': 'verification',
            'is this accurate': 'verification',
            'graph of thought': 'graph',
            'use graph': 'graph',
            'map out the connections': 'graph',
            'map connections': 'graph',
            'chain of thought': 'cot',
            'use cot': 'cot',
            'logical progression': 'cot',
            'creative mode': 'creative',
            'be creative': 'creative',
            'creative approach': 'creative',
            'react reasoning': 'react',
            'take action': 'react',
            'implement this': 'react',
            'step back and': 'step_back',
            'broader perspective': 'step_back',
            'big picture view': 'step_back',
            'multi-agent': 'multi_agent',
            'multiple perspectives': 'multi_agent',
            'different perspectives': 'multi_agent',
            'calculate this': 'calculation',
            'compute the': 'calculation',
            'math problem': 'calculation',
            'plan this': 'planning',
            'make a plan': 'planning',
            'plan of action': 'planning',
            'conversational mode': 'conversational',
            'just chat': 'conversational',
            'casual conversation': 'conversational'
        }
        
        # Check if any explicit request phrases are in the query
        for request, reasoning_type in explicit_requests.items():
            if request in query.lower():
                return reasoning_type
                
        return None
        
    def _assess_complexity(self, query: str) -> float:
        """
        Assess the complexity of a query.
        
        Returns:
            float: Complexity score (0.0 to 1.0)
        """
        # Base complexity on length (longer queries tend to be more complex)
        words = query.split()
        length_score = min(len(words) / 50.0, 1.0)  # Normalize to 0-1
        
        # Check for complexity indicators
        indicator_count = sum(1 for indicator in self.COMPLEXITY_INDICATORS if indicator in query.lower())
        indicator_score = min(indicator_count / 5.0, 1.0)  # Normalize to 0-1
        
        # Check for complex structures
        structure_score = 0.0
        if "?" in query:
            # Multiple questions indicate complexity
            question_count = query.count("?")
            structure_score += min(question_count / 3.0, 1.0) * 0.3
            
        if "if" in query.lower() and "then" in query.lower():
            # Conditional logic
            structure_score += 0.3
            
        # Check for lists or enumerations
        list_pattern = re.compile(r'\d+\.|•|\*|-|\((\d+|[a-z])\)|\[\d+\]')
        if list_pattern.search(query):
            structure_score += 0.3
            
        # Check for multiple clauses or sentence complexity
        clause_indicators = [', and ', ', but ', '; ', ', which ', ', because ', ', however ', ', therefore ']
        clause_count = sum(1 for indicator in clause_indicators if indicator in query.lower())
        clause_score = min(clause_count / 3.0, 1.0) * 0.3
        
        # Combined score (weighted)
        combined_score = length_score * 0.25 + indicator_score * 0.3 + structure_score * 0.25 + clause_score * 0.2
        return combined_score
        
    def _analyze_conversation_history(self, history: List[str]) -> Tuple[str, float]:
        """
        Analyze conversation history to determine context.
        
        Args:
            history: List of previous messages
            
        Returns:
            tuple: (dominant_reasoning_type, confidence_score)
        """
        # Initialize type counts
        type_counts = {reasoning_type: 0 for reasoning_type in self.REASONING_EMOJIS.keys()}
        
        # Analyze each message in history
        for message in history:
            # Score each reasoning type for this message
            for reasoning_type, keywords in self.REASONING_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in message.lower():
                        type_counts[reasoning_type] += 1
                        
        # Determine dominant type
        if not type_counts:
            return 'conversational', 0.5
            
        dominant_type = max(type_counts.items(), key=lambda x: x[1])
        if dominant_type[1] == 0:
            return 'conversational', 0.5
            
        # Calculate confidence based on dominance
        total_counts = sum(type_counts.values())
        confidence = dominant_type[1] / total_counts if total_counts > 0 else 0.5
        
        return dominant_type[0], confidence
        
    def _detect_domain(self, query: str) -> Optional[str]:
        """
        Detect the domain of the query to apply domain-specific reasoning preferences.
        
        Args:
            query: The user query
            
        Returns:
            Optional[str]: Domain-specific reasoning type or None
        """
        # Domain keywords
        domain_keywords = {
            'math': ['math', 'calculation', 'equation', 'formula', 'compute', 'calculate', 'problem', 'algebra', 'geometry'],
            'science': ['science', 'physics', 'chemistry', 'biology', 'scientific', 'experiment', 'theory', 'hypothesis'],
            'history': ['history', 'historical', 'ancient', 'century', 'era', 'period', 'civilization', 'timeline'],
            'politics': ['politics', 'political', 'government', 'policy', 'election', 'democracy', 'parliament', 'congress'],
            'art': ['art', 'artistic', 'painting', 'music', 'literature', 'creative', 'aesthetic', 'design', 'poetry'],
            'programming': ['code', 'programming', 'algorithm', 'function', 'software', 'developer', 'language', 'variable'],
            'philosophy': ['philosophy', 'philosophical', 'ethics', 'morality', 'existence', 'consciousness', 'metaphysics'],
            'news': ['news', 'current events', 'latest', 'recent', 'update', 'headline', 'media', 'press', 'journalist'],
            'business': ['business', 'company', 'corporate', 'strategy', 'market', 'management', 'finance', 'economics'],
            'technology': ['technology', 'tech', 'device', 'gadget', 'innovation', 'digital', 'computer', 'software']
        }
        
        # Check for domain keywords
        domain_scores = {domain: 0 for domain in domain_keywords.keys()}
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in query.lower():
                    domain_scores[domain] += 1
        
        # Get the domain with the highest score
        if any(domain_scores.values()):
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            if best_domain[1] > 0:
                return self.domain_preferences.get(best_domain[0])
        
        return None
        
    def _assess_continuation_likelihood(self, query: str, current_reasoning: str) -> float:
        """
        Assess how likely the current query is to continue the current reasoning mode.
        
        Args:
            query: The user query
            current_reasoning: The current reasoning type
            
        Returns:
            float: Likelihood of continuing the same reasoning (0.0 to 1.0)
        """
        # Check for continuation phrases
        continuation_phrases = [
            'continue', 'go on', 'proceed', 'next', 'furthermore', 'additionally',
            'building on this', 'following up', 'more on this', 'elaborate',
            'tell me more', 'expand on', 'further analysis', 'another aspect',
            'related to this', 'on this topic', 'in addition', 'moreover',
            'and also', 'what about', 'how about', 'and then', 'next step'
        ]
        
        # Check for topic continuation indicators
        continuation_score = 0.0
        
        # Check for explicit continuation phrases
        for phrase in continuation_phrases:
            if phrase in query.lower():
                continuation_score += 0.3
                break
                
        # Check for keywords related to the current reasoning type
        if current_reasoning in self.REASONING_KEYWORDS:
            for keyword in self.REASONING_KEYWORDS[current_reasoning]:
                if keyword in query.lower():
                    continuation_score += 0.2
                    break
        
        # Check for very short queries (often continuations)
        if len(query.split()) <= 4:
            continuation_score += 0.2
            
        # Cap at 1.0
        return min(continuation_score, 1.0)
    
    def _contains_math_expression(self, query: str) -> bool:
        """Check if the query contains a mathematical expression"""
        # Simple pattern to match basic math expressions
        math_pattern = re.compile(r'[\d\+\-\*\/\^\(\)=><]+')
        matches = math_pattern.findall(query)
        
        # Check if the matches look like math expressions (at least one operator and number)
        for match in matches:
            if (any(op in match for op in '+-*/^=><') and 
                any(c.isdigit() for c in match) and
                len(match) >= 3):
                return True
                
        # Check for math keywords with numbers
        math_keywords = ['calculate', 'compute', 'solve', 'evaluat', 'equation', 'formula', 'equal']
        if any(keyword in query.lower() for keyword in math_keywords):
            if re.search(r'\d', query):
                return True
                
        return False
    
    def _is_creative_request(self, query: str) -> bool:
        """Check if the query is asking for creative content"""
        creative_phrases = [
            'write a', 'create a', 'generate a', 'compose a', 'make up a',
            'story about', 'poem about', 'song lyrics', 'creative idea',
            'imagine a', 'invent a', 'fiction about', 'write me a',
            'be creative', 'use your imagination', 'fantasy about',
            'creative story', 'creative description', 'fictional scenario'
        ]
        
        return any(phrase in query.lower() for phrase in creative_phrases)
        
    def set_user_preference(self, user_id: str, reasoning_type: str) -> bool:
        """Set a user's preferred reasoning type"""
        self.user_preferences[user_id] = reasoning_type
        
        # Update cache if enabled
        if self.enable_cache:
            self.cache.set_user_preference(user_id, reasoning_type)
            
        return True
        
    def get_emoji_for_type(self, reasoning_type: str) -> str:
        """Get the emoji for a reasoning type"""
        return self.REASONING_EMOJIS.get(reasoning_type, '💬')  # Default to conversational
        
    def format_response_with_reasoning(self, 
                                     response: str, 
                                     reasoning_type: str,
                                     include_prefix: bool = True,
                                     include_suffix: bool = False) -> str:
        """
        Format a response with appropriate reasoning indicators.
        
        Args:
            response: The original response text
            reasoning_type: The reasoning type used
            include_prefix: Whether to include a reasoning type prefix
            include_suffix: Whether to include a reasoning type suffix
            
        Returns:
            str: The formatted response
        """
        emoji = self.get_emoji_for_type(reasoning_type)
        
        # Prepare formatted response
        formatted_response = response
        
        if include_prefix:
            # Add an appropriate prefix based on reasoning type
            prefixes = {
                'sequential': f"{emoji} **Sequential Thinking**: ",
                'rag': f"{emoji} **Information Retrieval**: ",
                'conversational': f"{emoji} ",  # Minimal prefix for conversational
                'knowledge': f"{emoji} **Knowledge Base**: ",
                'verification': f"{emoji} **Fact Check**: ",
                'creative': f"{emoji} **Creative Mode**: ",
                'calculation': f"{emoji} **Calculation**: ",
                'planning': f"{emoji} **Planning**: ",
                'graph': f"{emoji} **Graph-of-Thought**: ",
                'multi_agent': f"{emoji} **Multiple Perspectives**: ",
                'step_back': f"{emoji} **Step-Back Analysis**: ",
                'cot': f"{emoji} **Chain-of-Thought**: ",
                'react': f"{emoji} **ReAct Reasoning**: "
            }
            prefix = prefixes.get(reasoning_type, f"{emoji} ")
            formatted_response = f"{prefix}{formatted_response}"
            
        if include_suffix:
            # Add an appropriate suffix based on reasoning type
            suffix = f"\n\n*Response generated using {reasoning_type} reasoning {emoji}*"
            formatted_response = f"{formatted_response}{suffix}"
            
        return formatted_response

class ReasoningManager:
    """
    Orchestrates transitions between reasoning types and maintains context.
    """
    
    def __init__(self):
        """Initialize the reasoning manager"""
        self.detector = ReasoningDetector()
        # Keep track of active reasoning types by conversation
        self.active_reasoning = {}
        
    async def process_query(self, 
                         query: str, 
                         conversation_id: str = None,
                         user_id: str = None,
                         conversation_history: List[str] = None) -> Dict[str, Any]:
        """
        Process a query and determine the appropriate reasoning type.
        
        Args:
            query: The user query text
            conversation_id: Optional conversation/thread ID
            user_id: Optional user ID for personalized detection
            conversation_history: Optional list of previous messages
            
        Returns:
            Dict with the following keys:
            - reasoning_type: The detected reasoning type
            - emoji: The emoji indicator for the reasoning type
            - confidence: Confidence in the reasoning type selection
            - transition: Whether this represents a transition in reasoning
        """
        # Get current reasoning type if available
        current_reasoning = self.active_reasoning.get(conversation_id) if conversation_id else None
        
        # Detect the appropriate reasoning type
        reasoning_type, confidence = self.detector.detect_reasoning_type(
            query=query,
            conversation_history=conversation_history,
            user_id=user_id,
            current_reasoning=current_reasoning,
            conversation_id=conversation_id
        )
        
        # Get emoji for the reasoning type
        emoji = self.detector.get_emoji_for_type(reasoning_type)
        
        # Detect if this is a transition in reasoning
        is_transition = current_reasoning is not None and current_reasoning != reasoning_type
        
        # Update active reasoning
        if conversation_id:
            self.active_reasoning[conversation_id] = reasoning_type
            
        return {
            "reasoning_type": reasoning_type,
            "emoji": emoji,
            "confidence": confidence,
            "transition": is_transition
        }
        
    async def reset_conversation(self, conversation_id: str):
        """Reset reasoning state for a conversation."""
        if conversation_id in self.active_reasoning:
            del self.active_reasoning[conversation_id]
            
        # Also clear the cache if applicable
        if hasattr(self.detector, 'enable_cache') and self.detector.enable_cache:
            self.detector.cache.invalidate_conversation_cache(conversation_id)
            
    async def format_response(self, 
                           response: str, 
                           reasoning_type: str,
                           include_reasoning_details: bool = False) -> str:
        """
        Format the response based on the reasoning type.
        
        Args:
            response: The AI response text
            reasoning_type: The reasoning type used
            include_reasoning_details: Whether to include reasoning details
            
        Returns:
            Formatted response with appropriate indicators
        """
        # Format based on reasoning type
        return self.detector.format_response_with_reasoning(
            response=response,
            reasoning_type=reasoning_type,
            include_prefix=True,
            include_suffix=include_reasoning_details
        )

# Create a singleton instance
reasoning_manager = ReasoningManager() 