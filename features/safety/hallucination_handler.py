import logging
import re
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict

logger = logging.getLogger('hallucination_handler')
logger.setLevel(logging.INFO)

class HallucinationHandler:
    """
    Handler for detecting and mitigating hallucinations in AI responses
    using verification strategies and content awareness
    """
    def __init__(self, 
                response_cache=None, 
                llm_provider=None,
                verification_threshold: float = 0.7,
                enable_grounding: bool = True):
        """
        Initialize hallucination handler
        
        Args:
            response_cache: Cache for storing verified responses
            llm_provider: LLM provider for verification
            verification_threshold: Confidence threshold for verification
            enable_grounding: Whether to enable knowledge grounding
        """
        self.response_cache = response_cache
        self.llm_provider = llm_provider
        self.verification_threshold = verification_threshold
        self.enable_grounding = enable_grounding
        
        # Metrics
        self.metrics = {
            "hallucinations_detected": 0,
            "responses_verified": 0,
            "groundings_performed": 0,
            "cache_hits": 0
        }
        
        # Load common hallucination patterns
        self._load_hallucination_patterns()
    
    def _load_hallucination_patterns(self):
        """Load common hallucination patterns for quick checks"""
        self.hallucination_patterns = [
            r"I'm sorry, (but )?I don't have (information|knowledge) about",
            r"I don't have access to (real-time|current|live) information",
            r"I cannot browse the internet",
            r"As an AI (language model|assistant), I (don't|cannot|am not able to)",
            r"I don't have the ability to",
            r"I (can't|cannot|don't have the capability to) access",
            r"is not something I (have information about|can determine)",
            r"(without|lacking) (more|additional|specific) (information|context|details)",
            r"I'm not able to (verify|check|confirm)",
            r"(My knowledge|I) (has a cutoff|is limited to|only extends to)",
            r"I made (a mistake|an error)",
            r"I apologize for the (confusion|error|mistake|incorrect information)"
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.hallucination_patterns]
    
    async def verify_response(self, 
                            query: str, 
                            response: str,
                            context: Dict[str, Any] = None,
                            user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify response for hallucinations
        
        Args:
            query: User query
            response: AI response to verify
            context: Optional context data for verification
            user_id: Optional user ID for caching
            
        Returns:
            Verification result with confidence score
        """
        # Check for cached verification
        if self.response_cache and user_id:
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()
            response_hash = hashlib.md5(response.encode()).hexdigest()
            cache_key = f"verification:{query_hash}:{response_hash}"
            
            cached_result = await self.response_cache.get(
                cache_key,
                cache_type="response",
                user_id=user_id
            )
            
            if cached_result:
                self.metrics["cache_hits"] += 1
                return cached_result
        
        # Initial quick pattern-based check
        pattern_match_score = self._check_patterns(response)
        
        # Initialize verification result
        verification_result = {
            "verified": True,
            "confidence": 1.0,
            "hallucination_score": 0.0,
            "grounding_used": False,
            "verification_method": "pattern_check",
            "timestamp": time.time()
        }
        
        # If pattern match found potential hallucination, perform deeper verification
        if pattern_match_score > 0.3:
            verification_result["hallucination_score"] = pattern_match_score
            verification_result["confidence"] = 1.0 - pattern_match_score
            
            # If confidence below threshold, perform deeper verification
            if verification_result["confidence"] < self.verification_threshold:
                if self.llm_provider and hasattr(self.llm_provider, "verify_hallucination"):
                    # Use LLM for deeper verification
                    llm_verification = await self._verify_with_llm(query, response, context)
                    
                    # Update verification result
                    verification_result.update(llm_verification)
                    verification_result["verification_method"] = "llm_verification"
                
                # If grounding enabled and confidence still low, try grounding
                if self.enable_grounding and verification_result["confidence"] < self.verification_threshold:
                    grounding_result = await self._ground_response(query, response, context)
                    
                    if grounding_result["grounded"]:
                        verification_result["grounded_response"] = grounding_result["grounded_response"]
                        verification_result["confidence"] = max(
                            verification_result["confidence"],
                            grounding_result["confidence"]
                        )
                        verification_result["grounding_used"] = True
                        verification_result["grounding_sources"] = grounding_result.get("sources", [])
                        self.metrics["groundings_performed"] += 1
        
        # Set final verification status based on confidence
        verification_result["verified"] = verification_result["confidence"] >= self.verification_threshold
        
        # Update metrics
        if verification_result["verified"]:
            self.metrics["responses_verified"] += 1
        else:
            self.metrics["hallucinations_detected"] += 1
        
        # Cache verification result
        if self.response_cache and user_id:
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()
            response_hash = hashlib.md5(response.encode()).hexdigest()
            cache_key = f"verification:{query_hash}:{response_hash}"
            
            await self.response_cache.set(
                cache_key,
                verification_result,
                cache_type="response",
                ttl=3600,  # 1 hour
                user_id=user_id
            )
        
        return verification_result
    
    def _check_patterns(self, response: str) -> float:
        """
        Check response against common hallucination patterns
        
        Args:
            response: Response to check
            
        Returns:
            Hallucination score (0.0-1.0)
        """
        score = 0.0
        total_patterns = len(self.compiled_patterns)
        
        for pattern in self.compiled_patterns:
            if pattern.search(response):
                score += 1.0 / total_patterns
        
        return score
    
    async def _verify_with_llm(self, 
                            query: str, 
                            response: str, 
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Use LLM to verify response
        
        Args:
            query: User query
            response: AI response to verify
            context: Optional context data for verification
            
        Returns:
            Verification result
        """
        if not self.llm_provider or not hasattr(self.llm_provider, "verify_hallucination"):
            return {
                "verified": True,
                "confidence": 0.7,
                "verification_method": "fallback"
            }
        
        # Build verification prompt
        prompt = self._build_verification_prompt(query, response, context)
        
        try:
            # Call LLM for verification
            verification = await self.llm_provider.verify_hallucination(prompt)
            
            if isinstance(verification, dict):
                return {
                    "verified": verification.get("verified", True),
                    "confidence": verification.get("confidence", 0.7),
                    "explanation": verification.get("explanation", ""),
                    "issues": verification.get("issues", []),
                    "verification_method": "llm"
                }
            else:
                # Fallback for simple response
                return {
                    "verified": True,
                    "confidence": 0.7,
                    "verification_method": "llm_simple"
                }
        except Exception as e:
            logger.error(f"Error in LLM verification: {e}")
            return {
                "verified": True,
                "confidence": 0.6,
                "error": str(e),
                "verification_method": "error_fallback"
            }
    
    def _build_verification_prompt(self, 
                                query: str, 
                                response: str, 
                                context: Dict[str, Any] = None) -> str:
        """
        Build verification prompt for LLM
        
        Args:
            query: User query
            response: AI response to verify
            context: Optional context data for verification
            
        Returns:
            Verification prompt
        """
        prompt = (
            "Your task is to analyze an AI assistant's response for potential hallucinations or factual errors.\n\n"
            f"User query: {query}\n\n"
            f"AI response: {response}\n\n"
        )
        
        # Add context if available
        if context and isinstance(context, dict):
            # Extract relevant context for verification
            context_str = ""
            
            if "conversation_history" in context and context["conversation_history"]:
                context_str += "Previous conversation context:\n"
                for i, msg in enumerate(context["conversation_history"][-3:]):  # Last 3 messages
                    if "user" in msg and "bot" in msg:
                        context_str += f"User: {msg['user']}\nAssistant: {msg['bot']}\n"
            
            if context_str:
                prompt += f"\nAdditional context:\n{context_str}\n"
        
        prompt += (
            "\nPlease evaluate the response for:\n"
            "1. Factual correctness\n"
            "2. Consistency with the query\n"
            "3. Presence of made-up information\n"
            "4. Self-contradiction\n"
            "5. Logical errors\n\n"
            "Provide your analysis in the following JSON format:\n"
            "{\n"
            '  "verified": true/false,\n'
            '  "confidence": 0.0-1.0,\n'
            '  "explanation": "brief explanation",\n'
            '  "issues": ["issue1", "issue2"]\n'
            "}\n"
        )
        
        return prompt
    
    async def _ground_response(self, 
                             query: str, 
                             response: str,
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ground response in factual information
        
        Args:
            query: User query
            response: AI response to ground
            context: Optional context data
            
        Returns:
            Grounding result
        """
        # Default result if grounding fails
        default_result = {
            "grounded": False,
            "confidence": 0.5,
            "grounded_response": response,
            "sources": []
        }
        
        # If no LLM provider or no search method available, return default
        if not self.llm_provider or not hasattr(self.llm_provider, "search_information"):
            return default_result
        
        try:
            # Extract key claims from response
            claims = await self._extract_claims(response)
            
            if not claims:
                return default_result
            
            # Search for evidence for each claim
            sources = []
            for claim in claims:
                search_results = await self.llm_provider.search_information(claim)
                if search_results:
                    sources.extend(search_results)
            
            if not sources:
                return default_result
            
            # Ground the response with sources
            grounded_response = await self._integrate_sources(response, sources)
            
            return {
                "grounded": True,
                "confidence": 0.8,
                "grounded_response": grounded_response,
                "sources": sources[:3]  # Limit to top 3 sources
            }
        except Exception as e:
            logger.error(f"Error in grounding: {e}")
            return default_result
    
    async def _extract_claims(self, response: str) -> List[str]:
        """Extract key claims from response"""
        if not self.llm_provider:
            return []
            
        try:
            return await self.llm_provider.extract_claims(response)
        except:
            # Fallback to simple extraction
            sentences = response.split('.')
            return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    async def _integrate_sources(self, response: str, sources: List[Dict[str, str]]) -> str:
        """Integrate sources into response"""
        if not sources:
            return response
            
        # Simple integration - just add sources at the end
        grounded = response
        
        # Check if response already has sources section
        if "Sources:" not in response and "References:" not in response:
            grounded += "\n\nSources:\n"
            for i, source in enumerate(sources[:3], 1):
                title = source.get("title", "Source")
                url = source.get("url", "")
                if url:
                    grounded += f"{i}. {title}: {url}\n"
                else:
                    grounded += f"{i}. {title}\n"
        
        return grounded
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get handler metrics"""
        return dict(self.metrics)

def create_hallucination_handler(
    response_cache=None, 
    llm_provider=None, 
    verification_threshold: float = 0.7,
    enable_grounding: bool = True
) -> HallucinationHandler:
    """
    Create hallucination handler
    
    Args:
        response_cache: Cache for storing verified responses
        llm_provider: LLM provider for verification
        verification_threshold: Confidence threshold for verification
        enable_grounding: Whether to enable knowledge grounding
        
    Returns:
        Configured HallucinationHandler instance
    """
    return HallucinationHandler(
        response_cache=response_cache,
        llm_provider=llm_provider,
        verification_threshold=verification_threshold,
        enable_grounding=enable_grounding
    ) 