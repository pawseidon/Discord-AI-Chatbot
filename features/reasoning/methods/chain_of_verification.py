import os
import re
import json
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import time
import logging

# Updated imports for new architecture
from utils.token_utils import token_optimizer
from core.ai_provider import get_ai_provider

logger = logging.getLogger("chain_of_verification")

class VerificationStep:
    """Represents a single verification step in the Chain-of-Verification process"""
    def __init__(self, claim: str = "", verification: str = "", evidence: str = "", status: str = ""):
        self.claim = claim
        self.verification = verification
        self.evidence = evidence
        self.status = status  # "verified", "refuted", or "unknown"
        
    def __str__(self) -> str:
        result = f"Claim: {self.claim}\n"
        if self.verification:
            result += f"Verification: {self.verification}\n"
        if self.evidence:
            result += f"Evidence: {self.evidence}\n"
        if self.status:
            result += f"Status: {self.status}\n"
        return result
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "claim": self.claim,
            "verification": self.verification,
            "evidence": self.evidence,
            "status": self.status
        }

class ChainOfVerification:
    """
    Implementation of Chain-of-Verification (CoV) - a factual accuracy enhancement technique.
    
    Based on research in "Chain-of-Verification: Mitigating Hallucination in LLMs"
    (https://arxiv.org/abs/2309.11495)
    """
    
    def __init__(self, llm_provider=None, model_name: str = None):
        """
        Initialize the Chain-of-Verification system
        
        Args:
            llm_provider: Optional LLM provider (if None, default will be fetched)
            model_name: Optional model name to use
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        
        # CoV prompts
        self.generation_prompt = (
            """You are a helpful AI assistant responsible for generating accurate information.
            
            User question: {question}
            
            First, provide a direct answer to the question. Be detailed and informative while ensuring factual accuracy.
            
            Your response:
            """
        )
        
        self.claim_extraction_prompt = (
            """You are an expert at identifying factual claims in text.
            
            I need you to carefully extract all factual claims from the following text. These are statements
            presented as facts that can be verified or refuted with evidence.
            
            Focus on statements that:
            - Contain specific facts, dates, numbers, or statistics
            - Make assertions about events, people, places, or processes
            - State relationships or causal connections as facts
            - Define concepts or terms in a factual manner
            
            Avoid extracting:
            - Purely subjective opinions or value judgments
            - Hypothetical statements or questions
            - General observations without specific claims
            
            Text to analyze:
            {response}
            
            For each factual claim you identify, extract and number it. Only list the claims themselves without
            commentary. Aim for precision - don't combine multiple claims into one. Focus on claims that
            would benefit from verification.
            
            Format each claim on a new line starting with "Claim #: "
            
            CLAIMS:
            """
        )
        
        self.verification_prompt = (
            """You are a critical fact-checker responsible for verifying the accuracy of information.
            
            I need you to verify this specific claim:
            
            Claim: {claim}
            
            Context from search results:
            {evidence}
            
            Carefully verify the claim against the evidence provided. Consider:
            1. Is the claim factually accurate according to the evidence?
            2. Does the claim exaggerate, distort, or misrepresent what the evidence says?
            3. Does the claim make assertions that go beyond what the evidence supports?
            4. Does the claim contain specific details (dates, numbers, etc.) that can be validated?
            
            Provide your verification by:
            1. Analyzing the alignment between the claim and evidence
            2. Noting specific points of agreement or disagreement
            3. Assessing whether the claim is fully supported, partially supported, or unsupported
            
            Classification (choose one):
            - VERIFIED: Claim is accurate and fully supported by evidence
            - REFUTED: Claim contradicts available evidence
            - UNVERIFIED: Insufficient evidence to confirm or refute
            
            Your verification:
            """
        )
        
        self.rewrite_prompt = (
            """You are an expert at correcting inaccurate information while maintaining a helpful tone.
            
            Original response to question "{question}":
            {original_response}
            
            Verification results:
            {verification_results}
            
            Based on the verification results, please rewrite the original response to:
            1. Fix any refuted claims with accurate information
            2. Clarify any unverified claims by qualifying them appropriately
            3. Maintain all verified information
            4. Preserve the overall structure and tone of the original response
            
            The rewritten response should address the user's question clearly and accurately,
            without drawing attention to the corrections. Do not explain that you've made
            corrections - simply present the improved version as your response.
            
            Rewritten response:
            """
        )
    
    async def ensure_llm_provider(self):
        """Ensure LLM provider is available"""
        if self.llm_provider is None:
            self.llm_provider = await get_ai_provider(model=self.model_name)
            logger.info(f"Created LLM provider with model: {self.model_name}")
    
    async def generate_initial_response(self, question: str) -> str:
        """Generate the initial response to the question"""
        await self.ensure_llm_provider()
        
        prompt = self.generation_prompt.format(question=question)
        response = await self.llm_provider.async_call(prompt)
        return response
    
    async def extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from the response"""
        await self.ensure_llm_provider()
        
        prompt = self.claim_extraction_prompt.format(response=response)
        extraction_response = await self.llm_provider.async_call(prompt)
        
        # Parse the claims
        claims = []
        for line in extraction_response.split('\n'):
            match = re.search(r"Claim\s*\d+\s*:\s*(.*)", line)
            if match:
                claim = match.group(1).strip()
                if claim:
                    claims.append(claim)
        
        return claims
    
    async def verify_claim(self, claim: str) -> VerificationStep:
        """Verify a single claim using web search evidence"""
        await self.ensure_llm_provider()
        
        # Search for evidence
        search_query = f"{claim} facts evidence"
        
        if hasattr(self.llm_provider, "search_information"):
            evidence = await self.llm_provider.search_information(search_query)
        elif hasattr(self.llm_provider, "search_internet"):
            evidence = await self.llm_provider.search_internet(search_query)
        else:
            return VerificationStep(
                claim=claim,
                verification="Could not verify due to lack of search capability",
                evidence="No evidence available",
                status="unknown"
            )
        
        if not evidence or (isinstance(evidence, str) and "disabled" in evidence.lower()):
            return VerificationStep(
                claim=claim,
                verification="Could not verify due to lack of search access",
                evidence="No evidence available",
                status="unknown"
            )
        
        # Optimize evidence to reduce tokens
        if hasattr(token_optimizer, "clean_text") and hasattr(token_optimizer, "truncate_text"):
            evidence = token_optimizer.clean_text(evidence)
            evidence = token_optimizer.truncate_text(evidence, max_tokens=800)
        
        # Verify the claim
        prompt = self.verification_prompt.format(claim=claim, evidence=evidence)
        verification_text = await self.llm_provider.async_call(prompt)
        
        # Extract the classification
        status = "unknown"
        if "VERIFIED" in verification_text:
            status = "verified"
        elif "REFUTED" in verification_text:
            status = "refuted"
        
        return VerificationStep(
            claim=claim,
            verification=verification_text,
            evidence=evidence,
            status=status
        )
    
    async def rewrite_response(self, question: str, original_response: str, verification_steps: List[VerificationStep]) -> str:
        """Rewrite the response based on verification results"""
        await self.ensure_llm_provider()
        
        # Format verification results
        verification_results = ""
        for i, step in enumerate(verification_steps, 1):
            verification_results += f"Claim {i}: {step.claim}\n"
            verification_results += f"Status: {step.status.upper()}\n"
            verification_results += f"Verification: {step.verification}\n\n"
        
        # Rewrite the response
        prompt = self.rewrite_prompt.format(
            question=question,
            original_response=original_response,
            verification_results=verification_results
        )
        
        rewritten_response = await self.llm_provider.async_call(prompt)
        return rewritten_response
    
    async def verify(self, 
                  question: str, 
                  max_claims: int = 5, 
                  timeout: int = 60) -> Tuple[str, List[Dict[str, str]]]:
        """
        Perform the full Chain-of-Verification process
        
        Args:
            question: User question
            max_claims: Maximum number of claims to verify
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (verified response, verification steps)
        """
        start_time = time.time()
        
        try:
            # Generate initial response
            logger.info(f"Generating initial response for question: {question}")
            response = await asyncio.wait_for(
                self.generate_initial_response(question),
                timeout=timeout/3
            )
            
            # Extract claims
            logger.info("Extracting claims from response")
            claims = await asyncio.wait_for(
                self.extract_claims(response),
                timeout=timeout/4
            )
            
            # Limit the number of claims
            claims = claims[:max_claims]
            
            if not claims:
                logger.info("No claims extracted, returning original response")
                return response, []
            
            # Verify each claim
            logger.info(f"Verifying {len(claims)} claims")
            verification_steps = []
            verification_tasks = [self.verify_claim(claim) for claim in claims]
            
            remaining_time = timeout - (time.time() - start_time)
            steps = await asyncio.wait_for(
                asyncio.gather(*verification_tasks),
                timeout=max(remaining_time, 10)  # At least 10 seconds
            )
            verification_steps.extend(steps)
            
            # Rewrite the response based on verification
            logger.info("Rewriting response based on verification")
            remaining_time = timeout - (time.time() - start_time)
            verified_response = await asyncio.wait_for(
                self.rewrite_response(question, response, verification_steps),
                timeout=max(remaining_time, 10)  # At least 10 seconds
            )
            
            # Convert verification steps to dictionaries for return
            steps_dict = [step.to_dict() for step in verification_steps]
            
            return verified_response, steps_dict
            
        except asyncio.TimeoutError:
            logger.warning(f"Chain-of-Verification timed out after {time.time() - start_time:.2f} seconds")
            return response if 'response' in locals() else f"I apologize, but I couldn't process your question '{question}' in time.", []
        except Exception as e:
            logger.error(f"Error in Chain-of-Verification: {e}")
            return f"I apologize, but I encountered an error while answering your question '{question}'.", []

async def process_chain_of_verification(
    query: str,
    user_id: str = None,
    context: Dict[str, Any] = None,
    model: str = None,
    max_claims: int = 5,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Process a query using Chain-of-Verification reasoning
    
    Args:
        query: User query
        user_id: User ID (optional)
        context: Additional context (optional)
        model: Model to use (optional)
        max_claims: Maximum number of claims to verify
        timeout: Timeout in seconds
    
    Returns:
        Dictionary with verified response and reasoning details
    """
    logger.info(f"Processing Chain-of-Verification for query: {query}")
    
    # Get LLM provider
    llm_provider = await get_ai_provider(model=model)
    
    # Create Chain-of-Verification instance
    cov = ChainOfVerification(llm_provider=llm_provider, model_name=model)
    
    # Run verification
    try:
        verified_response, verification_steps = await cov.verify(
            question=query,
            max_claims=max_claims,
            timeout=timeout
        )
        
        # Format the response
        result = {
            "answer": verified_response,
            "method": "chain_of_verification",
            "method_emoji": "🔍✓",
            "verification_steps": verification_steps,
            "verified_claims_count": len([s for s in verification_steps if s.get("status") == "verified"]),
            "refuted_claims_count": len([s for s in verification_steps if s.get("status") == "refuted"]),
            "unverified_claims_count": len([s for s in verification_steps if s.get("status") == "unknown"]),
            "processing_time": time.time() - time.time()  # Will be filled with actual time by the router
        }
        
        return result
    except Exception as e:
        logger.error(f"Error in process_chain_of_verification: {e}")
        return {
            "answer": f"I encountered an error while processing your question using verification. Error: {str(e)}",
            "method": "chain_of_verification",
            "method_emoji": "❌",
            "error": str(e)
        }

class ChainOfVerificationReasoning:
    """Class-based implementation of Chain-of-Verification reasoning"""
    
    def __init__(self, model: str = None):
        """Initialize the reasoning method"""
        self.model = model
        self.cov = None
        
    async def initialize(self):
        """Initialize the CoV instance if needed"""
        if self.cov is None:
            llm_provider = await get_ai_provider(model=self.model)
            self.cov = ChainOfVerification(llm_provider=llm_provider, model_name=self.model)
    
    async def process(self, 
                    query: str,
                    user_id: str = None,
                    context: Dict[str, Any] = None,
                    max_claims: int = 5,
                    timeout: int = 60) -> Dict[str, Any]:
        """Process query with Chain-of-Verification"""
        await self.initialize()
        
        start_time = time.time()
        
        try:
            # Run verification
            verified_response, verification_steps = await self.cov.verify(
                question=query,
                max_claims=max_claims,
                timeout=timeout
            )
            
            # Format the response
            result = {
                "answer": verified_response,
                "method": "chain_of_verification",
                "method_emoji": "🔍✓",
                "verification_steps": verification_steps,
                "verified_claims_count": len([s for s in verification_steps if s.get("status") == "verified"]),
                "refuted_claims_count": len([s for s in verification_steps if s.get("status") == "refuted"]),
                "unverified_claims_count": len([s for s in verification_steps if s.get("status") == "unknown"]),
                "processing_time": time.time() - start_time
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in ChainOfVerificationReasoning.process: {e}")
            return {
                "answer": f"I encountered an error while processing your question using verification.",
                "method": "chain_of_verification",
                "method_emoji": "❌",
                "error": str(e)
            } 