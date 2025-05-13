import os
import re
import json
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from bot_utilities.token_utils import token_optimizer
from bot_utilities.ai_utils import search_internet, get_ai_provider
import time

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
    
    def __init__(self, api_key: str = None, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        """Initialize the Chain-of-Verification system"""
        self.api_key = api_key or os.environ.get("API_KEY")
        self.model_name = model_name
        
        # Set up the LLM
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name
        )
        
        # CoV prompts
        self.generation_prompt = PromptTemplate.from_template(
            """You are a helpful AI assistant responsible for generating accurate information.
            
            User question: {question}
            
            First, provide a direct answer to the question. Be detailed and informative while ensuring factual accuracy.
            
            Your response:
            """
        )
        
        self.claim_extraction_prompt = PromptTemplate.from_template(
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
        
        self.verification_prompt = PromptTemplate.from_template(
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
        
        self.rewrite_prompt = PromptTemplate.from_template(
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
    
    async def generate_initial_response(self, question: str) -> str:
        """Generate the initial response to the question"""
        prompt = self.generation_prompt.format(question=question)
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from the response"""
        prompt = self.claim_extraction_prompt.format(response=response)
        extraction_response = await self.llm.ainvoke(prompt)
        
        # Parse the claims
        claims = []
        for line in extraction_response.content.split('\n'):
            match = re.search(r"Claim\s*\d+\s*:\s*(.*)", line)
            if match:
                claim = match.group(1).strip()
                if claim:
                    claims.append(claim)
        
        return claims
    
    async def verify_claim(self, claim: str) -> VerificationStep:
        """Verify a single claim using web search evidence"""
        # Search for evidence
        search_query = f"{claim} facts evidence"
        evidence = await search_internet(search_query)
        
        if not evidence or evidence == "Internet access has been disabled by user":
            return VerificationStep(
                claim=claim,
                verification="Could not verify due to lack of search access",
                evidence="No evidence available",
                status="unknown"
            )
        
        # Optimize evidence to reduce tokens
        evidence = token_optimizer.clean_text(evidence)
        evidence = token_optimizer.truncate_text(evidence, max_tokens=800)
        
        # Verify the claim
        prompt = self.verification_prompt.format(claim=claim, evidence=evidence)
        verification_response = await self.llm.ainvoke(prompt)
        verification_text = verification_response.content
        
        # Extract the classification
        status = "unknown"
        if "VERIFIED" in verification_text:
            status = "verified"
        elif "REFUTED" in verification_text:
            status = "refuted"
        elif "UNVERIFIED" in verification_text:
            status = "unknown"
        
        # Extract the explanation (exclude the classification line)
        explanation = re.sub(r"Classification.*?:.*?\n", "", verification_text)
        explanation = explanation.strip()
        
        return VerificationStep(
            claim=claim,
            verification=explanation,
            evidence=evidence,
            status=status
        )
    
    async def rewrite_response(self, question: str, original_response: str, verification_steps: List[VerificationStep]) -> str:
        """Rewrite the response based on verification results"""
        # Format verification results
        verification_results = ""
        for i, step in enumerate(verification_steps):
            verification_results += f"Claim {i+1}: {step.claim}\n"
            verification_results += f"Status: {step.status.upper()}\n"
            verification_results += f"Verification: {step.verification}\n\n"
        
        # Create rewrite prompt
        prompt = self.rewrite_prompt.format(
            question=question,
            original_response=original_response,
            verification_results=verification_results
        )
        
        # Get rewritten response
        rewrite_response = await self.llm.ainvoke(prompt)
        return rewrite_response.content
    
    async def verify(
        self, 
        question: str, 
        max_claims: int = 5, 
        timeout: int = 60
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Process a question through the Chain-of-Verification:
        1. Generate an initial response
        2. Extract factual claims
        3. Verify each claim
        4. Rewrite the response based on verification
        
        Args:
            question: The user's question
            max_claims: Maximum number of claims to verify
            timeout: Timeout in seconds
            
        Returns:
            Tuple: (verified_response, verification_steps)
        """
        start_time = time.time()
        verification_steps = []
        
        try:
            # Generate initial response
            original_response = await self.generate_initial_response(question)
            
            # Check remaining time
            elapsed = time.time() - start_time
            if elapsed > timeout * 0.3:  # If initial generation took >30% of allowed time
                return original_response, []
            
            # Extract claims
            claims = await self.extract_claims(original_response)
            
            # Limit number of claims to verify
            claims = claims[:max_claims]
            
            # Verify each claim in parallel
            verification_tasks = []
            for claim in claims:
                task = asyncio.create_task(self.verify_claim(claim))
                verification_tasks.append(task)
            
            # Wait for all verifications with timeout
            remaining_time = max(1, timeout - (time.time() - start_time))
            verified_steps = await asyncio.gather(*verification_tasks, return_exceptions=True)
            
            # Process results, handling any exceptions
            for result in verified_steps:
                if isinstance(result, Exception):
                    # Create a fallback step for failed verifications
                    step = VerificationStep(
                        claim="Verification failed due to error",
                        verification=f"Error: {str(result)}",
                        evidence="",
                        status="unknown"
                    )
                else:
                    step = result
                verification_steps.append(step)
            
            # Check if there's enough time to rewrite
            elapsed = time.time() - start_time
            if elapsed > timeout * 0.8:  # If we've used >80% of time
                return original_response, [step.to_dict() for step in verification_steps]
            
            # Rewrite response
            verified_response = await self.rewrite_response(question, original_response, verification_steps)
            
            return verified_response, [step.to_dict() for step in verification_steps]
            
        except asyncio.TimeoutError:
            # If we timeout, return what we have
            return original_response, [step.to_dict() for step in verification_steps]
            
        except Exception as e:
            # If any other error occurs, fall back to the original response
            return f"I encountered an error during verification: {str(e)}", []

async def run_verification(
    question: str, 
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    max_claims: int = 5,
    timeout: int = 60
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Run Chain-of-Verification on a question
    
    Args:
        question: The user's question
        model: Model to use
        max_claims: Maximum number of claims to verify
        timeout: Timeout in seconds
        
    Returns:
        Tuple: (verified_response, verification_steps)
    """
    api_key = os.environ.get("API_KEY")
    cov = ChainOfVerification(api_key=api_key, model_name=model)
    return await cov.verify(question, max_claims, timeout) 