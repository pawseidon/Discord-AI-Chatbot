import os
import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from bot_utilities.token_utils import token_optimizer
from bot_utilities.rag_utils import RAGSystem
from bot_utilities.ai_utils import get_ai_provider

class RelevanceScore(BaseModel):
    """Model for document relevance scoring"""
    score: float = Field(description="Relevance score between 0.0 and 1.0")
    reasoning: str = Field(description="Reasoning behind the score")

class SelfReflectiveRAG:
    """Implementation of Self-Reflective RAG for improved response quality"""
    
    def __init__(self, server_id: str, api_key: str = None, model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        """Initialize the self-reflective RAG system"""
        self.server_id = server_id
        # Get base RAG system
        self.rag_system = RAGSystem(server_id)
        # Initialize model for reflection
        self.api_key = api_key or os.environ.get("API_KEY")
        self.model_name = model_name
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name=self.model_name
        )
        # Prepare the output parser
        self.parser = PydanticOutputParser(pydantic_object=RelevanceScore)
        self.reflection_log = []
        self.quality_metrics = ["relevance", "accuracy", "completeness", "consistency", "helpfulness"]
        
    async def evaluate_document_relevance(self, query: str, document: Document) -> RelevanceScore:
        """Evaluate the relevance of a document to the query"""
        # Create prompt for relevance assessment
        prompt_template = """You are an expert at evaluating search result relevance.
        
        Query: {query}
        
        Document Content: {content}
        
        Rate the relevance of this document to the query on a scale of 0.0 to 1.0, where:
        - 0.0 means completely irrelevant
        - 1.0 means perfectly relevant and provides exactly what the query is looking for
        
        Provide your reasoning for the score as well.
        
        {format_instructions}
        """
        
        format_instructions = self.parser.get_format_instructions()
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["query", "content"],
            partial_variables={"format_instructions": format_instructions}
        )
        
        # Format the document content to reduce tokens
        content = token_optimizer.clean_text(document.page_content)
        content = token_optimizer.truncate_text(content, max_tokens=500)
        
        # Create the formatted prompt
        formatted_prompt = prompt.format(query=query, content=content)
        
        # Get evaluation from the LLM
        response = await self.llm.ainvoke(formatted_prompt)
        
        try:
            # Parse the response into a RelevanceScore object
            relevance_score = self.parser.parse(response.content)
            return relevance_score
        except Exception as e:
            # If parsing fails, return a default score
            print(f"Error parsing relevance score: {e}")
            return RelevanceScore(score=0.5, reasoning="Failed to parse relevance score")
    
    async def query_with_reflection(self, query: str, threshold: float = 0.3, k: int = 5) -> Tuple[List[Document], List[RelevanceScore]]:
        """Query the RAG system with self-reflection on results"""
        # Get initial results from the base RAG system
        raw_results = await self.rag_system.query(query, k=k+2)  # Get a few extra for filtering
        
        if not raw_results:
            return [], []
        
        # Evaluate each document for relevance
        evaluations = []
        for doc in raw_results:
            score = await self.evaluate_document_relevance(query, doc)
            evaluations.append((doc, score))
        
        # Sort by relevance score
        evaluations.sort(key=lambda x: x[1].score, reverse=True)
        
        # Filter by threshold
        filtered_results = [(doc, score) for doc, score in evaluations if score.score >= threshold]
        
        # If no documents meet the threshold, take the top one to avoid empty results
        if not filtered_results and evaluations:
            filtered_results = [evaluations[0]]
        
        # Take top k after filtering
        final_results = filtered_results[:k]
        
        # Separate documents and scores
        documents = [doc for doc, _ in final_results]
        scores = [score for _, score in final_results]
        
        return documents, scores
    
    async def format_reflective_results(self, query: str, threshold: float = 0.3, k: int = 5) -> str:
        """Format retrieval results as context with reflection information"""
        documents, scores = await self.query_with_reflection(query, threshold, k)
        
        if not documents:
            return ""
        
        context = "Relevant information from knowledge base (with confidence scores):\n\n"
        
        for i, (doc, score) in enumerate(zip(documents, scores)):
            # Add document content with relevance information
            context += f"[Document {i+1} - Relevance: {score.score:.2f}]:\n"
            context += f"{doc.page_content}\n\n"
            
            # Add metadata about the document source if available
            if "source" in doc.metadata:
                context += f"Source: {doc.metadata['source']}\n"
            
            # Add a brief note about why this document was considered relevant
            context += f"Relevance: {score.reasoning}\n\n"
        
        return context
    
    async def reflect_on_response(self, query: str, response: str, reference_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Evaluate a response against quality metrics and provide reflection
        
        Args:
            query: The original user query
            response: The response to evaluate
            reference_data: Optional reference data to check against
            
        Returns:
            Dict containing reflection results and improvement suggestions
        """
        # Get AI provider for evaluation
        ai_provider = await get_ai_provider()
        
        # Create evaluation prompt
        prompt = f"""Evaluate this response to the following query:
        
Query: {query}

Response: {response}

Analyze this response and evaluate it on the following criteria (score 1-10):
1. Relevance: How well does the response address the query?
2. Accuracy: Are the facts and information correct?
3. Completeness: Does it fully address all aspects of the query?
4. Consistency: Is the response internally consistent?
5. Helpfulness: Is the response actually helpful to the user?

For each criterion, provide:
- Score (1-10)
- Brief explanation
- Specific suggestion for improvement

Finally, provide an overall assessment and concrete recommendations for how to improve similar responses in the future.
"""

        try:
            # Get evaluation
            result = await ai_provider.async_call(prompt, temperature=0.3)
            
            # Parse evaluation results
            parsed_result = self._parse_evaluation(result)
            
            # Log the reflection
            reflection_entry = {
                "timestamp": time.time(),
                "query": query,
                "response": response,
                "evaluation": parsed_result,
                "has_reference": reference_data is not None
            }
            self.reflection_log.append(reflection_entry)
            
            return parsed_result
            
        except Exception as e:
            print(f"Error during self-reflection: {e}")
            return {
                "error": str(e),
                "overall_score": 0,
                "improvements": ["Unable to perform self-reflection due to an error."]
            }
    
    def _parse_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
        """Parse the evaluation text into structured data"""
        metrics = {}
        improvements = []
        
        # Extract scores using regex
        score_pattern = r'(\d+)\/10|score:\s*(\d+)|rating:\s*(\d+)'
        
        for metric in self.quality_metrics:
            # Look for sections about each metric
            metric_pattern = re.compile(
                rf'{metric}.*?(?:rating|score).*?({score_pattern})', 
                re.IGNORECASE | re.DOTALL
            )
            match = metric_pattern.search(evaluation_text)
            
            if match:
                # Try to extract numeric score
                score_text = match.group(1)
                scores = re.findall(r'\d+', score_text)
                if scores:
                    metrics[metric] = int(scores[0])
                else:
                    metrics[metric] = 5  # Default middle score if parsing fails
            else:
                metrics[metric] = 5  # Default if metric not found
        
        # Extract overall assessment
        overall_pattern = re.compile(
            r'overall.*?assessment.*?(.*?)(?=recommendation|improvement|$)', 
            re.IGNORECASE | re.DOTALL
        )
        overall_match = overall_pattern.search(evaluation_text)
        overall_assessment = overall_match.group(1).strip() if overall_match else "No overall assessment provided."
        
        # Extract recommendations/improvements
        improvements_pattern = re.compile(
            r'(?:recommendation|improvement|suggestion)s?:?\s*(.*?)(?=\n\n|$)', 
            re.IGNORECASE | re.DOTALL
        )
        improvements_match = improvements_pattern.search(evaluation_text)
        
        if improvements_match:
            # Split by bullet points or numbers
            improvement_text = improvements_match.group(1)
            improvement_items = re.split(r'\n\s*[-•*]|\n\s*\d+\.', improvement_text)
            improvements = [item.strip() for item in improvement_items if item.strip()]
        
        # Calculate overall score
        overall_score = sum(metrics.values()) / len(metrics) if metrics else 0
        
        return {
            "metrics": metrics,
            "overall_score": overall_score,
            "overall_assessment": overall_assessment,
            "improvements": improvements
        }
    
    async def improve_response(self, query: str, original_response: str, evaluation: Dict[str, Any]) -> str:
        """
        Generate an improved response based on self-reflection
        
        Args:
            query: The original query
            original_response: The original response
            evaluation: The evaluation results
            
        Returns:
            An improved response
        """
        if "improvements" not in evaluation or not evaluation["improvements"]:
            return original_response
            
        # Get AI provider
        ai_provider = await get_ai_provider()
        
        # Create improvement prompt
        improvements_text = "\n".join([f"- {imp}" for imp in evaluation["improvements"]])
        
        prompt = f"""Here is a query and an initial response that needs improvement:

Query: {query}

Initial Response: {original_response}

Please improve this response based on the following suggestions:
{improvements_text}

Generate a new, improved response that addresses these suggestions while maintaining a natural, conversational tone.
"""

        try:
            # Get improved response
            improved_response = await ai_provider.async_call(prompt, temperature=0.4)
            return improved_response
        except Exception as e:
            print(f"Error generating improved response: {e}")
            return original_response
    
    def get_recent_reflections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent reflection entries"""
        return self.reflection_log[-limit:] if self.reflection_log else []
        
    def get_average_scores(self) -> Dict[str, float]:
        """Get average scores across all metrics"""
        if not self.reflection_log:
            return {metric: 0.0 for metric in self.quality_metrics}
            
        totals = {metric: 0.0 for metric in self.quality_metrics}
        count = 0
        
        for entry in self.reflection_log:
            if "evaluation" in entry and "metrics" in entry["evaluation"]:
                metrics = entry["evaluation"]["metrics"]
                for metric in self.quality_metrics:
                    if metric in metrics:
                        totals[metric] += metrics[metric]
                count += 1
        
        if count == 0:
            return {metric: 0.0 for metric in self.quality_metrics}
            
        return {metric: totals[metric] / count for metric in self.quality_metrics}

def create_reflective_rag(server_id: str = "global"):
    """
    Factory function to create a self-reflective RAG instance
    
    Args:
        server_id: The server ID to associate with this RAG instance, defaults to "global"
        
    Returns:
        SelfReflectiveRAG: A configured reflective RAG instance
    """
    return SelfReflectiveRAG(server_id=server_id) 