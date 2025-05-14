import os
import json
import re
import time
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("reflective_rag")

class RelevanceScore:
    """Model for document relevance scoring"""
    score: float
    reasoning: str
    
    def __init__(self, score: float, reasoning: str):
        self.score = score
        self.reasoning = reasoning

class ReflectiveRAG:
    """Implementation of Self-Reflective RAG for improved response quality"""
    
    def __init__(self, ai_provider=None, response_cache=None, knowledge_base=None):
        """
        Initialize the self-reflective RAG system
        
        Args:
            ai_provider: AI provider for reasoning and evaluation
            response_cache: Optional cache for responses
            knowledge_base: Knowledge base for document retrieval
        """
        self.ai_provider = ai_provider
        self.response_cache = response_cache
        self.knowledge_base = knowledge_base
        self.reflection_log = []
        self.quality_metrics = ["relevance", "accuracy", "completeness", "consistency", "helpfulness"]
        
    async def evaluate_document_relevance(self, query: str, document: Any) -> RelevanceScore:
        """
        Evaluate the relevance of a document to the query
        
        Args:
            query: User query
            document: Document to evaluate
            
        Returns:
            RelevanceScore with score and reasoning
        """
        # Create prompt for relevance assessment
        prompt = f"""You are an expert at evaluating search result relevance.
        
        Query: {query}
        
        Document Content: {document.page_content if hasattr(document, 'page_content') else str(document)}
        
        Rate the relevance of this document to the query on a scale of 0.0 to 1.0, where:
        - 0.0 means completely irrelevant
        - 1.0 means perfectly relevant and provides exactly what the query is looking for
        
        Provide your reasoning for the score as well.
        
        Format your response as a JSON object with 'score' (float) and 'reasoning' (string) fields.
        """
        
        # Get evaluation from the AI provider
        try:
            response = await self.ai_provider.generate_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            # Try to parse the response as JSON
            try:
                # Extract JSON from the response
                json_match = re.search(r'{.*}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_data = json.loads(json_str)
                    return RelevanceScore(
                        score=float(parsed_data.get("score", 0.5)),
                        reasoning=parsed_data.get("reasoning", "No reasoning provided")
                    )
            except Exception as e:
                logger.error(f"Error parsing relevance score: {e}")
            
            # Fallback: try to extract score directly from text
            score_match = re.search(r'score[:\s]+([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                return RelevanceScore(score=score, reasoning="Extracted from text")
                
            # If all extraction fails, return a default score
            return RelevanceScore(score=0.5, reasoning="Failed to parse relevance score")
            
        except Exception as e:
            logger.error(f"Error evaluating document relevance: {e}")
            return RelevanceScore(score=0.5, reasoning=f"Error in evaluation: {str(e)}")
    
    async def query_with_reflection(self, query: str, user_id: str, threshold: float = 0.3, k: int = 5) -> Tuple[List[Any], List[RelevanceScore]]:
        """
        Query the knowledge base with self-reflection on results
        
        Args:
            query: User query
            user_id: User ID
            threshold: Minimum relevance score threshold
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (relevant documents, relevance scores)
        """
        if not self.knowledge_base:
            logger.warning("No knowledge base provided for reflective RAG")
            return [], []
            
        # Get initial results from the knowledge base
        try:
            raw_results = await self.knowledge_base.search(
                query=query, 
                user_id=user_id, 
                limit=k+2  # Get a few extra for filtering
            )
            
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
            
        except Exception as e:
            logger.error(f"Error in query_with_reflection: {e}")
            return [], []
    
    async def process(self, 
                     query: str, 
                     user_id: str,
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query using reflective RAG
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            Dict containing the answer and thinking steps
        """
        context = context or {}
        
        # Query documents with reflection
        documents, scores = await self.query_with_reflection(
            query=query, 
            user_id=user_id, 
            threshold=0.3, 
            k=5
        )
        
        # Format documents as context for the AI
        formatted_context = self._format_documents_as_context(documents, scores)
        
        # Generate initial response based on the documents
        initial_prompt = f"""Answer the following question based on the provided information:

Question: {query}

{formatted_context}

Provide a comprehensive, accurate answer based exclusively on the information above. If the information doesn't contain enough to answer the question, acknowledge the limitations in your response.
"""
        
        initial_response = await self.ai_provider.generate_response(
            prompt=initial_prompt,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Reflect on the initial response
        reflection_results = await self.reflect_on_response(query, initial_response)
        
        # If the reflection indicates significant issues, improve the response
        if self._needs_improvement(reflection_results):
            improved_response = await self.improve_response(query, initial_response, reflection_results)
            final_response = improved_response
            used_reflection = True
        else:
            final_response = initial_response
            used_reflection = False
        
        # Prepare result
        result = {
            "answer": final_response,
            "thinking_steps": [
                {"type": "document_retrieval", "documents": [self._document_to_dict(doc) for doc in documents]},
                {"type": "relevance_scores", "scores": [{"score": s.score, "reasoning": s.reasoning} for s in scores]},
                {"type": "initial_response", "response": initial_response},
                {"type": "reflection", "evaluation": reflection_results},
                {"type": "used_reflection", "value": used_reflection}
            ],
            "method": "reflective_rag",
            "method_emoji": "🪞"
        }
        
        return result
    
    def _format_documents_as_context(self, documents: List[Any], scores: List[RelevanceScore]) -> str:
        """
        Format retrieval results as context with reflection information
        
        Args:
            documents: List of retrieved documents
            scores: List of relevance scores
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant information found."
        
        context = "Relevant information:\n\n"
        
        for i, (doc, score) in enumerate(zip(documents, scores)):
            # Add document content with relevance information
            context += f"[Document {i+1} - Relevance: {score.score:.2f}]:\n"
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            context += f"{content}\n\n"
            
            # Add metadata about the document source if available
            if hasattr(doc, "metadata") and doc.metadata and "source" in doc.metadata:
                context += f"Source: {doc.metadata['source']}\n"
            
        return context
    
    async def reflect_on_response(self, query: str, response: str) -> Dict[str, Any]:
        """
        Evaluate a response against quality metrics and provide reflection
        
        Args:
            query: The original user query
            response: The response to evaluate
            
        Returns:
            Dict containing reflection results and improvement suggestions
        """
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

Format your response as a JSON object with the criteria as keys, each containing 'score', 'explanation', and 'suggestion' fields, plus an 'overall' key with 'assessment' and 'recommendations' fields.
"""

        try:
            # Get evaluation from AI provider
            evaluation_text = await self.ai_provider.generate_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse evaluation results
            parsed_result = self._parse_evaluation(evaluation_text)
            
            # Log the reflection
            reflection_entry = {
                "timestamp": time.time(),
                "query": query,
                "response": response,
                "evaluation": parsed_result
            }
            self.reflection_log.append(reflection_entry)
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error during self-reflection: {e}")
            return {
                "error": str(e),
                "overall": {
                    "assessment": "Error occurred during evaluation",
                    "recommendations": "Retry with more specific query"
                }
            }
    
    def _parse_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
        """
        Parse evaluation response into structured format
        
        Args:
            evaluation_text: Raw evaluation text from AI
            
        Returns:
            Structured evaluation results
        """
        parsed_eval = {}
        
        try:
            # Try to parse as JSON first
            json_match = re.search(r'{.*}', evaluation_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_eval = json.loads(json_str)
                return parsed_eval
        except Exception as e:
            logger.debug(f"JSON parsing failed, trying pattern matching: {e}")
        
        # If JSON parsing fails, extract information using regex patterns
        try:
            # Extract scores for each metric
            for metric in self.quality_metrics:
                score_pattern = rf"{metric}.*?score.*?(\d+)"
                explanation_pattern = rf"{metric}.*?explanation:?\s*(.*?)(?=\n\w|$)"
                suggestion_pattern = rf"{metric}.*?suggestion:?\s*(.*?)(?=\n\w|$)"
                
                score_match = re.search(score_pattern, evaluation_text, re.IGNORECASE | re.DOTALL)
                explanation_match = re.search(explanation_pattern, evaluation_text, re.IGNORECASE | re.DOTALL)
                suggestion_match = re.search(suggestion_pattern, evaluation_text, re.IGNORECASE | re.DOTALL)
                
                score = int(score_match.group(1)) if score_match else 5
                explanation = explanation_match.group(1).strip() if explanation_match else ""
                suggestion = suggestion_match.group(1).strip() if suggestion_match else ""
                
                parsed_eval[metric] = {
                    "score": score,
                    "explanation": explanation,
                    "suggestion": suggestion
                }
            
            # Extract overall assessment
            overall_pattern = r"overall.*?assessment:?\s*(.*?)(?=\n\w|$)"
            recommend_pattern = r"recommendations?:?\s*(.*?)(?=\n\w|$)"
            
            overall_match = re.search(overall_pattern, evaluation_text, re.IGNORECASE | re.DOTALL)
            recommend_match = re.search(recommend_pattern, evaluation_text, re.IGNORECASE | re.DOTALL)
            
            overall = overall_match.group(1).strip() if overall_match else ""
            recommendations = recommend_match.group(1).strip() if recommend_match else ""
            
            parsed_eval["overall"] = {
                "assessment": overall,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error parsing evaluation text: {e}")
            # Provide default values for required fields
            if "overall" not in parsed_eval:
                parsed_eval["overall"] = {
                    "assessment": "Evaluation parsing failed",
                    "recommendations": "Review response manually"
                }
        
        return parsed_eval
    
    def _needs_improvement(self, evaluation: Dict[str, Any]) -> bool:
        """
        Determine if a response needs improvement based on evaluation
        
        Args:
            evaluation: Evaluation results
            
        Returns:
            True if response needs improvement
        """
        # Calculate average score across metrics
        total_score = 0
        count = 0
        
        for metric in self.quality_metrics:
            if metric in evaluation and "score" in evaluation[metric]:
                total_score += evaluation[metric]["score"]
                count += 1
        
        if count == 0:
            return True  # If no metrics were found, assume improvement needed
        
        average_score = total_score / count
        
        # If average score is below 7, suggest improvement
        return average_score < 7.0
    
    async def improve_response(self, query: str, original_response: str, evaluation: Dict[str, Any]) -> str:
        """
        Improve a response based on reflection evaluation
        
        Args:
            query: Original user query
            original_response: Original response
            evaluation: Evaluation results
            
        Returns:
            Improved response
        """
        # Construct improvement prompt
        prompt = f"""Your task is to improve this response based on the provided evaluation:

Original Query: {query}

Original Response: {original_response}

Evaluation:
"""
        # Add evaluation details
        for metric in self.quality_metrics:
            if metric in evaluation:
                metric_data = evaluation[metric]
                prompt += f"- {metric.capitalize()}: Score {metric_data.get('score', 'N/A')}/10\n"
                prompt += f"  Explanation: {metric_data.get('explanation', 'N/A')}\n"
                prompt += f"  Suggestion: {metric_data.get('suggestion', 'N/A')}\n"
        
        if "overall" in evaluation:
            prompt += f"\nOverall Assessment: {evaluation['overall'].get('assessment', '')}\n"
            prompt += f"Recommendations: {evaluation['overall'].get('recommendations', '')}\n"
        
        prompt += """
Now, rewrite the response to address all the improvement suggestions. The improved response should:
1. Fix all identified issues
2. Be more comprehensive and accurate
3. Address all aspects of the original query
4. Be well-structured and helpful
5. Maintain a friendly, conversational tone

Provide only the improved response without any additional commentary.
"""
        
        try:
            improved_response = await self.ai_provider.generate_response(
                prompt=prompt,
                temperature=0.4,
                max_tokens=1500
            )
            
            return improved_response
        except Exception as e:
            logger.error(f"Error improving response: {e}")
            return original_response
    
    def _document_to_dict(self, doc: Any) -> Dict[str, Any]:
        """Convert document to dictionary for serialization"""
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            return {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
        elif hasattr(doc, "__dict__"):
            return doc.__dict__
        else:
            return {"content": str(doc)}

async def process_reflective_rag(
    query: str,
    user_id: str,
    ai_provider=None,
    response_cache=None,
    knowledge_base=None,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a query using reflective RAG
    
    Args:
        query: User query
        user_id: User ID
        ai_provider: AI provider for reasoning
        response_cache: Optional cache for responses
        knowledge_base: Knowledge base for document retrieval
        context: Additional context
        
    Returns:
        Dict containing the answer and thinking steps
    """
    reflective_rag = ReflectiveRAG(
        ai_provider=ai_provider,
        response_cache=response_cache,
        knowledge_base=knowledge_base
    )
    
    return await reflective_rag.process(query, user_id, context) 