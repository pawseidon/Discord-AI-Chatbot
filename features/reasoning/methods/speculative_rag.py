import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("speculative_rag")

class SpeculativeRAG:
    """
    Implementation of Speculative RAG - a two-tiered approach using a drafting model
    to quickly generate retrieval queries and a verification model to ensure quality.
    
    Based on research in "Speculative Retrieval: Generating Retrieval Queries to Enhance LLM Reasoning"
    (https://arxiv.org/abs/2402.10169)
    """
    
    def __init__(self, ai_provider=None, response_cache=None, knowledge_base=None, 
                drafting_model=None, verification_model=None):
        """
        Initialize the speculative RAG system
        
        Args:
            ai_provider: AI provider for reasoning
            response_cache: Optional cache for responses
            knowledge_base: Knowledge base for document retrieval
            drafting_model: Optional faster model for generating queries
            verification_model: Optional more capable model for verification
        """
        self.ai_provider = ai_provider
        self.response_cache = response_cache
        self.knowledge_base = knowledge_base
        self.drafting_model = drafting_model or ai_provider
        self.verification_model = verification_model or ai_provider
        
        # Query generation prompt
        self.query_generation_prompt = """You are an expert search query generator. Your task is to generate multiple search queries
        that would help answer the user's question or solve their problem.
        
        User question: {question}
        
        Generate 3-5 different search queries that would help retrieve relevant information.
        Format each query on a new line starting with "Query: ".
        
        Your queries should:
        1. Be diverse and cover different aspects of the question
        2. Use different phrasings and keywords
        3. Range from specific to general
        4. Consider potential information needs that aren't explicitly stated
        
        QUERIES:
        """
        
        # Verification prompt
        self.verification_prompt = """You are an expert at evaluating information relevance. You need to verify if the retrieved information
        is actually helpful for answering the user's question.
        
        User question: {question}
        
        Retrieved information:
        {retrieved_info}
        
        Is this information relevant and helpful for answering the question? Rate each piece of information
        on a scale of 0-10, where:
        - 0-3: Not relevant, should be discarded
        - 4-6: Somewhat relevant but incomplete
        - 7-10: Highly relevant and helpful
        
        For each document, provide:
        1. Relevance score (0-10)
        2. Brief explanation of why it's relevant or not
        3. How it helps answer the original question
        
        EVALUATION:
        """
    
    async def generate_search_queries(self, question: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple search queries for the question using the drafting model
        
        Args:
            question: User question
            num_queries: Number of queries to generate
            
        Returns:
            List of generated search queries
        """
        # Format the prompt with the question
        prompt = self.query_generation_prompt.format(question=question)
        
        # Generate queries with the drafting model
        try:
            response = await self.drafting_model.generate_response(
                prompt=prompt,
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract queries using regex
            queries = re.findall(r"Query: (.+)", response)
            
            # Ensure we have the requested number of queries
            if len(queries) < num_queries:
                # Generate more generic queries based on keywords
                keywords = re.findall(r'\b[A-Za-z]{3,}\b', question)
                for keyword in keywords[:num_queries - len(queries)]:
                    queries.append(f"{keyword} information")
                    
            # Limit to requested number and remove duplicates
            unique_queries = []
            for query in queries:
                if query not in unique_queries:
                    unique_queries.append(query)
                    if len(unique_queries) >= num_queries:
                        break
                        
            return unique_queries[:num_queries]
            
        except Exception as e:
            logger.error(f"Error generating search queries: {e}")
            # Fallback to simple keyword extraction
            keywords = re.findall(r'\b[A-Za-z]{3,}\b', question)
            keywords = [k for k in keywords if len(k) > 3]
            return [f"{question}", f"{' '.join(keywords[:3])}", f"{keywords[0]} information"][:num_queries]
    
    async def verify_relevance(
        self, 
        question: str, 
        documents: List[Any]
    ) -> List[Tuple[Any, float, str]]:
        """
        Verify the relevance of retrieved documents using the verification model
        
        Args:
            question: User question
            documents: Retrieved documents
            
        Returns:
            List of tuples containing (document, relevance score, explanation)
        """
        if not documents:
            return []
            
        # Prepare retrieved information for verification
        retrieved_info = ""
        for i, doc in enumerate(documents):
            # Get document content
            if hasattr(doc, "page_content"):
                content = doc.page_content
            else:
                content = str(doc)
                
            # Truncate if too long
            if len(content) > 500:
                content = content[:500] + "..."
                
            retrieved_info += f"[Document {i+1}]:\n{content}\n\n"
            
        # Format the verification prompt
        prompt = self.verification_prompt.format(
            question=question,
            retrieved_info=retrieved_info
        )
        
        # Get verification from the model
        try:
            response_text = await self.verification_model.generate_response(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse the verification results
            verified_docs = []
            
            # Extract scores using regex
            score_pattern = r"\[Document (\d+)\].*?(\d+)\/10|score.*?(\d+)\/10|rating.*?(\d+)\/10|relevance.*?(\d+)\/10"
            scores = re.findall(score_pattern, response_text, re.IGNORECASE | re.DOTALL)
            
            # Create a mapping of document index to score and explanation
            doc_scores = {}
            for match in scores:
                # Find the first non-empty group (document number)
                doc_num = next((int(g) for g in match if g.isdigit()), 0)
                if doc_num == 0:
                    continue
                    
                # Find the score
                score_str = next((g for g in match[1:] if g.isdigit()), "0")
                score = float(score_str) / 10.0  # Normalize to 0-1
                
                # Extract explanation (text following the score until the next document)
                doc_start = response_text.find(f"[Document {doc_num}]")
                next_doc_start = response_text.find(f"[Document {doc_num+1}]")
                
                if next_doc_start == -1:
                    explanation_text = response_text[doc_start:]
                else:
                    explanation_text = response_text[doc_start:next_doc_start]
                    
                # Clean up explanation
                explanation = re.sub(r".*?(\d+)\/10", "", explanation_text).strip()
                explanation = re.split(r"\n\n", explanation)[0].strip()
                
                doc_scores[doc_num-1] = (score, explanation)
            
            # Match scores with documents
            for i, doc in enumerate(documents):
                if i in doc_scores:
                    score, explanation = doc_scores[i]
                    verified_docs.append((doc, score, explanation))
                else:
                    # Default score if not found in parsing
                    verified_docs.append((doc, 0.5, "No explicit evaluation available"))
                    
            # Sort by relevance score (descending)
            verified_docs.sort(key=lambda x: x[1], reverse=True)
            
            return verified_docs
            
        except Exception as e:
            logger.error(f"Error verifying document relevance: {e}")
            # Return documents with default score
            return [(doc, 0.5, "Verification error") for doc in documents]
    
    async def process(self, 
                    query: str, 
                    user_id: str,
                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query using Speculative RAG
        
        Args:
            query: User query
            user_id: User ID
            context: Additional context
            
        Returns:
            Dict containing the answer and thinking steps
        """
        context = context or {}
        
        # Generate multiple search queries
        search_queries = await self.generate_search_queries(query, num_queries=3)
        
        # Keep track of all documents and their sources
        all_documents = []
        query_results = {}
        
        # Query the knowledge base with each search query
        if self.knowledge_base:
            for search_query in search_queries:
                results = await self.knowledge_base.search(
                    query=search_query,
                    user_id=user_id,
                    limit=3
                )
                
                if results:
                    query_results[search_query] = results
                    all_documents.extend(results)
        
        # If no results from knowledge base, early return
        if not all_documents:
            # Generate a direct response
            answer = await self.ai_provider.generate_response(
                prompt=f"Answer this question as best you can: {query}",
                temperature=0.7,
                max_tokens=1000
            )
            
            return {
                "answer": answer,
                "thinking_steps": [
                    {"type": "search_queries", "queries": search_queries},
                    {"type": "document_retrieval", "count": 0, "note": "No documents found"}
                ],
                "method": "speculative_rag",
                "method_emoji": "🔮"
            }
        
        # Verify document relevance
        verified_docs = await self.verify_relevance(query, all_documents)
        
        # Filter by relevance threshold and limit the number of documents
        relevant_docs = [(doc, score, exp) for doc, score, exp in verified_docs if score >= 0.6]
        top_docs = relevant_docs[:5]
        
        # Format documents as context for the AI
        formatted_context = self._format_documents_as_context(top_docs)
        
        # Generate response based on the verified documents
        response_prompt = f"""Answer the following question based on the provided information:

Question: {query}

{formatted_context}

Provide a comprehensive, accurate answer based on the information above. If the information doesn't fully answer the question, acknowledge the limitations in your response.
"""
        
        answer = await self.ai_provider.generate_response(
            prompt=response_prompt,
            temperature=0.5,
            max_tokens=1000
        )
        
        # Prepare result
        result = {
            "answer": answer,
            "thinking_steps": [
                {"type": "search_queries", "queries": search_queries},
                {"type": "document_retrieval", "count": len(all_documents)},
                {"type": "document_verification", "verified_count": len(top_docs)},
                {"type": "relevant_documents", "documents": [
                    {"content": self._get_document_content(doc), "score": score, "explanation": exp}
                    for doc, score, exp in top_docs
                ]}
            ],
            "method": "speculative_rag",
            "method_emoji": "🔮"
        }
        
        return result
    
    def _format_documents_as_context(self, verified_docs: List[Tuple[Any, float, str]]) -> str:
        """
        Format verified documents as context for response generation
        
        Args:
            verified_docs: List of tuples (document, score, explanation)
            
        Returns:
            Formatted context string
        """
        if not verified_docs:
            return "No relevant information found."
        
        context = "Relevant information:\n\n"
        
        for i, (doc, score, explanation) in enumerate(verified_docs):
            content = self._get_document_content(doc)
            
            # Add document with relevance score
            context += f"[Document {i+1} - Relevance: {score:.2f}]:\n"
            context += f"{content}\n\n"
            
            # Add metadata if available
            if hasattr(doc, "metadata") and doc.metadata:
                if "source" in doc.metadata:
                    context += f"Source: {doc.metadata['source']}\n"
                    
        return context
    
    def _get_document_content(self, doc: Any) -> str:
        """Extract content from document object"""
        if hasattr(doc, "page_content"):
            return doc.page_content
        elif hasattr(doc, "content"):
            return doc.content
        else:
            return str(doc)

async def process_speculative_rag(
    query: str,
    user_id: str,
    ai_provider=None,
    response_cache=None,
    knowledge_base=None,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a query using Speculative RAG
    
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
    speculative_rag = SpeculativeRAG(
        ai_provider=ai_provider,
        response_cache=response_cache,
        knowledge_base=knowledge_base
    )
    
    return await speculative_rag.process(query, user_id, context) 