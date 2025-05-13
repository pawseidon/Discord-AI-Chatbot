import os
import re
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from bot_utilities.token_utils import token_optimizer
from bot_utilities.rag_utils import RAGSystem, get_server_rag

class SpeculativeRAG:
    """
    Implementation of Speculative RAG - a two-tiered approach using a drafting model
    to quickly generate retrieval queries and a verification model to ensure quality.
    
    Based on research in "Speculative Retrieval: Generating Retrieval Queries to Enhance LLM Reasoning"
    (https://arxiv.org/abs/2402.10169)
    """
    
    def __init__(self, server_id: str, api_key: str = None):
        """Initialize the speculative RAG system"""
        self.server_id = server_id
        self.api_key = api_key or os.environ.get("API_KEY")
        
        # Get the base RAG system
        self.rag_system = get_server_rag(server_id)
        
        # Initialize drafting model (smaller/faster)
        self.drafting_model = ChatGroq(
            api_key=self.api_key,
            model_name="llama-3-8b-8192" # Smaller model for drafting
        )
        
        # Initialize verification model (larger/more capable)
        self.verification_model = ChatGroq(
            api_key=self.api_key,
            model_name="meta-llama/llama-4-scout-17b-16e-instruct" # Larger model for verification
        )
        
        # Prompts for the speculative process
        self.query_generation_prompt = PromptTemplate.from_template(
            """You are an expert search query generator. Your task is to generate multiple search queries
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
        )
        
        self.verification_prompt = PromptTemplate.from_template(
            """You are an expert at evaluating information relevance. You need to verify if the retrieved information
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
        )
    
    async def generate_search_queries(self, question: str, num_queries: int = 3) -> List[str]:
        """Generate multiple search queries for the question using the drafting model"""
        # Format the prompt with the question
        prompt = self.query_generation_prompt.format(question=question)
        
        # Generate queries with the drafting model
        response = await self.drafting_model.ainvoke(prompt)
        response_text = response.content
        
        # Extract queries using regex
        queries = re.findall(r"Query: (.+)", response_text)
        
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
    
    async def verify_relevance(
        self, 
        question: str, 
        documents: List[Document]
    ) -> List[Tuple[Document, float, str]]:
        """Verify the relevance of retrieved documents using the verification model"""
        if not documents:
            return []
            
        # Prepare retrieved information for verification
        retrieved_info = ""
        for i, doc in enumerate(documents):
            # Optimize content to reduce tokens
            content = token_optimizer.clean_text(doc.page_content)
            content = token_optimizer.truncate_text(content, max_tokens=300)
            retrieved_info += f"[Document {i+1}]:\n{content}\n\n"
            
        # Format the verification prompt
        prompt = self.verification_prompt.format(
            question=question,
            retrieved_info=retrieved_info
        )
        
        # Get verification from the larger model
        response = await self.verification_model.ainvoke(prompt)
        response_text = response.content
        
        # Parse the verification results
        verified_docs = []
        current_doc_idx = 0
        
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
    
    async def query(
        self, 
        question: str, 
        num_search_queries: int = 3, 
        results_per_query: int = 3,
        relevance_threshold: float = 0.6
    ) -> List[Tuple[Document, float, str]]:
        """
        Perform a speculative RAG query process:
        1. Generate multiple search queries from the question
        2. Retrieve documents for each query
        3. Verify relevance with the verification model
        4. Return filtered and ranked results
        """
        # Generate search queries
        search_queries = await self.generate_search_queries(question, num_search_queries)
        
        # Collect all retrieved documents
        all_docs = []
        doc_sources = {}  # Track which query produced each document
        
        for query in search_queries:
            # Retrieve documents for this query
            results = await self.rag_system.query(query, k=results_per_query)
            
            for doc in results:
                # Use a hash of the content as a document identifier
                doc_id = hash(doc.page_content)
                
                if doc_id not in doc_sources:
                    all_docs.append(doc)
                    doc_sources[doc_id] = query
        
        # Verify relevance of all retrieved documents
        verified_docs = await self.verify_relevance(question, all_docs)
        
        # Filter by relevance threshold
        filtered_docs = [(doc, score, explanation) 
                        for doc, score, explanation in verified_docs 
                        if score >= relevance_threshold]
        
        # If nothing passes the threshold but we have documents, keep the best one
        if not filtered_docs and verified_docs:
            filtered_docs = [verified_docs[0]]
            
        return filtered_docs
    
    async def format_as_context(
        self, 
        question: str,
        num_search_queries: int = 3, 
        results_per_query: int = 3,
        relevance_threshold: float = 0.6,
        max_docs: int = 3
    ) -> str:
        """Query and format results as context for an LLM"""
        # Get verified and filtered documents
        verified_docs = await self.query(
            question, 
            num_search_queries, 
            results_per_query,
            relevance_threshold
        )
        
        # Limit to max_docs
        verified_docs = verified_docs[:max_docs]
        
        if not verified_docs:
            return ""
            
        # Format as context
        context = "Relevant information from knowledge base:\n\n"
        
        for i, (doc, score, explanation) in enumerate(verified_docs):
            # Clean and optimize content
            content = token_optimizer.clean_text(doc.page_content)
            content = token_optimizer.truncate_text(content, max_tokens=500)
            
            # Format with confidence score
            context += f"[Document {i+1} - Confidence: {score:.2f}]:\n{content}\n\n"
            
            # Add metadata about the document source if available
            if "source" in doc.metadata:
                context += f"Source: {doc.metadata['source']}\n"
                
            # Add relevance explanation from verification step
            context += f"Relevance: {explanation}\n\n"
            
        return context 