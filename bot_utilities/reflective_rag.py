import os
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from bot_utilities.token_utils import token_optimizer
from bot_utilities.rag_utils import RAGSystem

class RelevanceScore(BaseModel):
    """Model for document relevance scoring"""
    score: float = Field(description="Relevance score between 0.0 and 1.0")
    reasoning: str = Field(description="Reasoning behind the score")

class SelfReflectiveRAG:
    """RAG system with self-reflection capabilities for improved retrieval quality"""
    
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