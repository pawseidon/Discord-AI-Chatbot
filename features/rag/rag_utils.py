"""
Retrieval-Augmented Generation (RAG) utilities for Discord AI Chatbot.

This module provides utilities for RAG capabilities including 
document retrieval, context enhancement, and knowledge integration.
"""

import logging
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple, Union

# Set up logging
logger = logging.getLogger("rag_utils")

class RAGProcessor:
    """
    RAG Processor for enhancing AI responses with retrieved knowledge
    """
    
    def __init__(self, 
               knowledge_path: str = "knowledge",
               vector_db_path: str = "vector_db",
               chunk_size: int = 512,
               chunk_overlap: int = 50):
        """
        Initialize the RAG processor
        
        Args:
            knowledge_path: Path to knowledge documents
            vector_db_path: Path to vector database
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.knowledge_path = knowledge_path
        self.vector_db_path = vector_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create directories if they don't exist
        os.makedirs(knowledge_path, exist_ok=True)
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Initialize embeddings if available
        try:
            # Import optional dependencies
            from sentence_transformers import SentenceTransformer
            self.embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings_available = True
        except ImportError:
            logger.warning("SentenceTransformer not installed. Semantic search will be limited.")
            self.embeddings_model = None
            self.embeddings_available = False
    
    async def enhance_prompt(self, 
                          query: str, 
                          context: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance a prompt with retrieved knowledge
        
        Args:
            query: User query
            context: Optional context information
            
        Returns:
            Tuple of (enhanced prompt, retrieval info)
        """
        # Get relevant documents
        docs, retrieval_info = await self.retrieve_relevant_documents(query, context)
        
        # Format documents into context string
        if docs:
            context_str = "\n\n".join([f"[Document {i+1}]: {doc['content']}" for i, doc in enumerate(docs)])
            
            # Create enhanced prompt
            enhanced_prompt = (
                f"I'll answer the following query: {query}\n\n"
                f"Here's some relevant information to consider:\n{context_str}\n\n"
                f"Based on this information and my knowledge, I'll provide a comprehensive answer."
            )
            
            return enhanced_prompt, retrieval_info
        else:
            # No relevant documents found
            return query, retrieval_info
    
    async def retrieve_relevant_documents(self, 
                                       query: str, 
                                       context: Optional[Dict[str, Any]] = None, 
                                       top_k: int = 3) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query
            context: Optional context information
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (list of relevant documents, retrieval info)
        """
        retrieval_info = {
            "query": query,
            "method": "keyword",
            "documents_found": 0
        }
        
        # Simple keyword-based retrieval
        results = await self.keyword_search(query, top_k)
        
        # Use semantic search if available
        if self.embeddings_available:
            semantic_results = await self.semantic_search(query, top_k)
            
            # Combine results (prioritize semantic)
            combined_results = semantic_results.copy()
            for doc in results:
                if doc not in combined_results:
                    combined_results.append(doc)
            
            results = combined_results[:top_k]
            retrieval_info["method"] = "semantic+keyword"
        
        retrieval_info["documents_found"] = len(results)
        return results, retrieval_info
    
    async def keyword_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search
        
        Args:
            query: Search query
            top_k: Maximum number of results
            
        Returns:
            List of matching documents
        """
        results = []
        
        # Simple implementation for now
        # Convert query to keywords
        keywords = re.findall(r'\b\w+\b', query.lower())
        
        # Score documents based on keyword matches
        scores = {}
        
        for filename in os.listdir(self.knowledge_path):
            if filename.endswith('.txt') or filename.endswith('.md'):
                file_path = os.path.join(self.knowledge_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Count keyword matches
                    score = 0
                    for keyword in keywords:
                        score += content.lower().count(keyword)
                    
                    if score > 0:
                        scores[file_path] = score
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
        
        # Sort by score
        sorted_paths = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Get top results
        for path in sorted_paths[:top_k]:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                results.append({
                    "source": os.path.basename(path),
                    "content": content,
                    "score": scores[path]
                })
            except Exception as e:
                logger.error(f"Error reading file {path}: {e}")
        
        return results
    
    async def semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings
        
        Args:
            query: Search query
            top_k: Maximum number of results
            
        Returns:
            List of matching documents
        """
        if not self.embeddings_available:
            return []
        
        results = []
        try:
            # Encode query
            query_embedding = self.embeddings_model.encode(query)
            
            # Simple vector search implementation
            # In a real system, you'd use a vector database like FAISS
            embeddings = {}
            
            for filename in os.listdir(self.knowledge_path):
                if filename.endswith('.txt') or filename.endswith('.md'):
                    file_path = os.path.join(self.knowledge_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Get embedding for the content
                        content_embedding = self.embeddings_model.encode(content)
                        embeddings[file_path] = content_embedding
                    except Exception as e:
                        logger.error(f"Error reading or embedding file {file_path}: {e}")
            
            # Calculate similarities
            import numpy as np
            similarities = {}
            
            for path, embedding in embeddings.items():
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities[path] = similarity
            
            # Sort by similarity
            sorted_paths = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)
            
            # Get top results
            for path in sorted_paths[:top_k]:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    results.append({
                        "source": os.path.basename(path),
                        "content": content,
                        "score": float(similarities[path])
                    })
                except Exception as e:
                    logger.error(f"Error reading file {path}: {e}")
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
        
        return results
    
    async def add_document(self, content: str, source: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a document to the knowledge base
        
        Args:
            content: Document content
            source: Document source identifier
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        try:
            # Save document
            file_path = os.path.join(self.knowledge_path, f"{source}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Save metadata if provided
            if metadata:
                metadata_path = os.path.join(self.knowledge_path, f"{source}.meta.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f)
            
            logger.info(f"Added document: {source}")
            return True
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    async def get_document(self, source: str) -> Optional[Dict[str, Any]]:
        """
        Get a document from the knowledge base
        
        Args:
            source: Document source identifier
            
        Returns:
            Document with metadata or None if not found
        """
        file_path = os.path.join(self.knowledge_path, f"{source}.txt")
        metadata_path = os.path.join(self.knowledge_path, f"{source}.meta.json")
        
        try:
            if not os.path.exists(file_path):
                return None
            
            # Read content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Read metadata if available
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            return {
                "source": source,
                "content": content,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error retrieving document {source}: {e}")
            return None

# Global instance for singleton pattern
_rag_processor = None

async def get_rag_processor(**kwargs) -> RAGProcessor:
    """
    Get or create the global RAG processor instance
    
    Args:
        **kwargs: Arguments to pass to RAGProcessor constructor
        
    Returns:
        RAGProcessor instance
    """
    global _rag_processor
    
    if _rag_processor is None:
        _rag_processor = RAGProcessor(**kwargs)
        
    return _rag_processor 