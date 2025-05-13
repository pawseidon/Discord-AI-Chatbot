import os
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class RAGSystem:
    def __init__(self, server_id: str, embedding_model="all-MiniLM-L6-v2"):
        self.server_id = server_id
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.storage_dir = f"bot_data/knowledge_bases/{server_id}"
        os.makedirs(self.storage_dir, exist_ok=True)
        self.vector_store = None
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Load existing vector store if it exists"""
        if os.path.exists(f"{self.storage_dir}/faiss_index"):
            try:
                self.vector_store = FAISS.load_local(
                    f"{self.storage_dir}/faiss_index", 
                    self.embeddings
                )
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.vector_store = None
    
    async def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the knowledge base"""
        if not metadata:
            metadata = [{"source": "user_added"} for _ in texts]
        
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadata)
        ]
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        # Save the updated vector store
        self.vector_store.save_local(f"{self.storage_dir}/faiss_index")
        return len(documents)
    
    async def query(self, query_text: str, k: int = 3) -> List[Document]:
        """Query the knowledge base"""
        if self.vector_store is None:
            return []
        
        results = self.vector_store.similarity_search(query_text, k=k)
        return results
    
    async def format_results_as_context(self, results: List[Document]) -> str:
        """Format retrieval results as context for the model"""
        if not results:
            return ""
        
        context = "Relevant information from knowledge base:\n\n"
        for i, doc in enumerate(results):
            context += f"[Document {i+1}]:\n{doc.page_content}\n\n"
        
        return context 