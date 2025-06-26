
from typing import List, Dict, Any
from .core import RAGSystem

class AdvancedRAGFeatures:
    """
    Additional features for the RAG system
    """
    
    def __init__(self, rag_system: RAGSystem):
        self.rag = rag_system
    
    def batch_add_documents(self, file_paths: List[str], metadata_list: List[Dict] = None):
        """
        Add multiple documents in batch
        """
        if metadata_list is None:
            metadata_list = [{}] * len(file_paths)
        
        doc_ids = []
        for file_path, metadata in zip(file_paths, metadata_list):
            doc_id = self.rag.add_document(file_path, metadata)
            if doc_id:
                doc_ids.append(doc_id)
        
        return doc_ids
    
    def semantic_search_with_filters(self, query: str, filters: Dict[str, Any], top_k: int = 5):
        """
        Search with metadata filters
        """
        # This would require more complex Pinecone querying with filters
        # For now, we'll do post-filtering
        results = self.rag.search_similar(query, top_k * 2)  # Get more results to filter
        
        filtered_results = []
        for result in results:
            match = True
            for key, value in filters.items():
                if result['metadata'].get(key) != value:
                    match = False
                    break
            if match:
                filtered_results.append(result)
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results[:top_k]
    
    def get_document_summary(self, doc_id: str) -> Dict[str, Any]:
        """
        Get summary of a document
        """
        doc = self.rag.documents_collection.find_one({'doc_id': doc_id})
        if not doc:
            return None
        
        chunks = list(self.rag.chunks_collection.find({'doc_id': doc_id}))
        
        return {
            'doc_id': doc_id,
            'file_path': doc.get('file_path', ''),
            'chunk_count': len(chunks),
            'total_characters': len(doc.get('content', '')),
            'created_at': doc.get('created_at'),
            'metadata': doc.get('metadata', {})
        }
