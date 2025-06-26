from src.core import RAGSystem
from src.features import AdvancedRAGFeatures
from src.utils import setup_environment, create_sample_documents, display_search_results

def interactive_rag_demo():
    """
    Interactive demo for the RAG system
    """
    print("Setting up RAG system...")
    config = setup_environment()
    
    rag = RAGSystem(
        mongo_uri=config['MONGO_URI'],
        mongo_db_name=config['MONGO_DB_NAME'],
        pinecone_api_key=config['PINECONE_API_KEY'],
        pinecone_index_name=config['PINECONE_INDEX_NAME'],
        gemini_api_key=config['GEMINI_API_KEY']
    )
    
    # Create and add sample documents
    print("Creating sample documents...")
    sample_files = create_sample_documents()
    
    print("Adding documents to RAG system...")
    advanced_rag = AdvancedRAGFeatures(rag)
    doc_ids = advanced_rag.batch_add_documents(sample_files)
    
    print(f"Added {len(doc_ids)} documents successfully!")
    
    # Show stats
    stats = rag.get_document_stats()
    print(f"System Stats: {stats}")
    
    return rag, advanced_rag

if __name__ == "__main__":
    rag, advanced_rag = interactive_rag_demo()
    
    while True:
        print("\n--- RAG System Interactive Prompt ---")
        query = input("Enter your query (or type 'exit' to quit): ")
        
        if query.lower() == 'exit':
            break
        
        # Perform search and generate answer
        results = rag.query(query)
        
        print("\n--- Answer ---")
        print(results['answer'])
        
        print("\n--- Retrieved Chunks ---")
        display_search_results(results['retrieved_chunks'])
        
    print("Exiting RAG system.")
