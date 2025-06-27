import os
import json
import uuid
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Core libraries
from pymongo import MongoClient
import pinecone
from pinecone import Pinecone, ServerlessSpec

# For text processing and embeddings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel
import torch

# For document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tiktoken

# Utilities
import warnings
warnings.filterwarnings('ignore')

class RAGSystem:
    """
    Complete RAG system integrating MongoDB and Pinecone
    """
    
    def __init__(self, 
                 mongo_uri: str,
                 mongo_db_name: str,
                 pinecone_api_key: str,
                 pinecone_index_name: str,
                 gemini_api_key: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        # Initialize MongoDB
        self.mongo_client = MongoClient(mongo_uri)
        self.mongo_db = self.mongo_client[mongo_db_name]
        self.documents_collection = self.mongo_db['documents']
        self.chunks_collection = self.mongo_db['chunks']
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = pinecone_index_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Gemini if provided
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            self.has_llm = True
        else:
            self.has_llm = False
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Setup Pinecone index
        self._setup_pinecone_index()
        
    def _setup_pinecone_index(self):
        """Setup Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                # Create index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                print(f"Created Pinecone index: {self.index_name}")
            
            # Connect to index
            self.pinecone_index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            print(f"Error setting up Pinecone index: {e}")
    
    def add_document(self, file_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a document to the RAG system
        """
        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Load document based on file type
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                content = "\n".join([page.page_content for page in pages])
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                raise ValueError("Unsupported file type. Use .pdf or .txt files.")
            
            # Store document in MongoDB
            doc_metadata = {
                'doc_id': doc_id,
                'file_path': file_path,
                'content': content,
                'created_at': pd.Timestamp.now(),
                'metadata': metadata or {}
            }
            
            self.documents_collection.insert_one(doc_metadata)
            
            # Process and store chunks
            self._process_and_store_chunks(doc_id, content, metadata)
            
            print(f"Successfully added document: {doc_id}")
            return doc_id
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return None
    
    def _process_and_store_chunks(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """
        Process document into chunks and store in both MongoDB and Pinecone
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Process each chunk
        vectors_to_upsert = []
        mongo_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(chunk).tolist()
            
            # Prepare for Pinecone
            vectors_to_upsert.append({
                'id': chunk_id,
                'values': embedding,
                'metadata': {
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'text': chunk,
                    **(metadata or {})
                }
            })
            
            # Prepare for MongoDB
            mongo_chunks.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'chunk_index': i,
                'text': chunk,
                'embedding': embedding,
                'metadata': metadata or {},
                'created_at': pd.Timestamp.now()
            })
        
        # Batch upsert to Pinecone
        if vectors_to_upsert:
            self.pinecone_index.upsert(vectors=vectors_to_upsert)
        
        # Batch insert to MongoDB
        if mongo_chunks:
            self.chunks_collection.insert_many(mongo_chunks)
        
        print(f"Processed {len(chunks)} chunks for document {doc_id}")
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using Pinecone
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in Pinecone
            search_results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            results = []
            for match in search_results.matches:
                results.append({
                    'chunk_id': match.id,
                    'score': match.score,
                    'text': match.metadata.get('text', ''),
                    'doc_id': match.metadata.get('doc_id', ''),
                    'metadata': {k: v for k, v in match.metadata.items() 
                               if k not in ['text', 'doc_id']}
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate answer using Google Gemini with retrieved context
        """
        if not self.has_llm:
            return "Gemini API key not provided. Cannot generate answers."
        
        # Prepare context
        context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Create prompt
        prompt = f'''
Based on the following context, answer the question clearly and concisely. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Please provide a helpful and accurate answer based on the context provided:
        '''
        
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=500,
                top_p=0.8,
                top_k=40
            )
            
            # Generate response using Gemini
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error generating answer with Gemini: {e}")
            return "Error generating answer with Gemini."
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve and generate
        """
        # Search for relevant chunks
        retrieved_chunks = self.search_similar(question, top_k)
        
        # Generate answer
        answer = self.generate_answer(question, retrieved_chunks)
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_chunks': retrieved_chunks,
            'sources': [chunk['doc_id'] for chunk in retrieved_chunks]
        }
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored documents
        """
        doc_count = self.documents_collection.count_documents({})
        chunk_count = self.chunks_collection.count_documents({})
        
        # Get Pinecone stats
        index_stats = self.pinecone_index.describe_index_stats()
        
        return {
            'mongodb_documents': doc_count,
            'mongodb_chunks': chunk_count,
            'pinecone_vectors': index_stats.total_vector_count,
            'index_dimension': index_stats.dimension
        }
    
    def delete_document(self, doc_id: str):
        """
        Delete a document and all its chunks from both MongoDB and Pinecone
        """
        try:
            # Get all chunk IDs for this document
            chunks = list(self.chunks_collection.find({'doc_id': doc_id}, {'chunk_id': 1}))
            chunk_ids = [chunk['chunk_id'] for chunk in chunks]
            
            # Delete from MongoDB
            self.documents_collection.delete_one({'doc_id': doc_id})
            self.chunks_collection.delete_many({'doc_id': doc_id})
            
            # Delete from Pinecone
            if chunk_ids:
                self.pinecone_index.delete(ids=chunk_ids)
            
            print(f"Deleted document {doc_id} and {len(chunk_ids)} chunks")
            
        except Exception as e:
            print(f"Error deleting document: {e}")
