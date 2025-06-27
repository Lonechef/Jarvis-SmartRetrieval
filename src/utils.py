
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

def setup_environment():
    """
    Setup function with environment variables
    """
    # You'll need to set these environment variables or replace with your actual values
    load_dotenv()
    config = {
        'MONGO_URI': os.getenv('MONGO_URI', 'mongodb://localhost:27017/'),
        'MONGO_DB_NAME': os.getenv('MONGO_DB_NAME', 'rag_database'),
        'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY', 'your-pinecone-api-key'),
        'PINECONE_INDEX_NAME': os.getenv('PINECONE_INDEX_NAME', 'rag-index'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY', 'your-gemini-api-key')
    }
    print("Environment setup complete with the following configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("Make sure to set the environment variables in your .env file or system.")
    return config

def create_sample_documents():
    """
    Create sample documents for testing
    """
    docs = {
        'ai_basics.txt': '''
        Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
        Machine Learning is a subset of AI that enables machines to learn from data.
        Deep Learning uses neural networks with multiple layers to process information.
        Natural Language Processing (NLP) helps computers understand and generate human language.
        Computer Vision allows machines to interpret and understand visual information.
        ''',
        
        'data_science.txt': '''
        Data Science combines statistics, programming, and domain expertise.
        Data preprocessing is crucial for cleaning and preparing data for analysis.
        Exploratory Data Analysis (EDA) helps understand data patterns and relationships.
        Feature engineering involves creating relevant features for machine learning models.
        Model validation ensures that models generalize well to new data.
        ''',
        
        'cloud_computing.txt': '''
        Cloud Computing provides on-demand access to computing resources over the internet.
        Infrastructure as a Service (IaaS) provides virtualized computing infrastructure.
        Platform as a Service (PaaS) offers development platforms in the cloud.
        Software as a Service (SaaS) delivers software applications over the internet.
        Serverless computing allows running code without managing servers.
        '''
    }
    
    for filename, content in docs.items():
        with open(filename, 'w') as f:
            f.write(content)
    
    print("Created sample documents:", list(docs.keys()))
    return list(docs.keys())

def display_search_results(results: List[Dict[str, Any]]):
    """
    Display search results in a nice format for Jupyter
    """
    import pandas as pd
    
    if not results:
        print("No results found.")
        return
    
    df_data = []
    for i, result in enumerate(results, 1):
        df_data.append({
            'Rank': i,
            'Score': f"{result['score']:.4f}",
            'Text Preview': result['text'][:100] + "..." if len(result['text']) > 100 else result['text'],
            'Doc ID': result['doc_id'][:8] + "..." if len(result['doc_id']) > 8 else result['doc_id']
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
