
# End-to-End RAG System with MongoDB and Pinecone

This project implements a complete Retrieval-Augmented Generation (RAG) system using MongoDB for document storage and Pinecone for vector search. It's designed to be run from the command line and can be easily extended.

## Features

- **Document Ingestion:** Add documents from `.txt` and `.pdf` files.
- **Chunking and Embedding:** Automatically splits documents into manageable chunks and generates embeddings.
- **Vector Search:** Uses Pinecone for efficient similarity search.
- **Answer Generation:** Leverages Google's Gemini Pro to generate answers based on retrieved context.
- **MongoDB Integration:** Stores document metadata and chunks for persistence.

## Project Structure

```
rag_project/
├── src/
│   ├── core.py         # RAGSystem class
│   ├── features.py     # AdvancedRAGFeatures class
│   └── utils.py        # Helper functions
├── main.py             # Main entry point for the interactive demo
├── requirements.txt    # Project dependencies
├── .env.example        # Example environment file
├── .gitignore          # Git ignore file
└── README.md           # Project documentation
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd rag_project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment:**
    - Rename `.env.example` to `.env`.
    - Add your API keys and configuration to the `.env` file.

4.  **Run the interactive demo:**
    ```bash
    python main.py
    ```
