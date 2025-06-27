# Jarvis Smart Retrieval - End-to-End RAG System

This project implements a complete Retrieval-Augmented Generation (RAG) system with a FastAPI backend and a Next.js frontend. It uses MongoDB for document storage and Pinecone for vector search, providing a user-friendly interface for document upload and interactive querying.

## Features

-   **Document Ingestion:** Upload PDF files through the web interface.
-   **Chunking and Embedding:** Automatically splits documents into manageable chunks and generates embeddings.
-   **Vector Search:** Uses Pinecone for efficient similarity search.
-   **Answer Generation:** Leverages Google's Gemini Flash model to generate answers based on retrieved context.
-   **MongoDB Integration:** Stores document metadata and chunks for persistence.
-   **FastAPI Backend:** Exposes RAG functionalities via a RESTful API.
-   **Next.js Frontend:** Provides a modern web interface for PDF upload and chatbot interaction.

## Project Structure

```
.
├── rag_project/
│   ├── src/
│   │   ├── core.py         # RAGSystem class (core RAG logic)
│   │   ├── features.py     # AdvancedRAGFeatures class
│   │   └── utils.py        # Helper functions (environment setup, etc.)
│   ├── api.py              # FastAPI application for the backend API
│   ├── main.py             # (Optional) Original CLI entry point for testing
│   ├── requirements.txt    # Python dependencies for the backend
│   ├── .env.example        # Example environment file
│   ├── .gitignore          # Git ignore file
│   └── README.md           # Project documentation
└── jarvis-frontend/        # Next.js frontend application
    ├── public/
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx        # Main UI component (PDF upload, chat)
    │   │   └── services/
    │   │       └── api.ts      # Frontend API service for backend communication
    │   └── ... (other Next.js files)
    ├── package.json        # Frontend dependencies
    ├── next.config.js
    ├── tailwind.config.ts
    └── ... (other Next.js config files)
```

## How to Run

To get the Jarvis Smart Retrieval system up and running, you need to start both the FastAPI backend and the Next.js frontend.

### 1. Backend Setup (rag_project)

1.  **Navigate to the backend directory:**
    ```bash
    cd rag_project
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you encounter issues with `python-multipart`, install it separately: `pip install python-multipart`)*

3.  **Set up your environment variables:**
    -   Rename `.env.example` to `.env`.
    -   Add your MongoDB URI, Pinecone API Key, Pinecone Index Name, and Gemini API Key to the `.env` file.

4.  **Start the FastAPI backend server:**
    ```bash
    uvicorn api:app --reload
    ```
    The backend will typically run on `http://127.0.0.1:8000` or `http://localhost:8000`. Keep this terminal open.

### 2. Frontend Setup (jarvis-frontend)

1.  **Open a new terminal window.**

2.  **Navigate to the frontend directory:**
    ```bash
    cd jarvis-frontend
    ```

3.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```
    *(Note: If `axios` is not found, install it: `npm install axios`)*

4.  **Start the Next.js development server:**
    ```bash
    npm run dev
    ```
    The frontend will typically run on `http://localhost:3000`. Keep this terminal open.

### 3. Access the Application

Once both the backend and frontend servers are running, open your web browser and go to:

```
http://localhost:3000
```

You can now upload PDF documents and interact with the Jarvis Smart Retrieval chatbot!