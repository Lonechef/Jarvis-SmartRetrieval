from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.core import RAGSystem
from src.utils import setup_environment
import os
import shutil

app = FastAPI(
    title="Jarvis Smart Retrieval API",
    description="API for document retrieval and question answering using RAG system.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your Next.js frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system globally
rag_system: RAGSystem = None

@app.on_event("startup")
async def startup_event():
    global rag_system
    print("Initializing RAG system...")
    config = setup_environment()
    rag_system = RAGSystem(
        mongo_uri=config['MONGO_URI'],
        mongo_db_name=config['MONGO_DB_NAME'],
        pinecone_api_key=config['PINECONE_API_KEY'],
        pinecone_index_name=config['PINECONE_INDEX_NAME'],
        gemini_api_key=config['GEMINI_API_KEY']
    )
    print("RAG system initialized.")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        # Create a temporary directory to save the uploaded file
        upload_dir = "uploaded_pdfs"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add document to RAG system
        doc_id = rag_system.add_document(file_path)

        # Clean up the temporary file
        os.remove(file_path)
        os.rmdir(upload_dir) # Remove directory if empty

        if doc_id:
            return JSONResponse(content={"message": "PDF uploaded and processed successfully", "doc_id": doc_id})
        else:
            raise HTTPException(status_code=500, detail="Failed to process PDF with RAG system.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during PDF upload: {str(e)}")

@app.post("/query/")
async def query_rag(question: str):
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized.")
    
    try:
        results = rag_system.query(question)
        return JSONResponse(content={"answer": results['answer'], "retrieved_chunks": results['retrieved_chunks']})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during query: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "rag_system_initialized": rag_system is not None}
