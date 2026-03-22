from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import os

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.agent import SimpleRAGAgent
import src.llm_client as llm_client

app = FastAPI(title="PDF Document Q&A API", version="1.0.0")

# Allow CORS for local frontend (Vite's default port is 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances for the backend
processor = DocumentProcessor()
vector_store = None
rag_agent = None

class ChatRequest(BaseModel):
    query: str
    category_filter: str | None = None

@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    global vector_store, rag_agent
    try:
        # Avoid re-initialization if already set
        if vector_store is None:
            vector_store = VectorStore(collection_name="fastapi_docs")
            rag_agent = SimpleRAGAgent(llm_client, vector_store)
            print("VectorStore and RAGAgent initialized.")
    except Exception as e:
        print(f"Failed to initialize on startup: {e}")

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("Other")
):
    """Handles PDF file upload and processing."""
    global vector_store, rag_agent
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        # Ensure vector store is initialized (in case startup failed)
        if vector_store is None:
             vector_store = VectorStore(collection_name="fastapi_docs")
             rag_agent = SimpleRAGAgent(llm_client, vector_store)
             
        content = await file.read()
        
        # Process the document
        result = processor.process_document(content, category, file.filename)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=f"Processing failed: {result.get('error')}")
            
        chunks = result['chunks']
        chunk_metadata = result.get('chunk_metadata', [])
        
        # Prepare metadata for vector store
        vector_metadata = []
        for i, chunk_meta in enumerate(chunk_metadata):
            meta = {
                'category': category,
                'document_id': file.filename,
                'section': chunk_meta.get('section', f'Section {i+1}'),
                'chunk_index': i,
                'chunk_type': chunk_meta.get('chunk_type', 'content'),
                'confidence': chunk_meta.get('confidence', 0.5)
            }
            vector_metadata.append(meta)
            
        # Store in vector store
        store_success, store_message, ids = vector_store.add_documents(
            chunks, vector_metadata, None
        )
        
        if not store_success:
            raise HTTPException(status_code=500, detail=store_message)
            
        return {
            "success": True,
            "message": "Document processed and stored successfully",
            "metadata": result.get('metadata'),
            "chunks_stored": len(ids)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handles chat queries and streams the AI response."""
    global rag_agent
    
    if not rag_agent:
        raise HTTPException(status_code=400, detail="RAG system not initialized. Upload a document first.")
        
    def stream_generator():
        try:
            # Stream tokens from the agent
            for token in rag_agent.process_query_stream(
                query=request.query,
                category_filter=request.category_filter,
                top_k=5
            ):
                yield token
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Deletes a document chunks from vector store and clears session."""
    global vector_store, rag_agent
    try:
        if vector_store:
            success, message, count = vector_store.delete_documents_by_metadata(
                {"document_id": filename}
            )
            
            if rag_agent:
                rag_agent.clear_session()
            
            if success:
                return {"success": True, "message": f"Deleted {count} chunks for {filename}"}
            else:
                raise HTTPException(status_code=500, detail=message)
        else:
            return {"success": True, "message": "No vector store to clean"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
