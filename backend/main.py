"""FastAPI backend for on-the-fly RAG system."""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import tempfile
from typing import Dict, Any, List

from backend.pdf_processor import process_pdf
from backend.ollama_client import OllamaClient
from backend.rag_engine import RAGEngine


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class SourceInfo(BaseModel):
    page: int
    text: str
    relevance_score: float


class ItemResult(BaseModel):
    item: str
    answer: str
    sources: List[SourceInfo]


class QueryResponse(BaseModel):
    items: List[ItemResult]
    total_items: int


class StatusResponse(BaseModel):
    ollama_available: bool
    message: str


# Initialize FastAPI app
app = FastAPI(title="On-the-fly RAG", version="1.0.0")

# Global RAG engine state (per session)
current_rag_engine: RAGEngine = None
current_pdf_filename: str = None


# Create static directory if it doesn't exist
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)


@app.on_event("startup")
async def startup_event():
    """Initialize on app startup."""
    global current_rag_engine
    ollama_client = OllamaClient()

    if not ollama_client.is_available():
        raise RuntimeError("Ollama is not running at localhost:11434")

    required_embedding_model = "nomic-embed-text-v2-moe"
    if not ollama_client.has_model(required_embedding_model):
        raise RuntimeError(
            f"Required embedding model is not downloaded: {required_embedding_model}. "
            f"Run: ollama pull {required_embedding_model}"
        )

    current_rag_engine = RAGEngine()


@app.get("/health")
async def health_check() -> StatusResponse:
    """Check if system is healthy."""
    ollama_client = OllamaClient()
    is_available = ollama_client.is_available()
    
    return StatusResponse(
        ollama_available=is_available,
        message="Ollama is running" if is_available else "Ollama is not running at localhost:11434"
    )


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), chunk_tokens: int = Form(100)) -> Dict[str, Any]:
    """
    Upload and process a PDF file.
    
    Args:
        file: PDF file to upload
        
    Returns:
        Status message with number of chunks created
    """
    global current_rag_engine, current_pdf_filename
    
    # Validate file type
    if file.content_type not in ["application/pdf", "application/x-pdf"] or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Process PDF with user-specified chunk size (tokens)
        chunks = process_pdf(tmp_path, max_tokens_per_chunk=chunk_tokens)
        
        # Initialize RAG engine with chunks
        current_rag_engine = RAGEngine()
        current_rag_engine.add_chunks(chunks)
        current_pdf_filename = file.filename
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "message": f"PDF '{file.filename}' processed successfully with {len(chunks)} chunks",
            "chunk_tokens": chunk_tokens
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/query")
async def query_pdf(request: QueryRequest) -> QueryResponse:
    """
    Query the uploaded PDF with support for multiple items.
    
    Args:
        request: Query request with query text and top_k
        
    Returns:
        Query response with items, each containing answer and sources
    """
    global current_rag_engine
    
    if current_rag_engine is None or not current_rag_engine.chunks:
        raise HTTPException(status_code=400, detail="No PDF uploaded. Please upload a PDF first.")
    
    try:
        result = current_rag_engine.process_query(request.query, top_k=request.top_k)
        
        # Convert to Pydantic models
        items_results = []
        for item_result in result["items"]:
            sources = [SourceInfo(**source) for source in item_result["sources"]]
            items_results.append(ItemResult(
                item=item_result["item"],
                answer=item_result["answer"],
                sources=sources
            ))
        
        return QueryResponse(
            items=items_results,
            total_items=result["total_items"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/")
async def root():
    """Serve the frontend index page."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {"message": "Frontend not found"}


# Mount static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
