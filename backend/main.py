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


class RetrievedChunk(BaseModel):
    chunk_id: int
    text: str
    page_num: int
    score: float


class RetrievedItem(BaseModel):
    item: str
    retrieved: List[RetrievedChunk]


class RetrieveResponse(BaseModel):
    items: List[RetrievedItem]


class GenerateRequest(BaseModel):
    items: List[RetrievedItem]


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


@app.post("/query/retrieve")
async def retrieve_query(request: QueryRequest) -> RetrieveResponse:
    """Retrieve relevant chunks for each item in the query."""
    global current_rag_engine

    if current_rag_engine is None or not current_rag_engine.chunks:
        raise HTTPException(status_code=400, detail="No PDF uploaded. Please upload a PDF first.")

    try:
        raw_items = current_rag_engine.retrieve_items(request.query, top_k=request.top_k)
        items = [
            RetrievedItem(
                item=entry["item"],
                retrieved=[RetrievedChunk(**chunk) for chunk in entry["retrieved"]]
            )
            for entry in raw_items
        ]
        return RetrieveResponse(items=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieve failed: {str(e)}")


@app.post("/query/generate")
async def generate_query(request: GenerateRequest) -> QueryResponse:
    """Generate answers from pre-retrieved chunks."""
    global current_rag_engine

    if current_rag_engine is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded. Please upload a PDF first.")

    try:
        items_with_retrieved = [
            {"item": entry.item, "retrieved": [c.model_dump() for c in entry.retrieved]}
            for entry in request.items
        ]
        result = current_rag_engine.generate_answers(items_with_retrieved)

        items_results = []
        for item_result in result["items"]:
            sources = [SourceInfo(**source) for source in item_result["sources"]]
            items_results.append(ItemResult(
                item=item_result["item"],
                answer=item_result["answer"],
                sources=sources
            ))

        return QueryResponse(items=items_results, total_items=result["total_items"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generate failed: {str(e)}")


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
        items_with_retrieved = current_rag_engine.retrieve_items(request.query, top_k=request.top_k)
        result = current_rag_engine.generate_answers(items_with_retrieved)
        
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
    import sys
    import threading
    import webbrowser
    import uvicorn

    # プロジェクトルートを sys.path に追加（backend パッケージのインポートを解決）
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    host = "localhost"
    port = 8000
    url = f"http://{host}:{port}"

    print()
    print("========================================")
    print("  On-the-fly RAG System")
    print("========================================")
    print(f"  URL: {url}")
    print("  Ctrl+C で停止")
    print("========================================")
    print()

    def _open_browser():
        import time
        time.sleep(2)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)
