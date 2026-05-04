"""PDF processing module for RAG system."""

from pypdf import PdfReader
from typing import List, Dict, Any
import math


class PDFChunk:
    """Represents a chunk of text extracted from PDF."""
    
    def __init__(self, text: str, page_num: int, chunk_id: int):
        self.text = text
        # Representative page number (kept for backward compatibility)
        self.page_num = page_num
        # All page numbers covered by this chunk (initialized with representative)
        self.page_nums: List[int] = [page_num]
        self.chunk_id = chunk_id
        self.token_count = self._estimate_tokens()
    
    def _estimate_tokens(self) -> int:
        """Estimate token count using simplified method: char_count / 4."""
        return max(1, len(self.text) // 4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "text": self.text,
            "page_num": self.page_num,
            "page_nums": self.page_nums,
            "chunk_id": self.chunk_id,
            "token_count": self.token_count,
        }


def extract_pdf_text(pdf_path: str) -> Dict[int, str]:
    """
    Extract text from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary mapping page numbers to text content
    """
    text_by_page = {}
    
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                text_by_page[page_num] = text
    except Exception as e:
        raise ValueError(f"Error reading PDF: {str(e)}")
    
    if not text_by_page:
        raise ValueError("No text extracted from PDF")
    
    return text_by_page


def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    """
    Split text into chunks based on token count (simplified: char_count / 4).
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    max_chars = max_tokens * 4
    chunks = []
    
    # Split by sentences first to preserve readability
    sentences = text.replace("。", "。\n").replace(".", ".\n").split("\n")
    
    current_chunk = ""
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        # If adding this sentence exceeds max chars, save current chunk and start new one
        if len(current_chunk) + len(sentence) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence
    
    # Add remaining text
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def process_pdf(pdf_path: str, max_tokens_per_chunk: int = 100) -> List[PDFChunk]:
    """
    Process PDF file globally: extract text and chunk across pages.
    
    Args:
        pdf_path: Path to PDF file
        max_tokens_per_chunk: Maximum tokens per chunk
        
    Returns:
        List of PDFChunk objects with `page_nums` tracking all pages included
    """
    text_by_page = extract_pdf_text(pdf_path)
    max_chars = max_tokens_per_chunk * 4

    # Build sentence list across all pages with page tracking
    sentences_with_pages: List[tuple[str, int]] = []
    for page_num in sorted(text_by_page.keys()):
        page_text = text_by_page[page_num]
        # Sentence split consistent with chunk_text()
        page_sentences = page_text.replace("。", "。\n").replace(".", ".\n").split("\n")
        for s in page_sentences:
            if s.strip():
                sentences_with_pages.append((s, page_num))

    chunks: List[PDFChunk] = []
    global_chunk_id = 0
    current_text = ""
    current_pages: set[int] = set()

    for sentence, page_num in sentences_with_pages:
        if len(current_text) + len(sentence) > max_chars and current_text:
            # finalize current chunk
            rep_page = min(current_pages) if current_pages else page_num
            chunk = PDFChunk(current_text.strip(), rep_page, global_chunk_id)
            chunk.page_nums = sorted(current_pages) if current_pages else [rep_page]
            chunks.append(chunk)
            global_chunk_id += 1

            # start new chunk with current sentence
            current_text = sentence
            current_pages = {page_num}
        else:
            current_text += sentence
            current_pages.add(page_num)

    # Add remaining text
    if current_text.strip():
        rep_page = min(current_pages) if current_pages else (min(text_by_page.keys()) if text_by_page else 1)
        chunk = PDFChunk(current_text.strip(), rep_page, global_chunk_id)
        chunk.page_nums = sorted(current_pages) if current_pages else [rep_page]
        chunks.append(chunk)

    return chunks
