"""RAG engine for retrieval and generation."""

from typing import List, Dict, Any, Tuple
from backend.pdf_processor import PDFChunk
from backend.ollama_client import OllamaClient
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


class RAGEngine:
    """RAG engine for retrieval and generation with source citations."""
    
    def __init__(self, ollama_client: OllamaClient = None):
        self.ollama = ollama_client or OllamaClient()
        self.chunks: List[PDFChunk] = []
        self.embeddings: List[List[float]] = []
        self.embedding_dim: int | None = None
        self.embed_workers: int = 4
    
    def _extract_items_from_query(self, query: str) -> List[str]:
        """
        Extract items from query text.
        Splits query by comma (,).
        
        Args:
            query: Query text with items separated by commas
            
        Returns:
            List of extracted items
        """
        # Split by comma
        items = [item.strip() for item in query.split(',')]
        # Remove empty items
        return [item for item in items if item]
    
    def add_chunks(self, chunks: List[PDFChunk]) -> None:
        """
        Add PDF chunks and compute embeddings.
        
        Args:
            chunks: List of PDFChunk objects
        """
        self.chunks = chunks
        self.embeddings = []
        
        total = len(chunks)
        if total > 0:
            print(f"Starting embedding for {total} chunks with {self.embed_workers} workers...")
        
        # Define worker to embed a single chunk
        def _embed_worker(idx: int, chunk: PDFChunk) -> Tuple[int, List[float]]:
            try:
                pages_info = getattr(chunk, 'page_nums', None)
                if pages_info:
                    print(f"Embedding chunk {idx}/{total} (id={chunk.chunk_id}, pages={pages_info})")
                else:
                    print(f"Embedding chunk {idx}/{total} (id={chunk.chunk_id}, page={chunk.page_num})")
            except Exception:
                print(f"Embedding chunk {idx}/{total} (id={chunk.chunk_id})")
            try:
                emb = self.ollama.get_embedding(chunk.text)
                return (idx - 1, emb)
            except Exception as e:
                print(f"Warning: Failed to embed chunk {chunk.chunk_id}: {str(e)}")
                return (idx - 1, [])
        
        # Run workers in parallel
        results: List[Tuple[int, List[float]]] = []
        with ThreadPoolExecutor(max_workers=self.embed_workers) as executor:
            futures = [executor.submit(_embed_worker, idx, chunk) for idx, chunk in enumerate(chunks, start=1)]
            for future in as_completed(futures):
                results.append(future.result())
        
        # Restore original order
        results.sort(key=lambda x: x[0])
        self.embeddings = [emb for _, emb in results]
        
        # Track embedding dimension from first success
        if self.embedding_dim is None:
            for emb in self.embeddings:
                if emb:
                    self.embedding_dim = len(emb)
                    break
        
        # Normalize lengths / fill zeros for failures
        if self.embedding_dim and self.embedding_dim > 0:
            normalized: List[List[float]] = []
            for emb in self.embeddings:
                if not emb or len(emb) != self.embedding_dim:
                    normalized.append([0.0] * self.embedding_dim)
                else:
                    normalized.append(emb)
            self.embeddings = normalized
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Handle size mismatch or empty vectors
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        arr1 = np.array(vec1, dtype=float)
        arr2 = np.array(vec2, dtype=float)
        
        # Handle zero vectors
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(arr1, arr2) / (norm1 * norm2))
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for query.
        
        Args:
            query: User query
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of retrieved chunks with relevance scores
        """
        if not self.chunks or not self.embeddings:
            return []
        
        try:
            query_embedding = self.ollama.get_embedding(query)
            # If dimension not yet known (e.g., all chunk embeddings failed), set it now
            if self.embedding_dim is None:
                self.embedding_dim = len(query_embedding)
            # Normalize stored embeddings lengths (replace empties with zeros of correct dim)
            if self.embedding_dim and self.embedding_dim > 0:
                normalized_embeddings: List[List[float]] = []
                for emb in self.embeddings:
                    if not emb or len(emb) != self.embedding_dim:
                        normalized_embeddings.append([0.0] * self.embedding_dim)
                    else:
                        normalized_embeddings.append(emb)
                self.embeddings = normalized_embeddings
        except Exception as e:
            print(f"Warning: Failed to embed query: {str(e)}")
            return []
        
        # Compute similarities
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            score = self._cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((i, score))
        
        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:top_k]]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = similarities[[s[0] for s in similarities].index(idx)][1]
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "page_num": chunk.page_num,
                "score": score
            })
        
        return results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer based on query and retrieved chunks.
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved chunks from retrieve()
            
        Returns:
            Dictionary with answer and sources
        """
        if not retrieved_chunks:
            return {
                "answer": "申し訳ありません。関連する情報が見つかりませんでした。",
                "sources": []
            }
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[ソース {i} (ページ {chunk['page_num']})]:\n{chunk['text']}")
            sources.append({
                "page": chunk['page_num'],
                "text": chunk['text'],
                "relevance_score": round(chunk['score'], 3)
            })
        
        context = "\n\n".join(context_parts)
        
        # Build prompt for qwen3.5
        prompt = f"""以下の情報から、{query}に関連する部分だけを抽出し、簡潔に整理して回答してください。

情報: {context}

回答:"""
        
        try:
            answer = self.ollama.generate(prompt, model="qwen3.5:latest")
            answer = answer.strip()
        except Exception as e:
            answer = f"申し訳ありません。回答生成に失敗しました: {str(e)}"
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def process_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        End-to-end query processing with multi-item support.
        
        Args:
            query: User query (may contain multiple items with bullet points)
            top_k: Number of chunks to retrieve per item
            
        Returns:
            Dictionary with items list containing answers and sources for each item
        """
        # Extract items from query
        items = self._extract_items_from_query(query)
        
        # Process each item separately
        results_items = []
        
        for item in items:
            # Skip if item is too short or generic
            if len(item.strip()) < 2:
                continue
            
            # Retrieve relevant chunks for this item
            retrieved = self.retrieve(item, top_k=top_k)
            
            # Generate answer for this item
            answer_data = self.generate_answer(item, retrieved)
            
            results_items.append({
                "item": item.strip(),
                "answer": answer_data["answer"],
                "sources": answer_data["sources"]
            })
        
        # If no items were extracted or processed, treat as single query
        if not results_items:
            retrieved = self.retrieve(query, top_k=top_k)
            answer_data = self.generate_answer(query, retrieved)
            results_items.append({
                "item": query.strip(),
                "answer": answer_data["answer"],
                "sources": answer_data["sources"]
            })
        
        return {
            "items": results_items,
            "total_items": len(results_items)
        }
