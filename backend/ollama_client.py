"""Ollama REST API client for embeddings and generation."""

import requests
from typing import List
import json
import time


class OllamaClient:
    """Client for Ollama REST API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", retries: int = 5, retry_backoff: float = 0.5):
        self.base_url = base_url
        self.embedding_model = "nomic-embed-text-v2-moe"
        self.generation_model = "qwen3.5:latest"
        self.retries = max(1, int(retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        # Reuse HTTP connections for performance
        self.session = requests.Session()

    def list_models(self) -> List[str]:
        """
        List model names downloaded in Ollama.

        Returns:
            List of model names
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to list Ollama models: {str(e)}")

    def has_model(self, model_name: str) -> bool:
        """
        Check whether a model exists in Ollama local models.

        Args:
            model_name: Target model name (e.g. nomic-embed-text-v2-moe)

        Returns:
            True if model is available locally
        """
        available_models = self.list_models()

        # Accept both tagged and untagged names.
        # Example: `nomic-embed-text-v2-moe` should match
        # `nomic-embed-text-v2-moe:latest` from /api/tags.
        requested_base = model_name.split(":", 1)[0]
        for available in available_models:
            if available == model_name:
                return True
            available_base = available.split(":", 1)[0]
            if available_base == requested_base:
                return True
        return False
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text using Ollama.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                return data.get("embedding", [])
            except requests.exceptions.RequestException as e:
                last_err = e
                # Backoff before next try, unless it's the last attempt
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * (2 ** (attempt - 1)))
                else:
                    break
        raise ConnectionError(f"Failed to get embedding from Ollama after {self.retries} attempts: {str(last_err)}")
    
    def generate(self, prompt: str, model: str = "qwen3.5:latest") -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            model: Model name (default: qwen3.5:latest)
            
        Returns:
            Generated text
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=600
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to generate text from Ollama: {str(e)}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available and running.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
