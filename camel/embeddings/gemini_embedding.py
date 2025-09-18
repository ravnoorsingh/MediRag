"""Gemini embedding implementation for CAMEL framework."""
import os
from typing import Any, List
import google.generativeai as genai
from dotenv import load_dotenv

from camel.embeddings.base import BaseEmbedding

# Load environment variables
load_dotenv()

class GeminiEmbedding(BaseEmbedding[str]):
    """Google Gemini embedding model implementation.
    
    This class provides embedding capabilities using Google's Gemini embedding models
    through the Google AI API.
    
    Args:
        model_name (str): The name of the Gemini embedding model to use.
            Defaults to "models/embedding-001".
        api_key (str, optional): The Google API key. If not provided, will be
            loaded from environment variable GOOGLE_API_KEY.
    """
    
    def __init__(
        self,
        model_name: str = "models/embedding-001",
        api_key: str = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Gemini embedding-001 produces 768-dimensional vectors
        self._output_dim = 768
    
    def embed_list(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of text strings to embed.
            **kwargs (Any): Additional arguments (currently unused).
            
        Returns:
            List[List[float]]: List of embeddings, each as a list of floats.
        """
        embeddings = []
        
        for text in texts:
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"Warning: Failed to embed text: {e}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * self._output_dim)
                
        return embeddings
    
    def get_output_dim(self) -> int:
        """Return the output dimension of the embeddings.
        
        Returns:
            int: The dimensionality of the embedding vectors (768 for Gemini).
        """
        return self._output_dim