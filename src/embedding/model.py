"""
Embedding model using sentence-transformers.
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from loguru import logger

class EmbeddingModel:
    """
    Wrapper for sentence-transformers model to generate embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        logger.info(f"Loading embedding model {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        # Encode the text (SentenceTransformer already normalizes)
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.

        Args:
            texts: List of text strings to encode
            
        Returns:
            Array of embeddings, shape (len(texts), embedding_dim)
        """
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            logger.info(f"Successfully created embeddings of shape {embeddings.shape}")
            # Log a sample norm to verify normalization
            logger.info(f"Sample embedding norm: {np.linalg.norm(embeddings[0]):.6f}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise

    def split_text(self, text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
        """
        Split text into chunks for embedding.

        Args:
            text (str): Text to split.
            chunk_size (int): Maximum number of characters per chunk.
            overlap (int): Number of characters to overlap between chunks.

        Returns:
            List[str]: List of text chunks.
        """
        try:
            # Split text into sentences (simple approach)
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if current_length + sentence_length <= chunk_size:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    # Add current chunk if not empty
                    if current_chunk:
                        chunks.append('. '.join(current_chunk) + '.')
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > 0:
                        # Take last few sentences as overlap
                        overlap_text_length = 0
                        overlap_sentences = []
                        for s in reversed(current_chunk):
                            if overlap_text_length + len(s) <= overlap:
                                overlap_sentences.insert(0, s)
                                overlap_text_length += len(s)
                            else:
                                break
                        current_chunk = overlap_sentences
                    else:
                        current_chunk = []
                    
                    current_chunk.append(sentence)
                    current_length = sum(len(s) for s in current_chunk)
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
            
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to split text: {e}")
            raise 