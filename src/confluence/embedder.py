"""
Generic text embedder for creating embeddings from content.
"""

import logging
import os
from typing import Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

class TextEmbedder:
    """Embedder for creating embeddings from text content."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the model to use for embeddings
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Initialized TextEmbedder with model {model_name} and dimension {self.dimension}")
        
    def process_text(self, text: str, output_dir: str) -> Dict[str, Any]:
        """
        Process text content and create embeddings.
        
        Args:
            text: Text content to process
            output_dir: Directory to save the processed content
            
        Returns:
            Dictionary containing:
            - content: The processed text content
            - metadata: Content metadata
            - filepath: Path to saved content
        """
        try:
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            # Save content to file
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"content_{hash(text)}.txt")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Extract metadata
            metadata = {
                'title': 'Text Content',
                'content_id': str(hash(text))
            }
            
            return {
                'content': content,
                'metadata': metadata,
                'filepath': filepath
            }
            
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            raise
            
    def split_text(self, text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
        """
        Split text into chunks for embedding.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        try:
            # Split text into sentences
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