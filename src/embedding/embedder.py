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
        
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        try:
            logger.info(f"Encoding {len(texts)} texts")
            embeddings = self.model.encode(texts)
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise
            
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        try:
            logger.info(f"Creating embedding for text of length: {len(text)}")
            embedding = self.model.encode(text)
            logger.info(f"Created embedding with shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise
        
    def process_text(self, text: str, output_dir: str, title: str = None) -> Dict[str, Any]:
        """
        Process text content and create embeddings.
        
        Args:
            text: Text content to process
            output_dir: Directory to save the processed content
            title: Title of the content (optional)
            
        Returns:
            Dictionary containing:
            - content: The processed text content
            - metadata: Content metadata
            - filepath: Path to saved content
        """
        try:
            logger.info(f"Processing text of length: {len(text)}")
            logger.info(f"First 100 chars of input: {text[:100]}")
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            logger.info(f"Processed text length: {len(content)}")
            logger.info(f"First 100 chars of processed text: {content[:100]}")
            
            # Save content to file
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"content_{hash(text)}.txt")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Extract metadata
            metadata = {
                'title': title or 'Untitled',
                'content_id': str(hash(text))
            }
            
            return {
                'content': content,
                'metadata': metadata,
                'filepath': filepath
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise
            
    def split_text(self, text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
        """Split text into chunks of specified size with overlap.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        try:
            logger.info(f"Splitting text of length: {len(text)}")
            logger.info(f"First 100 chars of input: {text[:100]}")
            
            # Clean up the text first
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = ' '.join(text.split())  # Normalize whitespace
            
            # Split text into sentences using more robust sentence splitting
            sentences = []
            for sentence in text.split('.'):
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Only keep meaningful sentences
                    sentences.append(sentence)
            
            logger.info(f"Number of sentences: {len(sentences)}")
            if sentences:
                logger.info(f"First sentence: {sentences[0]}")
            
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
            
            logger.info(f"Created {len(chunks)} chunks")
            if chunks:
                logger.info(f"First chunk length: {len(chunks[0])}")
                logger.info(f"First chunk: {chunks[0][:200]}")
            else:
                logger.error("No chunks were created!")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            return [] 