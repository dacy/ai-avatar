"""
FAISS index implementation for vector similarity search.
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from loguru import logger

class FAISSIndex:
    """
    FAISS index for efficient vector similarity search.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize the FAISS index.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index to use (flat, ivf, hnsw)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metadata = []
        
        # Create the appropriate index based on the type
        if index_type == "flat":
            # Flat index with inner product (cosine similarity) for normalized vectors
            self.index = faiss.IndexFlatIP(dimension)
            logger.info(f"Created FAISS flat index with dimension {dimension}")
        elif index_type == "ivf":
            # IVF index - approximate search, faster but less accurate
            # We need some data to train this index, so we'll create it later
            self.index = None
            logger.info(f"Will create FAISS IVF index with dimension {dimension} after training")
        elif index_type == "hnsw":
            # HNSW index - hierarchical navigable small world graph, very fast and accurate
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors per node
            logger.info(f"Created FAISS HNSW index with dimension {dimension}")
        else:
            # Default to flat index
            logger.warning(f"Unknown index type '{index_type}', defaulting to flat index")
            self.index = faiss.IndexFlatIP(dimension)
            self.index_type = "flat"
    
    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Add embeddings and their metadata to the index.
        
        Args:
            embeddings: Numpy array of embedding vectors (already normalized by SentenceTransformer)
            metadata_list: List of metadata dictionaries for each embedding
        """
        if len(embeddings) == 0:
            logger.warning("No embeddings to add to the index")
            return
            
        if len(embeddings) != len(metadata_list):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) does not match number of metadata items ({len(metadata_list)})")
        
        # If we're using an IVF index and it's not initialized yet, create and train it
        if self.index_type == "ivf" and self.index is None:
            # Number of centroids - rule of thumb is sqrt(N) where N is the number of vectors
            n_centroids = min(int(np.sqrt(len(embeddings))), 256)
            # Create a quantizer
            quantizer = faiss.IndexFlatIP(self.dimension)
            # Create the IVF index
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_centroids)
            # Train the index
            logger.info(f"Training IVF index with {len(embeddings)} vectors and {n_centroids} centroids")
            self.index.train(embeddings)
            # Set the number of probes (how many centroids to visit during search)
            self.index.nprobe = min(n_centroids // 4, 64)  # Rule of thumb
        
        # Add the embeddings to the index
        logger.info(f"Adding {len(embeddings)} embeddings to the index")
        self.index.add(embeddings)
        
        # Store the metadata
        self.metadata.extend(metadata_list)
        logger.info(f"Added {len(metadata_list)} metadata items to the index")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_embedding: The query embedding vector (already normalized by SentenceTransformer)
            k: Number of results to return
            threshold: Minimum similarity score (0-1) to include in results
            
        Returns:
            List of dictionaries containing search results with metadata and scores
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no results to return")
            return []
            
        logger.info(f"Searching index with {self.index.ntotal} vectors")
        logger.info(f"Query embedding shape: {query_embedding.shape}")
        
        # Ensure the query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Log the query vector norm to verify it's normalized
        query_norm = np.linalg.norm(query_embedding)
        logger.info(f"Query vector (A) norm: {query_norm:.6f}")
        
        # Get a few vectors from the index to check their norms
        if self.index.ntotal > 0:
            for i in range(min(3, self.index.ntotal)):
                vector = self.index.reconstruct(i)
                norm = np.linalg.norm(vector)
                logger.info(f"Document vector {i} (B) norm: {norm:.6f}")
                
                # Calculate raw inner product between query and this document
                inner_product = np.dot(query_embedding.flatten(), vector)
                logger.info(f"Raw inner product between query and document {i}: {inner_product:.6f}")
        
        # Search the index
        similarities, indices = self.index.search(query_embedding, k)
        logger.info(f"Raw similarities from FAISS: {similarities}")
        logger.info(f"Raw indices from FAISS: {indices}")
        
        # Log details about the similarities
        logger.info(f"Similarity stats - min: {np.min(similarities):.6f}, max: {np.max(similarities):.6f}, mean: {np.mean(similarities):.6f}")
        
        # Prepare results
        results = []
        for i, (idx, similarity) in enumerate(zip(indices[0], similarities[0])):
            # Skip invalid indices (can happen if there are fewer than k items in the index)
            if idx == -1:
                logger.info(f"Skipping invalid index {idx}")
                continue
                
            # Skip results below the threshold
            if similarity < threshold:
                logger.info(f"Skipping result {idx} with similarity {similarity:.4f} below threshold {threshold}")
                continue
                
            # Get the metadata for this result
            metadata = self.metadata[idx].copy()
            
            # Add the similarity score
            metadata["score"] = float(similarity)
            
            # Add to results
            results.append(metadata)
            logger.info(f"Added result {idx} with similarity {similarity:.4f}")
        
        logger.info(f"Returning {len(results)} results")
        return results
    
    def save(self, index_path: str, metadata_path: str):
        """
        Save the index and metadata to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata
        """
        try:
            # Create directories if they don't exist
            logger.info(f"Creating directories for index and metadata files")
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            logger.info("Directories created successfully")
            
            # Save the index
            logger.info(f"Saving FAISS index to {index_path}")
            logger.info(f"Index type: {type(self.index).__name__}")
            logger.info(f"Index dimension: {self.index.d}")
            logger.info(f"Number of vectors: {self.index.ntotal}")
            faiss.write_index(self.index, index_path)
            logger.info("FAISS index saved successfully")
            
            # Save the metadata
            logger.info(f"Saving metadata to {metadata_path}")
            logger.info(f"Number of metadata items: {len(self.metadata)}")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info("Metadata saved successfully")
            
            # Verify files were created and have content
            if os.path.exists(index_path):
                index_size = os.path.getsize(index_path)
                logger.info(f"Index file exists with size {index_size} bytes")
            else:
                logger.error(f"Index file was not created at {index_path}")
            if os.path.exists(metadata_path):
                metadata_size = os.path.getsize(metadata_path)
                logger.info(f"Metadata file exists with size {metadata_size} bytes")
            else:
                logger.error(f"Metadata file was not created at {metadata_path}")
        except Exception as e:
            logger.error(f"Error in FAISSIndex.save(): {e}", exc_info=True)
            raise
    
    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> 'FAISSIndex':
        """
        Load the index and metadata from disk.
        
        Args:
            index_path: Path to the saved FAISS index
            metadata_path: Path to the saved metadata
            
        Returns:
            A new FAISSIndex instance with the loaded index and metadata
        """
        # Check if both files exist
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        # Load the index
        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        
        # Try to load metadata
        try:
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded {len(metadata)} metadata items")
            
            # Verify metadata length matches index size
            if len(metadata) != index.ntotal:
                logger.warning(f"Metadata length ({len(metadata)}) does not match index size ({index.ntotal})")
                # Create new metadata for missing entries
                while len(metadata) < index.ntotal:
                    metadata.append({"text": f"Document {len(metadata) + 1}", "source": "unknown"})
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {metadata_path}, creating default metadata")
            # Create default metadata for each vector in the index
            metadata = [{"text": f"Document {i+1}", "source": "unknown"} for i in range(index.ntotal)]
        
        # Create a new instance
        dimension = index.d
        instance = cls(dimension)
        
        # Replace the index and metadata
        instance.index = index
        instance.metadata = metadata
        
        # Determine the index type
        if isinstance(index, faiss.IndexFlatIP):
            instance.index_type = "flat"
        elif isinstance(index, faiss.IndexIVFFlat):
            instance.index_type = "ivf"
        elif isinstance(index, faiss.IndexHNSWFlat):
            instance.index_type = "hnsw"
        else:
            instance.index_type = "unknown"
        
        logger.info(f"Loaded FAISS {instance.index_type} index with {index.ntotal} vectors and dimension {dimension}")
        return instance 