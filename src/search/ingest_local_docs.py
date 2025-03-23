#!/usr/bin/env python3
"""
AI Avatar - Local Document Ingestion Script

This script processes local knowledge base documents,
cleans the text, splits it into chunks, and creates embeddings for the vector database.
"""

import os
import sys
import yaml
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import re

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import application modules
from src.embedding.model import EmbeddingModel
from src.search.faiss_index import FAISSIndex

def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise RuntimeError(f"Could not load configuration: {e}")

def setup_logging(log_level):
    """Set up logging configuration"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    logger.add(
        "logs/ingest.log",
        rotation="10 MB",
        level=log_level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ingest local documents into the vector database")
    parser.add_argument("--docs-dir", default="../docs/test_knowledge_base_docs", 
                        help="Directory containing the test knowledge base documents")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index from scratch")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Logging level")
    return parser.parse_args()

def clean_text(text):
    """Clean the text by removing extra whitespace and special characters"""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()

def split_into_chunks(text, chunk_size, chunk_overlap):
    """Split text into chunks of specified size with overlap"""
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence would exceed the chunk size and we already have content,
        # save the current chunk and start a new one with overlap
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Calculate how many sentences to keep for overlap
            overlap_size = 0
            overlap_sentences = []
            
            for s in reversed(current_chunk):
                if overlap_size + len(s) <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s) + 1  # +1 for the space
                else:
                    break
            
            current_chunk = overlap_sentences
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_size + 1  # +1 for the space
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def main():
    """Main entry point for the script"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()
    
    # Create data directories if they don't exist
    os.makedirs(config["storage"]["raw_dir"], exist_ok=True)
    os.makedirs(config["storage"]["processed_dir"], exist_ok=True)
    os.makedirs(config["storage"]["vector_dir"], exist_ok=True)
    
    # Get the docs directory
    docs_dir = os.path.abspath(args.docs_dir)
    if not os.path.exists(docs_dir):
        logger.error(f"Documents directory not found: {docs_dir}")
        sys.exit(1)
    
    logger.info(f"Processing documents from: {docs_dir}")
    
    # Get all text files in the directory
    doc_files = [f for f in os.listdir(docs_dir) if f.endswith('.txt')]
    logger.info(f"Found {len(doc_files)} document files")
    
    # Process documents
    logger.info("Processing documents...")
    processed_docs = []
    
    for doc_file in tqdm(doc_files, desc="Processing documents"):
        doc_path = os.path.join(docs_dir, doc_file)
        doc_id = os.path.splitext(doc_file)[0]
        
        # Read the document
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Save raw content
        raw_path = os.path.join(config["storage"]["raw_dir"], f"{doc_id}.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Clean the text
        cleaned_content = clean_text(content)
        
        # Split into chunks
        chunks = split_into_chunks(
            cleaned_content,
            chunk_size=config["embedding"]["chunk_size"],
            chunk_overlap=config["embedding"]["chunk_overlap"]
        )
        
        logger.debug(f"Document {doc_id} split into {len(chunks)} chunks")
        
        # Save processed chunks
        processed_path = os.path.join(config["storage"]["processed_dir"], f"{doc_id}.txt")
        with open(processed_path, "w", encoding="utf-8") as f:
            f.write("\n\n===CHUNK===\n\n".join(chunks))
        
        # Add to processed documents
        for i, chunk in enumerate(chunks):
            processed_docs.append({
                "id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "title": doc_id,  # Using the filename as the title
                "file_path": doc_path,
                "text": chunk
            })
    
    # Initialize embedding model
    logger.info("Initializing embedding model...")
    embedding_model = EmbeddingModel(model_name=config["embedding"]["model"])
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    texts = [doc["text"] for doc in processed_docs]
    embeddings = embedding_model.generate_embeddings(
        texts, 
        batch_size=config["embedding"]["batch_size"]
    )
    
    # Initialize FAISS index
    logger.info("Initializing FAISS index...")
    index_path = os.path.join(config["storage"]["vector_dir"], "faiss_index")
    metadata_path = os.path.join(config["storage"]["vector_dir"], "metadata.json")
    
    faiss_index = FAISSIndex(
        dimension=embedding_model.dimension,
        index_type=config["search"]["index_type"]
    )
    
    # Add embeddings to index
    logger.info("Adding embeddings to index...")
    faiss_index.add_embeddings(embeddings, processed_docs)
    
    # Save index and metadata
    logger.info("Saving index and metadata...")
    faiss_index.save(index_path, metadata_path)
    
    logger.info("Ingestion complete!")
    logger.info(f"Processed {len(doc_files)} documents into {len(processed_docs)} chunks")
    logger.info(f"Index saved to {index_path}")
    logger.info(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main() 