#!/usr/bin/env python3
"""
AI Avatar - Confluence Content Ingestion Script

This script connects to a Confluence instance, retrieves content,
processes it, and creates embeddings for the vector database.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import application modules
from src.confluence.client import ConfluenceClient
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
    parser = argparse.ArgumentParser(description="Ingest Confluence content into the vector database")
    parser.add_argument("--space", help="Confluence space key to ingest (overrides config)")
    parser.add_argument("--limit", type=int, help="Maximum number of pages to ingest")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index from scratch")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], help="Logging level")
    return parser.parse_args()

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
    
    # Initialize Confluence client
    logger.info("Initializing Confluence client...")
    confluence_client = ConfluenceClient(
        url=config["confluence"]["url"],
        username=config["confluence"]["username"],
        api_token=config["confluence"]["api_token"]
    )
    
    # Get space key from args or config
    space_key = args.space if args.space else config["confluence"]["space_key"]
    
    # Fetch pages from Confluence
    logger.info(f"Fetching pages from Confluence space: {space_key}")
    pages = confluence_client.get_pages(space_key, limit=args.limit)
    logger.info(f"Found {len(pages)} pages")
    
    # Process pages
    logger.info("Processing pages...")
    processed_pages = []
    for page in tqdm(pages, desc="Processing pages"):
        # Get page content
        content = confluence_client.get_page_content(page["id"])
        
        # Save raw content
        raw_path = os.path.join(config["storage"]["raw_dir"], f"{page['id']}.html")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Process content (clean HTML, split into chunks)
        chunks = confluence_client.process_content(content, 
                                                  chunk_size=config["embedding"]["chunk_size"],
                                                  chunk_overlap=config["embedding"]["chunk_overlap"])
        
        # Save processed chunks
        processed_path = os.path.join(config["storage"]["processed_dir"], f"{page['id']}.txt")
        with open(processed_path, "w", encoding="utf-8") as f:
            f.write("\n\n===CHUNK===\n\n".join(chunks))
        
        # Add to processed pages
        for i, chunk in enumerate(chunks):
            processed_pages.append({
                "id": f"{page['id']}_{i}",
                "page_id": page["id"],
                "title": page["title"],
                "url": page["url"],
                "text": chunk
            })
    
    # Initialize embedding model
    logger.info("Initializing embedding model...")
    embedding_model = EmbeddingModel(model_name=config["embedding"]["model"])
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    texts = [page["text"] for page in processed_pages]
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
    faiss_index.add_embeddings(embeddings, processed_pages)
    
    # Save index and metadata
    logger.info("Saving index and metadata...")
    faiss_index.save(index_path, metadata_path)
    
    logger.info("Ingestion complete!")
    logger.info(f"Processed {len(pages)} pages into {len(processed_pages)} chunks")
    logger.info(f"Index saved to {index_path}")
    logger.info(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main() 