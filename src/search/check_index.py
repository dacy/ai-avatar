#!/usr/bin/env python3
"""
Script to check if the FAISS index exists and has data in it.
"""

import os
import sys
import pickle
import faiss

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """
    Main entry point.
    """
    # Define paths
    index_path = "ai_avatar/data/voice_search_index.faiss"
    metadata_path = "ai_avatar/data/voice_search_index.metadata"
    
    # Check if index file exists
    if not os.path.exists(index_path):
        print(f"Index file not found: {index_path}")
        return False
    
    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        # Check if there's a file with .faiss.metadata extension
        alt_metadata_path = f"{index_path}.metadata"
        if os.path.exists(alt_metadata_path):
            print(f"Found alternative metadata file: {alt_metadata_path}")
            metadata_path = alt_metadata_path
        else:
            return False
    
    # Try to load the index
    try:
        print(f"Loading index from: {index_path}")
        index = faiss.read_index(index_path)
        print(f"Index dimension: {index.d}")
        print(f"Index size: {index.ntotal}")
        
        # Try to load the metadata
        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        print(f"Number of metadata entries: {len(metadata)}")
        
        # Print sample of metadata
        if metadata:
            print("\nSample metadata entry:")
            print(metadata[0])
        
        return True
    except Exception as e:
        print(f"Error loading index: {e}")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nIndex check {'successful' if success else 'failed'}") 