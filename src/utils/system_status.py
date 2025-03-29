import torch
import os
import yaml
import requests
import logging
from typing import Dict
from src.search.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)

def check_ollama_availability() -> bool:
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/version")
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Ollama not available: {str(e)}")
        return False

def get_system_status() -> Dict[str, str]:
    """Get system status information including PyTorch, CUDA, Ollama, and index details."""
    status = {
        "pytorch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "status": "operational",  # Default status
        "error_details": None  # Will be populated if there are errors
    }
    
    if torch.cuda.is_available():
        status.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0)
        })
    
    # Load config to get model names and paths
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Add embedding model name to status
        embedding_config = config.get("embedding", {})
        status["embedding_model"] = embedding_config.get("model_name", "Unknown")
        
        # Get QA model name based on configuration
        qa_config = config.get("qa", {})
        qa_provider = qa_config.get("provider", "local")
        qa_mode = qa_config.get("mode", "generative")
        
        if qa_mode == "generative":
            if qa_provider == "ollama":
                status["qa_model"] = qa_config.get("generative", {}).get("ollama", {}).get("model_name", "Unknown")
            else:  # local
                status["qa_model"] = qa_config.get("generative", {}).get("local", {}).get("model_name", "Unknown")
        else:  # extractive
            status["qa_model"] = qa_config.get("extractive", {}).get("model_name", "Unknown")
        
        # Check Ollama availability only if it's configured as the QA provider
        if qa_provider == "ollama" and qa_mode == "generative":
            ollama_available = check_ollama_availability()
            status["ollama_available"] = str(ollama_available)
            if not ollama_available:
                status["error_details"] = "Ollama service is not running. Please start Ollama to use the QA system."
        else:
            status["ollama_available"] = "not_configured"
        
        # Check if index exists and is valid
        index_path = config.get("search", {}).get("index_path")
        if not index_path:
            status["error_details"] = "Search index path not configured"
            status["index_exists"] = "false"
            status["index_working"] = "false"
        else:
            index_exists = os.path.exists(index_path)
            status["index_exists"] = str(index_exists)
            
            index_working = True
            if index_exists:
                try:
                    # Try to load the index to verify it's not corrupted
                    faiss_index = FAISSIndex.load(index_path, config["search"]["metadata_path"])
                    index_working = True
                except Exception as e:
                    logger.error(f"Error loading index: {e}")
                    index_working = False
                    status["error_details"] = f"Search index is corrupted: {str(e)}"
                    # Delete corrupted files
                    for file_path in [index_path, config["search"]["metadata_path"]]:
                        if os.path.exists(file_path):
                            os.remove(file_path)
            
            status["index_working"] = str(index_working)
        
        # Update overall status based on configuration
        if qa_provider == "ollama" and qa_mode == "generative":
            # If using Ollama, require it to be available
            status["status"] = "operational" if ollama_available and (not index_exists or index_working) else "degraded"
        else:
            # If not using Ollama, only check index status
            status["status"] = "operational" if not index_exists or index_working else "degraded"
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        status.update({
            "embedding_model": "Unknown",
            "qa_model": "Unknown",
            "ollama_available": "error",
            "index_exists": "false",
            "index_working": "false",
            "status": "error",
            "error_details": f"Failed to load system configuration: {str(e)}"
        })
    
    return status 