"""
Question answering module for extracting answers from context.
"""

from typing import Dict, Any
from .generative import GenerativeQA
from .local_generative import LocalGenerativeQA
from .extractive import ExtractiveQA
from loguru import logger
import os

def create_qa(config: Dict[str, Any], config_path: str = "config/config.yaml"):
    """
    Create a QA instance based on configuration.
    
    Args:
        config: Configuration dictionary containing QA settings
        config_path: Path to the configuration file
        
    Returns:
        QA instance (GenerativeQA or ExtractiveQA)
    """
    qa_config = config.get('qa', {})
    mode = qa_config.get('mode', 'generative')
    
    if mode == 'generative':
        provider = qa_config.get('provider', 'local')
        gen_config = qa_config.get('generative', {})
        
        if provider == 'ollama':
            ollama_config = gen_config.get('ollama', {})
            return GenerativeQA(
                model_name=ollama_config.get('model_name', 'deepseek-r1:1.5b'),
                api_url=ollama_config.get('api_url', 'http://localhost:11434')
            )
        elif provider == 'local':
            local_config = gen_config.get('local', {})
            
            return LocalGenerativeQA(
                model_name=local_config['model_name']
            )
        else:
            raise ValueError(f"Unsupported QA provider: {provider}")
            
    elif mode == 'extractive':
        ext_config = qa_config.get('extractive', {})
        return ExtractiveQA(
            model_name=ext_config.get('model_name', 'deepset/roberta-base-squad2'),
            max_length=ext_config.get('max_length', 100),
            confidence_threshold=ext_config.get('confidence_threshold', 0.5),
            use_gpu=ext_config.get('use_gpu', True)
        )
    else:
        raise ValueError(f"Unsupported QA mode: {mode}") 