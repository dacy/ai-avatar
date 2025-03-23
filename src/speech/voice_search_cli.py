#!/usr/bin/env python3
"""
Command-line interface for the Voice Search system.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any, Optional
from loguru import logger
import yaml

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_avatar.src.api.voice_search_api import VoiceSearchAPI


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Path to log file
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file logger if specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="1 week"
        )
    
    logger.info(f"Logging initialized with level {log_level}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Voice Search CLI")
    
    # Configuration
    parser.add_argument("--config", type=str, default="config/voice_search_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to log file")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--input-dir", type=str, required=True,
                              help="Directory containing documents to ingest")
    ingest_parser.add_argument("--output-index", type=str, required=True,
                              help="Path to save the index")
    ingest_parser.add_argument("--file-ext", type=str, default=".txt",
                              help="File extension to filter documents")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("--index", type=str, required=True,
                             help="Path to the index")
    query_parser.add_argument("--query", type=str, default=None,
                             help="Text query")
    query_parser.add_argument("--audio", type=str, default=None,
                             help="Path to audio file for voice query")
    query_parser.add_argument("--play-audio", action="store_true",
                             help="Play audio response")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--index", type=str, required=True,
                                  help="Path to the index")
    interactive_parser.add_argument("--play-audio", action="store_true",
                                  help="Play audio response")
    
    return parser.parse_args()


def ingest_documents(api: VoiceSearchAPI, input_dir: str, output_index: str, file_ext: str = ".txt") -> None:
    """
    Ingest documents into the search index.
    
    Args:
        api: Voice Search API instance
        input_dir: Directory containing documents to ingest
        output_index: Path to save the index
        file_ext: File extension to filter documents
    """
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Get list of documents
    documents = []
    metadatas = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(file_ext):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, input_dir)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    documents.append(content)
                    metadatas.append({
                        "source": rel_path,
                        "filename": file,
                        "path": file_path,
                        "created_at": time.time()
                    })
                    
                    logger.debug(f"Loaded document: {rel_path}")
                except Exception as e:
                    logger.error(f"Failed to load document {file_path}: {e}")
    
    if not documents:
        logger.error(f"No documents found in {input_dir} with extension {file_ext}")
        sys.exit(1)
    
    logger.info(f"Found {len(documents)} documents to ingest")
    
    # Add documents to the index
    doc_ids = api.add_documents(documents, metadatas)
    
    logger.info(f"Added {len(doc_ids)} documents to the index")
    
    # Save the index
    os.makedirs(os.path.dirname(os.path.abspath(output_index)), exist_ok=True)
    api.save_index(output_index)
    
    logger.info(f"Saved index to {output_index}")


def process_query(api: VoiceSearchAPI, query: Optional[str] = None, audio_path: Optional[str] = None, play_audio: bool = False) -> None:
    """
    Process a query and display the results.
    
    Args:
        api: Voice Search API instance
        query: Text query
        audio_path: Path to audio file for voice query
        play_audio: Whether to play audio response
    """
    if audio_path:
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            sys.exit(1)
        
        logger.info(f"Processing voice query from {audio_path}")
        result = api.process_voice_query(audio_path)
        
        if not result["success"]:
            logger.error(f"Failed to process voice query: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
        print(f"\nTranscription: {result['transcription']}")
    elif query:
        logger.info(f"Processing text query: {query}")
        result = api.process_text_query(query)
    else:
        logger.error("No query or audio file provided")
        sys.exit(1)
    
    # Display results
    print("\n" + "="*80)
    print(f"Query: {query or result.get('transcription', 'Unknown')}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nContext:")
    print("-"*80)
    print(result['context'])
    print("="*80)
    
    # Play audio response if requested
    if play_audio and result.get("audio_response"):
        try:
            import sounddevice as sd
            import soundfile as sf
            
            logger.info(f"Playing audio response from {result['audio_response']}")
            data, samplerate = sf.read(result['audio_response'])
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            logger.error(f"Failed to play audio response: {e}")


def interactive_mode(api: VoiceSearchAPI, play_audio: bool = False) -> None:
    """
    Run the system in interactive mode.
    
    Args:
        api: Voice Search API instance
        play_audio: Whether to play audio response
    """
    print("\nVoice Search Interactive Mode")
    print("Type 'exit' or 'quit' to exit")
    print("="*80)
    
    while True:
        try:
            query = input("\nEnter your query: ")
            
            if query.lower() in ["exit", "quit"]:
                break
            
            if not query:
                continue
            
            process_query(api, query=query, play_audio=play_audio)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
    
    print("\nExiting interactive mode")


def main() -> None:
    """
    Main entry point.
    """
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize API
    if args.command == "query" or args.command == "interactive":
        # Update index path in config
        if "search" not in config:
            config["search"] = {}
        config["search"]["index_path"] = args.index
    
    api = VoiceSearchAPI(config)
    
    # Execute command
    if args.command == "ingest":
        ingest_documents(api, args.input_dir, args.output_index, args.file_ext)
    elif args.command == "query":
        process_query(api, args.query, args.audio, args.play_audio)
    elif args.command == "interactive":
        interactive_mode(api, args.play_audio)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main() 