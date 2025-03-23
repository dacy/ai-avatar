"""
Voice Search API module that ties together all components of the system.
"""

import os
import tempfile
import json
from typing import Dict, List, Any, Optional, Union
from loguru import logger
import uuid
import time

from ai_avatar.src.embedding.model import EmbeddingModel
from ai_avatar.src.search.faiss_index import FAISSIndex
from ai_avatar.src.qa.extractor import QAExtractor
from ai_avatar.src.speech.asr import ASRProcessor
from ai_avatar.src.speech.tts import TTSProcessor


class VoiceSearchAPI:
    """
    Voice Search API that ties together all components of the system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Voice Search API.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        logger.info("Initializing Voice Search API components...")
        
        # Initialize embedding model
        embedding_config = config.get("embedding", {})
        self.embedding_model = EmbeddingModel(
            model_name=embedding_config.get("model_name", "all-MiniLM-L6-v2"),
            device=embedding_config.get("device", "cpu")
        )
        
        # Initialize FAISS index
        search_config = config.get("search", {})
        self.faiss_index = FAISSIndex(
            embedding_dim=self.embedding_model.get_embedding_dimension(),
            index_type=search_config.get("index_type", "Flat"),
            metric=search_config.get("metric", "cosine")
        )
        
        # Load index if path is provided
        index_path = search_config.get("index_path")
        if index_path and os.path.exists(index_path):
            logger.info(f"Loading FAISS index from {index_path}")
            self.faiss_index.load(index_path)
        
        # Initialize QA extractor
        qa_config = config.get("qa", {})
        self.qa_extractor = QAExtractor(
            model_name=qa_config.get("model_name", "deepset/roberta-base-squad2"),
            device=qa_config.get("device", "cpu")
        )
        
        # Initialize ASR processor
        asr_config = config.get("asr", {})
        self.asr_processor = ASRProcessor(
            engine=asr_config.get("engine", "whisper"),
            model=asr_config.get("model", "base"),
            language=asr_config.get("language", "en")
        )
        
        # Initialize TTS processor
        tts_config = config.get("tts", {})
        self.tts_processor = TTSProcessor(
            engine=tts_config.get("engine", "piper"),
            voice=tts_config.get("voice", "en_US-amy-medium"),
            speed=tts_config.get("speed", 1.0)
        )
        
        logger.info("Voice Search API initialized successfully")
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a document to the search index.
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            Document ID
        """
        # Generate embedding for the document
        embedding = self.embedding_model.generate_embedding(text)
        
        # Add to the index
        doc_id = self.faiss_index.add_embedding(embedding, text, metadata)
        
        logger.info(f"Added document to index with ID: {doc_id}")
        
        return doc_id
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Add multiple documents to the search index.
        
        Args:
            texts: List of document texts
            metadatas: List of document metadata
            
        Returns:
            List of document IDs
        """
        # Generate embeddings for the documents
        embeddings = self.embedding_model.generate_embeddings(texts)
        
        # Add to the index
        doc_ids = self.faiss_index.add_embeddings(embeddings, texts, metadatas)
        
        logger.info(f"Added {len(doc_ids)} documents to index")
        
        return doc_ids
    
    def save_index(self, path: str) -> None:
        """
        Save the search index to disk.
        
        Args:
            path: Path to save the index
        """
        self.faiss_index.save(path)
        logger.info(f"Saved search index to {path}")
    
    def search_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents using a text query.
        
        Args:
            query: Text query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.generate_embedding(query)
        
        # Search the index
        results = self.faiss_index.search(query_embedding, top_k)
        
        logger.info(f"Found {len(results)} results for query: {query}")
        
        return results
    
    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a question using the search index.
        
        Args:
            question: Question to answer
            top_k: Number of search results to use
            
        Returns:
            Answer dictionary
        """
        # Search for relevant documents
        search_results = self.search_text(question, top_k)
        
        if not search_results:
            logger.warning(f"No search results found for question: {question}")
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "confidence": 0.0,
                "context": "",
                "search_results": []
            }
        
        # Extract contexts from search results
        contexts = [result["text"] for result in search_results]
        
        # Extract answer from contexts
        answer_result = self.qa_extractor.extract_answer(question, contexts)
        
        # Prepare response
        response = {
            "answer": answer_result["answer"],
            "confidence": answer_result["confidence"],
            "context": answer_result["context"],
            "search_results": search_results
        }
        
        logger.info(f"Generated answer for question: {question}")
        logger.debug(f"Answer: {response['answer']}")
        
        return response
    
    def process_voice_query(self, audio_path: str) -> Dict[str, Any]:
        """
        Process a voice query.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        # Transcribe the audio
        logger.info(f"Transcribing audio from {audio_path}")
        transcription = self.asr_processor.transcribe_file(audio_path)
        
        if not transcription:
            logger.warning("Failed to transcribe audio")
            return {
                "success": False,
                "error": "Failed to transcribe audio",
                "transcription": "",
                "answer": "",
                "audio_response": None,
                "processing_time": time.time() - start_time
            }
        
        logger.info(f"Transcription: {transcription}")
        
        # Answer the question
        answer_result = self.answer_question(transcription)
        
        # Generate speech response
        response_audio_path = None
        if answer_result["answer"]:
            try:
                logger.info("Generating speech response")
                response_audio_path = self.tts_processor.synthesize_to_temp_file(answer_result["answer"])
                logger.info(f"Speech response generated: {response_audio_path}")
            except Exception as e:
                logger.error(f"Failed to generate speech response: {e}")
        
        # Prepare response
        response = {
            "success": True,
            "transcription": transcription,
            "answer": answer_result["answer"],
            "confidence": answer_result["confidence"],
            "context": answer_result["context"],
            "search_results": answer_result["search_results"],
            "audio_response": response_audio_path,
            "processing_time": time.time() - start_time
        }
        
        return response
    
    def process_text_query(self, query: str) -> Dict[str, Any]:
        """
        Process a text query.
        
        Args:
            query: Text query
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        # Answer the question
        answer_result = self.answer_question(query)
        
        # Generate speech response
        response_audio_path = None
        if answer_result["answer"]:
            try:
                logger.info("Generating speech response")
                response_audio_path = self.tts_processor.synthesize_to_temp_file(answer_result["answer"])
                logger.info(f"Speech response generated: {response_audio_path}")
            except Exception as e:
                logger.error(f"Failed to generate speech response: {e}")
        
        # Prepare response
        response = {
            "success": True,
            "query": query,
            "answer": answer_result["answer"],
            "confidence": answer_result["confidence"],
            "context": answer_result["context"],
            "search_results": answer_result["search_results"],
            "audio_response": response_audio_path,
            "processing_time": time.time() - start_time
        }
        
        return response 