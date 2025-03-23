#!/usr/bin/env python3
"""
Ollama-based QA system for the Voice Search application.
This script uses the Ollama API to extract answers from context using a local LLM.
"""

import argparse
import logging
import sys
import os
import time
import yaml
import re
from typing import Dict, Any, List, Optional, Tuple
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from loguru import logger
import json

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: The logging level to use.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        The configuration as a dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

class EmbeddingModel:
    """
    Class for generating embeddings using SentenceTransformers.
    """
    def __init__(self, model_name: str):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use.
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.
        
        Args:
            text: The text to generate an embedding for.
            
        Returns:
            The embedding as a numpy array.
        """
        return self.model.encode(text, show_progress_bar=False)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: The texts to generate embeddings for.
            
        Returns:
            The embeddings as a numpy array.
        """
        return self.model.encode(texts, show_progress_bar=False)
        
    def split_text(self, text: str, chunk_size: int = 512) -> List[str]:
        """
        Split text into chunks of specified size.
        
        Args:
            text: The text to split.
            chunk_size: Maximum size of each chunk.
            
        Returns:
            List of text chunks.
        """
        # Split text into sentences
        sentences = text.split('. ')
        
        # Combine sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Join current chunk and add to chunks list
                chunks.append('. '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks

class FAISSIndex:
    """
    Class for managing a FAISS index for vector search.
    """
    def __init__(self, dimension: int):
        """
        Initialize the FAISS index.
        
        Args:
            dimension: Dimension of the vectors to be indexed.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        Add embeddings and metadata to the index.
        
        Args:
            embeddings: The embeddings to add.
            metadata: The metadata associated with the embeddings.
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings and metadata must match")
        
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        
    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search the index for the k nearest neighbors to the query embedding.
        
        Args:
            query_embedding: The query embedding.
            k: Number of nearest neighbors to return.
            threshold: Threshold for similarity score.
            
        Returns:
            List of search results with metadata and scores
        """
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
            if distance <= threshold:
                result = self.metadata[index].copy()
                result['score'] = float(distance)
                results.append(result)
        
        return results
    
    def save(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save the index and metadata to disk.
        
        Args:
            index_path: Path to save the index to.
            metadata_path: Path to save the metadata to. If None, will use index_path + '.metadata'.
        """
        if metadata_path is None:
            metadata_path = index_path + ".metadata"
            
        # Save the FAISS index
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, index_path)
        logger.info("FAISS index saved successfully")
        
        # Save metadata as JSON
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info("Metadata saved successfully")
            
    @classmethod
    def load(cls, index_path: str, metadata_path: Optional[str] = None) -> "FAISSIndex":
        """
        Load an index and metadata from disk.
        
        Args:
            index_path: Path to load the index from.
            metadata_path: Path to load the metadata from. If None, will use index_path + '.metadata'.
            
        Returns:
            A FAISSIndex instance.
        """
        if metadata_path is None:
            # First try the correct name format (without .faiss extension)
            base_path = index_path.replace('.faiss', '')
            metadata_path = f"{base_path}.metadata"
            
            # If that doesn't exist, fall back to the old format
            if not os.path.exists(metadata_path):
                metadata_path = index_path + ".metadata"
            
        # Make sure the paths are absolute
        index_path = os.path.abspath(index_path)
        metadata_path = os.path.abspath(metadata_path)
        
        logger.info(f"Loading index from: {index_path}")
        logger.info(f"Loading metadata from: {metadata_path}")
        
        # Load the FAISS index
        index = faiss.read_index(index_path)
        logger.info("FAISS index loaded successfully")
        
        # Load metadata from JSON
        try:
            with open(metadata_path, "r", encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata with {len(metadata)} items")
        except FileNotFoundError:
            logger.warning(f"Metadata file not found: {metadata_path}, creating default metadata")
            metadata = [{"text": f"Document {i+1}", "source": "unknown"} for i in range(index.ntotal)]
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in metadata file: {metadata_path}, creating default metadata")
            metadata = [{"text": f"Document {i+1}", "source": "unknown"} for i in range(index.ntotal)]
        
        instance = cls(index.d)
        instance.index = index
        instance.metadata = metadata
        
        return instance

class OllamaQAExtractor:
    """
    QA extractor using Ollama for generating answers from context.
    """
    
    def __init__(self, model_name: str = "llama2", api_url: str = "http://localhost:11434"):
        """
        Initialize the QA extractor.
        
        Args:
            model_name: Name of the Ollama model to use
            api_url: URL of the Ollama API server
        """
        self.model_name = model_name
        self.api_url = api_url.rstrip('/')
        logger.info(f"Initialized Ollama QA extractor with model: {model_name}")
    
    def extract_answer(self, question: str, contexts: List[str]) -> str:
        """
        Extract an answer for a question from the given contexts.
        
        Args:
            question: The question to answer
            contexts: List of text contexts to use for answering
            
        Returns:
            Generated answer
        """
        try:
            # Combine contexts with the question into a prompt
            prompt = self._create_prompt(question, contexts)
            logger.info(f"Created prompt for question: {question}")
            logger.info(f"Using {len(contexts)} contexts")
            
            # Call Ollama API
            logger.info(f"Calling Ollama API with model: {self.model_name}")
            response = requests.post(
                f"{self.api_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            
            # Log the response
            response_data = response.json()
            logger.info(f"Ollama API response: {response_data}")
            
            # Extract answer from response
            answer = response_data.get('response', '').strip()
            
            if not answer:
                logger.warning("No answer received from Ollama API")
                return "I could not find a relevant answer in the provided context."
            
            # Clean up the answer
            # Remove thinking process if present
            if "<think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            
            # Remove "Answer:" prefix if present
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
            
            # Remove any leading/trailing quotes
            answer = answer.strip('"\'')
            
            # Post-process the answer
            answer = post_process_answer(answer)
            
            logger.info(f"Generated answer: {answer}")
            return answer
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            return f"Error calling Ollama API: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in extract_answer: {str(e)}", exc_info=True)
            return f"Unexpected error: {str(e)}"
    
    def _create_prompt(self, question: str, contexts: List[str]) -> str:
        """Create a prompt for the model."""
        context_text = "\n\n".join(contexts)
        
        return f"""You are a helpful AI assistant. Answer the following question based ONLY on the provided context. If you cannot find a relevant answer in the context, say so.

IMPORTANT:
- Provide a direct, concise answer without any thinking process or explanations
- Do not include any XML tags or special formatting
- Do not repeat the question or add any prefixes like "Answer:"
- If you cannot find a relevant answer, simply say "I could not find a relevant answer in the provided context."

Context:
{context_text}

Question: {question}"""

def search_index(query: str, index_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Search the FAISS index for relevant documents.
    
    Args:
        query: The search query
        index_path: Path to the FAISS index
        config: Configuration dictionary
        
    Returns:
        List of search results with metadata and scores
    """
    try:
        # Load the index
        faiss_index = FAISSIndex.load(index_path, config["search"]["metadata_path"])
        
        # Generate query embedding
        embedding_model = EmbeddingModel(config["embedding"]["model_name"])
        query_embedding = embedding_model.generate_embedding(query)
        
        # Search the index
        results = faiss_index.search(query_embedding, k=5, threshold=0.0)
        
        # Log results for debugging
        logger.info(f"Found {len(results)} results")
        for i, result in enumerate(results):
            logger.info(f"Result {i+1}: score={result.get('score', 0.0):.3f}, text={result.get('text', '')[:100]}...")
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching index: {e}")
        raise

def post_process_answer(answer: str) -> str:
    """
    Improve the readability of the answer.
    
    Args:
        answer: The answer to post-process.
        
    Returns:
        The post-processed answer.
    """
    # Remove any remaining XML tags
    answer = re.sub(r'<[^>]+>', '', answer)
    
    # Remove multiple newlines
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    
    # Remove "I don't know" statements if there's other content
    if len(answer) > 100 and ("I don't know" in answer or "I don't have enough information" in answer):
        answer = re.sub(r'I don\'t know[^.]*\.', '', answer)
        answer = re.sub(r'I don\'t have enough information[^.]*\.', '', answer)
    
    return answer.strip()

def verify_answer_factuality(answer: str, search_results: List[Dict[str, Any]], qa_extractor: OllamaQAExtractor) -> Dict[str, Any]:
    """
    Verify the factuality of an answer against the search results.
    
    Args:
        answer: The answer to verify.
        search_results: The search results to verify against.
        qa_extractor: The QA extractor to use for verification.
        
    Returns:
        A dictionary with verification results including:
        - is_factual: Whether the answer is factual
        - confidence: Confidence score for factuality
        - factuality_notes: List of factuality notes and issues
    """
    # Extract all text from search results to create verification context
    verification_text = ""
    for result in search_results:
        if "text" in result and result["text"]:
            verification_text += result["text"] + "\n\n"
    
    if not verification_text:
        return {
            "is_factual": False,
            "confidence": 0.0,
            "factuality_notes": "No context available for verification"
        }
    
    # Create a prompt to verify the factuality
    verification_prompt = f"""<instructions>
You are a fact-checking assistant. Your task is to analyze if the provided answer is supported by the context.
- ONLY consider what's explicitly stated in the context.
- For each claim in the answer, determine if it's:
  * SUPPORTED: The claim is directly supported by the context
  * UNSUPPORTED: The claim isn't found in the context
  * CONTRADICTED: The claim contradicts information in the context
- Give a confidence score (0-100%) for the factuality of the entire answer.
- List any specific issues or unsupported claims you find.
- Be strict in your assessment.
</instructions>

<context>
{verification_text[:4000]}
</context>

<answer_to_verify>
{answer}
</answer_to_verify>

Analyze the factuality of the above answer based ONLY on the provided context. 
Respond in this format:
FACTUALITY ASSESSMENT: [MOSTLY FACTUAL or PARTIALLY FACTUAL or NOT FACTUAL]
CONFIDENCE: [0-100%] 
ISSUES:
- [List any unsupported claims or issues]"""
    
    verification_result = qa_extractor.query_ollama(verification_prompt)
    
    # Parse verification result
    factuality = "NOT FACTUAL"
    confidence = 0.0
    factuality_notes = []
    
    if "MOSTLY FACTUAL" in verification_result:
        factuality = "MOSTLY FACTUAL"
        confidence = 0.8
    elif "PARTIALLY FACTUAL" in verification_result:
        factuality = "PARTIALLY FACTUAL"
        confidence = 0.5
    
    # Extract confidence if provided
    confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', verification_result)
    if confidence_match:
        try:
            confidence = int(confidence_match.group(1)) / 100.0
        except ValueError:
            pass
    
    # Extract issues
    issues_section = verification_result.split("ISSUES:")[-1] if "ISSUES:" in verification_result else ""
    if issues_section:
        issues = [issue.strip() for issue in re.findall(r'-\s*(.*?)(?=$|\n-)', issues_section)]
        if issues:
            factuality_notes = issues
    
    # Create factuality note based on assessment
    if factuality == "NOT FACTUAL":
        factuality_notes.insert(0, "This answer may contain information not supported by the available context.")
    elif factuality == "PARTIALLY FACTUAL":
        factuality_notes.insert(0, "This answer is partially supported by the available context.")
    
    return {
        "is_factual": factuality in ["MOSTLY FACTUAL", "PARTIALLY FACTUAL"],
        "confidence": confidence,
        "factuality_notes": "\n".join(["- " + note for note in factuality_notes]) if factuality_notes else ""
    }

def extract_answer(question: str, search_results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract an answer from search results using the Ollama API.
    
    Args:
        question: The question to answer.
        search_results: The search results to extract the answer from.
        config: Configuration dictionary.
        
    Returns:
        A dictionary containing:
        - answer: The main answer text
        - sources: List of source documents used
        - factuality_notes: Notes about the answer's factuality
        - confidence: Confidence score
    """
    try:
        qa_extractor = OllamaQAExtractor(model_name=config.get("model", "deepseek-r1:1.5b"))
        
        # Extract answer from contexts
        result = qa_extractor.extract_answer_from_contexts(question, search_results)
        if not result:
            return None
            
        # Get the raw answer
        raw_answer = result.get("answer", "")
        
        # Extract sources
        sources = []
        for res in search_results[:3]:  # Only include top 3 sources
            source = res.get("metadata", {}).get("source", "")
            if source and source not in sources:
                sources.append(source)
        
        # Verify factuality
        factuality_result = verify_answer_factuality(raw_answer, search_results, qa_extractor)
        
        # Structure the output
        structured_output = {
            "answer": raw_answer.strip(),
            "sources": sources,
            "factuality_notes": factuality_result.get("factuality_notes", ""),
            "confidence": search_results[0].get("score", 0) if search_results else 0
        }
        
        return structured_output
        
    except Exception as e:
        logging.error(f"Error extracting answer: {str(e)}", exc_info=True)
        return None

def interactive_mode(index_path: str, config: Dict[str, Any], model: str = "deepseek-r1:1.5b") -> None:
    """
    Run an interactive mode for querying the QA system.
    
    Args:
        index_path: Path to the FAISS index.
        config: Configuration dictionary.
        model: Name of the Ollama model to use.
    """
    # Update config with model name
    config["ollama_model"] = model
    
    # Check if debug mode is enabled
    debug_mode = config.get("debug_mode", False)
    if debug_mode:
        print("\nDEBUG MODE ENABLED - Showing detailed information about the search and answer process")
    
    print("\nWelcome to the Ollama-based Voice Search Interactive Mode!")
    print("Type 'exit' or 'quit' to exit.")
    print("Type your question and press Enter.\n")
    
    while True:
        try:
            query = input("Enter your query: ")
            if query.lower() in ["exit", "quit"]:
                break
                
            if not query.strip():
                continue
                
            start_time = time.time()
            
            # Search index
            logging.info(f"Searching for: {query}")
            search_results = search_index(query, index_path, config)
            
            if not search_results:
                print("No results found.")
                continue
                
            # Extract answer
            logging.info("Extracting answer...")
            result = extract_answer(query, search_results, config)
            
            answer = result["answer"]
            confidence = result["confidence"]
            
            end_time = time.time()
            
            # Print answer
            print("\n" + "=" * 80)
            
            # Create a confidence label based on the confidence score
            confidence_label = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
            
            print(f"Answer (confidence: {confidence:.2f} - {confidence_label}):")
            print("-" * 80)
            print(answer)
            print("=" * 80)
            
            if debug_mode:
                print(f"\nTime taken: {end_time - start_time:.2f} seconds")
                print(f"Number of search results: {len(search_results)}")
                
                if search_results:
                    print(f"Top result score: {search_results[0]['score']:.4f}")
                    source = "Unknown"
                    if "source" in search_results[0]:
                        source = search_results[0]["source"]
                    elif "metadata" in search_results[0] and "source" in search_results[0]["metadata"]:
                        source = search_results[0]["metadata"]["source"]
                    print(f"Top source: {source}")
                    
                    # Print text sample from top result
                    text_sample = search_results[0].get("text", "")[:150]
                    if text_sample:
                        print(f"Top result text sample: {text_sample}...")
                
                # Show answer quality metrics
                print("\nAnswer Quality Metrics:")
                
                # Analyze answer characteristics
                answer_length = len(answer)
                question_words = set(re.findall(r'\b\w+\b', query.lower()))
                answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
                common_words = question_words.intersection(answer_words)
                
                # Calculate lexical overlap (percentage of question words in answer)
                lexical_overlap = len(common_words) / len(question_words) if question_words else 0
                
                # Check for uncertainty markers
                uncertainty_markers = [
                    "I don't know", "I'm not sure", "cannot determine", "insufficient information",
                    "context doesn't provide", "not mentioned", "not specified", "unclear"
                ]
                contains_uncertainty = any(marker.lower() in answer.lower() for marker in uncertainty_markers)
                
                print(f"- Length: {answer_length} characters")
                print(f"- Lexical overlap with question: {lexical_overlap:.2f}")
                print(f"- Contains uncertainty markers: {'Yes' if contains_uncertainty else 'No'}")
                
                if "factuality notes" in answer.lower() or "issues include" in answer.lower():
                    print("- Factuality verification: Completed with issues")
                elif "this answer may contain information not supported" in answer.lower():
                    print("- Factuality verification: Failed")
                else:
                    print("- Factuality verification: Passed")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logging.error(f"Error: {e}")
            print(f"An error occurred: {e}")

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Ollama-based QA system for Voice Search")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--index", type=str, required=True, help="Path to the FAISS index")
    parser.add_argument("--model", type=str, default="deepseek-r1:1.5b", help="Name of the Ollama model to use")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (same as setting debug_mode=true in config)")
    
    return parser.parse_args()

def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Make sure debug_mode is set if passed as a command line argument
    if args.log_level.upper() == "DEBUG" or args.debug:
        config["debug_mode"] = True
        print(f"Debug mode enabled via command line argument")
    
    # Run interactive mode
    interactive_mode(args.index, config, args.model)

if __name__ == "__main__":
    main() 