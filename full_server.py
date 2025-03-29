import os
import yaml
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import requests
from flask import Flask, render_template, jsonify, request
import soundfile as sf
import io
import re
import markdown
from bs4 import BeautifulSoup
import numpy as np
import faiss
import json
import asyncio
import pickle
from datetime import datetime
from src.utils.system_status import get_system_status

print("Starting full_server.py...")

# Add current directory to path to import from src
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(f"Python path: {sys.path}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Logging initialized")

# Import from src.embedding since we're now inside the ai_avatar directory
from src.embedding.crawler import WebCrawler
from src.embedding.embedder import TextEmbedder

try:
    logger.info("Successfully imported WebCrawler and TextEmbedder from src.embedding")
except Exception as e:
    logger.error(f"Failed to import WebCrawler or TextEmbedder: {e}")
    raise

# Import embedding and index modules
try:
    from src.search.faiss_index import FAISSIndex
    logger.info("Successfully imported FAISSIndex")
except ImportError as e:
    logger.error(f"Failed to import FAISSIndex: {e}")
    raise

# Import QA extractors
from src.qa.extractive import ExtractiveQA
from src.qa.generative import GenerativeQA
from src.qa import create_qa  # Add this import

# Calculate static and template folder paths
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

# Initialize Flask app with correct paths
app = Flask(__name__, 
            static_folder=static_folder,
            template_folder=template_folder)

logger.info(f"Static folder: {static_folder}")
logger.info(f"Template folder: {template_folder}")

# Load Whisper model
try:
    import whisper
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None
    logger.warning("Voice input will be disabled until Whisper model is available")

# Load config
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(data_dir, exist_ok=True)

# Load config from config.yaml
config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update paths in config
config["search"]["index_path"] = os.path.join(data_dir, "vectors", "voice_search_index.faiss")
config["search"]["metadata_path"] = os.path.join(data_dir, "vectors", "voice_search_index.metadata")

# Create necessary subdirectories
os.makedirs(os.path.dirname(config["search"]["index_path"]), exist_ok=True)
os.makedirs(os.path.dirname(config["search"]["metadata_path"]), exist_ok=True)

# Create data directories
for dir_path in [
    config["storage"]["data_dir"],
    config["storage"]["raw_dir"],
    config["storage"]["processed_dir"],
    config["storage"]["vector_dir"]
]:
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Created directory: {dir_path}")

# Print paths for debugging
logger.info(f"Data directory: {data_dir}")
logger.info(f"Index path: {config['search']['index_path']}")
logger.info(f"Metadata path: {config['search']['metadata_path']}")
logger.info(f"Vector directory: {config['storage']['vector_dir']}")

# Initialize crawler and embedder
crawler = WebCrawler()
text_embedder = TextEmbedder()

# Initialize QA extractor based on config
logger.info("Step 3: Initializing QA extractor...")
qa_extractor = create_qa(config)  # Use the factory function
logger.info("QA extractor initialized successfully")

# Initialize embedding model and FAISS index
try:
    logger.info("Starting server initialization...")
    logger.info("Step 1: Initializing text embedder...")
    text_embedder = TextEmbedder(
        model_name=config["embedding"]["model_name"]
    )
    logger.info("Text embedder initialized successfully")
    
    # Load FAISS index if it exists
    logger.info("Step 2: Loading FAISS index...")
    global faiss_index
    if os.path.exists(config["search"]["index_path"]):
        logger.info(f"Found index at {config['search']['index_path']}")
        faiss_index = FAISSIndex.load(config["search"]["index_path"], config["search"]["metadata_path"])
        logger.info("FAISS index loaded successfully")
    else:
        logger.warning("No FAISS index found. Please add some Confluence pages first.")
        faiss_index = None
        logger.info("FAISS index set to None")
    
    logger.info("Server initialization completed successfully")
except Exception as e:
    logger.error(f"Error during server initialization: {e}", exc_info=True)
    # Delete corrupted files
    for file_path in [config["search"]["index_path"], config["search"]["metadata_path"]]:
        if os.path.exists(file_path):
            logger.info(f"Deleting potentially corrupted file: {file_path}")
            os.remove(file_path)
    logger.info("Corrupted files deleted. Please restart the server.")
    raise

@app.route('/status')
def status():
    """Check system status"""
    try:
        return jsonify(get_system_status())
    except Exception as e:
        logger.error(f"Error checking system status: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', system_status=get_system_status())

def format_source_path(path: str) -> str:
    """Format the source path to be more readable"""
    try:
        # Extract just the filename without path and extension
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        # Replace underscores and hyphens with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        return name.title()
    except Exception as e:
        logger.error(f"Error formatting source path: {e}")
        return path

@app.route('/search', methods=['POST'])
def search():
    """Handle search queries"""
    try:
        data = request.get_json()
        query_text = data.get('query')
        logger.info(f"Received search query: {query_text}")
        
        if not query_text:
            logger.error("No query text provided")
            return jsonify({'error': 'No query text provided'}), 400
            
        # Check if index exists
        if not os.path.exists(config["search"]["index_path"]):
            logger.error(f"Search index not found at {config['search']['index_path']}")
            return jsonify({'error': 'Search index not found. Please add some Confluence pages first.'}), 404
            
        # Search index using the FAISSIndex class
        try:
            logger.info("Searching index...")
            
            # Generate query embedding
            logger.info("Generating query embedding")
            query_embedding = text_embedder.encode_texts([query_text])[0]
            logger.info(f"Generated query embedding with shape {query_embedding.shape}")
            
            # Search the index
            logger.info("Searching index with query embedding")
            results = faiss_index.search(
                query_embedding, 
                k=config["search"]["max_results"], 
                threshold=config["search"]["similarity_threshold"]
            )
            logger.info(f"Found {len(results)} results")
            
            # Log the results for debugging
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}:")
                logger.info(f"  Score: {result.get('score', 0.0)}")
                logger.info(f"  URL: {result.get('url', 'N/A')}")
                logger.info(f"  Text: {result.get('text', 'N/A')[:100]}...")
        except Exception as e:
            logger.error(f"Error searching index: {e}", exc_info=True)
            error_msg = str(e).lower()
            if "corrupted" in error_msg or "invalid" in error_msg or "pickle" in error_msg:
                # Delete corrupted files
                for file_path in [config["search"]["index_path"], config["search"]["metadata_path"]]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                return jsonify({'error': 'Search index was corrupted. Please add your Confluence pages again to rebuild the index.'}), 500
            raise
            
        # Clean up results
        # Remove duplicates and default metadata
        seen_texts = set()
        cleaned_results = []
        for result in results:
            # Skip default metadata
            if result.get('source') == 'unknown':
                logger.info("Skipping result with unknown source")
                continue
                
            # Skip duplicates
            text = result.get('text', '').strip()
            if text in seen_texts:
                logger.info("Skipping duplicate text")
                continue
            seen_texts.add(text)
            
            # Clean up the result
            cleaned_result = {
                'text': text,
                'url': result.get('url', ''),
                'title': result.get('title', ''),  # Include title from metadata
                'score': result.get('score', 0.0)
            }
            cleaned_results.append(cleaned_result)
            
        # Generate answer using QA model
        logger.info("Generating answer using QA model")
        answer_result = qa_extractor.extract_answers_from_multiple_contexts(query_text, cleaned_results)
        
        # Handle different response formats
        if isinstance(answer_result, list) and len(answer_result) > 0:
            # If it's a list, take the first answer
            answer = answer_result[0].get('answer', '')
            confidence = answer_result[0].get('confidence', 0.0)
        else:
            # If it's a single result or empty
            answer = answer_result.get('answer', '') if isinstance(answer_result, dict) else ''
            confidence = answer_result.get('confidence', 0.0) if isinstance(answer_result, dict) else 0.0
        
        # Prepare response
        response = {
            'answer': answer,
            'sources': cleaned_results,  # Use cleaned results with titles
            'confidence': confidence
        }
        
        return jsonify(response)
            
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def convert_webm_to_wav(webm_path: str) -> str:
    """Convert WebM audio to WAV format using ffmpeg"""
    try:
        wav_path = webm_path.replace('.webm', '.wav')
        import subprocess
        subprocess.run([
            'ffmpeg', '-i', webm_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            wav_path
        ], check=True)
        return wav_path
    except Exception as e:
        logger.error(f"Error converting audio format: {e}")
        raise

@app.route('/api/ask', methods=['POST'])
def ask():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        logger.info("Received audio file")
        
        # Save audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name
            
        try:
            # Convert WebM to WAV
            wav_path = convert_webm_to_wav(temp_path)
            
            # Transcribe audio to text using Whisper
            if not whisper_model:
                return jsonify({'error': 'Whisper model not available'}), 500
                
            query_text = process_voice_query(wav_path)
            logger.info(f"Transcribed text: {query_text}")
            
            # Search for relevant documents
            search_results = search_index(query_text, config["search"]["index_path"], config)
            if not search_results:
                return jsonify({'error': 'No relevant documents found'}), 404
                
            # Extract answer from search results
            result = extract_answer(query_text, search_results, config)
            if not result:
                return jsonify({'error': 'Failed to generate response'}), 500
                
            # Convert response to speech
            audio_url = text_to_speech(result['answer'])
            logger.info(f"Generated audio URL: {audio_url}")
            
            # Format sources
            formatted_sources = [format_source_path(source) for source in result['sources']]
            
            # Convert answer to HTML
            answer_html = markdown.markdown(
                result['answer'],
                extensions=[
                    'markdown.extensions.extra',
                    'markdown.extensions.sane_lists',
                    'markdown.extensions.nl2br'
                ]
            )
            
            # Add proper paragraph spacing
            answer_html = re.sub(r'<p>', '<p style="margin-bottom: 1em;">', answer_html)
            
            # Add factuality notes if present
            if result['factuality_notes']:
                answer_html += f'<div class="factuality-notes"><h4>Factuality Notes:</h4><p>{result["factuality_notes"]}</p></div>'
            
            return jsonify({
                'question': query_text,
                'answer': answer_html,
                'source': formatted_sources[0] if formatted_sources else 'Unknown',
                'confidence': f"{result['confidence']:.2%}",
                'audio_url': audio_url,
                'html': True  # Flag to indicate that the answer contains HTML
            })
            
        finally:
            # Clean up temporary files
            os.unlink(temp_path)
            if 'wav_path' in locals():
                os.unlink(wav_path)
            
    except Exception as e:
        logger.error(f"Error processing audio query: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def text_to_speech(text: str) -> Optional[str]:
    """Convert text to speech using Ollama"""
    try:
        # Get TTS config
        tts_config = config.get("tts", {})
        provider = tts_config.get("provider", "ollama")
        
        if provider == "ollama":
            # Get Ollama TTS settings
            ollama_config = tts_config.get("ollama", {})
            model_name = ollama_config.get("model_name", "deepseek-r1:1.5b")
            api_url = ollama_config.get("api_url", "http://localhost:11434")
            
            response = requests.post(
                f"{api_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": f"Convert this text to speech: {text}",
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
            
        else:
            raise ValueError(f"Unsupported TTS provider: {provider}")
            
    except Exception as e:
        logger.error(f"Error converting text to speech: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    try:
        if not whisper_model:
            raise ValueError("Whisper model not loaded")
        result = whisper_model.transcribe(audio_path)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise

def extract_web_text(url):
    """Extract text content from a web page"""
    try:
        crawler = WebCrawler()
        result = asyncio.run(crawler.crawl_page(url))
        if not result:
            raise Exception(f"Failed to fetch content from {url}")
        return result['title'], result['content']
    except Exception as e:
        logger.error(f"Error extracting text from web page: {e}")
        raise

def split_text(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of specified size"""
    try:
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
        
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}")
        raise

@app.route('/process_web', methods=['POST'])
def process_web():
    """Process a web page and add it to the knowledge base"""
    global faiss_index  # Move global declaration to the top
    
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'})
        
        url = data['url']
        logger.info(f"Processing web page: {url}")
        
        # Crawl the page
        crawler = WebCrawler()
        page_data = asyncio.run(crawler.crawl_page(url))
        
        if not page_data:
            error_msg = "Failed to extract content"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
            
        if not page_data.get('content'):
            error_msg = "No content found in page data"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
            
        if page_data.get('title') == 'Error':
            error_msg = page_data.get('content', 'Unknown error occurred while extracting content')
            logger.error(f"Error from crawler: {error_msg}")
            return jsonify({'error': error_msg}), 400
            
        # Process the content
        logger.info(f"Processing content from URL: {url}")
        result = text_embedder.process_text(page_data['content'], config["storage"]["raw_dir"])
        
        # Generate embeddings for the content
        logger.info("Generating embeddings for content")
        chunks = text_embedder.split_text(result['content'])
        logger.info(f"Number of chunks: {len(chunks)}")
        if chunks:
            logger.info(f"First chunk length: {len(chunks[0])}")
            logger.info(f"First chunk content: {chunks[0][:200]}")
        else:
            error_msg = "No content chunks were generated"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
            
        embeddings = text_embedder.encode_texts(chunks)
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        # Prepare metadata for each chunk
        metadata_list = []
        for i, chunk in enumerate(chunks):
            metadata = {
                'page_id': result['metadata']['content_id'],
                'title': page_data['title'],
                'url': url,
                'chunk_index': i,
                'text': chunk
            }
            metadata_list.append(metadata)
        
        # Check if index exists
        if not os.path.exists(config["search"]["index_path"]):
            logger.info("Index not found, rebuilding from processed files")
            # Create new index
            faiss_index = FAISSIndex(dimension=config["embedding"]["dimension"])
            
            # Get all processed files
            processed_dir = os.path.join('data', 'processed')
            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)
            
            processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.json')]
            
            if not processed_files:
                logger.warning("No processed files found to rebuild index")
                return jsonify({'error': 'No processed files found to rebuild index'})
            
            # Process each file
            all_embeddings = []
            all_metadata = []
            
            for filename in processed_files:
                file_path = os.path.join(processed_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Generate embeddings for the content
                    file_chunks = text_embedder.split_text(data['content'])
                    file_embeddings = text_embedder.encode_texts(file_chunks)
                    
                    # Prepare metadata for each chunk
                    for i, chunk in enumerate(file_chunks):
                        metadata = {
                            'page_id': data.get('metadata', {}).get('content_id', ''),
                            'title': data.get('title', ''),
                            'url': data.get('url', ''),
                            'chunk_index': i,
                            'text': chunk
                        }
                        all_metadata.append(metadata)
                    
                    all_embeddings.append(file_embeddings)
                    logger.info(f"Processed {filename}")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    continue
            
            # Combine all embeddings
            if all_embeddings:
                combined_embeddings = np.vstack(all_embeddings)
                logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")
                logger.info(f"Total metadata items: {len(all_metadata)}")
                
                # Add to index
                faiss_index.add_embeddings(combined_embeddings, all_metadata)
                
                # Save the index
                faiss_index.save(config["search"]["index_path"], config["search"]["metadata_path"])
                logger.info(f"Index rebuilt and saved to {config['search']['index_path']}")
            else:
                logger.warning("No embeddings generated from processed files")
                return jsonify({'error': 'Failed to generate embeddings from processed files'})
        else:
            # Load existing index and add new content
            faiss_index = FAISSIndex.load(config["search"]["index_path"], config["search"]["metadata_path"])
            faiss_index.add_embeddings(embeddings, metadata_list)
            faiss_index.save(config["search"]["index_path"], config["search"]["metadata_path"])
        
        return jsonify({'message': 'Web page processed successfully'})
    except Exception as e:
        error_msg = str(e).lower()
        if "corrupted" in error_msg or "invalid" in error_msg or "pickle" in error_msg:
            # Delete corrupted files
            for file_path in [config["search"]["index_path"], config["search"]["metadata_path"]]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            error_msg = 'Search index was corrupted. Please add your web pages again to rebuild the index.'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
        logger.error(f"Error processing web page: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def rebuild_metadata():
    """Rebuild metadata from the FAISS index"""
    try:
        # Load the FAISS index
        faiss_index = FAISSIndex(config["search"]["index_path"])
        
        # Get all documents from the index
        documents = {}
        for i in range(faiss_index.index.ntotal):
            metadata = faiss_index.index.reconstruct(i)
            if isinstance(metadata, dict):
                url = metadata.get('url')
                if url and url not in documents:
                    documents[url] = {
                        'title': metadata.get('title', ''),
                        'url': url,
                        'last_updated': metadata.get('last_updated', '')
                    }
        
        # Save the rebuilt metadata
        with open(config["search"]["metadata_path"], 'wb') as f:
            pickle.dump(list(documents.values()), f)
            
        return list(documents.values())
    except Exception as e:
        logger.error(f"Error rebuilding metadata: {e}", exc_info=True)
        return []

@app.route('/indexed_documents')
def get_indexed_documents():
    """Get list of indexed documents"""
    try:
        # Get absolute path to data/processed directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        processed_dir = os.path.join(base_dir, 'data', 'processed')
        
        logger.info(f"Looking for documents in: {processed_dir}")
        
        if not os.path.exists(processed_dir):
            logger.warning(f"Directory does not exist: {processed_dir}")
            return jsonify({'documents': []})
            
        documents = []
        files = os.listdir(processed_dir)
        logger.info(f"Found {len(files)} files in processed directory")
        
        for filename in files:
            file_path = os.path.join(processed_dir, filename)
            if os.path.isfile(file_path):
                # Get file metadata
                stat = os.stat(file_path)
                last_modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                # Try to read the title from the file content
                title = filename  # Default to filename if no title found
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        logger.info(f"File content for {filename}:")
                        logger.info(f"First 100 chars: {content[:100]}")
                        
                        # Try to parse as JSON to get the title
                        try:
                            data = json.loads(content)
                            # logger.info(f"Parsed JSON data: {data}")
                            if isinstance(data, dict) and 'title' in data:
                                title = data['title']
                                logger.info(f"Found title in JSON: {title}")
                        except json.JSONDecodeError:
                            logger.info("File is not JSON format")
                            # If not JSON, try to find a title in the content
                            lines = content.split('\n')
                            for line in lines:
                                if line.strip().startswith('title:'):
                                    title = line.replace('title:', '').strip()
                                    logger.info(f"Found title in content: {title}")
                                    break
                except Exception as e:
                    logger.warning(f"Could not read title from {filename}: {e}")
                
                documents.append({
                    'title': title,
                    'url': filename,
                    'last_updated': last_modified
                })
                logger.info(f"Added document: {title}")
        
        logger.info(f"Returning {len(documents)} documents")
        return jsonify({'documents': documents})
        
    except Exception as e:
        logger.error(f"Error fetching indexed documents: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Log all registered routes
        logger.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            logger.info(f"  {rule.endpoint}: {rule.methods} {rule.rule}")
        
        # Get port from config or use default
        port = config.get("server", {}).get("port", 8000)
        host = config.get("server", {}).get("host", "0.0.0.0")
        debug = config.get("server", {}).get("debug", True)
        
        logger.info(f"Starting server on {host}:{port} (debug={debug})")
        logger.info("Server configuration:")
        logger.info(f"  Host: {host}")
        logger.info(f"  Port: {port}")
        logger.info(f"  Debug mode: {debug}")
        logger.info(f"  Log level: {config.get('server', {}).get('log_level', 'info')}")
        
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise 