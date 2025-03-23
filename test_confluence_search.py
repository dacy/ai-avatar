import requests
import logging
import os
import sys
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import time
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def wait_for_server(timeout=30, interval=2):
    """Wait for server to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get('http://localhost:8000/status')
            if response.ok:
                logging.info("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            logging.info("Waiting for server to start...")
        time.sleep(interval)
    return False

def clean_html_content(html_content: str) -> str:
    """Clean HTML content by removing scripts, styles, and formatting."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text content
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

async def get_web_content(url: str) -> str:
    """Get web content using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        content = await page.content()
        await browser.close()
        return clean_html_content(content)

async def test_confluence_search():
    """Test the Confluence search functionality"""
    try:
        # Use a simple test URL
        test_url = "https://www.atlassian.com/work-management/knowledge-sharing/documentation"
        test_query = 'what does coworker do?'
        logging.info("Starting test...")
        logging.info(f"Processing test page: {test_url}")
        
        # Get web content
        logging.info("Getting web content...")
        content = await get_web_content(test_url)
        if not content:
            raise Exception("Failed to scrape web content")
            
        # Process the page
        logging.info("Processing page...")
        logging.info(f"Making POST request to /process_confluence with content length: {len(content)}")
        response = requests.post(
            'http://localhost:8000/process_confluence',
            json={'url': test_url, 'content': content}
        )
        logging.info(f"Response status: {response.status_code}")
        response.raise_for_status()
        
        # Wait for index to be created
        logging.info("Waiting for index to be created...")
        time.sleep(2)
        
        # Test search with a specific query about the example.com page
        logging.info("Testing search...")
        logging.info(f"Making POST request to /search with query: {test_query}")
        search_response = requests.post(
            'http://localhost:8000/search',
            json={'query': test_query}
        )
        logging.info(f"Search response status: {search_response.status_code}")
        search_response.raise_for_status()
        
        # Print the search results
        results = search_response.json()
        logging.info("Search results:")
        logging.info(f"Answer: {results.get('answer', 'No answer found')}")
        logging.info(f"Sources: {results.get('sources', [])}")
        
        logging.info("Test completed successfully!")
        
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        raise

def remove_existing_index():
    """Remove existing index and metadata files"""
    try:
        # Get the data directory from config
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
        index_path = os.path.join(data_dir, "vectors", "voice_search_index.faiss")
        metadata_path = os.path.join(data_dir, "vectors", "voice_search_index.metadata")
        
        # Remove files if they exist
        for file_path in [index_path, metadata_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Removed existing file: {file_path}")
            else:
                logging.info(f"File does not exist: {file_path}")
                
        return True
    except Exception as e:
        logging.error(f"Error removing existing index: {e}")
        return False

def test_embedding_and_indexing():
    """Test embedding and indexing creation"""
    try:
        # Use a test URL
        test_url = "https://www.atlassian.com/work-management/knowledge-sharing/documentation"
        logging.info(f"Testing with URL: {test_url}")
        
        # Send request to process the page
        response = requests.post(
            'http://localhost:8000/process_confluence',
            json={'url': test_url}
        )
        
        # Check response
        if response.ok:
            logging.info("Successfully processed Confluence page")
            logging.info(f"Server response: {response.json()}")
        else:
            logging.error(f"Failed to process page: {response.status_code}")
            logging.error(f"Error message: {response.json()}")
            return False
            
        # Check if index and metadata files were created
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
        index_path = os.path.join(data_dir, "vectors", "voice_search_index.faiss")
        metadata_path = os.path.join(data_dir, "vectors", "voice_search_index.metadata")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            logging.info("Index and metadata files created successfully")
            logging.info(f"Index file size: {os.path.getsize(index_path)} bytes")
            logging.info(f"Metadata file size: {os.path.getsize(metadata_path)} bytes")
            return True
        else:
            logging.error("Index or metadata files were not created")
            return False
            
    except Exception as e:
        logging.error(f"Error in embedding and indexing test: {e}")
        return False

def test_text_search():
    """Test text input search"""
    try:
        # Test query - more specific to the content
        test_query = "What are the benefits of knowledge sharing in the workplace?"
        logging.info(f"Testing search with query: {test_query}")
        
        # Send search request
        response = requests.post(
            'http://localhost:8000/search',
            json={'query': test_query}
        )
        
        # Check response
        if response.ok:
            result = response.json()
            logging.info("Search successful!")
            logging.info(f"Answer: {result.get('answer', 'No answer found')}")
            logging.info(f"Confidence: {result.get('confidence', 'N/A')}")
            logging.info("Sources:")
            for source in result.get('sources', []):
                logging.info(f"  - {source.get('url', 'N/A')} (Score: {source.get('score', 'N/A')})")
            return True
        else:
            logging.error(f"Search failed: {response.status_code}")
            logging.error(f"Error message: {response.json()}")
            return False
            
    except Exception as e:
        logging.error(f"Error in text search test: {e}")
        return False

def main():
    """Main test function"""
    try:
        # Wait for server to be ready
        logging.info("Waiting for server to start...")
        if not wait_for_server():
            logging.error("Server failed to start within timeout")
            return False
            
        # Remove existing index
        logging.info("Removing existing index...")
        if not remove_existing_index():
            logging.error("Failed to remove existing index")
            return False
            
        # Test embedding and indexing
        logging.info("\nTesting embedding and indexing creation...")
        if not test_embedding_and_indexing():
            logging.error("Embedding and indexing test failed")
            return False
            
        # Test text search
        logging.info("\nTesting text search...")
        if not test_text_search():
            logging.error("Text search test failed")
            return False
            
        logging.info("\nAll tests completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 