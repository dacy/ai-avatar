#!/usr/bin/env python3
"""
Confluence page crawler to extract text content.
"""

import requests
from bs4 import BeautifulSoup
from loguru import logger

class ConfluenceCrawler:
    """
    Crawler to extract text content from Confluence pages.
    """
    
    def __init__(self):
        """Initialize the crawler."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_text(self, url: str) -> str:
        """
        Extract text content from a Confluence page.
        
        Args:
            url: URL of the Confluence page
            
        Returns:
            Extracted text content
        """
        try:
            # Fetch the page
            logger.info(f"Fetching page: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()
            
            # Extract text from main content area
            # This selector might need to be adjusted based on the specific Confluence page structure
            main_content = soup.find('main') or soup.find('div', {'id': 'main-content'}) or soup.find('article')
            
            if main_content:
                # Get text from main content
                text = main_content.get_text(separator=' ', strip=True)
            else:
                # Fallback to body text if main content not found
                text = soup.body.get_text(separator=' ', strip=True)
            
            # Clean up the text
            text = ' '.join(text.split())  # Remove extra whitespace
            
            logger.info(f"Successfully extracted {len(text)} characters")
            return text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch page: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return "" 