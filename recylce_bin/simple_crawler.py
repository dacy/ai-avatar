#!/usr/bin/env python3
"""
Simplified Confluence page crawler for extracting content.
"""

import logging
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
import re
import os
from urllib.parse import urlparse

class SimpleConfluenceCrawler:
    """Crawler for extracting content from Confluence pages."""
    
    def __init__(self, base_url: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the crawler.
        
        Args:
            base_url: Base URL of the Confluence instance (optional)
            username: Confluence username (optional)
            password: Confluence password or API token (optional)
        """
        self.base_url = base_url
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)
        
    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a Confluence page.
        
        Args:
            url: URL of the Confluence page
            
        Returns:
            Dictionary containing:
            - title: Page title
            - content: Main content text
            - metadata: Additional metadata (author, last modified, etc.)
        """
        try:
            # Set up authentication if credentials are provided
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)
            
            self.logger.info(f"Making request to {url}")
            
            # Make request to the page with a user agent to avoid blocks
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, auth=auth, headers=headers)
            response.raise_for_status()
            
            self.logger.info(f"Response status code: {response.status_code}")
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            self.logger.info(f"Extracted title: {title}")
            
            # Extract main content
            content = self._extract_main_content(soup)
            self.logger.info(f"Extracted content length: {len(content)}")
            
            # Extract metadata
            metadata = self._extract_metadata(soup)
            self.logger.info(f"Extracted metadata: {metadata}")
            
            return {
                'title': title,
                'content': content,
                'metadata': metadata,
                'url': url
            }
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                self.logger.error("Authentication failed. Please check your credentials.")
            elif e.response.status_code == 403:
                self.logger.error("Access denied. Please check your permissions.")
            else:
                self.logger.error(f"HTTP error occurred: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            raise
            
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the page title."""
        # Try multiple common title elements
        title_elem = (
            soup.find('h1', {'id': 'title-text'}) or
            soup.find('h1', {'class': 'title'}) or
            soup.find('h1') or
            soup.find('title')
        )
        
        if title_elem:
            return title_elem.get_text().strip()
        else:
            self.logger.warning("Could not find page title, using 'Untitled Page'")
            return "Untitled Page"
        
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content from the page."""
        # Try to find the main content container - ordered by specificity
        content_candidates = [
            # Confluence specific
            soup.find('div', {'id': 'main-content'}),
            soup.find('div', {'class': 'wiki-content'}),
            
            # Common content containers
            soup.find('article'),
            soup.find('main'),
            soup.find('div', {'id': 'content'}),
            soup.find('div', {'class': 'content'}),
            
            # More generic containers
            soup.find('div', {'role': 'main'}),
            soup.find('div', {'class': re.compile(r'(article|post|entry|main|content)')}),
            
            # Last resort - the body itself
            soup.find('body')
        ]
        
        # Use the first non-None candidate
        content_elem = next((elem for elem in content_candidates if elem is not None), None)
        
        if not content_elem:
            self.logger.warning("Could not find main content container")
            return ""
        
        # Log which container was found
        self.logger.info(f"Using content container: {content_elem.name}{' id=' + content_elem.get('id') if content_elem.get('id') else ''}{' class=' + str(content_elem.get('class')) if content_elem.get('class') else ''}")
            
        # Remove unwanted elements
        for unwanted in content_elem.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            unwanted.decompose()
        
        # Remove ads and other non-content elements
        for unwanted in content_elem.find_all(class_=re.compile(r'(ads?|banner|promo|sidebar|navigation|nav|menu|cookie|popup)')):
            unwanted.decompose()
            
        # Extract text
        text = content_elem.get_text(separator=' ', strip=True)
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Fix multiple newlines
        
        # If the content is too short, try to get more from p tags
        if len(text) < 100:
            self.logger.warning(f"Extracted text is short ({len(text)} chars), trying to extract from paragraphs")
            paragraphs = soup.find_all('p')
            if paragraphs:
                p_text = "\n\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                if len(p_text) > len(text):
                    self.logger.info(f"Using paragraph text instead ({len(p_text)} chars)")
                    text = p_text
        
        return text.strip()
        
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from the page."""
        metadata = {}
        
        # Try to find author - multiple possible locations
        author_candidates = [
            soup.find('a', {'class': 'url fn'}),  # Confluence style
            soup.find('meta', {'name': 'author'}),  # HTML meta
            soup.find(attrs={'class': re.compile(r'author|byline')}),  # Common author classes
        ]
        
        for author_elem in author_candidates:
            if author_elem:
                if author_elem.name == 'meta':
                    metadata['author'] = author_elem.get('content', '')
                else:
                    metadata['author'] = author_elem.get_text().strip()
                break
            
        # Try to find last modified date
        date_candidates = [
            soup.find('time', {'class': 'date'}),  # Confluence style 
            soup.find('meta', {'property': 'article:modified_time'}),  # Open Graph
            soup.find('meta', {'name': 'last-modified'}),  # HTML meta
            soup.find(attrs={'class': re.compile(r'date|time|modified|published')})  # Common date classes
        ]
        
        for date_elem in date_candidates:
            if date_elem:
                if date_elem.name == 'meta':
                    metadata['last_modified'] = date_elem.get('content', '')
                elif date_elem.get('datetime'):
                    metadata['last_modified'] = date_elem.get('datetime', '')
                else:
                    metadata['last_modified'] = date_elem.get_text().strip()
                break
            
        # Try to find labels/tags
        tag_candidates = [
            soup.find_all('a', {'class': 'label'}),  # Confluence style
            soup.find_all('meta', {'property': 'article:tag'}),  # Open Graph
            soup.find_all(attrs={'class': re.compile(r'tag|category|label')}),  # Common tag classes
        ]
        
        for tags in tag_candidates:
            if tags:
                if tags[0].name == 'meta':
                    metadata['tags'] = [tag.get('content', '') for tag in tags]
                else:
                    metadata['tags'] = [tag.get_text().strip() for tag in tags]
                break
                
        # Add page description if available
        desc_elem = soup.find('meta', {'name': 'description'}) or soup.find('meta', {'property': 'og:description'})
        if desc_elem:
            metadata['description'] = desc_elem.get('content', '')
            
        return metadata
        
    def save_to_file(self, content: Dict[str, Any], output_dir: str) -> str:
        """
        Save extracted content to a file.
        
        Args:
            content: Dictionary containing extracted content
            output_dir: Directory to save the file
            
        Returns:
            Path to the saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename from title
        filename = re.sub(r'[^\w\s-]', '', content['title'].lower())
        filename = re.sub(r'[-\s]+', '_', filename)
        filepath = os.path.join(output_dir, f"{filename}.txt")
        
        # Format content
        text = f"Title: {content['title']}\n\n"
        text += f"URL: {content['url']}\n\n"
        
        if content['metadata']:
            text += "Metadata:\n"
            for key, value in content['metadata'].items():
                text += f"- {key}: {value}\n"
            text += "\n"
            
        text += "Content:\n"
        text += content['content']
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        self.logger.info(f"Saved content to {filepath}")
        return filepath 