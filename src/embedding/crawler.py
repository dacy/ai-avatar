import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from playwright.async_api import async_playwright
import html5lib
from urllib.parse import urlparse
import json
import os
from pathlib import Path
import yaml
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfluenceCrawler:
    """A general-purpose web crawler that converts web pages into structured JSON format."""
    
    def __init__(self, base_url: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize the crawler.
        
        Args:
            base_url: Optional base URL for relative paths.
            config_path: Path to the config.yaml file.
        """
        self.base_url = base_url.rstrip('/') if base_url else None
        
        # Load config
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     "config", "config.yaml")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create necessary directories
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories for storing data."""
        # Create all required directories
        for dir_path in [
            self.config["storage"]["data_dir"],
            self.config["storage"]["raw_dir"],
            self.config["storage"]["processed_dir"],
            self.config["storage"]["vector_dir"]
        ]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
            
    def _get_page_id(self, url: str) -> str:
        """Extract a unique page ID from the URL."""
        parsed = urlparse(url)
        # Use the full path as the ID to ensure uniqueness
        path = parsed.path.strip('/')
        if not path:
            path = 'index'
        # Replace special characters with underscores
        page_id = re.sub(r'[^a-zA-Z0-9_-]', '_', path)
        return page_id
        
    def _save_page_data(self, page_data: Dict[str, Any], url: str):
        """Save page data to appropriate directories."""
        page_id = self._get_page_id(url)
        
        # Save raw HTML
        raw_path = os.path.join(self.config["storage"]["raw_dir"], f"{page_id}.html")
        if "raw_html" in page_data:
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(page_data["raw_html"])
            logger.info(f"Saved raw HTML to {raw_path}")
        
        # Save processed data (JSON structure)
        processed_path = os.path.join(self.config["storage"]["processed_dir"], f"{page_id}.json")
        with open(processed_path, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data to {processed_path}")
        
    def _normalize_url(self, url: str) -> str:
        """Normalize the URL by adding the base URL if needed."""
        if self.base_url and not url.startswith(('http://', 'https://')):
            return f"{self.base_url}/{url.lstrip('/')}"
        return url
        
    async def _fetch_page(self, url: str, context) -> Optional[str]:
        """Asynchronously fetch a web page's content."""
        page = await context.new_page()
        try:
            logger.info(f"Fetching page: {url}")
            await page.goto(url)
            await page.wait_for_load_state('networkidle')
            content = await page.content()
            logger.info(f"Successfully fetched {url}")
            return content
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
        finally:
            await page.close()
            
    def _parse_html(self, html_content: Optional[str]) -> tuple[str, str, Dict[str, Any]]:
        """Parse HTML content and extract structured content."""
        if not html_content:
            return "Untitled", "", {"title": "Untitled", "content": []}
        
        try:
            document = html5lib.parse(html_content)
            result = []
            seen_texts = set()  # To avoid duplicates
            title = "Untitled"
            structure = {"title": "Untitled", "content": []}
            current_section = None
            
            def should_skip_element(elem) -> bool:
                """Check if the element should be skipped."""
                # Skip script and style tags
                if elem.tag in ['{http://www.w3.org/1999/xhtml}script', 
                              '{http://www.w3.org/1999/xhtml}style',
                              '{http://www.w3.org/1999/xhtml}noscript',
                              '{http://www.w3.org/1999/xhtml}iframe']:
                    return True
                # Skip empty elements or elements with only whitespace
                if not any(text.strip() for text in elem.itertext()):
                    return True
                # Skip navigation and footer elements
                if any(attr.endswith('role') and value in ['navigation', 'banner', 'complementary'] 
                      for attr, value in elem.items()):
                    return True
                return False
            
            def process_element(elem, depth=0):
                """Process an element and its children recursively."""
                nonlocal title, current_section
                
                if should_skip_element(elem):
                    return
                
                # Try to find the title
                if not title or title == "Untitled":
                    # Check meta title first
                    meta_title = document.find('.//{http://www.w3.org/1999/xhtml}meta[@property="og:title"]')
                    if meta_title is not None:
                        for attr, value in meta_title.items():
                            if attr.endswith('content'):
                                title = value.strip()
                                structure["title"] = title
                                break
                    
                    # If no meta title, check h1
                    if title == "Untitled":
                        h1 = document.find('.//{http://www.w3.org/1999/xhtml}h1')
                        if h1 is not None and h1.text:
                            title = h1.text.strip()
                            structure["title"] = title
                
                # Handle headings
                if elem.tag in ['{http://www.w3.org/1999/xhtml}h1', '{http://www.w3.org/1999/xhtml}h2', 
                              '{http://www.w3.org/1999/xhtml}h3', '{http://www.w3.org/1999/xhtml}h4']:
                    text = elem.text.strip() if elem.text else ""
                    if text:
                        # Create new section
                        current_section = {
                            "type": "section",
                            "heading": text,
                            "content": [],
                            "level": int(elem.tag[-1])
                        }
                        structure["content"].append(current_section)
                        # Add heading to flattened text with proper markdown formatting
                        result.append("  " * (depth-1) + "#" * int(elem.tag[-1]) + f" {text}")
                        seen_texts.add(text)
                
                # Handle paragraphs
                elif elem.tag == '{http://www.w3.org/1999/xhtml}p':
                    text = ' '.join(t.strip() for t in elem.itertext() if t.strip())
                    if text and text not in seen_texts:
                        # Add paragraph to structure
                        para_section = {
                            "type": "paragraph",
                            "content": text
                        }
                        if current_section:
                            current_section["content"].append(para_section)
                        else:
                            structure["content"].append(para_section)
                        # Add to flattened text
                        result.append(text)
                        seen_texts.add(text)
                
                # Handle lists
                elif elem.tag in ['{http://www.w3.org/1999/xhtml}ul', '{http://www.w3.org/1999/xhtml}ol']:
                    list_items = []
                    for item in elem.findall('.//{http://www.w3.org/1999/xhtml}li'):
                        item_text = ' '.join(t.strip() for t in item.itertext() if t.strip())
                        if item_text and item_text not in seen_texts:
                            list_items.append(item_text)
                            # Add list item to flattened text
                            prefix = "1. " if elem.tag == '{http://www.w3.org/1999/xhtml}ol' else "- "
                            result.append("  " * depth + prefix + item_text)
                            seen_texts.add(item_text)
                    
                    if list_items:
                        list_section = {
                            "type": "list",
                            "items": list_items,
                            "ordered": elem.tag == '{http://www.w3.org/1999/xhtml}ol'
                        }
                        if current_section:
                            current_section["content"].append(list_section)
                        else:
                            structure["content"].append(list_section)
                
                # Handle links
                elif elem.tag == '{http://www.w3.org/1999/xhtml}a':
                    text = ' '.join(t.strip() for t in elem.itertext() if t.strip())
                    if text and text not in seen_texts:
                        href = None
                        for attr, value in elem.items():
                            if attr.endswith('href'):
                                href = value
                                break
                        if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                            # Add link to structure
                            link_section = {
                                "type": "link",
                                "text": text,
                                "url": href
                            }
                            if current_section:
                                current_section["content"].append(link_section)
                            else:
                                structure["content"].append(link_section)
                            # Add to flattened text
                            result.append(f"[{text}]({href})")
                            seen_texts.add(text)
                
                # Process children
                for child in elem:
                    process_element(child, depth + 1)
            
            # Start processing from the body tag
            body = document.find('.//{http://www.w3.org/1999/xhtml}body')
            if body is not None:
                process_element(body)
            else:
                # Fallback to processing the entire document
                process_element(document)
            
            # Join lines with proper spacing
            flattened_text = '\n'.join(result)
            
            # Clean up extra whitespace while preserving structure
            flattened_text = re.sub(r'\n\s*\n', '\n\n', flattened_text)
            
            return title, flattened_text, structure
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            return "Untitled", "", {"title": "Untitled", "content": []}
            
    async def crawl_page(self, url: str) -> Dict[str, Any]:
        """Crawl a single web page and extract its content.
        
        Args:
            url: The URL of the web page to crawl.
            
        Returns:
            A dictionary containing:
            - title: The page title
            - content: The page content in markdown format
            - url: The normalized URL
            - metadata: Additional metadata about the page
            - structure: The structured content of the page
        """
        url = self._normalize_url(url)
        
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                context = await browser.new_context()
                html_content = await self._fetch_page(url, context)
                
                if not html_content:
                    raise Exception(f"Failed to fetch content from {url}")
                
                title, content, structure = self._parse_html(html_content)
                
                # Extract additional metadata
                metadata = {
                    'url': url,
                    'content_length': len(content),
                    'has_links': '[' in content and '](' in content,
                    'structure_type': 'json',
                    'content_id': self._get_page_id(url)
                }
                
                # Prepare page data
                page_data = {
                    'title': title,
                    'content': content,
                    'url': url,
                    'metadata': metadata,
                    'structure': structure
                }
                
                # Save the data
                self._save_page_data(page_data, url)
                
                return page_data
                
            finally:
                await browser.close()
                
    async def crawl_pages(self, urls: list[str], max_concurrent: int = 5) -> list[Dict[str, Any]]:
        """Crawl multiple web pages concurrently.
        
        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum number of concurrent browser instances
            
        Returns:
            List of dictionaries containing the crawled content for each page
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                # Create browser contexts
                n_contexts = min(len(urls), max_concurrent)
                contexts = [await browser.new_context() for _ in range(n_contexts)]
                
                # Create tasks for each URL
                tasks = []
                for i, url in enumerate(urls):
                    context = contexts[i % len(contexts)]
                    task = self.crawl_page(url)
                    tasks.append(task)
                
                # Gather results
                results = await asyncio.gather(*tasks)
                return results
                
            finally:
                await browser.close()
                
    def crawl(self, urls: list[str], max_concurrent: int = 5) -> list[Dict[str, Any]]:
        """Synchronous wrapper for crawling multiple pages.
        
        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum number of concurrent browser instances
            
        Returns:
            List of dictionaries containing the crawled content for each page
        """
        return asyncio.run(self.crawl_pages(urls, max_concurrent)) 