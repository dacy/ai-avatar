import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from playwright.async_api import async_playwright, BrowserType
import html5lib
from urllib.parse import urlparse
import json
import os
from pathlib import Path
import yaml
import re
import random

# Initialize logger
logger = logging.getLogger(__name__)

# Configure logging
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class WebCrawler:
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
        
        # Log browser configuration
        browser_config = self.config.get("browser", {})
        logger.info("Browser configuration loaded:")
        logger.info(f"  User data directory: {browser_config.get('user_data_dir', 'Not set')}")
        logger.info(f"  Headless: {browser_config.get('headless', True)}")
        
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
        
    async def _fetch_page(self, page: Any, url: str) -> str:
        """Fetch and extract content from a page."""
        try:

            
            # Navigate to the page
            logger.info(f"Fetching {url}")
            await page.goto(url)
            await page.wait_for_load_state('networkidle')
            # If browser is not headless, wait for user input before starting
            if not self.config.get("browser", {}).get("headless", True):
                input("Please log in to the browser. Press Enter when you're done...")
            # Get the page content
            content = await page.content()
            logger.info(f"Successfully fetched {url}")
            
            return content
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return ""
            
    def _parse_html(self, html_content: Optional[str]) -> tuple[str, str, Dict[str, Any]]:
        """Parse HTML content and extract structured content."""
        if not html_content:
            return "Untitled", "", {"title": "Untitled", "content": []}
        
        try:
            document = html5lib.parse(html_content)
            result = []
            seen_texts = set()  # To avoid duplicates
            title = "Untitled"
            
            def should_skip_element(elem) -> bool:
                """Check if the element should be skipped."""
                # Skip script and style tags
                if elem.tag in ['{http://www.w3.org/1999/xhtml}script', 
                              '{http://www.w3.org/1999/xhtml}style']:
                    return True
                # Skip empty elements or elements with only whitespace
                if not any(text.strip() for text in elem.itertext()):
                    return True
                return False
            
            def process_element(elem, depth=0):
                """Process an element and its children recursively."""
                nonlocal title
                
                if should_skip_element(elem):
                    return
                
                # Try to find the title
                if title == "Untitled":
                    # Check meta title first
                    meta_title = document.find('.//{http://www.w3.org/1999/xhtml}meta[@property="og:title"]')
                    if meta_title is not None:
                        for attr, value in meta_title.items():
                            if attr.endswith('content'):
                                title = value.strip()
                                break
                    
                    # If no meta title, check h1
                    if title == "Untitled":
                        h1 = document.find('.//{http://www.w3.org/1999/xhtml}h1')
                        if h1 is not None and h1.text:
                            title = h1.text.strip()
                
                # Handle text content
                if hasattr(elem, 'text') and elem.text:
                    text = elem.text.strip()
                    if text and text not in seen_texts:
                        # Check if this is an anchor tag
                        if elem.tag == '{http://www.w3.org/1999/xhtml}a':
                            href = None
                            for attr, value in elem.items():
                                if attr.endswith('href'):
                                    href = value
                                    break
                            if href and not href.startswith(('#', 'javascript:')):
                                # Format as markdown link
                                link_text = f"[{text}]({href})"
                                result.append(link_text)
                                seen_texts.add(text)
                        else:
                            result.append(text)
                            seen_texts.add(text)
                
                # Process children
                for child in elem:
                    process_element(child, depth + 1)
                
                # Handle tail text
                if hasattr(elem, 'tail') and elem.tail:
                    tail = elem.tail.strip()
                    if tail and tail not in seen_texts:
                        result.append(tail)
                        seen_texts.add(tail)
            
            # Start processing from the body tag
            body = document.find('.//{http://www.w3.org/1999/xhtml}body')
            if body is not None:
                process_element(body)
            else:
                # Fallback to processing the entire document
                process_element(document)
            
            # Filter out common unwanted patterns
            filtered_result = []
            for line in result:
                # Skip lines that are likely to be noise
                if any(pattern in line.lower() for pattern in [
                    'var ', 
                    'function()', 
                    '.js',
                    '.css',
                    'google-analytics',
                    'disqus',
                    '{',
                    '}'
                ]):
                    continue
                filtered_result.append(line)
            
            # Join lines with proper spacing
            flattened_text = ' '.join(filtered_result)
            
            # Clean up extra whitespace
            flattened_text = re.sub(r'\s+', ' ', flattened_text).strip()
            return title, flattened_text
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            return "Untitled", "", {"title": "Untitled", "content": []}
            
    async def _get_browser_context(self, playwright) -> Any:
        """Get a browser context with if configured."""
        browser_config = self.config.get("browser", {})
        user_data_dir = browser_config.get("user_data_dir", "")
        headless = browser_config.get("headless", False)

        # Log configuration
        logger.info("Browser context configuration:")
        logger.info(f"  User data directory: {user_data_dir}")
        logger.info(f"  Headless: {headless}")
        logger.info(f"  User data directory exists: {os.path.exists(user_data_dir) if user_data_dir else False}")

        # Common browser arguments
        browser_args = [
            '--disable-web-security',  # Disable web security for testing
            '--no-sandbox',  # Required for some environments
            '--disable-setuid-sandbox',  # Required for some environments
            '--disable-dev-shm-usage',  # Overcome limited resource problems
            '--disable-accelerated-2d-canvas',  # Disable GPU hardware acceleration
            '--disable-gpu',  # Disable GPU hardware acceleration
            '--window-size=1920,1080',  # Set a standard window size
            '--disable-blink-features=AutomationControlled',  # Hide automation
            '--disable-features=IsolateOrigins,site-per-process',  # Disable site isolation
            '--disable-site-isolation-trials',  # Disable site isolation trials
            '--disable-features=IsolateOrigins,site-per-process,AutomationControlled',  # Additional anti-bot flags
            '--disable-blink-features',  # Disable automation flags
            '--disable-features=AutomationControlled',  # Disable automation control
            '--disable-web-security',  # Disable web security
            '--disable-features=IsolateOrigins,site-per-process,AutomationControlled,WebSecurity',  # Comprehensive flags
            '--remote-debugging-port=9222',  # Enable remote debugging
        ]

        # Common user agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'

        if user_data_dir and os.path.exists(user_data_dir):
            # Try Chrome first
            try:
                logger.info(f"Browser arguments: {browser_args}")
                
                # Launch browser with persistent context
                context = await playwright.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=headless,
                    args=browser_args,
                    java_script_enabled=True,  # Explicitly enable JavaScript
                    bypass_csp=True,  # Bypass Content Security Policy
                    user_agent=user_agent,  # Set user agent
                    viewport={'width': 1920, 'height': 1080},  # Set viewport
                    extra_http_headers={  # Add common headers
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Cache-Control': 'max-age=0',
                        'DNT': '1',  # Do Not Track
                        'Sec-Ch-Ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
                        'Sec-Ch-Ua-Mobile': '?0',
                        'Sec-Ch-Ua-Platform': '"Windows"'
                    }
                )
                
                return context
            except Exception as e:
                logger.error(f"Failed to launch Chrome with persistent context: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error details: {str(e)}")
                # Fall back to default browser
                logger.info("Falling back to default browser without persistent context")
                browser = await playwright.chromium.launch(headless=headless, args=browser_args)
                context = await browser.new_context(
                    java_script_enabled=True,
                    bypass_csp=True,
                    user_agent=user_agent,
                    viewport={'width': 1920, 'height': 1080},
                    extra_http_headers={
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Cache-Control': 'max-age=0',
                        'DNT': '1',  # Do Not Track
                        'Sec-Ch-Ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
                        'Sec-Ch-Ua-Mobile': '?0',
                        'Sec-Ch-Ua-Platform': '"Windows"'
                    }
                )
                return context
        else:
            # Use default browser without persistent context
            logger.info("No user data directory found or directory does not exist, using regular launch")
            browser = await playwright.chromium.launch(headless=headless, args=browser_args)
            context = await browser.new_context(
                java_script_enabled=True,
                bypass_csp=True,
                user_agent=user_agent,
                viewport={'width': 1920, 'height': 1080},
                extra_http_headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Cache-Control': 'max-age=0',
                    'DNT': '1',  # Do Not Track
                    'Sec-Ch-Ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
                    'Sec-Ch-Ua-Mobile': '?0',
                    'Sec-Ch-Ua-Platform': '"Windows"'
                }
            )
            return context

    async def crawl_page(self, url: str, context: Any = None) -> Dict[str, Any]:
        """Crawl a single web page and extract its content.
        
        Args:
            url: The URL of the web page to crawl.
            context: Optional browser context to use. If None, a new context will be created.
            
        Returns:
            A dictionary containing:
            - title: The page title
            - content: The page content in markdown format
            - url: The normalized URL
            - metadata: Additional metadata about the page
            - structure: The structured content of the page
        """
        url = self._normalize_url(url)
        
        # If no context provided, create one
        should_close_context = context is None
        playwright = None
        if context is None:
            playwright = await async_playwright().start()
            context = await self._get_browser_context(playwright)
        
        try:
            # Create a new page from the context
            page = await context.new_page()
            try:
                # Add random delay between requests (1-3 seconds)
                await asyncio.sleep(1 + random.random() * 2)
                
                # Add random mouse movements to appear more human-like
                await page.mouse.move(
                    random.randint(0, 500),
                    random.randint(0, 500)
                )
                
                # Add random viewport size variation
                await page.set_viewport_size({
                    'width': 1920 + random.randint(-100, 100),
                    'height': 1080 + random.randint(-100, 100)
                })
                
                # Add random user agent variation
                await page.set_extra_http_headers({
                    'User-Agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.{random.randint(0, 9999)} Safari/537.36'
                })
                
                # Fetch the page content
                html_content = await self._fetch_page(page, url)
                
                if not html_content:
                    error_msg = f"No content extracted from page: {url}"
                    logger.error(error_msg)
                    return {
                        'title': 'Error',
                        'content': error_msg,
                        'url': url,
                        'metadata': {
                            'url': url,
                            'error': error_msg,
                            'content_length': 0,
                            'has_links': False,
                            'structure_type': 'error',
                            'content_id': self._get_page_id(url)
                        }
                    }
                
                # Parse the HTML content
                title, content = self._parse_html(html_content)
                
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
                    'metadata': metadata
                }
                
                # Save the data
                self._save_page_data(page_data, url)
                
                return page_data
                
            except Exception as e:
                error_msg = f"Error processing page {url}: {str(e)}"
                logger.error(error_msg)
                return {
                    'title': 'Error',
                    'content': error_msg,
                    'url': url,
                    'metadata': {
                        'url': url,
                        'error': error_msg,
                        'content_length': 0,
                        'has_links': False,
                        'structure_type': 'error',
                        'content_id': self._get_page_id(url)
                    }
                }
            finally:
                # Always close the page
                await page.close()
                
        finally:
            # Close context and playwright if we created them
            if should_close_context:
                try:
                    await context.close()
                except Exception as e:
                    logger.warning(f"Error closing context: {e}")
                if playwright:
                    try:
                        await playwright.stop()
                    except Exception as e:
                        logger.warning(f"Error stopping playwright: {e}")
                    
   
                
