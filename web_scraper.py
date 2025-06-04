import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import re
from urllib.parse import urljoin, urlparse
import time

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_url(self, url: str) -> dict:
        """
        Scrape content from URL and convert to markdown
        
        Args:
            url (str): URL to scrape
            
        Returns:
            dict: Contains 'content', 'title', 'url', 'status'
        """
        try:
            # Validate URL
            if not self._is_valid_url(url):
                return {
                    'content': '',
                    'title': '',
                    'url': url,
                    'status': 'error',
                    'message': 'Invalid URL format'
                }
            
            print(f"Fetching URL: {url}")
            
            # Make request
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            print(f"Response status: {response.status_code}")
            print(f"Content length: {len(response.content)} bytes")
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            print(f"Extracted title: {title}")
            
            # Remove unwanted elements (less aggressive)
            self._clean_soup(soup)
            
            # Extract main content based on site type
            main_content = self._extract_main_content(soup, url)
            
            if not main_content:
                print("No main content found, using body")
                main_content = soup.find('body') or soup
            
            # Convert to markdown
            markdown_content = self._html_to_markdown(main_content)
            
            # Clean markdown
            cleaned_content = self._clean_markdown(markdown_content)
            
            print(f"Final content length: {len(cleaned_content)} characters")
            print(f"Content preview: {cleaned_content[:200]}...")
            
            if not cleaned_content.strip():
                return {
                    'content': '',
                    'title': title,
                    'url': url,
                    'status': 'error',
                    'message': 'No meaningful content could be extracted from the page'
                }
            
            return {
                'content': cleaned_content,
                'title': title,
                'url': url,
                'status': 'success',
                'message': 'Content scraped successfully'
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return {
                'content': '',
                'title': '',
                'url': url,
                'status': 'error',
                'message': f'Request failed: {str(e)}'
            }
        except Exception as e:
            print(f"General error: {e}")
            return {
                'content': '',
                'title': '',
                'url': url,
                'status': 'error',
                'message': f'Scraping failed: {str(e)}'
            }
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
            # Clean Wikipedia titles
            if 'Wikipedia' in title:
                title = title.replace(' - Wikipedia', '').replace(' — Wikipedia', '')
            return title
        
        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return "Untitled"
    
    def _clean_soup(self, soup: BeautifulSoup):
        """Remove unwanted HTML elements (less aggressive)"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Remove navigation, but be more selective
        for nav in soup.find_all("nav"):
            nav.decompose()
        
        # Remove specific unwanted elements but preserve main content
        unwanted_selectors = [
            '[class*="advertisement"]',
            '[class*="ad-"]',
            '[id*="ad"]',
            '.social-share',
            '.cookie-notice',
            '.popup'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
    
    def _extract_main_content(self, soup: BeautifulSoup, url: str):
        """Extract main content based on site structure"""
        
        # Wikipedia-specific content extraction
        if 'wikipedia.org' in url:
            # Wikipedia main content is in div with id="mw-content-text"
            wiki_content = soup.find('div', id='mw-content-text')
            if wiki_content:
                print("Found Wikipedia main content area")
                return wiki_content
        
        # Try common content selectors in order of preference
        content_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.main-content',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '#content',
            '#main-content',
            '.container .content'
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                print(f"Found content using selector: {selector}")
                return content
        
        # If no specific content area found, try to find the largest text block
        body = soup.find('body')
        if body:
            print("Using body as fallback")
            return body
            
        return soup
    
    def _html_to_markdown(self, element) -> str:
        """Convert HTML to markdown"""
        try:
            # Convert with more permissive settings
            markdown = md(
                str(element), 
                heading_style="ATX",
                bullets="-",
                strong_em_style="*",
                strip=['a', 'img'],  # Remove links and images but keep text
                convert=['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'ul', 'ol', 'blockquote', 'div', 'span']
            )
            return markdown
        except Exception as e:
            print(f"Markdown conversion error: {e}")
            # Fallback to plain text extraction
            return element.get_text(separator='\n', strip=True)
    
    def _clean_markdown(self, content: str) -> str:
        """Clean and format markdown content (less aggressive)"""
        if not content:
            return ""
        
        # Normalize line breaks
        content = re.sub(r'\r\n|\r', '\n', content)
        
        # Remove excessive whitespace but preserve paragraph structure
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove empty links but preserve text
        content = re.sub(r'\[\]\([^)]*\)', '', content)
        content = re.sub(r'\[([^\]]+)\]\(\)', r'\1', content)  # Keep text from empty links
        
        # Clean up formatting issues
        content = re.sub(r'\*\*\s*\*\*', '', content)
        content = re.sub(r'\_\_\s*\_\_', '', content)
        
        # Remove lines that are just whitespace or minimal punctuation
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Keep lines that have actual content
            if (stripped and 
                len(stripped) > 2 and  # Minimum length
                not re.match(r'^[\s\*\-\_\|\+\=\#\[\]]*$', stripped) and  # Not just formatting chars
                not re.match(r'^[\(\)\[\]\{\}]*$', stripped)):  # Not just brackets
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines).strip()
        
        # Final cleanup - remove multiple consecutive empty lines
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result
    
    def save_to_file(self, content: str, filename: str) -> bool:
        """Save content to markdown file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False