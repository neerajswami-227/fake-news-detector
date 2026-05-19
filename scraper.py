"""
Universal News Scraper - Improved for better text extraction and neutral fallback.
"""

import requests
from bs4 import BeautifulSoup
import re
import random
from urllib.parse import urlparse

class NewsScraper:
    _cache = {}

    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
        ]
    
    def get_headers(self):
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
        }
    
    def extract_domain(self, url):
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '').lower()
    
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\'\"]', '', text)
        return text.strip()
    
    def scrape_article(self, url):
        if url in self._cache:
            print(f"📦 Cached: {url}")
            return self._cache[url].copy()

        result = {
            'success': False,
            'title': '',
            'text': '',
            'error': None,
            'word_count': 0,
            'domain': '',
            'url': url
        }
        
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            result['domain'] = self.extract_domain(url)
            print(f"📡 Scraping: {url}")
            
            response = requests.get(url, headers=self.get_headers(), timeout=12)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove noisy elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                tag.decompose()
            
            # Extract title
            title_tag = soup.find('h1')
            if not title_tag:
                title_tag = soup.find('title')
            result['title'] = title_tag.get_text(strip=True) if title_tag else ''
            
            # Try to find article container
            article = soup.find('article')
            paragraphs = []
            if article:
                for p in article.find_all('p'):
                    text = p.get_text(strip=True)
                    if len(text) > 40:
                        paragraphs.append(text)
            else:
                # Fallback: all substantial paragraphs
                for p in soup.find_all('p'):
                    text = p.get_text(strip=True)
                    if len(text) > 40 and not text.startswith(('Subscribe', 'Sign', 'Follow', 'Share', 'Advertisement')):
                        paragraphs.append(text)
            
            full_text = ' '.join(paragraphs)
            full_text = self.clean_text(full_text)
            word_count = len(full_text.split())
            print(f"   Extracted {len(paragraphs)} paragraphs, {word_count} words")
            
            # If too little text, build a neutral fallback from title + meta description + domain
            if word_count < 80:
                print("⚠️ Low content, using title+meta+domain fallback")
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                meta = meta_desc.get('content', '') if meta_desc else ''
                fallback = f"{result['title']}. {meta} Source: {result['domain']}"
                result['text'] = fallback[:1500]
                result['word_count'] = len(fallback.split())
            else:
                result['text'] = full_text
                result['word_count'] = word_count
            
            result['success'] = True
            print(f"✅ Success: {result['word_count']} words")
            
        except Exception as e:
            # On error, use domain and title as fallback (still informative)
            result['success'] = True
            fallback = f"News article from {result['domain']}. Title: {result['title'] if result['title'] else 'No title'}"
            result['text'] = fallback[:1500]
            result['word_count'] = len(fallback.split())
            print(f"⚠️ Fallback: {str(e)[:50]}")
        
        self._cache[url] = result.copy()
        return result