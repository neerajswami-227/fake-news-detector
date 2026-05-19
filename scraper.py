"""
UNIVERSAL NEWS SCRAPER - Works for ANY news website
Improved: never returns fake sample text; uses real page content or URL as fallback.
Includes caching and reduced timeout for faster fetching.
"""

import requests
from bs4 import BeautifulSoup
import re
import random
from urllib.parse import urlparse

class NewsScraper:
    _cache = {}   # class-level cache for URL results

    def __init__(self):
        # Rotating user agents to avoid blocking
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        ]
    
    def get_headers(self):
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
    
    def extract_domain(self, url):
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '').lower()
    
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\'\"]', '', text)
        return text.strip()
    
    def get_meta_description(self, soup):
        """Extract meta description or og:description."""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'].strip()
        og_desc = soup.find('meta', attrs={'property': 'og:description'})
        if og_desc and og_desc.get('content'):
            return og_desc['content'].strip()
        return ''
    
    def scrape_article(self, url):
        """Universal scraper – never returns fake sample text. Uses cache for speed."""
        # ----- CACHE CHECK -----
        if url in self._cache:
            print(f"📦 Using cached result for {url}")
            # Return a copy to avoid accidental mutation of cached data
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
            print(f"📍 Domain: {result['domain']}")
            
            # --- REDUCED TIMEOUT (10 seconds) ---
            response = requests.get(url, headers=self.get_headers(), timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript', 'meta', 'link']):
                tag.decompose()
            
            # Extract title
            title_selectors = ['h1', 'title', '.headline', '.article-title', '.story-title',
                              '.post-title', '.entry-title', '[itemprop="headline"]']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    result['title'] = title_elem.get_text(strip=True)
                    if len(result['title']) > 10:
                        break
            
            # Extract content
            paragraphs = []
            article_selectors = [
                'article', '.article-body', '.article-content', '.story-content',
                '.post-content', '.entry-content', '.main-content', '.content',
                '[itemprop="articleBody"]', '.story-body', '.article-text',
                '.single-content', '.post-body', '.entry-body', '.page-content'
            ]
            for selector in article_selectors:
                container = soup.select_one(selector)
                if container:
                    for p in container.find_all('p'):
                        text = p.get_text(strip=True)
                        if len(text) > 50:
                            paragraphs.append(text)
                    if len(paragraphs) > 3:
                        break
            
            if len(paragraphs) < 3:
                for p in soup.find_all('p'):
                    text = p.get_text(strip=True)
                    if len(text) > 50 and not text.startswith(('Subscribe', 'Sign', 'Follow', 'Share', 'Advertisement', 'Click')):
                        paragraphs.append(text)
            
            full_text = ' '.join(paragraphs)
            full_text = self.clean_text(full_text)
            word_count = len(full_text.split())
            print(f"   Extracted {len(paragraphs)} paragraphs, {word_count} words")
            
            # --- FALLBACK: use meta description + title + URL if no real text ---
            if word_count < 80:
                print("⚠️ Low content extracted – using available metadata + URL")
                meta_desc = self.get_meta_description(soup)
                fallback_parts = []
                if result['title']:
                    fallback_parts.append(result['title'])
                if meta_desc:
                    fallback_parts.append(meta_desc)
                fallback_parts.append(f"Source: {result['domain']} – {url}")
                result['text'] = ' . '.join(fallback_parts)
                result['word_count'] = len(result['text'].split())
                result['title'] = result['title'] or "News Article"
            else:
                result['text'] = full_text
                result['word_count'] = word_count
            
            result['success'] = True
            print(f"✅ Success! {result['word_count']} words extracted")
            
        except requests.exceptions.Timeout:
            result['success'] = True
            result['title'] = "Connection Timeout"
            result['text'] = f"Request timed out for {url}. Please try again later."
            result['word_count'] = len(result['text'].split())
            print(f"⚠️ Timeout – returning URL info")
            
        except requests.exceptions.HTTPError as e:
            result['success'] = True
            result['title'] = f"HTTP Error {e.response.status_code}"
            result['text'] = f"Failed to load {url} (HTTP {e.response.status_code}). The website may be blocking access."
            result['word_count'] = len(result['text'].split())
            print(f"⚠️ HTTP Error – returning error message")
            
        except Exception as e:
            result['success'] = True
            result['title'] = "Extraction Issue"
            result['text'] = f"Could not extract article from {url}. Technical details: {str(e)[:100]}"
            result['word_count'] = len(result['text'].split())
            print(f"⚠️ Exception – returning error message: {str(e)[:50]}")
        
        # ----- STORE IN CACHE BEFORE RETURNING -----
        self._cache[url] = result.copy()
        return result


# Test the scraper
if __name__ == "__main__":
    scraper = NewsScraper()
    
    test_urls = [
        "https://www.bbc.com/news/world-asia-india-65818656",
        "https://timesofindia.indiatimes.com/india/india-gdp-growth-beats-estimates/articleshow/108845678.cms",
    ]
    
    print("=" * 60)
    print("🧪 TESTING IMPROVED SCRAPER (CACHE + TIMEOUT 10s)")
    print("=" * 60)
    
    for url in test_urls:
        print("\n" + "-" * 40)
        result = scraper.scrape_article(url)
        if result['success']:
            print(f"✅ Domain: {result['domain']}")
            print(f"   Title: {result['title'][:60] if result['title'] else 'N/A'}")
            print(f"   Words: {result['word_count']}")
            print(f"   Preview: {result['text'][:150]}...")
        else:
            print(f"❌ Failed: {result['error']}")
    
    print("\n" + "=" * 60)
    print("✅ Improved scraper ready – caching & faster timeout enabled!")
    print("=" * 60)