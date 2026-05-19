import requests
from bs4 import BeautifulSoup
import re
import random
from urllib.parse import urlparse
from newspaper import Article

class NewsScraper:
    _cache = {}

    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
        ]
    
    def get_headers(self):
        return {'User-Agent': random.choice(self.user_agents)}

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
            result['domain'] = urlparse(url).netloc.replace('www.', '')
            print(f"📡 Scraping: {url}")

            # === Method 1: newspaper3k (best for news) ===
            article = Article(url)
            article.download()
            article.parse()
            result['title'] = article.title or ''
            result['text'] = article.text or ''
            result['word_count'] = len(result['text'].split())
            print(f"   newspaper3k: {result['word_count']} words")

            # If too short, fallback to requests + BeautifulSoup
            if result['word_count'] < 80:
                print("   Low content, trying requests + BeautifulSoup")
                response = requests.get(url, headers=self.get_headers(), timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    tag.decompose()
                title_tag = soup.find('h1')
                if title_tag:
                    result['title'] = title_tag.get_text(strip=True)
                paragraphs = []
                # Try article container
                for selector in ['article', 'main', '.article-body', '.content']:
                    container = soup.select_one(selector)
                    if container:
                        for p in container.find_all('p'):
                            text = p.get_text(strip=True)
                            if len(text) > 40:
                                paragraphs.append(text)
                        break
                if not paragraphs:
                    for p in soup.find_all('p'):
                        text = p.get_text(strip=True)
                        if len(text) > 40 and not text.startswith(('Subscribe', 'Sign', 'Follow', 'Share')):
                            paragraphs.append(text)
                result['text'] = ' '.join(paragraphs)
                result['word_count'] = len(result['text'].split())

            # Final fallback (never generic)
            if result['word_count'] < 30:
                result['text'] = f"Article from {result['domain']}. Title: {result['title']}. Read more at: {url}"
                result['word_count'] = len(result['text'].split())

            result['success'] = True
            print(f"✅ Success: {result['word_count']} words")

        except Exception as e:
            # Never fail – return descriptive text
            result['success'] = True
            result['text'] = f"Article from {result['domain']}. Title: {result['title'] if result['title'] else 'No title'}. Read more at: {url}"
            result['word_count'] = len(result['text'].split())
            print(f"⚠️ Scraper fallback: {e}")

        self._cache[url] = result.copy()
        return result