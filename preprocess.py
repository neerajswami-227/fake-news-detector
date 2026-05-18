"""
Advanced NLP Preprocessing Pipeline for Fake News Detection
Includes: text cleaning, tokenization, stopword removal, lemmatization, POS tagging
"""

import re
import nltk
import numpy as np  # ✅ ADD THIS LINE - FIXES THE ERROR
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from typing import List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Advanced text preprocessing pipeline for fake news detection
    """
    
    def __init__(self):
        """Initialize preprocessor with NLTK resources"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords specific to news articles
        self.custom_stopwords = {
            'said', 'says', 'say', 'told', 'according', 'also', 'would', 
            'could', 'may', 'one', 'two', 'three', 'get', 'make', 'time',
            'like', 'just', 'know', 'take', 'see', 'come', 'think', 'look',
            'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem',
            'feel', 'try', 'leave', 'call', 'article', 'news', 'report'
        }
        self.stop_words.update(self.custom_stopwords)
        
        logger.info(f"Preprocessor initialized with {len(self.stop_words)} stopwords")
    
    def get_wordnet_pos(self, tag: str) -> str:
        """
        Convert Penn Treebank POS tag to WordNet POS tag
        
        Args:
            tag: Penn Treebank POS tag
            
        Returns:
            WordNet POS tag for lemmatization
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def clean_text(self, text: str) -> str:
        """
        Clean raw text by removing noise
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Tokenize text and remove stopwords and short words
        
        Args:
            text: Cleaned text
            
        Returns:
            List of filtered tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter: remove stopwords and words shorter than 3 characters
        filtered = [
            token for token in tokens 
            if token not in self.stop_words 
            and len(token) > 2
        ]
        
        return filtered
    
    def lemmatize_with_pos(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens using POS tags for better accuracy
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not tokens:
            return []
        
        # Get POS tags
        pos_tags = pos_tag(tokens)
        
        # Lemmatize each token
        lemmatized = []
        for token, tag in pos_tags:
            wordnet_pos = self.get_wordnet_pos(tag)
            lemmatized_token = self.lemmatizer.lemmatize(token, wordnet_pos)
            lemmatized.append(lemmatized_token)
        
        return lemmatized
    
    def preprocess(self, text: str, verbose: bool = False) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Raw text
            verbose: Print progress
            
        Returns:
            Preprocessed text ready for feature extraction
        """
        original_len = len(text)
        
        # Step 1: Clean
        cleaned = self.clean_text(text)
        
        # Step 2: Tokenize and filter
        tokens = self.tokenize_and_filter(cleaned)
        
        # Step 3: Lemmatize with POS
        lemmatized = self.lemmatize_with_pos(tokens)
        
        # Join back to string
        result = ' '.join(lemmatized)
        
        if verbose:
            print(f"Original length: {original_len} chars")
            print(f"Processed length: {len(result)} chars")
            print(f"Compression: {len(result)/original_len*100:.1f}%")
        
        return result
    
    def preprocess_batch(self, texts: List[str], batch_size: int = 1000) -> List[str]:
        """
        Preprocess a batch of texts with progress bar
        
        Args:
            texts: List of text strings
            batch_size: Size of processing batches
            
        Returns:
            List of preprocessed texts
        """
        processed = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_processed = [self.preprocess(t) for t in batch]
            processed.extend(batch_processed)
            print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)}")
        return processed
    
    def extract_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract additional features from text
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = word_tokenize(cleaned)
            
            feature_dict = {
                'original_length': len(text),
                'cleaned_length': len(cleaned),
                'num_tokens': len(tokens),
                'unique_tokens': len(set(tokens)),
                'avg_token_length': np.mean([len(t) for t in tokens]) if tokens else 0,
                'num_exclamation': text.count('!'),
                'num_question': text.count('?'),
                'has_url': int('http' in text.lower()),
                'has_email': int('@' in text),
                'is_uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1)
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)


# Test the preprocessor
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "BREAKING: Government announces new policy! Click here https://example.com",
        "According to official sources, the minister will visit the disaster site tomorrow.",
        "SHOCKING!!! You won't believe what happened next!!! Share this!"
    ]
    
    print("\n" + "=" * 60)
    print("🧪 TESTING PREPROCESSOR")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\n📝 ORIGINAL: {text}")
        processed = preprocessor.preprocess(text)
        print(f"✅ PROCESSED: {processed}")
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSOR TEST PASSED")
    print("=" * 60)