# hindi_preprocess.py
import re

# Optional: try to import Indic NLP Library for better tokenization
try:
    from indicnlp.tokenize import indic_tokenize
    from indicnlp.normalize import indic_normalize
    INDIC_AVAILABLE = True
    normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer("hi")
except ImportError:
    INDIC_AVAILABLE = False
    print("⚠️ Indic NLP Library not installed. Using basic tokenization.")

# Hindi stopwords set
HINDI_STOPWORDS = {
    'और', 'एक', 'यह', 'उस', 'इस', 'को', 'ने', 'से', 'के', 'में',
    'है', 'था', 'थी', 'थे', 'थीं', 'कर', 'गया', 'गई', 'गए', 'हुए',
    'लिए', 'साथ', 'बिना', 'तक', 'कि', 'की', 'के', 'वह', 'वे', 'ये',
    'इन', 'उन', 'उनके', 'उनकी', 'उन्हें', 'उन्होंने', 'करना', 'करते',
    'होना', 'हुआ', 'हुई', 'हुए', 'रहा', 'रही', 'रहे', 'जा', 'जाता',
    'जाती', 'जाते', 'पर', 'अपना', 'अपनी', 'अपने', 'मैं', 'तू', 'तुम',
    'आप', 'हम', 'हमारा', 'हमारी', 'हमारे'
}

def clean_hindi_text(text):
    """Clean Hindi text, convert to string, remove extra spaces."""
    if not isinstance(text, str):
        # Convert to string if not already
        if pd.isna(text):   # pandas NaN
            return ""
        text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Optional: normalize Unicode
    if INDIC_AVAILABLE:
        text = normalizer.normalize(text)
    return text

def tokenize_hindi(text):
    """Tokenize Hindi text."""
    if not text:
        return []
    if INDIC_AVAILABLE:
        return indic_tokenize.trivial_tokenize(text)
    else:
        # Simple fallback: split by spaces
        return text.split()

def remove_hindi_stopwords(tokens):
    """Remove stopwords and short tokens."""
    return [t for t in tokens if t not in HINDI_STOPWORDS and len(t) > 1]

def stem_hindi_word(word):
    """Simple rule-based stemmer for Hindi."""
    suffixes = ['ना', 'नी', 'ने', 'ई', 'ए', 'े', 'ी', 'ा', 'ू', 'ौ', 'ै']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word

def stem_hindi_tokens(tokens):
    """Apply stemming to all tokens."""
    return [stem_hindi_word(token) for token in tokens]

def preprocess_hindi(text):
    """Complete preprocessing pipeline for Hindi text."""
    # Safety: handle non-string
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)
    cleaned = clean_hindi_text(text)
    if not cleaned:
        return ""
    tokens = tokenize_hindi(cleaned)
    no_stopwords = remove_hindi_stopwords(tokens)
    stemmed = stem_hindi_tokens(no_stopwords)
    return ' '.join(stemmed)

# Import pandas only for isna check inside the function
import pandas as pd