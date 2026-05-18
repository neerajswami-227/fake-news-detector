# language_utils.py
from langdetect import detect, DetectorFactory
from googletrans import Translator
import re

DetectorFactory.seed = 0
translator = Translator()

def detect_language(text):
    """Detect language of input text. Returns 'hi' for Hindi, 'en' for English."""
    if not text:
        return 'en'
    try:
        lang = detect(text)
        return lang
    except:
        # Fallback: check for Devanagari characters
        if re.search(r'[\u0900-\u097F]', text):
            return 'hi'
        return 'en'

def is_hindi(text):
    """Check if text contains Devanagari script."""
    return bool(re.search(r'[\u0900-\u097F]', text))

def translate_to_english(text):
    """Translate Hindi text to English using Google Translate."""
    if not text:
        return ""
    # If already English, return as is
    if detect_language(text) == 'en':
        return text
    try:
        translation = translator.translate(text, dest='en')
        return translation.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # fallback to original