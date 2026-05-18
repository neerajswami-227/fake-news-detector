# verify_setup.py
import sys
print(f"Python: {sys.version}")

packages = ['numpy', 'pandas', 'sklearn', 'nltk', 'flask', 'flask_login', 'flask_sqlalchemy', 'bcrypt', 'requests', 'bs4']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg}")
    except ImportError:
        print(f"❌ {pkg} - NOT INSTALLED")

import nltk
try:
    nltk.data.find('tokenizers/punkt')
    print("✅ NLTK punkt")
except:
    print("❌ NLTK punkt missing")