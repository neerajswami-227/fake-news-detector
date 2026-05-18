# train_hindi_model.py (fixed)
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Import Hindi preprocessor (safe)
try:
    from hindi_preprocess import preprocess_hindi
except ImportError:
    print("⚠️ Using fallback: identity preprocessor")
    def preprocess_hindi(text):
        return str(text)

print("="*60)
print("Training Hindi Fake News Detection Model")
print("="*60)

# Load dataset
dataset_path = 'data/hindi_dataset.csv'
if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found at {dataset_path}")
    exit(1)

df = pd.read_csv(dataset_path)
print(f"✅ Loaded {len(df)} articles")

# Identify text and label columns
possible_text_cols = ['text', 'news', 'content', 'article']
possible_label_cols = ['label', 'class', 'is_fake', 'target']

text_col = None
label_col = None
for col in possible_text_cols:
    if col in df.columns:
        text_col = col
        break
for col in possible_label_cols:
    if col in df.columns:
        label_col = col
        break

if text_col is None or label_col is None:
    print("❌ Could not identify text/label columns. Columns:", df.columns.tolist())
    exit(1)

print(f"📝 Using text column: '{text_col}', label column: '{label_col}'")

# Clean and prepare
df = df.dropna(subset=[text_col]).reset_index(drop=True)
df[text_col] = df[text_col].astype(str)
print(f"After cleaning: {len(df)} articles")

# Convert labels to binary if needed
if df[label_col].dtype == 'object':
    unique = df[label_col].unique()
    mapping = {}
    for val in unique:
        low = str(val).lower()
        if low in ['real', 'true', 'genuine']:
            mapping[val] = 0
        elif low in ['fake', 'false', 'hoax']:
            mapping[val] = 1
    if mapping:
        df[label_col] = df[label_col].map(mapping)
    else:
        print(f"⚠️ Unknown label values: {unique}. Please check.")
        exit(1)

# Preprocess
print("Preprocessing Hindi text...")
df['processed_text'] = df[text_col].apply(preprocess_hindi)

# Split
X = df['processed_text']
y = df[label_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# Save
os.makedirs('models', exist_ok=True)
with open('models/hindi_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/hindi_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("\n✅ Hindi model and vectorizer saved to models/")