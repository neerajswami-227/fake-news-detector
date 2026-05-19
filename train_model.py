"""
PHASE 3: Advanced Model Training for Fake News Detection
Trains on Kaggle Fake/Real News dataset (Fake.csv + True.csv)
Optimised: max_features=5000, joblib compression, only LR + Passive Aggressive
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import warnings
import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 70)
print("🤖 PHASE 3: ADVANCED MODEL TRAINING (Kaggle Dataset – Optimised, No RF)")
print("=" * 70)

# ============================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================
print("\n📂 STEP 1: Loading Kaggle Dataset...")
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')
print(f"   Fake: {len(fake_df):,} | Real: {len(true_df):,}")

fake_df['label'] = 1
true_df['label'] = 0
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"   Total: {len(df):,} articles")

# ============================================
# STEP 2: TEXT PREPROCESSING
# ============================================
print("\n🔧 STEP 2: Preprocessing text...")
df['full_text'] = df['title'] + " " + df['text']

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(lemmatized)

df['processed_text'] = df['full_text'].apply(preprocess_text)
df = df[df['processed_text'].str.strip() != ''].reset_index(drop=True)
print(f"   After preprocessing: {len(df):,} articles")

# ============================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================
X = df['processed_text'].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================
# STEP 4: TF-IDF VECTORIZATION (max_features=5000)
# ============================================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    max_df=0.85,
    stop_words='english'
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"   Features: {X_train_tfidf.shape[1]:,}")

# ============================================
# STEP 5: MODELS (ONLY LR + PASSIVE AGGRESSIVE)
# ============================================
models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {'C': [0.1, 0.5, 1.0, 2.0, 5.0], 'solver': ['liblinear', 'lbfgs']}
    },
    'Passive Aggressive': {
        'model': PassiveAggressiveClassifier(random_state=42, max_iter=1000),
        'params': {'C': [0.01, 0.1, 0.5, 1.0], 'loss': ['hinge', 'squared_hinge']}
    }
}

# ============================================
# STEP 6: TRAIN & EVALUATE
# ============================================
results = []
best_model = None
best_score = 0
best_name = ""
best_vectorizer = None

for name, config in models.items():
    print(f"\n📌 Training {name}...")
    gs = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1', n_jobs=-1)
    gs.fit(X_train_tfidf, y_train)
    model = gs.best_estimator_
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"   Params: {gs.best_params_}")
    print(f"   Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name
        best_vectorizer = vectorizer

# ============================================
# STEP 7: RESULTS & CONFUSION MATRIX
# ============================================
print("\n" + "="*70)
print("📊 MODEL COMPARISON")
print("="*70)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print(f"\n🏆 BEST MODEL: {best_name} (F1 = {best_score:.4f})")

y_pred_best = best_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:\n   REAL   FAKE\nR {cm[0,0]:5d} {cm[0,1]:5d}\nF {cm[1,0]:5d} {cm[1,1]:5d}")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['REAL', 'FAKE']))

# ============================================
# STEP 8: SAVE MODELS (COMPRESSED .joblib)
# ============================================
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/model.joblib', compress=3)
joblib.dump(best_vectorizer, 'models/vectorizer.joblib', compress=3)
print("\n✅ Saved compressed models: models/model.joblib, models/vectorizer.joblib")

# Optional .pkl backup
import pickle
with open('models/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(best_vectorizer, f)
print("✅ Also saved .pkl versions (backup)")

# ============================================
# STEP 9: VISUALISATIONS (Optional)
# ============================================
print("\n📈 Creating visualisations...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for i, m in enumerate(metrics):
    axes[0].bar(results_df['Model'], results_df[m], color=['#667eea', '#764ba2'])
    axes[0].set_ylabel(m)
axes[0].set_title('Model Comparison')
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', xticklabels=['REAL','FAKE'], yticklabels=['REAL','FAKE'], ax=axes[1])
axes[1].set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('models/model_comparison.png', dpi=150)
print("   Saved: models/model_comparison.png")

if hasattr(best_model, 'predict_proba'):
    y_prob = best_model.predict_proba(X_test_tfidf)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('models/roc_curve.png', dpi=150)
    print("   Saved: models/roc_curve.png")

# ============================================
# STEP 10: SAVE METRICS SUMMARY
# ============================================
metrics_summary = {
    'best_model': best_name,
    'accuracy': float(accuracy_score(y_test, y_pred_best)),
    'precision': float(precision_score(y_test, y_pred_best)),
    'recall': float(recall_score(y_test, y_pred_best)),
    'f1_score': float(best_score),
    'confusion_matrix': cm.tolist(),
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'features_count': int(X_train_tfidf.shape[1])
}
with open('models/metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print("   Saved: models/metrics_summary.json")

print("\n" + "="*70)
print("✅ PHASE 3 COMPLETE (Random Forest removed)")
print("="*70)
print(f"\n🏆 Best Model: {best_name}")
print(f"📊 Accuracy: {accuracy_score(y_test, y_pred_best)*100:.2f}%")
print("📁 Output: models/model.joblib, models/vectorizer.joblib")
print("="*70)