"""
Train Logistic Regression Model for Fake News Detection
Comprehensive training with evaluation, visualizations, and feature importance
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("=" * 70)
print("🚀 Training Logistic Regression Model for Fake News Detection")
print("=" * 70)

# ============================================
# 1. Load and prepare dataset
# ============================================
print("\n📂 Loading dataset (first 5000 rows each)...")
fake_df = pd.read_csv('data/Fake.csv', nrows=5000)
true_df = pd.read_csv('data/True.csv', nrows=5000)
print(f"   Fake articles: {len(fake_df):,}")
print(f"   Real articles: {len(true_df):,}")

fake_df['label'] = 1
true_df['label'] = 0
df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['full_text'] = df['title'] + " " + df['text']
print(f"   Total articles: {len(df):,}")
print(f"   Class distribution: Real={(df['label']==0).sum()}, Fake={(df['label']==1).sum()}")

# ============================================
# 2. Text preprocessing
# ============================================
print("\n🔧 Preprocessing text (cleaning, stopword removal, lemmatization)...")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# Additional custom stopwords
extra_stopwords = {'said', 'says', 'say', 'told', 'according', 'also', 'would', 'could', 'may'}
stop_words.update(extra_stopwords)

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and short words
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    # Lemmatize
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(lemmatized)

print("   Applying preprocessing to all articles...")
df['processed_text'] = df['full_text'].apply(preprocess_text)
# Remove empty rows
df = df[df['processed_text'].str.strip() != ''].reset_index(drop=True)
print(f"   After preprocessing: {len(df):,} articles")

# ============================================
# 3. Train-test split
# ============================================
print("\n✂️ Splitting data into train (80%) and test (20%)...")
X = df['processed_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")

# ============================================
# 4. TF-IDF vectorization
# ============================================
print("\n📝 Creating TF-IDF features (max_features=5000, ngram_range=(1,2))...")
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
print(f"   Feature matrix shape: {X_train_tfidf.shape}")
print(f"   Number of features: {X_train_tfidf.shape[1]:,}")

# ============================================
# 5. Train Logistic Regression (with optional grid search)
# ============================================
print("\n🤖 Training Logistic Regression model...")
# Simple model first
model = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
print("   Model training complete.")

# Optional grid search for hyperparameter tuning (commented out to keep fast, but you can uncomment)
# print("\n   Performing Grid Search for hyperparameters...")
# param_grid = {'C': [0.1, 0.5, 1.0, 2.0, 5.0], 'solver': ['liblinear', 'lbfgs']}
# grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), param_grid, cv=5, scoring='f1')
# grid.fit(X_train_tfidf, y_train)
# model = grid.best_estimator_
# print(f"   Best params: {grid.best_params_}")

# ============================================
# 6. Evaluation on test set
# ============================================
print("\n📊 Evaluating model on test set...")
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1])

print(f"\n   ✅ Accuracy:  {accuracy:.4f}")
print(f"   ✅ Precision: {precision:.4f}")
print(f"   ✅ Recall:    {recall:.4f}")
print(f"   ✅ F1-Score:  {f1:.4f}")
print(f"   ✅ ROC-AUC:   {roc_auc:.4f}")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\n   Confusion Matrix:")
print(f"                 Predicted")
print(f"                 REAL    FAKE")
print(f"   Actual REAL   {tn:5d}   {fp:5d}")
print(f"   Actual FAKE   {fn:5d}   {tp:5d}")

# ============================================
# 7. Feature importance (coefficients)
# ============================================
print("\n🔍 Extracting feature importance (top indicators for FAKE and REAL)...")
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]
# Top 20 features for FAKE (positive coefficients)
top_fake_idx = np.argsort(coef)[-20:][::-1]
top_fake_features = [(feature_names[i], coef[i]) for i in top_fake_idx]
# Top 20 features for REAL (negative coefficients)
top_real_idx = np.argsort(coef)[:20]
top_real_features = [(feature_names[i], coef[i]) for i in top_real_idx]

print("\n   📌 Top 10 indicators of FAKE news:")
for i, (word, score) in enumerate(top_fake_features[:10]):
    print(f"      {i+1}. '{word}' (score: {score:.4f})")
print("\n   📌 Top 10 indicators of REAL news:")
for i, (word, score) in enumerate(top_real_features[:10]):
    print(f"      {i+1}. '{word}' (score: {score:.4f})")

# ============================================
# 8. Save model and vectorizer
# ============================================
print("\n💾 Saving model and vectorizer...")
os.makedirs('models', exist_ok=True)
# Save as compressed joblib (recommended)
joblib.dump(model, 'models/model.joblib', compress=3)
joblib.dump(vectorizer, 'models/vectorizer.joblib', compress=3)
print("   ✅ Saved as .joblib (compressed)")
# Also save as .pkl for compatibility
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("   ✅ Saved as .pkl (backup)")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Score': [accuracy, precision, recall, f1, roc_auc]
})
metrics_df.to_csv('models/training_metrics.csv', index=False)
print("   ✅ Saved metrics to models/training_metrics.csv")

# ============================================
# 9. Create visualizations
# ============================================
print("\n📈 Creating visualizations...")

# Plot 1: Confusion Matrix Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150)
print("   ✅ Saved confusion_matrix.png")

# Plot 2: ROC Curve
plt.figure(figsize=(7,6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('models/roc_curve.png', dpi=150)
print("   ✅ Saved roc_curve.png")

# Plot 3: Feature Importance (Top 20 words for FAKE)
plt.figure(figsize=(10,6))
words = [w for w, _ in top_fake_features[:20]]
scores = [s for _, s in top_fake_features[:20]]
colors = ['#d62728'] * 20
plt.barh(words, scores, color=colors)
plt.xlabel('Coefficient Value')
plt.title('Top 20 Indicators of FAKE News')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('models/feature_importance_fake.png', dpi=150)
print("   ✅ Saved feature_importance_fake.png")

# Plot 4: Feature Importance (Top 20 words for REAL)
plt.figure(figsize=(10,6))
words_real = [w for w, _ in top_real_features[:20]]
scores_real = [s for _, s in top_real_features[:20]]
colors_real = ['#2ca02c'] * 20
plt.barh(words_real, scores_real, color=colors_real)
plt.xlabel('Coefficient Value')
plt.title('Top 20 Indicators of REAL News')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('models/feature_importance_real.png', dpi=150)
print("   ✅ Saved feature_importance_real.png")

# ============================================
# 10. Save summary as JSON
# ============================================
import json
summary = {
    'model_type': 'LogisticRegression',
    'max_features': 5000,
    'ngram_range': [1,2],
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'roc_auc': float(roc_auc),
    'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
}
with open('models/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("   ✅ Saved training_summary.json")

# ============================================
# 11. Final summary
# ============================================
print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\n🏆 Model: Logistic Regression")
print(f"📊 Accuracy: {accuracy*100:.2f}%")
print(f"📊 F1-Score: {f1*100:.2f}%")
print(f"📊 ROC-AUC: {roc_auc:.3f}")
print("\n📁 All files saved in 'models/' directory:")
print("   - model.joblib, vectorizer.joblib (compressed)")
print("   - model.pkl, vectorizer.pkl (backup)")
print("   - training_metrics.csv (metrics table)")
print("   - confusion_matrix.png (heatmap)")
print("   - roc_curve.png (ROC curve)")
print("   - feature_importance_fake.png, feature_importance_real.png")
print("   - training_summary.json (JSON summary)")
print("\n" + "=" * 70)