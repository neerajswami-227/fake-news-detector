"""
PHASE 3: Advanced Model Training for Fake News Detection
Trains on Kaggle Fake/Real News dataset (Fake.csv + True.csv)
Optimized: max_features=10000, compressed joblib output
"""

import pandas as pd
import numpy as np
import joblib  # ← instead of pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
# from sklearn.ensemble import RandomForestClassifier
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

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 70)
print("🤖 PHASE 3: ADVANCED MODEL TRAINING (Kaggle Dataset - Optimized)")
print("=" * 70)

# ============================================
# STEP 1: LOAD AND PREPARE DATA (Kaggle)
# ============================================
print("\n📂 STEP 1: Loading Kaggle Dataset...")

fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

print(f"   Fake news articles: {len(fake_df):,}")
print(f"   Real news articles: {len(true_df):,}")

fake_df['label'] = 1
true_df['label'] = 0

df = pd.concat([fake_df, true_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   Total articles: {len(df):,}")
print(f"   Real news (0): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
print(f"   Fake news (1): {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.1f}%)")

# ============================================
# STEP 2: TEXT PREPROCESSING
# ============================================
print("\n🔧 STEP 2: Preprocessing text...")

df['full_text'] = df['title'] + " " + df['text']

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(lemmatized)

print("   Applying preprocessing...")
df['processed_text'] = df['full_text'].apply(preprocess_text)
df = df[df['processed_text'].str.strip() != ''].reset_index(drop=True)
print(f"   After preprocessing: {len(df):,} articles")

# ============================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================
print("\n✂️ STEP 3: Creating Train-Test Split...")

X = df['processed_text'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {len(X_train):,} samples")
print(f"   Test set: {len(X_test):,} samples")

# ============================================
# STEP 4: TF-IDF VECTORIZATION (Reduced features)
# ============================================
print("\n📝 STEP 4: Creating TF-IDF Features...")

vectorizer = TfidfVectorizer(
    max_features=10000,           # ← REDUCED from 15000 to 10000
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
# STEP 5: DEFINE MODELS
# ============================================
print("\n🤖 STEP 5: Initializing Models...")

models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {'C': [0.1, 0.5, 1.0, 2.0, 5.0], 'solver': ['liblinear', 'lbfgs']}
    },
    'Passive Aggressive': {
        'model': PassiveAggressiveClassifier(random_state=42, max_iter=1000),
        'params': {'C': [0.01, 0.1, 0.5, 1.0], 'loss': ['hinge', 'squared_hinge']}
    },
    
}

# ============================================
# STEP 6: TRAIN AND EVALUATE MODELS
# ============================================
print("\n🏋️ STEP 6: Training and Evaluating Models...")
print("-" * 70)

results = []
best_model = None
best_score = 0
best_name = ""
best_vectorizer = None

for name, config in models.items():
    print(f"\n📌 Training {name}...")
    grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train_tfidf, y_train)
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   Best params: {grid_search.best_params_}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Best Params': str(grid_search.best_params_)
    })
    
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name
        best_vectorizer = vectorizer

# ============================================
# STEP 7: RESULTS SUMMARY
# ============================================
print("\n" + "=" * 70)
print("📊 STEP 7: MODEL COMPARISON SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print(f"\n🏆 BEST MODEL: {best_name} with F1-Score: {best_score:.4f}")

# ============================================
# STEP 8: CONFUSION MATRIX & REPORT
# ============================================
y_pred_best = best_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_best)

print(f"\n   Confusion Matrix:")
print(f"                 Predicted")
print(f"                 REAL    FAKE")
print(f"   Actual REAL   {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"   Actual FAKE   {cm[1,0]:5d}   {cm[1,1]:5d}")

tn, fp, fn, tp = cm.ravel()
print(f"\n   Detailed Metrics:")
print(f"   True Negatives (correct REAL): {tn}")
print(f"   False Positives (REAL marked FAKE): {fp}")
print(f"   False Negatives (FAKE marked REAL): {fn}")
print(f"   True Positives (correct FAKE): {tp}")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=['REAL', 'FAKE']))

# ============================================
# STEP 9: SAVE MODEL AND VECTORIZER (Compressed joblib)
# ============================================
print("\n💾 STEP 9: Saving Model and Vectorizer (compressed joblib)...")

os.makedirs('models', exist_ok=True)

# Save best model with compression level 3
joblib.dump(best_model, 'models/model.joblib', compress=3)
print(f"   ✅ Model saved to: models/model.joblib (compressed)")

# Save vectorizer with compression
joblib.dump(best_vectorizer, 'models/vectorizer.joblib', compress=3)
print(f"   ✅ Vectorizer saved to: models/vectorizer.joblib (compressed)")

# For backward compatibility, also save as .pkl if you want (optional)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(best_vectorizer, f)
print(f"   ✅ Also saved .pkl versions for compatibility")

results_df.to_csv('models/training_results.csv', index=False)
print(f"   ✅ Results saved to: models/training_results.csv")

# ============================================
# STEP 10: CREATE VISUALIZATIONS
# ============================================
print("\n📈 STEP 10: Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics_list):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(results_df['Model'], results_df[metric], color=['#667eea', '#764ba2', '#f093fb'])
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('models/model_comparison.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: models/model_comparison.png")

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'], ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title(f'Confusion Matrix - {best_name}')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: models/confusion_matrix.png")

if hasattr(best_model, 'predict_proba'):
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    y_pred_proba = best_model.predict_proba(X_test_tfidf)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title(f'ROC Curve - {best_name}')
    ax3.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('models/roc_curve.png', dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: models/roc_curve.png (AUC = {roc_auc:.3f})")

# ============================================
# STEP 11: FEATURE IMPORTANCE (if Logistic Regression)
# ============================================
if best_name == 'Logistic Regression':
    print("\n🔍 STEP 11: Analyzing Feature Importance...")
    feature_names = best_vectorizer.get_feature_names_out()
    coefficients = best_model.coef_[0]
    top_fake_idx = np.argsort(coefficients)[-20:][::-1]
    top_fake_features = [(feature_names[i], coefficients[i]) for i in top_fake_idx]
    top_real_idx = np.argsort(coefficients)[:20]
    top_real_features = [(feature_names[i], coefficients[i]) for i in top_real_idx]
    print("\n   📌 Top 10 indicators of FAKE news:")
    for i, (feature, coef) in enumerate(top_fake_features[:10]):
        print(f"      {i+1}. '{feature}' (score: {coef:.4f})")
    print("\n   📌 Top 10 indicators of REAL news:")
    for i, (feature, coef) in enumerate(top_real_features[:10]):
        print(f"      {i+1}. '{feature}' (score: {coef:.4f})")
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients}).sort_values('coefficient', ascending=False)
    feature_importance_df.to_csv('models/feature_importance.csv', index=False)
    print("\n   ✅ Saved: models/feature_importance.csv")

# ============================================
# STEP 12: SAVE METRICS SUMMARY
# ============================================
metrics_summary = {
    'best_model': best_name,
    'accuracy': float(accuracy_score(y_test, y_pred_best)),
    'precision': float(precision_score(y_test, y_pred_best)),
    'recall': float(recall_score(y_test, y_pred_best)),
    'f1_score': float(f1_score(y_test, y_pred_best)),
    'confusion_matrix': cm.tolist(),
    'training_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'features_count': int(X_train_tfidf.shape[1])
}
with open('models/metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print("   ✅ Saved: models/metrics_summary.json")

# ============================================
# COMPLETION
# ============================================
print("\n" + "=" * 70)
print("✅ PHASE 3 COMPLETE!")
print("=" * 70)
print(f"\n🏆 Best Model: {best_name}")
print(f"📊 Accuracy: {accuracy_score(y_test, y_pred_best)*100:.2f}%")
print(f"📊 F1-Score: {f1_score(y_test, y_pred_best)*100:.2f}%")
print("\n📁 Output files saved in 'models/' directory.")
print("   - model.joblib, vectorizer.joblib (compressed)")
print("   - model.pkl, vectorizer.pkl (backup)")
print("=" * 70)