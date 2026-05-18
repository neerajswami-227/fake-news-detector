"""
PHASE 3: Advanced Model Training for Fake News Detection
Trains and compares: Logistic Regression, Passive Aggressive Classifier, Random Forest
Includes hyperparameter tuning, cross-validation, and comprehensive evaluation
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline
import warnings
import json
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("=" * 70)
print("🤖 PHASE 3: ADVANCED MODEL TRAINING")
print("=" * 70)

# ============================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================
print("\n📂 STEP 1: Loading Preprocessed Data...")

df = pd.read_csv('data/preprocessed_dataset.csv')
print(f"   ✅ Loaded {len(df):,} articles")

# Use preprocessed text for training
X = df['processed_text'].values
y = df['label'].values

# Check class distribution
print(f"\n📊 Class Distribution:")
print(f"   Real News (0): {(y == 0).sum():,} ({(y == 0).mean()*100:.1f}%)")
print(f"   Fake News (1): {(y == 1).sum():,} ({(y == 1).mean()*100:.1f}%)")

# ============================================
# STEP 2: TRAIN-TEST SPLIT
# ============================================
print("\n✂️ STEP 2: Creating Train-Test Split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {len(X_train):,} samples")
print(f"   Test set: {len(X_test):,} samples")

# ============================================
# STEP 3: TF-IDF VECTORIZATION
# ============================================
print("\n📝 STEP 3: Creating TF-IDF Features...")

vectorizer = TfidfVectorizer(
    max_features=10000,           # Top 10,000 features
    ngram_range=(1, 2),           # Unigrams and bigrams
    sublinear_tf=True,            # Use 1+log(tf)
    min_df=3,                     # Ignore terms appearing in <3 docs
    max_df=0.85,                  # Ignore terms appearing in >85% docs
    stop_words='english'          # Remove English stopwords
)

# Fit on training data only
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"   Feature matrix shape: {X_train_tfidf.shape}")
print(f"   Number of features: {X_train_tfidf.shape[1]:,}")

# ============================================
# STEP 4: DEFINE MODELS
# ============================================
print("\n🤖 STEP 4: Initializing Models...")

models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.1, 0.5, 1.0, 2.0, 5.0],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    'Passive Aggressive': {
        'model': PassiveAggressiveClassifier(random_state=42, max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 0.5, 1.0],
            'loss': ['hinge', 'squared_hinge']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    }
}

# ============================================
# STEP 5: TRAIN AND EVALUATE MODELS
# ============================================
print("\n🏋️ STEP 5: Training and Evaluating Models...")
print("-" * 70)

results = []
best_model = None
best_score = 0
best_name = ""

for name, config in models.items():
    print(f"\n📌 Training {name}...")
    
    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        config['model'],
        config['params'],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_tfidf, y_train)
    
    # Best model from grid search
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
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
    
    # Track best model
    if f1 > best_score:
        best_score = f1
        best_model = model
        best_name = name
        best_vectorizer = vectorizer

# ============================================
# STEP 6: RESULTS SUMMARY
# ============================================
print("\n" + "=" * 70)
print("📊 STEP 6: MODEL COMPARISON SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print(f"\n🏆 BEST MODEL: {best_name} with F1-Score: {best_score:.4f}")

# ============================================
# STEP 7: CONFUSION MATRIX FOR BEST MODEL
# ============================================
print("\n📊 STEP 7: Confusion Matrix Analysis...")

y_pred_best = best_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_best)

print(f"\n   Confusion Matrix:")
print(f"                 Predicted")
print(f"                 REAL    FAKE")
print(f"   Actual REAL   {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"   Actual FAKE   {cm[1,0]:5d}   {cm[1,1]:5d}")

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
print(f"\n   Detailed Metrics:")
print(f"   True Negatives (correct REAL): {tn}")
print(f"   False Positives (REAL marked FAKE): {fp}")
print(f"   False Negatives (FAKE marked REAL): {fn}")
print(f"   True Positives (correct FAKE): {tp}")

# ============================================
# STEP 8: CLASSIFICATION REPORT
# ============================================
print("\n📋 STEP 8: Detailed Classification Report...")
print("\n" + classification_report(y_test, y_pred_best, target_names=['REAL', 'FAKE']))

# ============================================
# STEP 9: SAVE MODEL AND VECTORIZER
# ============================================
print("\n💾 STEP 9: Saving Model and Vectorizer...")

with open('models/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"   ✅ Model saved to: models/model.pkl")

with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(best_vectorizer, f)
print(f"   ✅ Vectorizer saved to: models/vectorizer.pkl")

# Save results to CSV
results_df.to_csv('models/training_results.csv', index=False)
print(f"   ✅ Results saved to: models/training_results.csv")

# ============================================
# STEP 10: CREATE VISUALIZATIONS
# ============================================
print("\n📈 STEP 10: Creating Visualizations...")

# Figure 1: Model Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(results_df['Model'], results_df[metric], color=['#667eea', '#764ba2', '#f093fb'])
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('models/model_comparison.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: models/model_comparison.png")

# Figure 2: Confusion Matrix Heatmap
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=['REAL', 'FAKE'], 
            yticklabels=['REAL', 'FAKE'],
            ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title(f'Confusion Matrix - {best_name}')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: models/confusion_matrix.png")

# Figure 3: ROC Curve (if model supports probability)
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
# STEP 11: FEATURE IMPORTANCE (for Logistic Regression)
# ============================================
print("\n🔍 STEP 11: Analyzing Feature Importance...")

if best_name == 'Logistic Regression':
    feature_names = best_vectorizer.get_feature_names_out()
    coefficients = best_model.coef_[0]
    
    # Get top 20 features for FAKE news (positive coefficients)
    top_fake_idx = np.argsort(coefficients)[-20:][::-1]
    top_fake_features = [(feature_names[i], coefficients[i]) for i in top_fake_idx]
    
    # Get top 20 features for REAL news (negative coefficients)
    top_real_idx = np.argsort(coefficients)[:20]
    top_real_features = [(feature_names[i], coefficients[i]) for i in top_real_idx]
    
    print("\n   📌 Top 10 indicators of FAKE news:")
    for i, (feature, coef) in enumerate(top_fake_features[:10]):
        print(f"      {i+1}. '{feature}' (score: {coef:.4f})")
    
    print("\n   📌 Top 10 indicators of REAL news:")
    for i, (feature, coef) in enumerate(top_real_features[:10]):
        print(f"      {i+1}. '{feature}' (score: {coef:.4f})")
    
    # Save feature importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', ascending=False)
    feature_importance_df.to_csv('models/feature_importance.csv', index=False)
    print(f"\n   ✅ Saved: models/feature_importance.csv")

# ============================================
# STEP 12: SAVE MODEL METRICS
# ============================================
print("\n📝 STEP 12: Saving Model Metrics...")

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

import json
with open('models/metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print(f"   ✅ Saved: models/metrics_summary.json")

# ============================================
# COMPLETION
# ============================================
print("\n" + "=" * 70)
print("✅ PHASE 3 COMPLETE!")
print("=" * 70)
print(f"\n🏆 Best Model: {best_name}")
print(f"📊 Accuracy: {accuracy_score(y_test, y_pred_best)*100:.2f}%")
print(f"📊 F1-Score: {f1_score(y_test, y_pred_best)*100:.2f}%")
print(f"\n📁 Output files saved in 'models/' directory:")
print(f"   - model.pkl (trained model)")
print(f"   - vectorizer.pkl (TF-IDF vectorizer)")
print(f"   - training_results.csv (model comparison)")
print(f"   - model_comparison.png (bar chart)")
print(f"   - confusion_matrix.png (heatmap)")
print(f"   - metrics_summary.json (metrics)")
if best_name == 'Logistic Regression':
    print(f"   - feature_importance.csv (top features)")
print("\n" + "=" * 70)