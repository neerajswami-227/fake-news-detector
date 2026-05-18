# evaluate_hindi.py
import pandas as pd
import pickle
from hindi_preprocess import preprocess_hindi
from sklearn.metrics import accuracy_score

# Load Hindi model
with open('models/hindi_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/hindi_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load test data (adjust path)
test_df = pd.read_csv('data/hindi_test.csv')
X_test = test_df['text'].apply(preprocess_hindi)
y_test = test_df['label']

X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Hindi Model Accuracy: {accuracy:.4f}")