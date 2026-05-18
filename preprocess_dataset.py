# preprocess_dataset.py
import pandas as pd
from preprocess import TextPreprocessor
import time

print("=" * 60)
print("🔄 APPLYING PREPROCESSING TO DATASET")
print("=" * 60)

# Load dataset
df = pd.read_csv('data/raw_dataset.csv')
print(f"\n📂 Loaded {len(df)} articles")

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Apply preprocessing
print("\n⚙️ Processing texts...")
start_time = time.time()

# Process each text
processed_texts = []
for i, text in enumerate(df['text']):
    processed = preprocessor.preprocess(text)
    processed_texts.append(processed)
    if (i + 1) % 1000 == 0:
        print(f"   Processed {i+1}/{len(df)} articles")

df['processed_text'] = processed_texts

# Calculate stats
df['original_length'] = df['text'].str.len()
df['processed_length'] = df['processed_text'].str.len()
df['compression_ratio'] = df['processed_length'] / df['original_length']

end_time = time.time()
print(f"\n✅ Preprocessing complete in {end_time - start_time:.2f} seconds")

print(f"\n📊 Preprocessing Statistics:")
print(f"   Average original length: {df['original_length'].mean():.0f} chars")
print(f"   Average processed length: {df['processed_length'].mean():.0f} chars")
print(f"   Average compression: {df['compression_ratio'].mean():.1%}")

# Save preprocessed dataset
df.to_csv('data/preprocessed_dataset.csv', index=False)
print(f"\n💾 Saved to: data/preprocessed_dataset.csv")

print(f"\n📝 Sample Results:")
for i in range(3):
    print(f"\n   Article {i+1}:")
    print(f"   Original: {df['text'].iloc[i][:80]}...")
    print(f"   Processed: {df['processed_text'].iloc[i][:80]}...")

print("\n" + "=" * 60)
print("✅ PHASE 2 COMPLETE")
print("=" * 60)