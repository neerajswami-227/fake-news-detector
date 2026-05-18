import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/raw_dataset.csv')

print("=" * 60)
print("📊 DATASET EXPLORATION")
print("=" * 60)

print(f"\nDataset Shape: {df.shape}")
print(f"\nClass Distribution:")
print(df['label'].value_counts().to_string())
print(f"\nText Statistics:")
print(df['text_length'].describe().to_string())

print(f"\nSample Real News (label=0):")
print(df[df['label']==0]['text'].iloc[0][:200])

print(f"\nSample Fake News (label=1):")
print(df[df['label']==1]['text'].iloc[0][:200])