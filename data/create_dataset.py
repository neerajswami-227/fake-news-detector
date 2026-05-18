"""
Fake News Dataset Generator
Creates a balanced dataset of real and fake news articles
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 60)
print("📊 FAKE NEWS DATASET GENERATOR")
print("=" * 60)

# ============= REAL NEWS TEMPLATES =============
REAL_NEWS_TEMPLATES = [
    # Political News
    "According to the Ministry of External Affairs, India and the United States have signed a new defense agreement aimed at strengthening bilateral ties.",
    "The Election Commission announced that voting for the upcoming state assembly elections will take place in seven phases starting next month.",
    "Official government data released today shows that India's GDP grew by 7.2% in the last quarter, beating market expectations.",
    
    # Health News  
    "The World Health Organization confirmed that the new COVID-19 variant shows reduced severity compared to previous strains based on preliminary data.",
    "A study published in The Lancet medical journal found that regular exercise reduces the risk of heart disease by up to 35%.",
    "The Ministry of Health reported that vaccination coverage has reached 95% among eligible adults across the country.",
    
    # Economic News
    "The Reserve Bank of India kept the repo rate unchanged at 6.5% citing concerns over inflation and global economic uncertainty.",
    "Finance Minister Nirmala Sitharaman announced a ₹10,000 crore stimulus package for the MSME sector in today's press conference.",
    "Stock markets closed at record highs today with the Sensex crossing 75,000 points for the first time in history.",
    
    # Technology News
    "ISRO successfully launched its latest satellite mission from the Sriharikota spaceport at 9:30 AM this morning.",
    "Microsoft announced a $3 billion investment in India to expand its cloud computing infrastructure over the next three years.",
    "Scientists have discovered a new exoplanet in the habitable zone of its star, according to research published in Nature Astronomy.",
    
    # Sports News
    "Team India defeated Australia by 5 wickets in the thrilling series finale to win the Border-Gavaskar trophy.",
    "The International Olympic Committee announced that cricket will be included in the 2028 Los Angeles Olympics.",
    
    # Education News
    "The Central Board of Secondary Education announced that Class 10 and 12 board exams will begin on February 15.",
    "University Grants Commission released new guidelines for online education and examination standards."
]

# ============= FAKE NEWS TEMPLATES =============
FAKE_NEWS_TEMPLATES = [
    # Political Misinformation
    "BREAKING: The Election Commission has been compromised! Secret documents reveal massive vote manipulation planned - SHARE BEFORE DELETED!",
    "SHOCKING EXPOSURE: The Prime Minister secretly signed away India's sovereignty in a hidden treaty! The media won't report this!",
    "CONSPIRACY REVEALED: Foreign powers are controlling Indian politicians through blackmail! The truth they don't want you to know!",
    
    # Health Misinformation
    "URGENT: WHO admits COVID vaccines cause permanent DNA mutation! This doctor reveals the truth that big pharma is hiding!",
    "WARNING: New study proves 5G towers are causing brain tumors! Governments are covering up the evidence! Share to spread awareness!",
    "MIRACLE CURE: This ancient herb eliminates cancer in 3 days! Doctors hate this simple trick! Watch before they take it down!",
    
    # Economic Misinformation
    "YOUR MONEY IS NOT SAFE! The government is planning to confiscate all bank accounts over ₹50,000! Official document leaked!",
    "ALERT: RBI secretly demonetizing ₹2000 notes TOMORROW! Convert your cash immediately or lose everything! 100% confirmed!",
    
    # Sensational Claims
    "EXPOSED: Celebrity caught in human trafficking ring! The mainstream media is protecting their own! Full evidence inside!",
    "THIS WILL CHANGE EVERYTHING! What NASA just discovered at the edge of our solar system will shock the world!",
    "TRENDING: Major news network caught fabricating stories for 10 years! Whistleblower releases damning evidence!",
    "SHOCKING: World leaders are actually AI-controlled robots! Former intelligence officer reveals the truth!"
]

def generate_real_article():
    """Generate a realistic real news article with proper structure"""
    template = random.choice(REAL_NEWS_TEMPLATES)
    
    # Add journalistic elements
    if random.random() > 0.5:
        attribution = random.choice([
            "According to official sources, ", "Government data shows ", 
            "In a press release, ", "The report states that "
        ])
        return attribution + template[0].lower() + template[1:]
    
    return template

def generate_fake_article():
    """Generate a fake news article with sensational patterns"""
    template = random.choice(FAKE_NEWS_TEMPLATES)
    
    # Add urgency and emotional manipulation
    if random.random() > 0.6:
        template = "🚨 " + template + " 🚨"
    
    # Add call to action
    if random.random() > 0.7:
        template += " Share this with everyone you know!"
    
    return template

def create_dataset(num_samples: int = 10000):
    """
    Create balanced dataset of real and fake news
    
    Args:
        num_samples: Total number of samples (half real, half fake)
    """
    half = num_samples // 2
    
    print(f"\n📝 Generating {num_samples} news articles...")
    print(f"   Real news: {half}")
    print(f"   Fake news: {half}\n")
    
    texts = []
    labels = []
    
    # Generate real news
    for i in range(half):
        texts.append(generate_real_article())
        labels.append(0)
        if (i + 1) % 1000 == 0:
            print(f"   Generated {i+1}/{half} real articles")
    
    # Generate fake news
    for i in range(half):
        texts.append(generate_fake_article())
        labels.append(1)
        if (i + 1) % 1000 == 0:
            print(f"   Generated {i+1}/{half} fake articles")
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add metadata
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    df['created_at'] = datetime.now().strftime('%Y-%m-%d')
    
    return df

# Create dataset
df = create_dataset(num_samples=10000)

# Save
df.to_csv('data/raw_dataset.csv', index=False)
print(f"\n💾 Dataset saved to: data/raw_dataset.csv")
print(f"\n📊 DATASET STATISTICS:")
print(f"   Total samples: {len(df):,}")
print(f"   Real news: {len(df[df['label']==0]):,}")
print(f"   Fake news: {len(df[df['label']==1]):,}")
print(f"   Balance: {len(df[df['label']==0])/len(df)*100:.1f}% Real, {len(df[df['label']==1])/len(df)*100:.1f}% Fake")
print(f"   Avg text length: {df['text_length'].mean():.0f} chars")
print(f"   Avg word count: {df['word_count'].mean():.0f} words")

print("\n" + "=" * 60)
print("✅ PHASE 1 COMPLETE")
print("=" * 60)