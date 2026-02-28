import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.features.text_preprocessor import preprocess_dataframe

# Initialize VADER
VADER = SentimentIntensityAnalyzer()

def vader_sentiment(text: str) -> dict:
    """Get VADER sentiment scores for a text."""
    if not isinstance(text, str) or len(text) == 0:
        return {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
    scores = VADER.polarity_scores(text)
    return {
        'pos': scores['pos'],
        'neg': scores['neg'],
        'neu': scores['neu'],
        'compound': scores['compound']
    }

def add_vader_features(df: pd.DataFrame, text_col: str, prefix: str) -> pd.DataFrame:
    """Add VADER sentiment features for a text column."""
    print(f"😊 Computing VADER sentiment for '{text_col}'...")

    if text_col not in df.columns:
        print(f"⚠️ Column '{text_col}' not found, skipping...")
        return df

    sentiments = df[text_col].fillna('').apply(vader_sentiment)
    df[f'{prefix}_vader_pos'] = sentiments.apply(lambda x: x['pos'])
    df[f'{prefix}_vader_neg'] = sentiments.apply(lambda x: x['neg'])
    df[f'{prefix}_vader_neu'] = sentiments.apply(lambda x: x['neu'])
    df[f'{prefix}_vader_compound'] = sentiments.apply(lambda x: x['compound'])

    print(f"✅ VADER features added for '{text_col}'!")
    return df

def add_distilbert_features(df: pd.DataFrame, text_col: str, prefix: str) -> pd.DataFrame:
    """Add DistilBERT sentiment features for a text column."""
    print(f"🤖 Computing DistilBERT sentiment for '{text_col}'...")

    try:
        from transformers import pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512
        )

        def get_distilbert_score(text):
            if not isinstance(text, str) or len(text.strip()) == 0:
                return 0.0
            try:
                result = classifier(text[:512])[0]
                score = result['score']
                return score if result['label'] == 'POSITIVE' else -score
            except:
                return 0.0

        print("  ⏳ This may take a few minutes...")
        df[f'{prefix}_distilbert_score'] = df[text_col].fillna('').apply(get_distilbert_score)
        print(f"✅ DistilBERT features added for '{text_col}'!")

    except Exception as e:
        print(f"⚠️ DistilBERT failed: {e}")
        print("  Filling with zeros...")
        df[f'{prefix}_distilbert_score'] = 0.0

    return df

def add_text_stats(df: pd.DataFrame, text_col: str, prefix: str) -> pd.DataFrame:
    """Add basic text statistics as features."""
    print(f"📊 Computing text stats for '{text_col}'...")

    if text_col not in df.columns:
        return df

    texts = df[text_col].fillna('')
    df[f'{prefix}_word_count'] = texts.apply(lambda x: len(x.split()))
    df[f'{prefix}_char_count'] = texts.apply(len)
    df[f'{prefix}_avg_word_length'] = texts.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )

    print(f"✅ Text stats added for '{text_col}'!")
    return df

def run_nlp_pipeline(df: pd.DataFrame, use_distilbert: bool = False) -> pd.DataFrame:
    """Full NLP feature pipeline."""
    print("\n🚀 Starting NLP feature pipeline...\n")

    # Preprocess text columns
    text_cols = ['overview', 'tagline']
    df = preprocess_dataframe(df, text_cols)

    # VADER sentiment on overview
    df = add_vader_features(df, 'overview', 'overview')

    # VADER sentiment on tagline
    df = add_vader_features(df, 'tagline', 'tagline')

    # Text statistics
    df = add_text_stats(df, 'overview', 'overview')
    df = add_text_stats(df, 'tagline', 'tagline')

    # DistilBERT (optional - takes longer)
    if use_distilbert:
        df = add_distilbert_features(df, 'overview', 'overview')

    print(f"\n✅ NLP pipeline complete! Shape: {df.shape}")
    return df

if __name__ == "__main__":
    from pathlib import Path

    print("📂 Loading data...")
    df = pd.read_csv("data/processed/movies_final.csv")
    print(f"✅ Loaded {len(df)} rows")

    # Run NLP pipeline (without DistilBERT for speed)
    df = run_nlp_pipeline(df, use_distilbert=False)

    # Save
    Path("data/processed").mkdir(exist_ok=True)
    df.to_csv("data/processed/movies_nlp.csv", index=False)
    print("\n💾 Saved to data/processed/movies_nlp.csv")

    # Show sample
    nlp_cols = [c for c in df.columns if 'vader' in c or 'word_count' in c]
    print(f"\n📋 NLP Features created: {nlp_cols}")
    print(df[['title'] + nlp_cols].head(5))