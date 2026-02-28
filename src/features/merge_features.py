import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed")

def merge_all_features():
    """Merge tabular + NLP features into one final feature set."""
    print("\n🚀 Merging all features...\n")

    # Load tabular features
    print("📂 Loading tabular features...")
    tabular_df = pd.read_csv(PROCESSED_DATA_PATH / "features.csv")
    print(f"✅ Tabular: {tabular_df.shape}")

    # Load NLP features
    print("📂 Loading NLP features...")
    nlp_df = pd.read_csv(PROCESSED_DATA_PATH / "movies_nlp.csv")
    print(f"✅ NLP: {nlp_df.shape}")

    # NLP columns to add
    nlp_cols = [
        'title',
        'overview_vader_pos',
        'overview_vader_neg',
        'overview_vader_neu',
        'overview_vader_compound',
        'tagline_vader_pos',
        'tagline_vader_neg',
        'tagline_vader_neu',
        'tagline_vader_compound',
        'overview_word_count',
        'tagline_word_count'
    ]

    # Keep only available NLP cols
    available_nlp = [c for c in nlp_cols if c in nlp_df.columns]
    nlp_subset = nlp_df[available_nlp].copy()

    # Merge on title
    print("🔗 Merging on title...")
    merged = tabular_df.merge(nlp_subset, on='title', how='left')

    # Fill any missing NLP values with 0
    nlp_feature_cols = [c for c in available_nlp if c != 'title']
    for col in nlp_feature_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    print(f"✅ Merged shape: {merged.shape}")

    # Save
    merged.to_csv(PROCESSED_DATA_PATH / "features_full.csv", index=False)
    print(f"💾 Saved to data/processed/features_full.csv")

    print(f"\n📋 All feature columns:")
    feature_cols = [c for c in merged.columns if c not in ['title', 'vote_average']]
    print(feature_cols)

    return merged

if __name__ == "__main__":
    df = merge_all_features()
    print(f"\n✅ Final merged dataset: {df.shape}")
    print(f"📊 Features: {len(df.columns) - 2} (excluding title and target)")