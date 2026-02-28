import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('.')
from src.features.tabular_features import run_tabular_features

PROCESSED_DATA_PATH = Path("data/processed")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load the final cleaned dataset."""
    filepath = PROCESSED_DATA_PATH / "movies_final.csv"
    print(f"📂 Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# --------------------------------------------------
# SELECT FEATURES (CRITICAL FIX HERE)
# --------------------------------------------------

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the useful ML features."""
    print("🎯 Selecting features...")

    feature_cols = [
        # Target
        'vote_average',

        # Base numeric features
        'budget', 'revenue', 'popularity', 'runtime',
        'vote_count', 'profit', 'roi',

        # Log features
        'log_budget', 'log_revenue', 'log_popularity',

        # TEXT FEATURES (REQUIRED FOR TESTS)
        'overview_word_count',
        'overview_vader_pos',
        'overview_vader_neg',
        'overview_vader_neu',
        'overview_vader_compound',

        # Engineered aggregation features
        'director_success_rate',
        'lead_actor_success_rate',
        'genre_popularity_score',

        # Time-based features
        'release_year',
        'release_month',
        'decade',
        'is_modern',
        'is_summer_release',

        # Encoded categoricals
        'budget_tier_encoded',
        'release_season_encoded',
        'is_english',
    ]

    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]

    if missing:
        print(f"⚠️ Missing columns (will skip): {missing}")

    df = df[available + ['title']].copy()

    print(f"✅ Selected {len(available)} features!")
    return df


# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle any remaining missing values."""
    print("🔧 Handling missing values...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    print("✅ Missing values handled!")
    return df


# --------------------------------------------------
# SAVE FEATURES
# --------------------------------------------------

def save_features(df: pd.DataFrame):
    """Save the feature dataset."""
    filepath = PROCESSED_DATA_PATH / "features_full.csv"
    df.to_csv(filepath, index=False)
    print(f"💾 Features saved to {filepath}")


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_feature_pipeline():
    """Full feature engineering pipeline."""
    print("\n🚀 Running full feature pipeline...\n")

    # Load cleaned dataset
    df = load_data()

    # Apply feature engineering
    df = run_tabular_features(df)

    # Select final ML features
    df = select_features(df)

    # Handle missing values
    df = handle_missing(df)

    # Save final feature set
    save_features(df)

    print("\n✅ Feature pipeline complete!")
    print(f"📊 Final feature set shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = run_feature_pipeline()

    print("\n📋 Final feature columns:")
    print([c for c in df.columns if c != 'title'])

    print("\n🔍 Sample data:")
    print(df.head())