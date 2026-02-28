import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed")

def load_clean_data() -> pd.DataFrame:
    """Load the cleaned dataset."""
    filepath = PROCESSED_DATA_PATH / "movies_clean.csv"
    print(f"📂 Loading cleaned data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers from numeric columns."""
    print("🔍 Removing outliers...")
    initial = len(df)

    # Remove movies with unrealistic budgets
    if 'budget' in df.columns:
        df = df[df['budget'] >= 0]

    # Remove movies with unrealistic revenue
    if 'revenue' in df.columns:
        df = df[df['revenue'] >= 0]

    # Remove movies with unrealistic runtime
    if 'runtime' in df.columns:
        df = df[(df['runtime'] >= 30) & (df['runtime'] <= 360)]

    # Remove rating outliers (keep between 1 and 10)
    if 'vote_average' in df.columns:
        df = df[(df['vote_average'] >= 1) & (df['vote_average'] <= 10)]

    print(f"✅ Outliers removed: {initial} → {len(df)} rows")
    return df

def fix_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data types for each column."""
    print("🔧 Fixing data types...")

    # Convert release_date to datetime
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month

    # Convert budget and revenue to float
    if 'budget' in df.columns:
        df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0)
    if 'revenue' in df.columns:
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)

    # Convert runtime to float
    if 'runtime' in df.columns:
        df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(df['runtime'].median())

    print("✅ Data types fixed!")
    return df

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add some basic derived features."""
    print("➕ Adding basic features...")

    # Budget to revenue ratio
    if 'budget' in df.columns and 'revenue' in df.columns:
        df['profit'] = df['revenue'] - df['budget']
        df['roi'] = df.apply(
            lambda x: x['revenue'] / x['budget'] if x['budget'] > 0 else 0, axis=1
        )

    # Overview length as a feature
    if 'overview' in df.columns:
        df['overview_length'] = df['overview'].fillna('').apply(len)

    # Tagline presence
    if 'tagline' in df.columns:
        df['has_tagline'] = df['tagline'].fillna('').apply(lambda x: 1 if len(str(x)) > 0 else 0)

    print("✅ Basic features added!")
    return df

def save_final(df: pd.DataFrame):
    """Save the final cleaned dataset."""
    filepath = PROCESSED_DATA_PATH / "movies_final.csv"
    df.to_csv(filepath, index=False)
    print(f"💾 Final dataset saved to {filepath}")

def run_cleaning():
    """Full cleaning pipeline."""
    print("\n🚀 Starting cleaning pipeline...\n")
    df = load_clean_data()
    df = remove_outliers(df)
    df = fix_datatypes(df)
    df = add_basic_features(df)
    save_final(df)
    print(f"\n✅ Cleaning complete! Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = run_cleaning()
    print(df[['title', 'vote_average', 'profit', 'roi']].head(10))