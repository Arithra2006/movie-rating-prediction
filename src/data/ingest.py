import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed")

def load_tmdb(filepath: str) -> pd.DataFrame:
    print(f"📂 Loading TMDB movies...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"✅ TMDB: {len(df)} rows, columns: {list(df.columns)}")
    return df

def load_imdb(filepath: str) -> pd.DataFrame:
    print(f"📂 Loading IMDB dataset...")
    df = pd.read_csv(filepath, encoding='latin-1')
    print(f"✅ IMDB: {len(df)} rows, columns: {list(df.columns)}")
    return df

def merge_all(tmdb_df: pd.DataFrame, imdb_df: pd.DataFrame) -> pd.DataFrame:
    print("🔗 Merging TMDB + IMDB...")
    tmdb_df['title_clean'] = tmdb_df['title'].str.strip().str.lower()
    imdb_df['title_clean'] = imdb_df['Series_Title'].str.strip().str.lower()
    merged = tmdb_df.merge(imdb_df, on='title_clean', how='left')
    print(f"✅ Merged: {len(merged)} rows, {len(merged.columns)} columns")
    return merged

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹 Cleaning dataset...")
    initial_rows = len(df)
    df = df.copy()
    df = df.drop_duplicates(subset=['title'])
    df = df.dropna(subset=['vote_average'])
    df = df[df['vote_average'] > 0]
    if 'vote_count' in df.columns:
        df = df[df['vote_count'] >= 10]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].fillna('')
    print(f"✅ Cleaned: {initial_rows} → {len(df)} rows")
    return df

def save_processed(df: pd.DataFrame, filename: str):
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DATA_PATH / filename
    df.to_csv(filepath, index=False)
    print(f"💾 Saved to {filepath}")

def run_ingestion():
    print("\n🚀 Starting data ingestion...\n")
    tmdb_df = load_tmdb("data/raw/tmdb_5000_movies.csv")
    imdb_df = load_imdb("data/raw/imdb_top_1000.csv")
    df = merge_all(tmdb_df, imdb_df)
    df = basic_clean(df)
    save_processed(df, "movies_clean.csv")
    print("\n✅ Ingestion complete!")
    return df

if __name__ == "__main__":
    df = run_ingestion()
    print(f"\n📊 Final dataset shape: {df.shape}")
    print(df[['title', 'vote_average']].head(10))