import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(".")
from src.features.tabular_features import run_tabular_features

PROCESSED_DATA_PATH = Path("data/processed")


def load_data() -> pd.DataFrame:
    filepath = PROCESSED_DATA_PATH / "movies_final.csv"
    df = pd.read_csv(filepath)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "vote_average",
        "budget", "revenue", "popularity", "runtime",
        "vote_count", "profit", "roi",
        "log_budget", "log_revenue", "log_popularity",

        # Overview features
        "overview_length",
        "overview_word_count",
        "overview_vader_pos",
        "overview_vader_neg",
        "overview_vader_neu",
        "overview_vader_compound",

        # Tagline features
        "has_tagline",
        "tagline_word_count",
        "tagline_vader_pos",
        "tagline_vader_neg",
        "tagline_vader_neu",
        "tagline_vader_compound",

        # Aggregations
        "director_success_rate",
        "lead_actor_success_rate",
        "genre_popularity_score",

        # Time features
        "release_year",
        "release_month",
        "decade",
        "is_modern",
        "is_summer_release",

        # Encoded
        "budget_tier_encoded",
        "is_english",
    ]

    available = [c for c in feature_cols if c in df.columns]
    df = df[available + ["title"]].copy()
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df


def save_features(df: pd.DataFrame):
    filepath = PROCESSED_DATA_PATH / "features_full.csv"
    df.to_csv(filepath, index=False)


def run_feature_pipeline():
    df = load_data()
    df = run_tabular_features(df)
    df = select_features(df)
    df = handle_missing(df)
    save_features(df)
    return df


if __name__ == "_main_":
    df = run_feature_pipeline()
    print(df.shape)