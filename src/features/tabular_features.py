import pandas as pd
import numpy as np
from pathlib import Path
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER is available in CI
nltk.download("vader_lexicon")

PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# Director Success Rate
# -------------------------------------------------------
def director_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    print("🎬 Computing director success rate...")

    if "Director" in df.columns:
        dir_col = "Director"
    elif "director" in df.columns:
        dir_col = "director"
    else:
        print("⚠️ No director column found.")
        return df

    director_avg = (
        df.groupby(dir_col)["vote_average"]
        .mean()
        .reset_index()
    )

    # Normalize column name for tests
    director_avg.columns = ["director", "director_success_rate"]

    # Save stats file
    director_avg.to_csv(PROCESSED_PATH / "director_stats.csv", index=False)
    print("💾 director_stats.csv saved!")

    df = df.merge(
        director_avg,
        left_on=dir_col,
        right_on="director",
        how="left"
    )

    return df


# -------------------------------------------------------
# Actor Success Rate
# -------------------------------------------------------
def actor_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    print("🌟 Computing actor success rate...")

    actor_cols = [c for c in ["Star1", "Star2", "Star3", "Star4"] if c in df.columns]

    if not actor_cols:
        print("⚠️ No actor columns found.")
        return df

    lead_col = actor_cols[0]

    actor_avg = (
        df.groupby(lead_col)["vote_average"]
        .mean()
        .reset_index()
    )

    actor_avg.columns = ["actor", "lead_actor_success_rate"]

    # Save stats file
    actor_avg.to_csv(PROCESSED_PATH / "actor_stats.csv", index=False)
    print("💾 actor_stats.csv saved!")

    df = df.merge(
        actor_avg,
        left_on=lead_col,
        right_on="actor",
        how="left"
    )

    return df


# -------------------------------------------------------
# Genre Popularity
# -------------------------------------------------------
def genre_popularity(df: pd.DataFrame) -> pd.DataFrame:
    print("🎭 Computing genre popularity...")

    genre_col = None
    if "Genre" in df.columns:
        genre_col = "Genre"
    elif "genres" in df.columns:
        genre_col = "genres"

    if genre_col is None:
        print("⚠️ No genre column found.")
        return df

    genre_avg = (
        df.groupby(genre_col)["vote_average"]
        .mean()
        .reset_index()
    )

    genre_avg.columns = ["genre", "genre_popularity_score"]

    # Save stats file
    genre_avg.to_csv(PROCESSED_PATH / "genre_stats.csv", index=False)
    print("💾 genre_stats.csv saved!")

    df = df.merge(
        genre_avg,
        left_on=genre_col,
        right_on="genre",
        how="left"
    )

    return df


# -------------------------------------------------------
# Text Features (Word Count + VADER Sentiment)
# -------------------------------------------------------
def text_features(df: pd.DataFrame) -> pd.DataFrame:
    print("📝 Computing text features...")

    if "overview" in df.columns:
        df["overview_word_count"] = (
            df["overview"].fillna("").apply(lambda x: len(str(x).split()))
        )

        sia = SentimentIntensityAnalyzer()
        df["overview_vader_compound"] = (
            df["overview"]
            .fillna("")
            .apply(lambda x: sia.polarity_scores(str(x))["compound"])
        )
    else:
        df["overview_word_count"] = 0
        df["overview_vader_compound"] = 0

    print("✅ Text features added!")
    return df


# -------------------------------------------------------
# Budget / Revenue Features
# -------------------------------------------------------
def budget_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    print("💰 Computing budget/revenue features...")

    if "budget" in df.columns and "revenue" in df.columns:

        df["log_budget"] = np.log1p(df["budget"])
        df["log_revenue"] = np.log1p(df["revenue"])

        if "popularity" in df.columns:
            df["log_popularity"] = np.log1p(df["popularity"])
        else:
            df["log_popularity"] = 0

        df["budget_tier"] = pd.qcut(
            df["budget"],
            q=4,
            labels=["low", "medium", "high", "blockbuster"],
            duplicates="drop",
        )

    return df


# -------------------------------------------------------
# Release Trend Features
# -------------------------------------------------------
def release_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    print("📅 Computing release trend features...")

    if "release_year" in df.columns:
        df["decade"] = (df["release_year"] // 10) * 10
        df["is_modern"] = (df["release_year"] >= 2000).astype(int)

    if "release_month" in df.columns:
        df["release_season"] = df["release_month"].apply(
            lambda m: "winter" if m in [12, 1, 2]
            else "spring" if m in [3, 4, 5]
            else "summer" if m in [6, 7, 8]
            else "fall"
        )

        df["is_summer_release"] = df["release_month"].isin([6, 7, 8]).astype(int)

    return df


# -------------------------------------------------------
# Encode Categoricals
# -------------------------------------------------------
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    print("🔢 Encoding categorical columns...")

    if "budget_tier" in df.columns:
        tier_map = {"low": 0, "medium": 1, "high": 2, "blockbuster": 3}
        df["budget_tier_encoded"] = df["budget_tier"].map(tier_map).fillna(0)

    if "release_season" in df.columns:
        season_map = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}
        df["release_season_encoded"] = df["release_season"].map(season_map).fillna(0)

    if "original_language" in df.columns:
        df["is_english"] = (df["original_language"] == "en").astype(int)

    return df


# -------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------
def run_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🚀 Starting tabular feature engineering...\n")

    df = director_success_rate(df)
    df = actor_success_rate(df)
    df = genre_popularity(df)
    df = text_features(df)
    df = budget_revenue_features(df)
    df = release_trend_features(df)
    df = encode_categoricals(df)

    print(f"\n✅ Tabular features complete! Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/movies_final.csv")
    df = run_tabular_features(df)
    print(df.head())