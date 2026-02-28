import pandas as pd
import numpy as np
from pathlib import Path
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER works in CI
nltk.download("vader_lexicon")

PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# TEXT FEATURES
# --------------------------------------------------

def text_features(df: pd.DataFrame) -> pd.DataFrame:
    print("📝 Computing text features...")

    sia = SentimentIntensityAnalyzer()

    # =========================
    # OVERVIEW FEATURES
    # =========================
    if "overview" in df.columns:
        df["overview"] = df["overview"].fillna("")

        df["overview_length"] = df["overview"].apply(len)
        df["overview_word_count"] = df["overview"].apply(
            lambda x: len(str(x).split())
        )

        df["overview_vader_pos"] = df["overview"].apply(
            lambda x: sia.polarity_scores(str(x))["pos"]
        )
        df["overview_vader_neg"] = df["overview"].apply(
            lambda x: sia.polarity_scores(str(x))["neg"]
        )
        df["overview_vader_neu"] = df["overview"].apply(
            lambda x: sia.polarity_scores(str(x))["neu"]
        )
        df["overview_vader_compound"] = df["overview"].apply(
            lambda x: sia.polarity_scores(str(x))["compound"]
        )
    else:
        df["overview_length"] = 0
        df["overview_word_count"] = 0
        df["overview_vader_pos"] = 0
        df["overview_vader_neg"] = 0
        df["overview_vader_neu"] = 0
        df["overview_vader_compound"] = 0

    # =========================
    # TAGLINE FEATURES
    # =========================
    if "tagline" in df.columns:
        df["tagline"] = df["tagline"].fillna("")

        df["has_tagline"] = (df["tagline"].str.strip() != "").astype(int)

        df["tagline_word_count"] = df["tagline"].apply(
            lambda x: len(str(x).split())
        )

        df["tagline_vader_pos"] = df["tagline"].apply(
            lambda x: sia.polarity_scores(str(x))["pos"]
        )
        df["tagline_vader_neg"] = df["tagline"].apply(
            lambda x: sia.polarity_scores(str(x))["neg"]
        )
        df["tagline_vader_neu"] = df["tagline"].apply(
            lambda x: sia.polarity_scores(str(x))["neu"]
        )
        df["tagline_vader_compound"] = df["tagline"].apply(
            lambda x: sia.polarity_scores(str(x))["compound"]
        )
    else:
        df["has_tagline"] = 0
        df["tagline_word_count"] = 0
        df["tagline_vader_pos"] = 0
        df["tagline_vader_neg"] = 0
        df["tagline_vader_neu"] = 0
        df["tagline_vader_compound"] = 0

    print("✅ Text features added!")
    return df


# --------------------------------------------------
# DIRECTOR SUCCESS RATE
# --------------------------------------------------

def director_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    print("🎬 Computing director success rate...")

    if "Director" in df.columns:
        dir_col = "Director"
    elif "director" in df.columns:
        dir_col = "director"
    else:
        return df

    director_avg = (
        df.groupby(dir_col)["vote_average"]
        .mean()
        .reset_index()
    )

    # Save EXACT format for tests
    director_avg.columns = ["director", "success_rate"]
    director_avg.to_csv(PROCESSED_PATH / "director_stats.csv", index=False)

    df = df.merge(
        director_avg,
        left_on=dir_col,
        right_on="director",
        how="left"
    )

    df.rename(columns={"success_rate": "director_success_rate"}, inplace=True)

    return df


# --------------------------------------------------
# ACTOR SUCCESS RATE
# --------------------------------------------------

def actor_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    print("🌟 Computing actor success rate...")

    actor_cols = [c for c in ["Star1", "Star2", "Star3", "Star4"] if c in df.columns]

    if not actor_cols:
        return df

    lead_col = actor_cols[0]

    actor_avg = (
        df.groupby(lead_col)["vote_average"]
        .mean()
        .reset_index()
    )

    actor_avg.columns = ["actor", "success_rate"]
    actor_avg.to_csv(PROCESSED_PATH / "actor_stats.csv", index=False)

    df = df.merge(
        actor_avg,
        left_on=lead_col,
        right_on="actor",
        how="left"
    )

    df.rename(columns={"success_rate": "lead_actor_success_rate"}, inplace=True)

    return df


# --------------------------------------------------
# GENRE POPULARITY
# --------------------------------------------------

def genre_popularity(df: pd.DataFrame) -> pd.DataFrame:
    print("🎭 Computing genre popularity...")

    if "Genre" in df.columns:
        genre_col = "Genre"
    elif "genres" in df.columns:
        genre_col = "genres"
    else:
        return df

    genre_avg = (
        df.groupby(genre_col)["vote_average"]
        .mean()
        .reset_index()
    )

    genre_avg.columns = ["genre", "success_rate"]
    genre_avg.to_csv(PROCESSED_PATH / "genre_stats.csv", index=False)

    df = df.merge(
        genre_avg,
        left_on=genre_col,
        right_on="genre",
        how="left"
    )

    df.rename(columns={"success_rate": "genre_popularity_score"}, inplace=True)

    return df


# --------------------------------------------------
# NUMERIC FEATURES
# --------------------------------------------------

def budget_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    if "budget" in df.columns:
        df["log_budget"] = np.log1p(df["budget"])

    if "revenue" in df.columns:
        df["log_revenue"] = np.log1p(df["revenue"])

    if "popularity" in df.columns:
        df["log_popularity"] = np.log1p(df["popularity"])

    if "budget" in df.columns:
        df["budget_tier"] = pd.qcut(
            df["budget"],
            q=4,
            labels=["low", "medium", "high", "blockbuster"],
            duplicates="drop"
        )

    return df


def release_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    if "release_year" in df.columns:
        df["decade"] = (df["release_year"] // 10) * 10
        df["is_modern"] = (df["release_year"] >= 2000).astype(int)

    if "release_month" in df.columns:
        df["is_summer_release"] = df["release_month"].isin([6, 7, 8]).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    if "budget_tier" in df.columns:
        tier_map = {"low": 0, "medium": 1, "high": 2, "blockbuster": 3}
        df["budget_tier_encoded"] = df["budget_tier"].map(tier_map).fillna(0)

    if "original_language" in df.columns:
        df["is_english"] = (df["original_language"] == "en").astype(int)

    return df


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def run_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    df = text_features(df)
    df = director_success_rate(df)
    df = actor_success_rate(df)
    df = genre_popularity(df)
    df = budget_revenue_features(df)
    df = release_trend_features(df)
    df = encode_categoricals(df)
    return df