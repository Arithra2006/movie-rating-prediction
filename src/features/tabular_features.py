import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


def director_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average rating per director and save stats."""
    print("🎬 Computing director success rate...")

    if 'Director' in df.columns:
        dir_col = 'Director'
    elif 'director' in df.columns:
        dir_col = 'director'
    else:
        print("⚠️ No director column found, skipping...")
        return df

    director_avg = (
        df.groupby(dir_col)['vote_average']
        .mean()
        .reset_index()
    )

    director_avg.columns = [dir_col, 'director_success_rate']

    # ✅ Save director stats
    director_avg.to_csv(PROCESSED_PATH / "director_stats.csv", index=False)
    print("💾 director_stats.csv saved!")

    df = df.merge(director_avg, on=dir_col, how='left')
    print("✅ Director success rate added!")
    return df


def actor_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average rating per lead actor and save stats."""
    print("🌟 Computing actor success rate...")

    actor_cols = [c for c in ['Star1', 'Star2', 'Star3', 'Star4'] if c in df.columns]

    if not actor_cols:
        print("⚠️ No actor columns found, skipping...")
        return df

    lead_col = actor_cols[0]

    actor_avg = (
        df.groupby(lead_col)['vote_average']
        .mean()
        .reset_index()
    )

    actor_avg.columns = [lead_col, 'lead_actor_success_rate']

    # ✅ Save actor stats
    actor_avg.to_csv(PROCESSED_PATH / "actor_stats.csv", index=False)
    print("💾 actor_stats.csv saved!")

    df = df.merge(actor_avg, on=lead_col, how='left')
    print("✅ Actor success rate added!")
    return df


def genre_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate genre popularity score."""
    print("🎭 Computing genre popularity...")

    genre_col = None
    if 'Genre' in df.columns:
        genre_col = 'Genre'
    elif 'genres' in df.columns:
        genre_col = 'genres'

    if genre_col is None:
        print("⚠️ No genre column found, skipping...")
        return df

    genre_avg = (
        df.groupby(genre_col)['vote_average']
        .mean()
        .reset_index()
    )

    genre_avg.columns = [genre_col, 'genre_popularity_score']
    df = df.merge(genre_avg, on=genre_col, how='left')

    print("✅ Genre popularity added!")
    return df


def budget_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add budget and revenue ratio features."""
    print("💰 Computing budget/revenue features...")

    if 'budget' in df.columns and 'revenue' in df.columns:

        df['log_budget'] = np.log1p(df['budget'])
        df['log_revenue'] = np.log1p(df['revenue'])

        if 'popularity' in df.columns:
            df['log_popularity'] = np.log1p(df['popularity'])
        else:
            df['log_popularity'] = 0

        df['budget_tier'] = pd.qcut(
            df['budget'],
            q=4,
            labels=['low', 'medium', 'high', 'blockbuster'],
            duplicates='drop'
        )

    print("✅ Budget/revenue features added!")
    return df


def release_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add release year and month trend features."""
    print("📅 Computing release trend features...")

    if 'release_year' in df.columns:
        df['decade'] = (df['release_year'] // 10) * 10
        df['is_modern'] = (df['release_year'] >= 2000).astype(int)

    if 'release_month' in df.columns:
        df['release_season'] = df['release_month'].apply(
            lambda m: 'winter' if m in [12, 1, 2]
            else 'spring' if m in [3, 4, 5]
            else 'summer' if m in [6, 7, 8]
            else 'fall'
        )

        df['is_summer_release'] = df['release_month'].isin([6, 7, 8]).astype(int)

    print("✅ Release trend features added!")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns."""
    print("🔢 Encoding categorical columns...")

    if 'budget_tier' in df.columns:
        tier_map = {'low': 0, 'medium': 1, 'high': 2, 'blockbuster': 3}
        df['budget_tier_encoded'] = df['budget_tier'].map(tier_map).fillna(0)

    if 'release_season' in df.columns:
        season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
        df['release_season_encoded'] = df['release_season'].map(season_map).fillna(0)

    if 'original_language' in df.columns:
        df['is_english'] = (df['original_language'] == 'en').astype(int)

    print("✅ Categorical encoding done!")
    return df


def run_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all tabular feature engineering steps."""
    print("\n🚀 Starting tabular feature engineering...\n")

    df = director_success_rate(df)
    df = actor_success_rate(df)
    df = genre_popularity(df)
    df = budget_revenue_features(df)
    df = release_trend_features(df)
    df = encode_categoricals(df)

    print(f"\n✅ Tabular features done! Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/movies_final.csv")
    df = run_tabular_features(df)
    print(df.head())