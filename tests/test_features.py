import pytest
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed")
MODEL_PATH = Path("models")

class TestFeatureEngineering:
    """Tests for feature engineering pipeline."""

    def test_features_full_exists(self):
        """Check features_full.csv exists."""
        assert (PROCESSED_DATA_PATH / "features_full.csv").exists()

    def test_vader_features_exist(self):
        """Check VADER sentiment features are present."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "features_full.csv")
        vader_cols = [
            'overview_vader_pos', 'overview_vader_neg',
            'overview_vader_neu', 'overview_vader_compound',
            'tagline_vader_pos', 'tagline_vader_neg',
            'tagline_vader_neu', 'tagline_vader_compound'
        ]
        for col in vader_cols:
            assert col in df.columns, f"Missing VADER feature: {col}"

    def test_vader_values_in_range(self):
        """Check VADER scores are within valid range."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "features_full.csv")
        assert df['overview_vader_compound'].min() >= -1.0
        assert df['overview_vader_compound'].max() <= 1.0
        assert df['overview_vader_pos'].min() >= 0.0
        assert df['overview_vader_pos'].max() <= 1.0

    def test_engineered_features_exist(self):
        """Check engineered tabular features exist."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "features_full.csv")
        engineered_cols = [
            'profit', 'roi', 'log_budget', 'log_revenue',
            'log_popularity', 'overview_length', 'overview_word_count',
            'tagline_word_count', 'decade', 'is_modern',
            'is_summer_release', 'has_tagline', 'is_english'
        ]
        for col in engineered_cols:
            assert col in df.columns, f"Missing engineered feature: {col}"

    def test_log_features_non_negative(self):
        """Check log features are non-negative."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "features_full.csv")
        assert df['log_budget'].min() >= 0
        assert df['log_revenue'].min() >= 0
        assert df['log_popularity'].min() >= 0

    def test_binary_features_valid(self):
        """Check binary features only contain 0 or 1."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "features_full.csv")
        binary_cols = ['is_modern', 'is_summer_release', 'has_tagline', 'is_english']
        for col in binary_cols:
            unique_vals = set(df[col].unique())
            assert unique_vals.issubset({0, 1}), \
                f"Binary feature {col} has invalid values: {unique_vals}"

    def test_director_stats_exist(self):
        """Check director lookup table exists."""
        assert (PROCESSED_DATA_PATH / "director_stats.csv").exists()

    def test_actor_stats_exist(self):
        """Check actor lookup table exists."""
        assert (PROCESSED_DATA_PATH / "actor_stats.csv").exists()

    def test_genre_stats_exist(self):
        """Check genre lookup table exists."""
        assert (PROCESSED_DATA_PATH / "genre_stats.csv").exists()

    def test_director_stats_valid(self):
        """Check director stats are valid."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "director_stats.csv")
        assert 'director' in df.columns
        assert 'success_rate' in df.columns
        assert df['success_rate'].min() >= 0
        assert df['success_rate'].max() <= 10

    def test_feature_cols_saved(self):
        """Check feature_cols.pkl exists."""
        import joblib
        assert (MODEL_PATH / "feature_cols.pkl").exists()
        feature_cols = joblib.load(MODEL_PATH / "feature_cols.pkl")
        assert len(feature_cols) > 0, "feature_cols is empty!"
        assert isinstance(feature_cols, list), "feature_cols should be a list!"

    def test_scaler_saved(self):
        """Check scaler.pkl exists."""
        assert (MODEL_PATH / "scaler.pkl").exists()

    def test_word_count_positive(self):
        """Check word counts are positive."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "features_full.csv")
        assert df['overview_word_count'].min() >= 0
        assert df['tagline_word_count'].min() >= 0