import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Test data paths
PROCESSED_DATA_PATH = Path("data/processed")
RAW_DATA_PATH = Path("data/raw")

class TestDataIngestion:
    """Tests for data ingestion pipeline."""

    def test_processed_data_exists(self):
        """Check that processed data files exist."""
        assert (PROCESSED_DATA_PATH / "movies_clean.csv").exists(), \
            "movies_clean.csv not found!"
        assert (PROCESSED_DATA_PATH / "movies_final.csv").exists(), \
            "movies_final.csv not found!"

    def test_movies_final_not_empty(self):
        """Check that movies_final.csv has data."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "movies_final.csv")
        assert len(df) > 0, "movies_final.csv is empty!"
        assert len(df) > 100, "Dataset too small — less than 100 movies!"

    def test_required_columns_exist(self):
        """Check that all required columns are present."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "movies_final.csv")
        required_cols = [
            'title', 'budget', 'revenue', 'popularity',
            'runtime', 'vote_count', 'vote_average',
            'Director', 'Star1', 'Genre', 'release_year', 'release_month'
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_vote_average_range(self):
        """Check that vote_average is within valid range."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "movies_final.csv")
        assert df['vote_average'].min() >= 0, "vote_average below 0!"
        assert df['vote_average'].max() <= 10, "vote_average above 10!"

    def test_no_duplicate_titles(self):
        """Check for duplicate movie titles."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "movies_final.csv")
        duplicate_count = df['title'].duplicated().sum()
        assert duplicate_count < len(df) * 0.1, \
            f"Too many duplicate titles: {duplicate_count}"

    def test_missing_values_acceptable(self):
        """Check that critical columns don't have too many missing values."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "movies_final.csv")
        critical_cols = ['vote_average', 'budget', 'revenue', 'runtime']
        for col in critical_cols:
            missing_pct = df[col].isnull().sum() / len(df)
            assert missing_pct < 0.5, \
                f"Column {col} has {missing_pct:.1%} missing values!"

    def test_features_file_exists(self):
        """Check that features file exists."""
        assert (PROCESSED_DATA_PATH / "features_full.csv").exists(), \
            "features_full.csv not found!"

    def test_features_has_correct_shape(self):
        """Check features file has correct number of columns."""
        df = pd.read_csv(PROCESSED_DATA_PATH / "features_full.csv")
        assert df.shape[1] >= 30, \
            f"Features file has too few columns: {df.shape[1]}"