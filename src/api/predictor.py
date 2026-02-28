import numpy as np
import pandas as pd
import joblib
import sys
sys.path.append('.')
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_PATH = Path("models")
PROCESSED_DATA_PATH = Path("data/processed")

# Initialize VADER
VADER = SentimentIntensityAnalyzer()

# Load lookup tables at startup
def load_lookup_tables():
    try:
        director_stats = pd.read_csv(PROCESSED_DATA_PATH / "director_stats.csv")
        director_lookup = dict(zip(director_stats['director'], director_stats['success_rate']))

        actor_stats = pd.read_csv(PROCESSED_DATA_PATH / "actor_stats.csv")
        actor_lookup = dict(zip(actor_stats['actor'], actor_stats['success_rate']))

        genre_stats = pd.read_csv(PROCESSED_DATA_PATH / "genre_stats.csv")
        genre_lookup = dict(zip(genre_stats['genre'], genre_stats['success_rate']))

        print("✅ Lookup tables loaded!")
        print(f"   Directors: {len(director_lookup)}")
        print(f"   Actors: {len(actor_lookup)}")
        print(f"   Genres: {len(genre_lookup)}")

        return director_lookup, actor_lookup, genre_lookup
    except Exception as e:
        print(f"⚠️ Could not load lookup tables: {e}")
        return {}, {}, {}

DIRECTOR_LOOKUP, ACTOR_LOOKUP, GENRE_LOOKUP = load_lookup_tables()

class MoviePredictor:
    """Handles all prediction logic."""

    def __init__(self):
        self.xgb_model = None
        self.lgbm_model = None
        self.meta_model = None
        self.scaler = None
        self.feature_cols = None
        self.explainer = None
        self.is_loaded = False

    def load_models(self):
        """Load all models from disk."""
        print("📂 Loading models...")
        try:
            self.xgb_model = joblib.load(MODEL_PATH / "xgb_model.pkl")
            self.lgbm_model = joblib.load(MODEL_PATH / "lgbm_model.pkl")
            self.meta_model = joblib.load(MODEL_PATH / "meta_model.pkl")
            self.scaler = joblib.load(MODEL_PATH / "scaler.pkl")
            self.feature_cols = joblib.load(MODEL_PATH / "feature_cols.pkl")
            self.explainer = joblib.load(MODEL_PATH / "shap_explainer.pkl")
            self.is_loaded = True
            print("✅ All models loaded!")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.is_loaded = False

    def extract_features(self, movie_input) -> np.ndarray:
        """Extract features from movie input."""

        # VADER sentiment on overview and tagline
        overview_scores = VADER.polarity_scores(movie_input.overview or "")
        tagline_scores = VADER.polarity_scores(movie_input.tagline or "")

        # Real lookup values instead of hardcoded 6.5
        global_mean = 6.5
        director_rate = DIRECTOR_LOOKUP.get(movie_input.director, global_mean)
        actor_rate = ACTOR_LOOKUP.get(movie_input.star1, global_mean)
        genre_rate = GENRE_LOOKUP.get(movie_input.genre, global_mean)

        # Build feature dict matching training features
        features = {
            # Tabular features
            'budget': movie_input.budget,
            'revenue': movie_input.revenue,
            'popularity': movie_input.popularity,
            'runtime': movie_input.runtime,
            'vote_count': movie_input.vote_count,
            'profit': movie_input.revenue - movie_input.budget,
            'roi': movie_input.revenue / movie_input.budget if movie_input.budget > 0 else 0,
            'log_budget': np.log1p(movie_input.budget),
            'log_revenue': np.log1p(movie_input.revenue),
            'log_popularity': np.log1p(movie_input.popularity),
            'overview_length': len(movie_input.overview or ""),
            'overview_word_count': len((movie_input.overview or "").split()),
            'tagline_word_count': len((movie_input.tagline or "").split()),

            # ✅ Real lookup values now!
            'director_success_rate': director_rate,
            'lead_actor_success_rate': actor_rate,
            'genre_popularity_score': genre_rate,

            'release_year': movie_input.release_year,
            'release_month': movie_input.release_month,
            'decade': (movie_input.release_year // 10) * 10,
            'is_modern': 1 if movie_input.release_year >= 2000 else 0,
            'is_summer_release': 1 if movie_input.release_month in [6, 7, 8] else 0,
            'has_tagline': 1 if len(movie_input.tagline or "") > 0 else 0,
            'budget_tier_encoded': self._get_budget_tier(movie_input.budget),
            'release_season_encoded': self._get_season(movie_input.release_month),
            'is_english': 1 if movie_input.original_language == 'en' else 0,

            # VADER sentiment features
            'overview_vader_pos': overview_scores['pos'],
            'overview_vader_neg': overview_scores['neg'],
            'overview_vader_neu': overview_scores['neu'],
            'overview_vader_compound': overview_scores['compound'],
            'tagline_vader_pos': tagline_scores['pos'],
            'tagline_vader_neg': tagline_scores['neg'],
            'tagline_vader_neu': tagline_scores['neu'],
            'tagline_vader_compound': tagline_scores['compound'],
        }

        # Debug: print any still-missing features
        missing = [col for col in self.feature_cols if col not in features]
        if missing:
            print(f"⚠️ Missing features: {missing}")

        # Create feature array in correct order
        feature_array = np.array([features[col] for col in self.feature_cols])
        return feature_array.reshape(1, -1)

    def _get_budget_tier(self, budget: float) -> int:
        """Get budget tier encoding."""
        if budget < 10_000_000:
            return 0  # low
        elif budget < 50_000_000:
            return 1  # medium
        elif budget < 150_000_000:
            return 2  # high
        else:
            return 3  # blockbuster

    def _get_season(self, month: float) -> int:
        """Get season encoding."""
        month = int(month)
        if month in [12, 1, 2]:
            return 0  # winter
        elif month in [3, 4, 5]:
            return 1  # spring
        elif month in [6, 7, 8]:
            return 2  # summer
        else:
            return 3  # fall

    def get_shap_explanation(self, X_scaled: np.ndarray) -> dict:
        """Get SHAP explanation for a prediction."""
        try:
            shap_values = self.explainer.shap_values(X_scaled)
            explanation = {}
            for i, col in enumerate(self.feature_cols):
                explanation[col] = round(float(shap_values[0][i]), 4)

            # Return top 5 features by absolute SHAP value
            top_features = dict(
                sorted(explanation.items(),
                       key=lambda x: abs(x[1]),
                       reverse=True)[:5]
            )
            return top_features
        except Exception as e:
            return {"error": str(e)}

    def predict(self, movie_input) -> dict:
        """Make a single prediction."""
        if not self.is_loaded:
            raise ValueError("Models not loaded!")

        # Extract features
        X = self.extract_features(movie_input)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict with stacking
        xgb_pred = self.xgb_model.predict(X_scaled)[0]
        lgbm_pred = self.lgbm_model.predict(X_scaled)[0]
        meta_X = np.column_stack([[xgb_pred], [lgbm_pred]])
        final_pred = self.meta_model.predict(meta_X)[0]

        # Clip to valid rating range
        final_pred = float(np.clip(final_pred, 1.0, 10.0))

        # Get confidence level
        if final_pred >= 7.0:
            confidence = "High"
        elif final_pred >= 5.0:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Get SHAP explanation
        top_features = self.get_shap_explanation(X_scaled)

        return {
            "title": movie_input.title,
            "predicted_rating": round(final_pred, 2),
            "confidence": confidence,
            "top_features": top_features
        }

    def predict_batch(self, movies: list) -> list:
        """Make batch predictions."""
        return [self.predict(movie) for movie in movies]


# Global predictor instance
predictor = MoviePredictor()