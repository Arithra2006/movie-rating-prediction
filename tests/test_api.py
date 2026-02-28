import pytest
from fastapi.testclient import TestClient
import sys
sys.path.append('.')
from src.api.main import app
from src.api.predictor import predictor

# Load models once before all tests
predictor.load_models()

client = TestClient(app)

# Sample movie payload
SAMPLE_MOVIE = {
    "title": "The Dark Knight",
    "budget": 185000000,
    "revenue": 1004558444,
    "popularity": 123.5,
    "runtime": 152,
    "vote_count": 27000,
    "overview": "Batman raises the stakes in his war on crime.",
    "tagline": "Why so serious?",
    "release_year": 2008,
    "release_month": 7,
    "original_language": "en",
    "director": "Christopher Nolan",
    "star1": "Christian Bale",
    "genre": "Action"
}

class TestAPI:
    """Tests for FastAPI endpoints."""

    def test_root_endpoint(self):
        """Check root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_endpoint(self):
        """Check health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_predict_endpoint_status(self):
        """Check predict endpoint returns 200."""
        response = client.post("/predict", json=SAMPLE_MOVIE)
        assert response.status_code == 200

    def test_predict_returns_rating(self):
        """Check prediction response has predicted_rating."""
        response = client.post("/predict", json=SAMPLE_MOVIE)
        data = response.json()
        assert "predicted_rating" in data

    def test_predict_rating_in_range(self):
        """Check predicted rating is between 1 and 10."""
        response = client.post("/predict", json=SAMPLE_MOVIE)
        data = response.json()
        rating = data['predicted_rating']
        assert 1.0 <= rating <= 10.0, f"Rating out of range: {rating}"

    def test_predict_returns_confidence(self):
        """Check prediction response has confidence."""
        response = client.post("/predict", json=SAMPLE_MOVIE)
        data = response.json()
        assert "confidence" in data
        assert data['confidence'] in ["High", "Medium", "Low"]

    def test_predict_returns_top_features(self):
        """Check prediction response has top_features."""
        response = client.post("/predict", json=SAMPLE_MOVIE)
        data = response.json()
        assert "top_features" in data
        assert len(data['top_features']) > 0

    def test_predict_returns_title(self):
        """Check prediction response echoes title."""
        response = client.post("/predict", json=SAMPLE_MOVIE)
        data = response.json()
        assert data['title'] == SAMPLE_MOVIE['title']

    def test_predict_missing_field_returns_422(self):
        """Check that missing required field returns validation error."""
        bad_payload = SAMPLE_MOVIE.copy()
        del bad_payload['title']
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_batch_endpoint(self):
        """Check batch prediction endpoint works."""
        batch_payload = {"movies": [SAMPLE_MOVIE, SAMPLE_MOVIE]}
        response = client.post("/predict/batch", json=batch_payload)
        assert response.status_code == 200

    def test_predict_batch_returns_list(self):
        """Check batch prediction returns list of results."""
        """Check batch prediction returns list of results."""
        batch_payload = {"movies": [SAMPLE_MOVIE, SAMPLE_MOVIE]}
        response = client.post("/predict/batch", json=batch_payload)
        data = response.json()
        # API returns {'predictions': [...], 'total': N}
        assert "predictions" in data
        assert isinstance(data['predictions'], list)
        assert len(data['predictions']) == 2

    def test_different_movies_different_ratings(self):
        """Check that different movies get different ratings."""
        movie2 = SAMPLE_MOVIE.copy()
        movie2['title'] = "Low Budget Film"
        movie2['budget'] = 100000
        movie2['revenue'] = 200000
        movie2['vote_count'] = 50
        movie2['popularity'] = 1.0
        movie2['director'] = "Unknown Director"

        response1 = client.post("/predict", json=SAMPLE_MOVIE)
        response2 = client.post("/predict", json=movie2)

        rating1 = response1.json()['predicted_rating']
        rating2 = response2.json()['predicted_rating']

        assert rating1 != rating2, "Different movies should get different ratings!"