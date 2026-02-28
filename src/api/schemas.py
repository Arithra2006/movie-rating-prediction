from pydantic import BaseModel, Field
from typing import Optional, List

class MovieInput(BaseModel):
    """Input schema for a single movie prediction."""

    title: str = Field(..., description="Movie title")
    budget: float = Field(0.0, description="Production budget in USD")
    revenue: float = Field(0.0, description="Box office revenue in USD")
    popularity: float = Field(5.0, description="TMDB popularity score")
    runtime: float = Field(120.0, description="Runtime in minutes")
    vote_count: float = Field(100.0, description="Number of votes")
    overview: str = Field("", description="Movie overview/description")
    tagline: str = Field("", description="Movie tagline")
    release_year: float = Field(2020.0, description="Release year")
    release_month: float = Field(6.0, description="Release month (1-12)")
    original_language: str = Field("en", description="Original language code")
    director: str = Field("", description="Director name")
    star1: str = Field("", description="Lead actor")
    genre: str = Field("", description="Movie genre")

    class Config:
        json_schema_extra = {
            "example": {
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
        }

class MoviePrediction(BaseModel):
    """Output schema for a single movie prediction."""
    title: str
    predicted_rating: float
    confidence: str
    top_features: dict

class BatchMovieInput(BaseModel):
    """Input schema for batch predictions."""
    movies: List[MovieInput]

class BatchMoviePrediction(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[MoviePrediction]
    total: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str