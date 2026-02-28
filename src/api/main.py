from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append('.')

from src.api.schemas import (
    MovieInput, MoviePrediction,
    BatchMovieInput, BatchMoviePrediction,
    HealthResponse
)
from src.api.predictor import predictor

# Create FastAPI app
app = FastAPI(
    title="🎬 Movie Rating Prediction API",
    description="Predict movie ratings using XGBoost + LightGBM Stacking Ensemble",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("🚀 Starting Movie Rating Prediction API...")
    predictor.load_models()
    print("✅ API ready!")

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint — health check."""
    return HealthResponse(
        status="running",
        model_loaded=predictor.is_loaded,
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "models not loaded",
        model_loaded=predictor.is_loaded,
        version="1.0.0"
    )

@app.post("/predict", response_model=MoviePrediction)
async def predict_single(movie: MovieInput):
    """
    Predict rating for a single movie.

    Returns predicted rating between 1-10 with SHAP explanation.
    """
    try:
        result = predictor.predict(movie)
        return MoviePrediction(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchMoviePrediction)
async def predict_batch(batch: BatchMovieInput):
    """
    Predict ratings for multiple movies at once.

    Maximum 50 movies per request.
    """
    if len(batch.movies) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 movies per batch request"
        )
    try:
        results = predictor.predict_batch(batch.movies)
        predictions = [MoviePrediction(**r) for r in results]
        return BatchMoviePrediction(
            predictions=predictions,
            total=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
async def get_features():
    """Get list of features used by the model."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")
    return {
        "features": predictor.feature_cols,
        "total": len(predictor.feature_cols)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)