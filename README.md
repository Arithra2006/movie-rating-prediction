# 🎬 Movie Rating Prediction System
> End-to-End ML System with NLP, Ensemble Models, SHAP Explainability & API Deployment

[![CI/CD Pipeline](https://github.com/Arithra2006/movie-rating-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Arithra2006/movie-rating-prediction/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

- *Author:* Arithra Mayur 
- *Contact:* arithramayur@gmail.com
- *Live Demo:* [Coming Soon on Render]

---

## 📌 Overview

A production-ready, end-to-end machine learning system that predicts movie ratings using ensemble learning, NLP-based sentiment analysis, and SHAP explainability — deployed via FastAPI backend and Streamlit frontend.

This project goes beyond a basic notebook by implementing:
- Real NLP feature extraction from movie descriptions
- Stacking ensemble (XGBoost + LightGBM + Ridge meta-learner)
- Per-prediction SHAP explanations
- REST API with live Streamlit dashboard
- Full CI/CD pipeline with GitHub Actions

---

## 🏆 Model Performance

| Model | RMSE | MAE | R² | Improvement over Baseline |
|-------|------|-----|----|--------------------------|
| Linear Regression (Baseline) | 0.7168 | 0.5551 | 0.3533 | — |
| XGBoost | 0.6760 | 0.4967 | 0.4249 | +20.3% R² |
| LightGBM | 0.6668 | 0.4918 | 0.4403 | +24.6% R² |
| Stacking Ensemble | 0.6664 | 0.4935 | 0.4410 | +24.8% R² |
| *Tuned Stacking Ensemble* | *0.6503* | *0.4815* | *0.4677* | *+32.4% R²* |

> 📈 *32.4% improvement in R² over baseline* through ensemble stacking and Optuna hyperparameter tuning.

> 💡 Note: Movie ratings are inherently subjective and noisy. Even Netflix-scale systems struggle with high R² on this problem. The system correctly identifies high-quality directors and actors, and predicts above-average ratings for acclaimed filmmakers like Christopher Nolan.

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| ML Models | XGBoost, LightGBM, scikit-learn |
| NLP / Sentiment | VADER, DistilBERT (HuggingFace) |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Hyperparameter Tuning | Optuna |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest (33 tests) |
| Datasets | TMDB 5000 + IMDB Top 1000 |

---

## 🏗️ System Architecture
Data Ingestion → Feature Engineering → NLP Pipeline → Model Training → API Deployment
↓                  ↓                  ↓               ↓               ↓
TMDB + IMDB      Director/Actor      VADER + BERT     XGBoost +       FastAPI +
datasets       success rates       sentiment        LightGBM        Streamlit
features         Stacking
### Pipeline Stages

*1. Data Ingestion*
- Merged TMDB 5000 Movies + IMDB Top 1000 datasets
- Handled missing values via median/mode imputation
- Combined dataset gives richer metadata than either source alone

*2. Feature Engineering*
- Director success rate (historical average rating per director)
- Lead actor success rate (historical average rating per actor)
- Genre popularity score
- Budget/revenue ratio, ROI, profit
- Log-transformed financial features
- Release season and decade features

*3. NLP Sentiment Pipeline*
- VADER sentiment scores on movie overviews and taglines
- DistilBERT embeddings for semantic features
- 8 VADER features + word count features per text field

*4. Model Training*
- XGBoost and LightGBM base models with 5-fold cross-validation
- Ridge meta-learner stacking on OOF predictions
- Optuna hyperparameter tuning with MLflow experiment tracking
- 20+ metrics logged per experiment run

*5. SHAP Explainability*
- Per-prediction SHAP waterfall charts
- Global feature importance summary plots
- Top 5 contributing features returned with every prediction

*6. API Deployment*
- FastAPI REST API with /predict and /predict/batch endpoints
- Pydantic request/response validation
- Streamlit dashboard with interactive movie input form

---

## 📁 Project Structure
movie-rating-prediction/
├── src/
│   ├── data/               # Data ingestion & cleaning
│   │   ├── ingest.py
│   │   └── clean.py
│   ├── features/           # Feature engineering & NLP
│   │   ├── tabular_features.py
│   │   ├── nlp_features.py
│   │   ├── text_preprocessor.py
│   │   └── feature_pipeline.py
│   ├── models/             # XGBoost, LightGBM, stacking
│   │   ├── xgb_model.py
│   │   ├── stacking.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── explainability/     # SHAP generation
│   │   ├── shap_explainer.py
│   │   └── plots.py
│   └── api/                # FastAPI backend
│       ├── main.py
│       ├── schemas.py
│       └── predictor.py
├── dashboard/              # Streamlit frontend
│   ├── app.py
│   └── components.py
├── tests/                  # pytest unit tests
│   ├── test_ingest.py      # 8 tests
│   ├── test_features.py    # 13 tests
│   └── test_api.py         # 12 tests
├── reports/                # Model evaluation plots
├── notebooks/              # EDA notebook
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
└── requirements.txt
---

## ⚡ Quick Start

### 1. Clone the repo
- bash
- git clone https://github.com/Arithra2006/movie-rating-prediction.git
- cd movie-rating-prediction
2. Install dependencies
- pip install -r requirements.txt
3. Run the pipeline
# Ingest and clean data
- python src/data/ingest.py
- python src/data/clean.py

# Feature engineering
- python src/features/feature_pipeline.py

# Train models
- python src/models/train.py

# Evaluate models
- python src/models/evaluate.py
4. Start the API
- uvicorn src.api.main:app --reload
5. Start the dashboard
- cd dashboard
- streamlit run app.py
6. Run with Docker
- docker-compose up --build
🧪 Testing
# Run all 33 tests
- pytest tests/ -v

# Run individual test suites
- pytest tests/test_ingest.py -v    # 8 tests
- pytest tests/test_features.py -v  # 13 tests
- pytest tests/test_api.py -v       # 12 tests
Test Coverage:
✅ Data ingestion & validation
✅ Feature engineering pipeline
✅ VADER sentiment features
✅ API endpoints (predict, batch, health)
✅ Request validation & error handling
🔍 API Usage
Single Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Inception",
    "budget": 160000000,
    "revenue": 836836967,
    "popularity": 96.0,
    "runtime": 148,
    "vote_count": 32000,
    "overview": "A thief who steals corporate secrets through dream-sharing technology.",
    "tagline": "Your mind is the scene of the crime.",
    "release_year": 2010,
    "release_month": 7,
    "original_language": "en",
    "director": "Christopher Nolan",
    "star1": "Leonardo DiCaprio",
    "genre": "Sci-Fi"
  }'
Response
{
  "title": "Inception",
  "predicted_rating": 7.47,
  "confidence": "High",
  "top_features": {
    "vote_count": 0.5072,
    "lead_actor_success_rate": 0.3474,
    "runtime": 0.3020,
    "director_success_rate": 0.2505,
    "overview_length": -0.1847
  }
}
📊 Key Insights from SHAP Analysis
- vote_count is the strongest predictor — movies with more votes tend to be better known and rated higher
- director_success_rate and lead_actor_success_rate significantly push predictions for acclaimed filmmakers
- runtime correlates positively — longer films tend to be rated higher (epic films)
- overview_length has a slight negative effect — overly long descriptions may indicate complex or niche films

🎓 Key Learnings
- How to combine NLP features with tabular ML in a unified pipeline
- Ensemble stacking mechanics — when and why it outperforms single models
- SHAP theory and practical implementation for regression explainability
- FastAPI design patterns — request validation, async endpoints, error handling
- MLflow experiment tracking and model registry for reproducible ML
- End-to-end Docker Compose orchestration for multi-service ML applications
- Director/actor historical performance as a strong signal for rating prediction

🔮 Future Enhancements
- Collaborative filtering layer for personalized recommendations
- Temporal features: release season trends, franchise sequel effects
- A/B testing framework to compare model versions in production
- User feedback loop to retrain model on new rating data
- Larger dataset (full TMDB API) for improved R²

📄 License
- MIT License — feel free to use this project for learning and portfolio purposes.