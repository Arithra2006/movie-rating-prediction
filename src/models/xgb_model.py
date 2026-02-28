import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

MODEL_PATH = Path("models")

def get_xgb_model(params: dict = None):
    """Get XGBoost model with given or default params."""
    default_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    if params:
        default_params.update(params)
    return xgb.XGBRegressor(**default_params)

def train_xgb(X_train, y_train, X_val, y_val, params: dict = None):
    """Train XGBoost model and log to MLflow."""
    print("🚀 Training XGBoost model...")

    model = get_xgb_model(params)

    with mlflow.start_run(run_name="XGBoost", nested=True):
        # Train
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae = mean_absolute_error(y_val, val_preds)
        r2 = r2_score(y_val, val_preds)

        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_r2", r2)
        mlflow.xgboost.log_model(model, "xgb_model")

        print(f"✅ XGBoost Results:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   R²:   {r2:.4f}")

    return model, val_preds

def save_xgb_model(model, filename: str = "xgb_model.pkl"):
    """Save XGBoost model to disk."""
    MODEL_PATH.mkdir(exist_ok=True)
    filepath = MODEL_PATH / filename
    joblib.dump(model, filepath)
    print(f"💾 XGBoost model saved to {filepath}")

def load_xgb_model(filename: str = "xgb_model.pkl"):
    """Load XGBoost model from disk."""
    filepath = MODEL_PATH / filename
    model = joblib.load(filepath)
    print(f"📂 XGBoost model loaded from {filepath}")
    return model

if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("movie_rating_test")
    model, preds = train_xgb(X_train, y_train, X_val, y_val)
    print("✅ XGBoost test passed!")