import numpy as np
import pandas as pd
import mlflow
import sys
sys.path.append('.')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

from src.models.xgb_model import get_xgb_model, train_xgb
from src.models.lgbm_model import get_lgbm_model, train_lgbm
from src.models.stacking import train_stacking, save_stacking_models

MODEL_PATH = Path("models")
PROCESSED_DATA_PATH = Path("data/processed")

def load_features() -> pd.DataFrame:
    """Load the feature dataset."""
    filepath = PROCESSED_DATA_PATH / "features_full.csv"
    print(f"📂 Loading features from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def prepare_data(df: pd.DataFrame):
    """Prepare X and y for training."""
    print("🔧 Preparing training data...")

    # Target column
    target = 'vote_average'

    # Drop non-feature columns
    drop_cols = [target, 'title']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[target].values

    print(f"✅ Features: {X.shape}, Target: {y.shape}")
    return X, y, feature_cols

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    MODEL_PATH.mkdir(exist_ok=True)
    joblib.dump(scaler, MODEL_PATH / "scaler.pkl")
    print("✅ Features scaled and scaler saved!")
    return X_train_scaled, X_test_scaled, scaler

def run_training():
    """Full training pipeline."""
    print("\n🚀 Starting full training pipeline...\n")

    # Load features
    df = load_features()

    # Prepare data
    X, y, feature_cols = prepare_data(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save feature columns
    joblib.dump(feature_cols, MODEL_PATH / "feature_cols.pkl")

    # Set MLflow experiment
    mlflow.set_experiment("movie_rating_prediction")

    with mlflow.start_run(run_name="Full_Training_Pipeline"):

        # Log dataset info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Train XGBoost
        print("\n--- XGBoost ---")
        xgb_model, xgb_val_preds = train_xgb(
            X_train_scaled, y_train,
            X_test_scaled, y_test
        )

        # Train LightGBM
        print("\n--- LightGBM ---")
        lgbm_model, lgbm_val_preds = train_lgbm(
            X_train_scaled, y_train,
            X_test_scaled, y_test
        )

        # Train Stacking Ensemble
        print("\n--- Stacking Ensemble ---")
        meta_model, xgb_model, lgbm_model, final_preds = train_stacking(
            xgb_model, lgbm_model,
            X_train_scaled, y_train,
            X_test_scaled, y_test
        )

        # Save all models
        save_stacking_models(meta_model, xgb_model, lgbm_model)

    print("\n🎉 Training pipeline complete!")
    return meta_model, xgb_model, lgbm_model, scaler, feature_cols

if __name__ == "__main__":
    meta_model, xgb_model, lgbm_model, scaler, feature_cols = run_training()
    print(f"\n📋 Features used: {feature_cols}")
    print(f"\n✅ All models saved to models/ folder!")