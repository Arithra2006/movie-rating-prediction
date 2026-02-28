import numpy as np
import pandas as pd
import mlflow
import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

MODEL_PATH = Path("models")

def generate_oof_predictions(model, X_train, y_train, n_splits=5):
    """Generate out-of-fold predictions for stacking."""
    print(f"🔄 Generating OOF predictions with {n_splits} folds...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr = X_train[train_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[train_idx]
        X_val = X_train[val_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[val_idx]
        y_tr = y_train[train_idx] if isinstance(y_train, np.ndarray) else y_train.iloc[train_idx]
        
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)
        print(f"  ✅ Fold {fold+1}/{n_splits} done")
    
    return oof_preds

def train_stacking(xgb_model, lgbm_model, X_train, y_train, X_test, y_test):
    """Train stacking ensemble with Ridge meta-learner."""
    print("\n🚀 Training Stacking Ensemble...\n")

    with mlflow.start_run(run_name="Stacking_Ensemble", nested=True):

        # Step 1: Generate OOF predictions for meta-learner training
        print("📊 Step 1: Generating OOF predictions...")
        xgb_oof = generate_oof_predictions(xgb_model, X_train, y_train)
        lgbm_oof = generate_oof_predictions(lgbm_model, X_train, y_train)

        # Step 2: Retrain base models on full training data
        print("\n📊 Step 2: Retraining base models on full data...")
        xgb_model.fit(X_train, y_train)
        lgbm_model.fit(X_train, y_train)
        print("✅ Base models retrained!")

        # Step 3: Generate test predictions from base models
        print("\n📊 Step 3: Generating test predictions...")
        xgb_test_preds = xgb_model.predict(X_test)
        lgbm_test_preds = lgbm_model.predict(X_test)

        # Step 4: Train Ridge meta-learner
        print("\n📊 Step 4: Training Ridge meta-learner...")
        meta_X_train = np.column_stack([xgb_oof, lgbm_oof])
        meta_X_test = np.column_stack([xgb_test_preds, lgbm_test_preds])

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_X_train, y_train)
        print("✅ Ridge meta-learner trained!")

        # Step 5: Final predictions
        final_preds = meta_model.predict(meta_X_test)

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, final_preds))
        mae = mean_absolute_error(y_test, final_preds)
        r2 = r2_score(y_test, final_preds)

        # Log to MLflow
        mlflow.log_metric("ensemble_rmse", rmse)
        mlflow.log_metric("ensemble_mae", mae)
        mlflow.log_metric("ensemble_r2", r2)

        print(f"\n🎯 Stacking Ensemble Results:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   R²:   {r2:.4f}")

    return meta_model, xgb_model, lgbm_model, final_preds

def save_stacking_models(meta_model, xgb_model, lgbm_model):
    """Save all stacking models."""
    MODEL_PATH.mkdir(exist_ok=True)
    joblib.dump(meta_model, MODEL_PATH / "meta_model.pkl")
    joblib.dump(xgb_model, MODEL_PATH / "xgb_model.pkl")
    joblib.dump(lgbm_model, MODEL_PATH / "lgbm_model.pkl")
    print("💾 All stacking models saved!")

def load_stacking_models():
    """Load all stacking models."""
    meta_model = joblib.load(MODEL_PATH / "meta_model.pkl")
    xgb_model = joblib.load(MODEL_PATH / "xgb_model.pkl")
    lgbm_model = joblib.load(MODEL_PATH / "lgbm_model.pkl")
    print("📂 All stacking models loaded!")
    return meta_model, xgb_model, lgbm_model

def predict_stacking(meta_model, xgb_model, lgbm_model, X):
    """Make predictions using stacking ensemble."""
    xgb_preds = xgb_model.predict(X)
    lgbm_preds = lgbm_model.predict(X)
    meta_X = np.column_stack([xgb_preds, lgbm_preds])
    return meta_model.predict(meta_X)

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    import lightgbm as lgb

    X, y = make_regression(n_samples=500, n_features=10, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    lgbm_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

    mlflow.set_experiment("movie_rating_test")
    with mlflow.start_run(run_name="Stacking_Test"):
        meta_model, xgb_model, lgbm_model, preds = train_stacking(
            xgb_model, lgbm_model, X_train, y_train, X_test, y_test
        )

    save_stacking_models(meta_model, xgb_model, lgbm_model)
    print("\n✅ Stacking test passed!")