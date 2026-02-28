import numpy as np
import pandas as pd
import optuna
import sys
sys.path.append('.')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path

MODEL_PATH = Path("models")
PROCESSED_DATA_PATH = Path("data/processed")

# Suppress optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_full_features():
    """Load the full feature dataset with NLP features."""
    print("📂 Loading full features...")
    df = pd.read_csv(PROCESSED_DATA_PATH / "features_full.csv")
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    target = 'vote_average'
    drop_cols = [target, 'title']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[target].values

    return X, y, feature_cols

def tune_xgboost(X_train, y_train, n_trials: int = 50):
    """Tune XGBoost hyperparameters using Optuna."""
    print(f"\n🔍 Tuning XGBoost with {n_trials} trials...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='r2', n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"✅ Best XGBoost R²: {study.best_value:.4f}")
    print(f"✅ Best params: {study.best_params}")
    return study.best_params

def tune_lightgbm(X_train, y_train, n_trials: int = 50):
    """Tune LightGBM hyperparameters using Optuna."""
    print(f"\n🔍 Tuning LightGBM with {n_trials} trials...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='r2', n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"✅ Best LightGBM R²: {study.best_value:.4f}")
    print(f"✅ Best params: {study.best_params}")
    return study.best_params

def retrain_with_best_params(X_train, y_train, X_test, y_test,
                              xgb_params, lgbm_params, feature_cols, scaler):
    """Retrain models with best params and stack them."""
    print("\n🚀 Retraining with best params...")

    from src.models.stacking import train_stacking, save_stacking_models

    # Add required fixed params
    xgb_params['random_state'] = 42
    xgb_params['n_jobs'] = -1
    xgb_params['verbosity'] = 0

    lgbm_params['random_state'] = 42
    lgbm_params['n_jobs'] = -1
    lgbm_params['verbose'] = -1

    xgb_model = xgb.XGBRegressor(**xgb_params)
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)

    import mlflow
    mlflow.set_experiment("movie_rating_tuned")

    with mlflow.start_run(run_name="Tuned_Stacking"):
        meta_model, xgb_model, lgbm_model, final_preds = train_stacking(
            xgb_model, lgbm_model,
            X_train, y_train,
            X_test, y_test
        )

    # Save everything
    save_stacking_models(meta_model, xgb_model, lgbm_model)
    joblib.dump(scaler, MODEL_PATH / "scaler.pkl")
    joblib.dump(feature_cols, MODEL_PATH / "feature_cols.pkl")

    # Final metrics
    rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    r2 = r2_score(y_test, final_preds)
    print(f"\n🎯 Final Tuned Model:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")

    return meta_model, xgb_model, lgbm_model

def run_tuning():
    """Full tuning pipeline."""
    print("\n🚀 Starting hyperparameter tuning pipeline...\n")

    # Load data
    X, y, feature_cols = load_full_features()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Tune XGBoost
    xgb_best_params = tune_xgboost(X_train_scaled, y_train, n_trials=50)

    # Tune LightGBM
    lgbm_best_params = tune_lightgbm(X_train_scaled, y_train, n_trials=50)

    # Retrain with best params
    retrain_with_best_params(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        xgb_best_params, lgbm_best_params,
        feature_cols, scaler
    )

    # Update SHAP explainer
    print("\n🔍 Updating SHAP explainer...")
    xgb_model = joblib.load(MODEL_PATH / "xgb_model.pkl")
    from src.explainability.shap_explainer import get_shap_explainer
    explainer = get_shap_explainer(xgb_model, X_train_scaled, "xgb")
    joblib.dump(explainer, MODEL_PATH / "shap_explainer.pkl")
    print("✅ SHAP explainer updated!")

    print("\n🎉 Tuning pipeline complete!")

if __name__ == "__main__":
    run_tuning()