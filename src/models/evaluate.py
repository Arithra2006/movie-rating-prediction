import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from pathlib import Path
import joblib

MODEL_PATH = Path("models")
PROCESSED_DATA_PATH = Path("data/processed")

def compute_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """Compute regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 {model_name} Metrics:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   R²:   {r2:.4f}")

    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'r2': r2}

def baseline_model(X_train, y_train, X_test, y_test) -> dict:
    """Train and evaluate a baseline linear regression model."""
    print("📏 Training baseline (Linear Regression)...")
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    preds = baseline.predict(X_test)
    return compute_metrics(y_test, preds, "Baseline (Linear Regression)")

def plot_predictions(y_true, y_pred, model_name: str = "Model"):
    """Plot actual vs predicted ratings."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, color='steelblue')
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        'r--', lw=2, label='Perfect Prediction'
    )
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title(f'{model_name} — Actual vs Predicted')
    plt.legend()
    plt.tight_layout()

    # Save plot
    Path("reports").mkdir(exist_ok=True)
    filepath = f"reports/{model_name.replace(' ', '_')}_predictions.png"
    plt.savefig(filepath)
    print(f"📊 Plot saved to {filepath}")
    plt.show()

def plot_residuals(y_true, y_pred, model_name: str = "Model"):
    """Plot residuals."""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, color='coral')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} — Residual Plot')
    plt.tight_layout()

    filepath = f"reports/{model_name.replace(' ', '_')}_residuals.png"
    plt.savefig(filepath)
    print(f"📊 Residual plot saved to {filepath}")
    plt.show()

def compare_models(results: list) -> pd.DataFrame:
    """Compare multiple models in a table."""
    df = pd.DataFrame(results)
    df = df.sort_values('rmse')
    print("\n🏆 Model Comparison:")
    print(df.to_string(index=False))
    return df

def run_evaluation():
    """Full evaluation pipeline."""
    print("\n🚀 Starting evaluation pipeline...\n")

    from sklearn.model_selection import train_test_split
    from src.models.stacking import predict_stacking

    # Load features
    df = pd.read_csv(PROCESSED_DATA_PATH / "features.csv")

    # Prepare data
    target = 'vote_average'
    drop_cols = [target, 'title']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load scaler
    scaler = joblib.load(MODEL_PATH / "scaler.pkl")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Load models
    meta_model, xgb_model, lgbm_model = (
        joblib.load(MODEL_PATH / "meta_model.pkl"),
        joblib.load(MODEL_PATH / "xgb_model.pkl"),
        joblib.load(MODEL_PATH / "lgbm_model.pkl")
    )

    results = []

    # Baseline
    baseline_result = baseline_model(X_train_scaled, y_train, X_test_scaled, y_test)
    results.append(baseline_result)

    # XGBoost
    xgb_preds = xgb_model.predict(X_test_scaled)
    results.append(compute_metrics(y_test, xgb_preds, "XGBoost"))

    # LightGBM
    lgbm_preds = lgbm_model.predict(X_test_scaled)
    results.append(compute_metrics(y_test, lgbm_preds, "LightGBM"))

    # Stacking
    final_preds = predict_stacking(meta_model, xgb_model, lgbm_model, X_test_scaled)
    results.append(compute_metrics(y_test, final_preds, "Stacking Ensemble"))

    # Compare
    compare_models(results)

    # Plots
    plot_predictions(y_test, final_preds, "Stacking Ensemble")
    plot_residuals(y_test, final_preds, "Stacking Ensemble")

    print("\n✅ Evaluation complete!")
    return results

if __name__ == "__main__":
    results = run_evaluation()