import numpy as np
import pandas as pd
import shap
import joblib
import sys
sys.path.append('.')
from pathlib import Path

MODEL_PATH = Path("models")
PROCESSED_DATA_PATH = Path("data/processed")

def load_models_and_data():
    """Load trained models, scaler and feature data."""
    print("📂 Loading models and data...")

    # Load models
    xgb_model = joblib.load(MODEL_PATH / "xgb_model.pkl")
    lgbm_model = joblib.load(MODEL_PATH / "lgbm_model.pkl")
    meta_model = joblib.load(MODEL_PATH / "meta_model.pkl")
    scaler = joblib.load(MODEL_PATH / "scaler.pkl")
    feature_cols = joblib.load(MODEL_PATH / "feature_cols.pkl")

    # Load features
    df = pd.read_csv(PROCESSED_DATA_PATH / "features.csv")

    print("✅ Models and data loaded!")
    return xgb_model, lgbm_model, meta_model, scaler, feature_cols, df

def get_shap_explainer(model, X_train, model_type: str = "xgb"):
    """Create SHAP explainer for a model."""
    print(f"🔍 Creating SHAP explainer for {model_type}...")

    if model_type == "xgb":
        explainer = shap.TreeExplainer(model)
    elif model_type == "lgbm":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X_train[:100])

    print(f"✅ SHAP explainer created!")
    return explainer

def compute_shap_values(explainer, X, feature_cols):
    """Compute SHAP values for given data."""
    print("⚙️ Computing SHAP values...")

    shap_values = explainer.shap_values(X)

    # Create a DataFrame for easier handling
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    print(f"✅ SHAP values computed! Shape: {shap_df.shape}")
    return shap_values, shap_df

def explain_single_prediction(explainer, X_single, feature_cols, feature_values):
    """Explain a single prediction."""
    print("🔍 Explaining single prediction...")

    shap_values = explainer.shap_values(X_single)

    # Create explanation dict
    explanation = {}
    for i, col in enumerate(feature_cols):
        explanation[col] = {
            'shap_value': float(shap_values[0][i]),
            'feature_value': float(feature_values[i])
        }

    # Sort by absolute SHAP value
    explanation = dict(
        sorted(explanation.items(),
               key=lambda x: abs(x[1]['shap_value']),
               reverse=True)
    )

    print("✅ Single prediction explained!")
    return shap_values, explanation

def get_feature_importance(shap_values, feature_cols) -> pd.DataFrame:
    """Get global feature importance from SHAP values."""
    importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    })
    importance = importance.sort_values('mean_abs_shap', ascending=False)
    return importance

def run_shap_analysis():
    """Full SHAP analysis pipeline."""
    print("\n🚀 Starting SHAP analysis...\n")

    from sklearn.model_selection import train_test_split

    # Load everything
    xgb_model, lgbm_model, meta_model, scaler, feature_cols, df = load_models_and_data()

    # Prepare data
    target = 'vote_average'
    drop_cols = [target, 'title']
    X = df[[c for c in df.columns if c not in drop_cols]].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create SHAP explainer for XGBoost
    explainer = get_shap_explainer(xgb_model, X_train_scaled, "xgb")

    # Compute SHAP values on test set (use first 200 for speed)
    shap_values, shap_df = compute_shap_values(
        explainer, X_test_scaled[:200], feature_cols
    )

    # Get feature importance
    importance = get_feature_importance(shap_values, feature_cols)
    print("\n🏆 Top 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))

    # Save explainer and shap values
    joblib.dump(explainer, MODEL_PATH / "shap_explainer.pkl")
    shap_df.to_csv("data/processed/shap_values.csv", index=False)
    print("\n💾 SHAP explainer and values saved!")

    return explainer, shap_values, shap_df, feature_cols, X_test_scaled

if __name__ == "__main__":
    explainer, shap_values, shap_df, feature_cols, X_test = run_shap_analysis()
    print("\n✅ SHAP analysis complete!")