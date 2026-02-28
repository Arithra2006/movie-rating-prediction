import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import sys
sys.path.append('.')
from pathlib import Path

MODEL_PATH = Path("models")

def plot_summary(shap_values, X_test, feature_cols, save: bool = True):
    """Plot SHAP summary plot — global feature importance."""
    print("📊 Generating SHAP summary plot...")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_cols,
        show=False
    )
    plt.title("SHAP Feature Importance Summary")
    plt.tight_layout()

    if save:
        Path("reports").mkdir(exist_ok=True)
        plt.savefig("reports/shap_summary.png", bbox_inches='tight')
        print("💾 Summary plot saved to reports/shap_summary.png")

    plt.show()
    plt.close()

def plot_waterfall(explainer, X_single, feature_cols,
                   feature_values, prediction: float,
                   save: bool = True, index: int = 0):
    """Plot SHAP waterfall plot for a single prediction."""
    print("📊 Generating SHAP waterfall plot...")

    shap_values = explainer.shap_values(X_single)
    expected_value = explainer.expected_value

    # Create SHAP explanation object
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=X_single[0],
        feature_names=feature_cols
    )

    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP Waterfall — Predicted Rating: {prediction:.2f}")
    plt.tight_layout()

    if save:
        Path("reports").mkdir(exist_ok=True)
        filepath = f"reports/shap_waterfall_{index}.png"
        plt.savefig(filepath, bbox_inches='tight')
        print(f"💾 Waterfall plot saved to {filepath}")

    plt.show()
    plt.close()

def plot_bar_importance(shap_values, feature_cols, top_n: int = 15, save: bool = True):
    """Plot bar chart of mean absolute SHAP values."""
    print("📊 Generating SHAP bar importance plot...")

    importance = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=True).tail(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance['feature'], importance['mean_abs_shap'], color='steelblue')
    plt.xlabel('Mean |SHAP Value|')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()

    if save:
        Path("reports").mkdir(exist_ok=True)
        plt.savefig("reports/shap_bar_importance.png", bbox_inches='tight')
        print("💾 Bar importance plot saved to reports/shap_bar_importance.png")

    plt.show()
    plt.close()

def generate_all_plots():
    """Generate all SHAP plots."""
    print("\n🚀 Generating all SHAP plots...\n")

    from sklearn.model_selection import train_test_split

    # Load everything
    xgb_model = joblib.load(MODEL_PATH / "xgb_model.pkl")
    scaler = joblib.load(MODEL_PATH / "scaler.pkl")
    feature_cols = joblib.load(MODEL_PATH / "feature_cols.pkl")
    explainer = joblib.load(MODEL_PATH / "shap_explainer.pkl")

    # Load data
    df = pd.read_csv("data/processed/features.csv")
    target = 'vote_average'
    drop_cols = [target, 'title']
    X = df[[c for c in df.columns if c not in drop_cols]].values
    y = df[target].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_test_scaled = scaler.transform(X_test)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_scaled[:200])

    # 1. Summary plot
    plot_summary(shap_values, X_test_scaled[:200], feature_cols)

    # 2. Bar importance plot
    plot_bar_importance(shap_values, feature_cols)

    # 3. Waterfall for first prediction
    prediction = xgb_model.predict(X_test_scaled[:1])[0]
    plot_waterfall(
        explainer,
        X_test_scaled[:1],
        feature_cols,
        X_test_scaled[0],
        prediction,
        index=0
    )

    print("\n✅ All SHAP plots generated!")

if __name__ == "__main__":
    generate_all_plots()