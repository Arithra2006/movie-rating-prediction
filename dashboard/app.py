import streamlit as st
import requests
import json

from components import (
    render_header,
    render_movie_form,
    render_prediction_result,
    render_shap_chart,
    render_history_table
)

# Config
API_URL = "http://localhost:8000"
st.set_page_config(
    page_title="🎬 Movie Rating Predictor",
    page_icon="🎬",
    layout="wide"
)

# Session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Header
render_header()

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    if health.status_code == 200:
        st.success("✅ API Connected")
    else:
        st.error("❌ API not responding")
except:
    st.error("❌ Cannot connect to API — make sure FastAPI is running on port 8000")
    st.stop()

# Movie input form
movie_data = render_movie_form()

# Predict button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🎬 Predict Rating", use_container_width=True, type="primary")

if predict_btn:
    with st.spinner("🤖 Analyzing movie..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=movie_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                # Show result
                render_prediction_result(result)

                # Show SHAP chart
                render_shap_chart(result['top_features'])

                # Save to history
                st.session_state.history.append(result)

            else:
                st.error(f"❌ API Error: {response.json()}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Prediction history
render_history_table(st.session_state.history)