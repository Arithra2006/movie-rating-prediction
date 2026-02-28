import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_header():
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🎬 Movie Rating Predictor</h1>
            <p style='color: gray;'>Powered by XGBoost + LightGBM Stacking Ensemble</p>
        </div>
    """, unsafe_allow_html=True)

def render_movie_form():
    """Render the movie input form."""
    st.subheader("📝 Enter Movie Details")

    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input("Movie Title", value="The Dark Knight")
        director = st.text_input("Director", value="Christopher Nolan")
        star1 = st.text_input("Lead Actor", value="Christian Bale")
        genre = st.selectbox("Genre", [
            "Action", "Adventure", "Animation", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Horror", "Mystery",
            "Romance", "Sci-Fi", "Thriller", "Western"
        ])
        original_language = st.selectbox("Language", ["en", "fr", "es", "de", "ja", "ko", "hi"])
        overview = st.text_area("Movie Overview", value="Batman raises the stakes in his war on crime.")
        tagline = st.text_input("Tagline", value="Why so serious?")

    with col2:
        budget = st.number_input("Budget ($)", min_value=0, value=185000000, step=1000000)
        revenue = st.number_input("Revenue ($)", min_value=0, value=1004558444, step=1000000)
        popularity = st.slider("Popularity Score", 0.0, 300.0, 123.5)
        runtime = st.number_input("Runtime (minutes)", min_value=0, value=152)
        vote_count = st.number_input("Vote Count", min_value=0, value=27000)
        release_year = st.number_input("Release Year", min_value=1900, max_value=2030, value=2008)
        release_month = st.selectbox("Release Month", list(range(1, 13)),
                                      format_func=lambda x: pd.Timestamp(2024, x, 1).strftime('%B'))

    return {
        "title": title,
        "budget": budget,
        "revenue": revenue,
        "popularity": popularity,
        "runtime": runtime,
        "vote_count": vote_count,
        "overview": overview,
        "tagline": tagline,
        "release_year": release_year,
        "release_month": release_month,
        "original_language": original_language,
        "director": director,
        "star1": star1,
        "genre": genre
    }

def render_prediction_result(result: dict):
    """Render the prediction result."""
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    rating = result['predicted_rating']
    confidence = result['confidence']

    # Color based on rating
    if rating >= 7.0:
        color = "#2ecc71"  # green
    elif rating >= 5.0:
        color = "#f39c12"  # orange
    else:
        color = "#e74c3c"  # red

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div style='text-align:center; padding:20px; background:#1e1e1e; border-radius:10px;'>
                <h2 style='color:{color}; font-size:48px;'>{rating}</h2>
                <p style='color:gray;'>Predicted Rating</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style='text-align:center; padding:20px; background:#1e1e1e; border-radius:10px;'>
                <h2 style='color:{color}; font-size:48px;'>{confidence}</h2>
                <p style='color:gray;'>Confidence</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        stars = "⭐" * int(round(rating / 2))
        st.markdown(f"""
            <div style='text-align:center; padding:20px; background:#1e1e1e; border-radius:10px;'>
                <h2 style='font-size:32px;'>{stars}</h2>
                <p style='color:gray;'>Out of 10</p>
            </div>
        """, unsafe_allow_html=True)

def render_shap_chart(top_features: dict):
    """Render SHAP feature importance chart."""
    st.markdown("---")
    st.subheader("🔍 Why this rating? (SHAP Explanation)")

    features = list(top_features.keys())
    values = list(top_features.values())
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition='outside'
    ))

    fig.update_layout(
        title="Feature Contributions to Predicted Rating",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis_title="Feature",
        height=400,
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font_color="white",
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333")
    )

    st.plotly_chart(fig, use_container_width=True)

def render_history_table(history: list):
    """Render prediction history table."""
    if not history:
        return

    st.markdown("---")
    st.subheader("📊 Prediction History")

    df = pd.DataFrame(history)[['title', 'predicted_rating', 'confidence']]
    df.columns = ['Movie', 'Predicted Rating', 'Confidence']
    st.dataframe(df, use_container_width=True)