# Streamlit App for Country Recommendation Based on Grouped Composite Indicators
import streamlit as st
import pandas as pd
import numpy as np
import pycountry
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Page config
st.set_page_config(page_title="Country Recommender", layout="wide")
st.title("🌍 Country Recommender Using Category Priorities")

@st.cache_data

def load_and_prepare_data():
    df = pd.read_excel("Capstone Final Dataset 2019-2025_v2.xlsx", sheet_name="sheet 1")

    df = df.loc[:, df.isnull().mean() < 0.9]
    df = df[df.isnull().mean(axis=1) < 0.9].reset_index(drop=True)

    topics = ["Politics", "Family", "Leisure time", "Work", "Friends", "Religion"]
    weights = {
        "Very important": 5,
        "Rather important": 4,
        "Not very important": 2,
        "Not at all important": 1
    }
    columns_to_drop = []
    created_indices = []

    for topic in topics:
        base_cols = [f"{label} in life: {topic}" for label in weights]
        available_cols = [col for col in base_cols if col in df.columns]
        if len(available_cols) >= 2:  # Create index if at least 2 response types exist
            numerator = sum(weights[label] * df.get(f"{label} in life: {topic}", 0).fillna(0) for label in weights if f"{label} in life: {topic}" in df.columns)
            denominator = sum(df.get(f"{label} in life: {topic}", 0).fillna(0) for label in weights if f"{label} in life: {topic}" in df.columns)
            index_name = f"{topic.replace(' ', '_')}_Importance_Index"
            df[index_name] = numerator / denominator.replace(0, np.nan)
            created_indices.append((topic, index_name))
            drop_cols = available_cols + [f"{x}: Important in life: {topic}" for x in ["Don't know", "No answer"]]
            columns_to_drop.extend([col for col in drop_cols if col in df.columns])

    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df = df[df['Cantril ladder score'].notna()].copy()
    df_meta = df[["Entity", "Year"]].copy()
    y = df["Cantril ladder score"].copy()
    X = df.drop(columns=["Entity", "Year", "Cantril ladder score"], errors='ignore')

    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
    X_scaled.insert(0, "Entity", df_meta["Entity"].values)
    return X_scaled, df_meta, created_indices

# Load and prep data
df_scaled, df_meta, created_indices = load_and_prepare_data()

# Define available categories from created indices
indicator_categories = {
    f"{topic} Importance": [index_name] for topic, index_name in created_indices
}

st.markdown("### 🎛️ Rate Category Importance")
ratings = {}
for cat in indicator_categories:
    ratings[cat] = st.slider(f"Importance of {cat}", 1, 5, 3)

submitted = st.button("Generate Rankings")

if submitted:
    selected_indicators = []
    weights = []
    for cat, val in ratings.items():
        for ind in indicator_categories[cat]:
            if ind in df_scaled.columns:
                selected_indicators.append(ind)
                weights.append(val ** 2)

    if not selected_indicators:
        st.error("No valid indicators found for selected categories.")
        st.stop()

    X = df_scaled[selected_indicators]
    weights = np.array(weights) / np.sum(np.abs(weights))
    scores = X @ weights

    df_scaled['Preference Score'] = scores
    top = df_scaled.sort_values('Preference Score', ascending=False)[['Entity', 'Preference Score']].head(10)
    st.markdown("### 🥇 Top 10 Countries")
    st.dataframe(top)

    # Summary table
    st.markdown("### 📋 Category Summary for Top 3")
    top3 = df_scaled.sort_values('Preference Score', ascending=False).head(3)
    for _, row in top3.iterrows():
        st.markdown(f"#### {row['Entity']}")
        summary = {}
        for cat, inds in indicator_categories.items():
            valid_inds = [ind for ind in inds if ind in row]
            if valid_inds:
                avg = row[valid_inds].mean()
                grade = "A" if avg > 1.0 else "B" if avg > 0.5 else "C" if avg > -0.5 else "D" if avg > -1.0 else "F"
                summary[cat] = grade
        st.write(summary)

    # Map
    def get_iso3(name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except:
            return None
    df_scaled['ISO3'] = df_scaled['Entity'].apply(get_iso3)
    map_df = df_scaled.dropna(subset=['ISO3'])
    fig = px.choropleth(
        map_df,
        locations='ISO3',
        color='Preference Score',
        hover_name='Entity',
        color_continuous_scale='YlGnBu',
        title='Country Preference Scores'
    )
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Clustering visual
    st.markdown("### 🔍 Clustering Countries")
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_data = df_scaled[selected_indicators]
    labels = kmeans.fit_predict(cluster_data)
    df_scaled['Cluster'] = labels

    fig2 = px.scatter(cluster_data, x=selected_indicators[0], y=selected_indicators[1], color=df_scaled['Cluster'].astype(str), hover_name=df_scaled['Entity'])
    fig2.update_layout(title="KMeans Clustering", height=500)
    st.plotly_chart(fig2, use_container_width=True)

    silhouette = silhouette_score(cluster_data, labels)
    st.info(f"Silhouette Score: {silhouette:.3f}")

    # Download
    csv = df_scaled.to_csv(index=False)
    st.download_button("📥 Download Results", data=csv, file_name="ranked_countries.csv", mime="text/csv")
