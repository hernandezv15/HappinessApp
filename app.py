import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data

def load_data():
    df = pd.read_csv("OECD data", encoding='utf-8')
    df = df.dropna(subset=["OBS_VALUE"])[["Reference area", "Measure", "OBS_VALUE"]]
    df.columns = ["Country", "Indicator", "Value"]
    df_wide = df.pivot_table(index="Country", columns="Indicator", values="Value", aggfunc="mean").reset_index()
    return df_wide

# Impute and standardize
@st.cache_data


def prepare_data(df):
    df_cleaned = df.loc[:, df.isnull().mean() < 0.2]
    df_cleaned = df_cleaned[df_cleaned.isnull().mean(axis=1) < 0.3]
    imputer = KNNImputer(n_neighbors=5)
    df_numeric = df_cleaned.select_dtypes(include='number')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns, index=df_cleaned.index)
    df_imputed.insert(0, 'Country', df_cleaned['Country'].values)
    return df_imputed

# Page setup
st.set_page_config(page_title="Country Recommender", layout="wide")
st.title("🌍 Country Recommendation Based on Your Priorities")
st.markdown("Rate what's important to you on the sliders below, and click **'Show Recommendations'** to view your best-matched countries.")

# Load and prep
raw = load_data()
df_imputed = prepare_data(raw)
features = df_imputed.drop(columns=['Country'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
df_scaled['Country'] = df_imputed['Country']

# Organize indicators by category
indicator_categories = {
    "Social": ["Feeling lonely", "Feeling safe at night", "Life satisfaction"],
    "Economic": ["Employment rate", "Household net adjusted disposable income", "Gender wage gap"],
    "Environmental": ["Air pollution PM2.5 exposure", "Access to green spaces", "Urban population exposure to air pollution", "Waste recycling rate"],
    "Health": ["Life expectancy", "Mental health issues"],
    "Education": ["Educational attainment", "Adult skills"]
}

ratings = {}
with st.form("priority_form"):
    st.subheader("📋 Rate What's Important to You (1 = Least, 5 = Most)")
    for category, indicators in indicator_categories.items():
        st.markdown(f"**{category} Indicators**")
        for ind in indicators:
            if ind in df_scaled.columns:
                ratings[ind] = st.slider(ind, 1, 5, 3)

    submitted = st.form_submit_button("Show Recommendations")

if submitted:
    selected_indicators = [k for k, v in ratings.items() if v > 0]
    weights = np.array([ratings[k] for k in selected_indicators])

    if len(selected_indicators) >= 1:
        X = df_scaled[selected_indicators]
        score = X @ weights
        df_imputed['Preference Score'] = score
        top_countries = df_imputed.sort_values("Preference Score", ascending=False).head(3)

        st.subheader("🌟 Recommended Countries for You")
        st.dataframe(top_countries[['Country', 'Preference Score'] + selected_indicators])

        st.markdown("### Why These Countries?")
        for _, row in top_countries.iterrows():
            reasons = ", ".join([f"{col}: {row[col]:.2f}" for col in selected_indicators])
            st.markdown(f"- **{row['Country']}** → {reasons}")

        st.subheader("📊 PCA Projection")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        df_proj = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_proj['Country'] = df_imputed['Country']
        df_proj['Cluster'] = labels
        fig, ax = plt.subplots()
        sns.scatterplot(data=df_proj, x='PC1', y='PC2', hue='Cluster', style='Country', palette='Set2', ax=ax)
        st.pyplot(fig)

        st.subheader("📌 Cluster Averages for Selected Indicators")
        df_imputed['Cluster'] = labels
        st.dataframe(df_imputed.groupby('Cluster')[selected_indicators].mean().T.style.highlight_max(axis=1))
    else:
        st.info("Please rate at least one indicator to get recommendations.")
