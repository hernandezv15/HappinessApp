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
    df_cleaned = df.loc[:, df.isnull().mean() < 0.4]
    df_cleaned = df_cleaned[df_cleaned.isnull().mean(axis=1) < 0.5]
    imputer = KNNImputer(n_neighbors=5)
    df_numeric = df_cleaned.select_dtypes(include='number')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns, index=df_cleaned.index)
    df_imputed.insert(0, 'Country', df_cleaned['Country'].values)
    return df_imputed

# Page setup
st.title("Country Clustering and Recommendation Tool")

# Load and prep
raw = load_data()
df_imputed = prepare_data(raw)
features = df_imputed.drop(columns=['Country'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Regression
if 'Life satisfaction' in features.columns:
    y = features['Life satisfaction']
    X_reg = features.drop(columns='Life satisfaction')
    reg = LinearRegression().fit(X_reg, y)
    coeffs = pd.Series(reg.coef_, index=X_reg.columns)
else:
    reg = None
    X_reg = features
    coeffs = pd.Series([0]*features.shape[1], index=features.columns)

# Top features selection
st.sidebar.header("Pick Top 3 Important Indicators")
all_feats = coeffs.abs().sort_values(ascending=False).index.tolist()
selected_feats = st.sidebar.multiselect("Choose 3 indicators:", all_feats, default=all_feats[:3])

if len(selected_feats) == 3:
    X_top = X_reg[selected_feats]
    X_top_scaled = StandardScaler().fit_transform(X_top)

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_top_scaled)
    df_imputed['Cluster'] = labels

    # Recommendation: find closest 3 to cluster centers
    distances = kmeans.transform(X_top_scaled)
    closest_idx = np.argsort(distances.min(axis=1))[:3]
    recommended = df_imputed.iloc[closest_idx][['Country'] + selected_feats + ['Cluster']]

    st.subheader("Recommended Countries Based on Selected Indicators")
    st.dataframe(recommended)

    st.markdown("### Why These Countries?")
    for _, row in recommended.iterrows():
        summary = f"- **{row['Country']}**: " + \
                  ", ".join([f"{feat} = {row[feat]:.2f}" for feat in selected_feats]) + \
                  f" (Cluster {int(row['Cluster'])})"
        st.markdown(summary)

    # Visualizations
    st.subheader("PCA Projection")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_top_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels
    pca_df['Country'] = df_imputed['Country']
    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', style='Country', palette='Set2', ax=ax)
    st.pyplot(fig)

    st.subheader("Cluster Averages")
    cluster_means = df_imputed.groupby('Cluster')[selected_feats].mean().T
    st.dataframe(cluster_means.style.highlight_max(axis=1))
else:
    st.info("Please select exactly 3 indicators to generate recommendations.")
