import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pycountry
import plotly.express as px

# Load data
@st.cache_data

def load_data():
    df = pd.read_csv("OECD data")
    df = df.dropna(subset=["OBS_VALUE"])[["Reference area", "Measure", "OBS_VALUE", "TIME_PERIOD"]]
    df.columns = ["Country", "Indicator", "Value", "Year"]
    df = df.loc[df.groupby(['Country', 'Indicator'])['Year'].idxmax()]
    df_wide = df.pivot_table(index="Country", columns="Indicator", values="Value", aggfunc="mean").reset_index()
    st.write("Available Indicators:", df_wide.columns.tolist())
    return df_wide

if submitted:
    st.write("User Ratings:", ratings)
    selected_indicators = []
    weights = []
    for category, score in ratings.items():
        if score > 0:
            for indicator in indicator_categories[category]:
                if indicator in df_scaled.columns:
                    direction = indicator_direction.get(indicator, "high")
                    weight = score if direction == "high" else -score
                    selected_indicators.append(indicator)
                    weights.append(weight)
    st.write("Selected Indicators:", selected_indicators)
    st.write("Weights:", weights)
    if selected_indicators:
        X = df_scaled[selected_indicators]
        weights = np.array(weights) / np.sum(np.abs(weights))
        score = X @ weights
        df_imputed['Preference Score'] = score
        st.write("Top 10 Countries:", df_imputed.sort_values("Preference Score", ascending=False)[["Country", "Preference Score"]].head(10))
