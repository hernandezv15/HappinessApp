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

# Define indicator categories and directions
indicator_categories = {
    "Overall Well-being": ["Life expectancy at birth", "Perceived health as positive", "Negative affect balance"],
    "Civic Engagement & Education": ["Voter turnout", "Having a say in government", "Adult literacy skills", "Students with low skills in reading, mathematics and science"],
    "Environmental Quality": ["Exposed to air pollution", "Exposure to extreme temperature"],
    "Economic Stability": ["Employment rate", "Households and NPISHs net adjusted disposable income per capita", "Gender wage gap"],
    "Accessibility": ["Housing affordability", "Households living in overcrowded conditions", "Households with internet access at home"],
    "Safety and Belonging": ["Social support", "Satisfaction with personal relationships", "Feeling lonely", "Feeling safe at night"]
}

indicator_direction = {
    "Life expectancy at birth": "high",
    "Perceived health as positive": "high",
    "Negative affect balance": "low",
    "Voter turnout": "high",
    "Having a say in government": "high",
    "Adult literacy skills": "high",
    "Students with low skills in reading, mathematics and science": "low",
    "Exposed to air pollution": "low",
    "Exposure to extreme temperature": "low",
    "Employment rate": "high",
    "Households and NPISHs net adjusted disposable income per capita": "high",
    "Gender wage gap": "low",
    "Housing affordability": "high",
    "Households living in overcrowded conditions": "low",
    "Households with internet access at home": "high",
    "Social support": "high",
    "Satisfaction with personal relationships": "high",
    "Feeling lonely": "low",
    "Feeling safe at night": "high"
}

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("OECD data")
        df = df.dropna(subset=["OBS_VALUE"])[["Reference area", "Measure", "OBS_VALUE", "TIME_PERIOD"]]
        df.columns = ["Country", "Indicator", "Value", "Year"]
        # Select the most recent year for each country-indicator pair
        df = df.loc[df.groupby(['Country', 'Indicator'])['Year'].idxmax()]
        df_wide = df.pivot_table(index="Country", columns="Indicator", values="Value", aggfunc="mean").reset_index()
        # Debug: Show available indicators
        st.markdown("**Available Indicators:**")
        st.write(df_wide.columns.tolist())
        return df_wide
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Impute and standardize
@st.cache_data
def prepare_data(df):
    try:
        # Remove columns and rows with too many missing values
        df_cleaned = df.loc[:, df.isnull().mean() < 0.5]  # Reduced threshold for more data
        df_cleaned = df_cleaned[df_cleaned.isnull().mean(axis=1) < 0.5]
        # Impute missing values
        imputer = KNNImputer(n_neighbors=10)  # Increased for better imputation
        df_numeric = df_cleaned.select_dtypes(include='number')
        df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns, index=df_cleaned.index)
        df_imputed.insert(0, 'Country', df_cleaned['Country'].values)
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_imputed.drop(columns=['Country']))
        df_scaled = pd.DataFrame(features_scaled, columns=df_numeric.columns, index=df_cleaned.index)
        df_scaled['Country'] = df_cleaned['Country']
        # Debug: Show variances
        st.markdown("**Feature Variances:**")
        st.write(df_scaled.select_dtypes(include='number').var())
        return df_imputed, df_scaled
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return df, df

def assign_grade(score, percentiles):
    if score >= percentiles[90]: return "A+"
    elif score >= percentiles[80]: return "A"
    elif score >= percentiles[60]: return "B"
    elif score >= percentiles[40]: return "C"
    elif score >= percentiles[20]: return "D"
    else: return "F"

def assign_aesthetic_rank(index):
    if index < 3: return "🌟"  # Top 3
    elif index < 6: return "😊"  # 4th-6th
    else: return "😐"  # 7th-10th

# Page setup
st.set_page_config(page_title="Country Recommender", layout="wide")
st.title("🌍 Country Recommendation Based on Your Priorities")
st.markdown("Rate what's important to you in each category. We’ll recommend countries that perform well in those areas.")

# Load and prep
df_raw = load_data()
if df_raw.empty:
    st.stop()
df_imputed, df_scaled = prepare_data(df_raw)

# Collect user ratings
ratings = {}
with st.form("priority_form"):
    st.subheader("📋 Rate Category Importance (1 = Least, 5 = Most)")
    for category,
