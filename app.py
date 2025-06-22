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
    df = df.dropna(subset=["OBS_VALUE"])[["Reference area", "Measure", "OBS_VALUE"]]
    df.columns = ["Country", "Indicator", "Value"]
    df_wide = df.pivot_table(index="Country", columns="Indicator", values="Value", aggfunc="mean").reset_index()
    return df_wide

# Impute and standardize
@st.cache_data

def prepare_data(df):
    df_cleaned = df.loc[:, df.isnull().mean() < 0.7]
    df_cleaned = df_cleaned[df_cleaned.isnull().mean(axis=1) < 0.7]
    imputer = KNNImputer(n_neighbors=5)
    df_numeric = df_cleaned.select_dtypes(include='number')
    df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns, index=df_cleaned.index)
    df_imputed.insert(0, 'Country', df_cleaned['Country'].values)
    return df_imputed

# Define desired direction for each indicator
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

# Group indicators by broad categories
indicator_categories = {
    "Overall Well-being": ["Life expectancy at birth", "Perceived health as positive", "Negative affect balance"],
    "Civic Engagement & Education": ["Voter turnout", "Having a say in government", "Adult literacy skills", "Students with low skills in reading, mathematics and science"],
    "Environmental Quality": ["Exposed to air pollution", "Exposure to extreme temperature"],
    "Economic Stability": ["Employment rate", "Households and NPISHs net adjusted disposable income per capita", "Gender wage gap"],
    "Accessibility": ["Housing affordability", "Households living in overcrowded conditions", "Households with internet access at home"],
    "Safety and Belonging": ["Social support", "Satisfaction with personal relationships", "Feeling lonely", "Feeling safe at night"]
}

# Page setup
st.set_page_config(page_title="Country Recommender", layout="wide")
st.title("🌍 Country Recommendation Based on Your Priorities")
st.markdown("Rate what's important to you in each category. We’ll recommend countries that perform well in those areas.")

# Load and prep
df_raw = load_data()
df_imputed = prepare_data(df_raw)
features = df_imputed.drop(columns=['Country'])
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
df_scaled = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
df_scaled['Country'] = df_imputed['Country']

# Collect user ratings
ratings = {}
with st.form("priority_form"):
    st.subheader("📋 Rate Category Importance (1 = Least, 5 = Most)")
    for category, indicators in indicator_categories.items():
        st.markdown(f"**{category}**")
        ratings[category] = st.slider(f"Importance of {category.lower()}", 1, 5, 3)
    submitted = st.form_submit_button("Show Recommendations")

if submitted:
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

    if selected_indicators:
        X = df_scaled[selected_indicators]
        score = X @ weights
        df_imputed['Preference Score'] = score
        top_countries = df_imputed.sort_values("Preference Score", ascending=False).head(3)

        st.subheader("🌟 Recommended Countries for You")

        for _, row in top_countries.iterrows():
            st.markdown(f"### 🏷️ {row['Country']}")
            grades = {}
            summary = []
            for category, indicators in indicator_categories.items():
                if any(ind in selected_indicators for ind in indicators):
                    cat_values = row[indicators].values
                    direction_adjusted = []
                    for ind in indicators:
                        if ind in selected_indicators:
                            val = row[ind]
                            if indicator_direction.get(ind) == "low":
                                val = -val
                            direction_adjusted.append(val)
                    avg_score = np.mean(direction_adjusted) if direction_adjusted else 0
                    grade = "A" if avg_score > 1.0 else "B" if avg_score > 0.5 else "C" if avg_score > -0.5 else "D" if avg_score > -1.0 else "F"
                    grades[category] = grade
                    if grade in ["A", "B"]:
                        summary.append(f"Excels in {category.lower()}")
                    elif grade in ["D", "F"]:
                        summary.append(f"Lags in {category.lower()}")
            st.markdown("**Category Grades:**")
            st.write(grades)
            st.markdown("**Summary:**")
            st.write(", ".join(summary))
            st.markdown("---")

        # Regional interactive map
        st.subheader("🗺️ Click a Country on the Map to View Category Grades")

        df_imputed['ISO3'] = df_imputed['Country'].apply(lambda x: pycountry.countries.lookup(x).alpha_3 if pycountry.countries.get(name=x) else None)
        df_viz = df_imputed.dropna(subset=['ISO3'])
        fig = px.choropleth(
            df_viz,
            locations="ISO3",
            color="Preference Score",
            hover_name="Country",
            color_continuous_scale="YlGnBu",
            title="Preference Scores by Country",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please rate at least one category to get recommendations.")
