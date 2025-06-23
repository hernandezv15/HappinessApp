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
        return df_wide
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Impute and standardize
@st.cache_data
def prepare_data(df):
    try:
        # Remove columns and rows with too many missing values
        df_cleaned = df.loc[:, df.isnull().mean() < 0.7]
        df_cleaned = df_cleaned[df_cleaned.isnull().mean(axis=1) < 0.7]
        # Impute missing values
        imputer = KNNImputer(n_neighbors=10)
        df_numeric = df_cleaned.select_dtypes(include='number')
        df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns, index=df_cleaned.index)
        # Median imputation for sparse indicators
        for col in ["Feeling lonely", "Negative affect balance"]:
            if col in df_cleaned.columns:
                df_imputed[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        df_imputed.insert(0, 'Country', df_cleaned['Country'].values)
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_imputed.drop(columns=['Country']))
        df_scaled = pd.DataFrame(features_scaled, columns=df_numeric.columns, index=df_cleaned.index)
        df_scaled['Country'] = df_cleaned['Country']
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
                    weight = score ** 2 if direction == "high" else -(score ** 2)  # Exponential weighting
                    selected_indicators.append(indicator)
                    weights.append(weight)
    
    if selected_indicators:
        try:
            # Calculate preference score
            X = df_scaled[selected_indicators]
            weights = np.array(weights) / np.sum(np.abs(weights))  # Normalize weights
            score = X @ weights
            df_imputed['Preference Score'] = score
            
            # Calculate percentiles for grading
            percentiles = np.percentile(df_imputed['Preference Score'], [20, 40, 60, 80, 90])
            
            # Top 10 countries with grades and aesthetic ranks
            st.subheader("🌟 Top 10 Recommended Countries")
            top_countries = df_imputed.sort_values("Preference Score", ascending=False)[["Country", "Preference Score"]].head(10).copy()
            top_countries['Grade'] = top_countries['Preference Score'].apply(lambda x: assign_grade(x, percentiles))
            top_countries['Aesthetic Rank'] = [assign_aesthetic_rank(i) for i in range(len(top_countries))]
            st.dataframe(top_countries)
            
            # Detailed data for top 3 with category grades
            st.subheader("📊 Detailed Data for Top 3 Countries")
            top_3 = df_imputed.sort_values("Preference Score", ascending=False).head(3)
            for _, row in top_3.iterrows():
                st.markdown(f"### 🏷️ {row['Country']}")
                grades = {}
                summary = []
                for category, indicators in indicator_categories.items():
                    if any(ind in selected_indicators for ind in indicators):
                        cat_values = [row[ind] for ind in indicators if ind in selected_indicators]
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
                st.markdown("**Indicator Values:**")
                st.dataframe(row[["Country"] + selected_indicators + ["Preference Score"]].to_frame().T)
                st.markdown("---")
            
            # Regional interactive map with country hover detail
            st.subheader("🗺️ Country Scores Map with Expanded Info")
            df_imputed['ISO3'] = df_imputed['Country'].apply(
                lambda x: pycountry.countries.lookup(x).alpha_3 if pycountry.countries.get(name=x) else None
            )
            if df_imputed['ISO3'].isnull().any():
                st.warning("Some countries could not be mapped to ISO-3 codes. Check country names in dataset.")
            df_viz = df_imputed.dropna(subset=['ISO3'])
            
            # Create hover text
            hover_text = []
            for i, row in df_viz.iterrows():
                text = f"<b>{row['Country']}</b><br>Preference Score: {row['Preference Score']:.2f}<br>"
                for cat, inds in indicator_categories.items():
                    cat_vals = []
                    for ind in inds:
                        if ind in row:
                            val = row[ind]
                            if indicator_direction.get(ind) == "low":
                                val = -val
                            cat_vals.append(val)
                    if cat_vals:
                        avg = np.mean(cat_vals)
                        grade = "A" if avg > 1.0 else "B" if avg > 0.5 else "C" if avg > -0.5 else "D" if avg > -1.0 else "F"
                        text += f"{cat}: {grade}<br>"
                hover_text.append(text)
            
            df_viz['hover'] = hover_text
            
            fig = px.choropleth(
                df_viz,
                locations="ISO3",
                color="Preference Score",
                hover_name="Country",
                hover_data={"ISO3": False, "Preference Score": True, "hover": True},
                color_continuous_scale="YlGnBu",
                title="Preference Scores by Country with Grades"
            )
            fig.update_traces(hovertemplate='%{customdata[0]}')
            fig.update_layout(
                margin={"r":0,"t":50,"l":0,"b":0},
