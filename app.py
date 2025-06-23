import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Define indicator categories and directions based on available data
indicator_categories = {
    "Overall Well-being": ["Life expectancy at birth", "Perceived health as positive"],
    "Economic Stability": ["Employment rate", "Households and NPISHs net adjusted disposable income per capita"],
    "Accessibility": ["Housing affordability", "Households living in overcrowded conditions", "Households with internet access at home"],
    "Safety and Belonging": ["Feeling lonely"]
}

indicator_direction = {
    "Life expectancy at birth": "high",
    "Perceived health as positive": "high",
    "Employment rate": "high",
    "Households and NPISHs net adjusted disposable income per capita": "high",
    "Housing affordability": "high",
    "Households living in overcrowded conditions": "low",
    "Households with internet access at home": "high",
    "Feeling lonely": "low"
}

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("OECD data.csv")
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

@st.cache_data
def prepare_data(df):
    try:
        # Remove columns and rows with too many missing values
        df_cleaned = df.loc[:, df.isnull().mean() < 0.5]
        df_cleaned = df_cleaned[df_cleaned.isnull().mean(axis=1) < 0.5]
        
        # Impute missing values
        imputer = KNNImputer(n_neighbors=5)
        numeric_columns = df_cleaned.select_dtypes(include='number').columns
        df_imputed = df_cleaned.copy()
        df_imputed[numeric_columns] = imputer.fit_transform(df_cleaned[numeric_columns])
        
        # Scale features
        scaler = StandardScaler()
        df_scaled = df_imputed.copy()
        df_scaled[numeric_columns] = scaler.fit_transform(df_imputed[numeric_columns])
        
        # Debug: Show variances and data preview
        st.markdown("**Feature Variances:**")
        st.write(df_scaled[numeric_columns].var())
        st.markdown("**Imputed Data Preview:**")
        st.dataframe(df_scaled.head())
        
        return df_imputed, df_scaled
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return df, df

# Streamlit app
st.title("Country Happiness Recommender")

# Load and prepare data
df_wide = load_data()
if df_wide.empty:
    st.stop()

df_imputed, df_scaled = prepare_data(df_wide)

# Collect user ratings
st.markdown("**Rate the importance of each category (1-5):**")
ratings = {}
for category in indicator_categories:
    ratings[category] = st.slider(f"{category}", 1, 5, 3)

# Submit button
if st.button("Get Recommendations"):
    # Debug: Show user ratings
    st.markdown("**User Ratings:**")
    st.json(ratings)
    
    # Calculate weights based on user ratings
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
    
    # Debug: Show selected indicators and weights
    st.markdown("**Selected Indicators:**")
    st.write(selected_indicators)
    st.markdown("**Weights:**")
    st.write(weights)
    
    if selected_indicators:
        try:
            # Calculate preference score
            X = df_scaled[selected_indicators]
            weights = np.array(weights) / np.sum(np.abs(weights))  # Normalize weights
            score = X @ weights
            df_imputed['Preference Score'] = score
            
            # Show top countries
            top_countries = df_imputed.sort_values("Preference Score", ascending=False)[["Country", "Preference Score"]].head(10)
            st.markdown("**Top 10 Countries:**")
            st.dataframe(top_countries)
            
            # Detailed data for top 3
            top_3 = df_imputed.sort_values("Preference Score", ascending=False).head(3)
            st.markdown("**Detailed Data for Top 3 Countries:**")
            st.dataframe(top_3[["Country"] + selected_indicators + ["Preference Score"]])
        except Exception as e:
            st.error(f"Error calculating recommendations: {e}")
    else:
        st.warning("No valid indicators selected. Please adjust ratings.")

# Test preset button for debugging
if st.button("Test Economic Stability Priority"):
    ratings = {cat: 1 for cat in indicator_categories}
    ratings["Economic Stability"] = 5
    st.markdown("**Preset Ratings:**")
    st.json(ratings)
    
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
    
    st.markdown("**Selected Indicators (Preset):**")
    st.write(selected_indicators)
    st.markdown("**Weights (Preset):**")
    st.write(weights)
    
    if selected_indicators:
        X = df_scaled[selected_indicators]
        weights = np.array(weights) / np.sum(np.abs(weights))
        score = X @ weights
        df_imputed['Preference Score'] = score
        top_countries = df_imputed.sort_values("Preference Score", ascending=False)[["Country", "Preference Score"]].head(10)
        st.markdown("**Top 10 Countries (Preset):**")
        st.dataframe(top_countries)
