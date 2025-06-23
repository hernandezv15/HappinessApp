import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import plotly.express as px

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

# ISO-3 codes for OECD countries (for mapping)
oecd_iso_codes = {
    "Australia": "AUS", "Austria": "AUT", "Belgium": "BEL", "Canada": "CAN", "Chile": "CHL",
    "Czechia": "CZE", "Denmark": "DNK", "Estonia": "EST", "Finland": "FIN", "France": "FRA",
    "Germany": "DEU", "Greece": "GRC", "Hungary": "HUN", "Iceland": "ISL", "Ireland": "IRL",
    "Israel": "ISR", "Italy": "ITA", "Japan": "JPN", "Korea": "KOR", "Latvia": "LVA",
    "Lithuania": "LTU", "Luxembourg": "LUX", "Mexico": "MEX", "Netherlands": "NLD",
    "New Zealand": "NZL", "Norway": "NOR", "Poland": "POL", "Portugal": "PRT",
    "Slovak Republic": "SVK", "Slovenia": "SVN", "Spain": "ESP", "Sweden": "SWE",
    "Switzerland": "CHE", "Turkey": "TUR", "United Kingdom": "GBR", "United States": "USA",
    "Colombia": "COL", "Costa Rica": "CRI"
}

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

@st.cache_data
def prepare_data(df):
    try:
        # Remove columns and rows with too many missing values
        df_cleaned = df.loc[:, df.isnull().mean() < 0.5]
        df_cleaned = df_cleaned[df_cleaned.isnull().mean(axis=1) < 0.5]
        
        # Impute missing values
        imputer = KNNImputer(n_neighbors=10)
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

# Streamlit app
st.title("Country Happiness Recommender")

# Load and prepare data
df_wide = load_data()
if df_wide.empty:
    st.stop()

df_imputed, df_scaled = prepare_data(df_wide)

# Collect user ratings
st.subheader("Rate the importance of each category (1-5)")
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
                    weight = score ** 2 if direction == "high" else -(score ** 2)  # Exponential weighting
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
            
            # Calculate percentiles for grading
            percentiles = np.percentile(df_imputed['Preference Score'], [20, 40, 60, 80, 90])
            
            # Top 10 countries with grades and aesthetic ranks
            st.subheader("Top 10 Recommended Countries")
            top_countries = df_imputed.sort_values("Preference Score", ascending=False)[["Country", "Preference Score"]].head(10).copy()
            top_countries['Grade'] = top_countries['Preference Score'].apply(lambda x: assign_grade(x, percentiles))
            top_countries['Aesthetic Rank'] = [assign_aesthetic_rank(i) for i in range(len(top_countries))]
            st.dataframe(top_countries)
            
            # Detailed data for top 3
            st.subheader("Detailed Data for Top 3 Countries")
            top_3 = df_imputed.sort_values("Preference Score", ascending=False).head(3)
            st.dataframe(top_3[["Country"] + selected_indicators + ["Preference Score"]])
            
            # Interactive map
            st.subheader("Interactive Map of Recommendations")
            df_map = df_imputed.copy()
            df_map['ISO3'] = df_map['Country'].map(oecd_iso_codes)
            df_map['Hover_Text'] = df_map.apply(
                lambda row: f"Country: {row['Country']}<br>Preference Score: {row['Preference Score']:.2f}<br>" +
                            f"Employment rate: {row.get('Employment rate', 'N/A')}<br>" +
                            f"Income per capita: {row.get('Households and NPISHs net adjusted disposable income per capita', 'N/A')}<br>" +
                            f"Life expectancy: {row.get('Life expectancy at birth', 'N/A')}<br>" +
                            f"Feeling lonely: {row.get('Feeling lonely', 'N/A')}%",
                axis=1
            )
            fig = px.choropleth(
                df_map,
                locations="ISO3",
                color="Preference Score",
                hover_name="Country",
                hover_data={"Preference Score": ":.2f", "Hover_Text": True},
                color_continuous_scale=px.colors.sequential.Plasma,
                title="Recommended Countries Based on Your Preferences",
                projection="natural earth"
            )
            fig.update_geos(showcountries=True, countrycolor="Black")
            fig.update_layout(
                margin={"r":0,"t":50,"l":0,"b":0},
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Full dataset table
            st.subheader("Full Country Dataset")
            st.dataframe(df_imputed)
            csv_imputed = df_imputed.to_csv(index=False)
            st.download_button(
                label="Download Full Dataset (Imputed)",
                data=csv_imputed,
                file_name="oecd_imputed_data.csv",
                mime="text/csv"
            )
            
            # Debug: Show raw and scaled values for top 3
            st.markdown("**Raw Data for Top 3 Countries:**")
            st.dataframe(df_imputed.loc[top_3.index, ["Country"] + selected_indicators])
            st.markdown("**Scaled Data for Top 3 Countries:**")
            st.dataframe(df_scaled.loc[top_3.index, ["Country"] + selected_indicators])
        except Exception as e:
            st.error(f"Error calculating recommendations or generating outputs: {e}")
    else:
        st.warning("No valid indicators selected. Please adjust ratings.")
