import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("Capstone Final Dataset 2019-2025_v2.xlsx", sheet_name='sheet 1')
    df = df[[
        'Entity',
        'Households with internet access at home',
        'Satisfaction with personal relationships',
        'Very important in life: Leisure time',
        'Very important in life: Politics',
        'Very important in life: Family',
        'Rather important in life: Leisure time',
        'Rather important in life: Politics',
        'Rather important in life: Family',
        'Not very important in life: Leisure time',
        'Not very important in life: Politics',
        'Not very important in life: Family',
        'Not at all important in life: Leisure time',
        'Not at all important in life: Politics',
        'Not at all important in life: Family',
        '16.1.4 - Proportion of population that feel safe walking alone around the area they live after dark (%) - VC_SNS_WALN_DRK - Bot',
        'Life expectancy - Sex: all - Age: 0 - Variant: estimates'
    ]]
    df = df.dropna(subset=['Entity'])
    df = df[~df['Entity'].str.contains(r'\(|World|income', regex=True)]

    df.columns = [
        'Country', 'Internet Access', 'Relationship Satisfaction',
        'Very Leisure', 'Very Politics', 'Very Family',
        'Rather Leisure', 'Rather Politics', 'Rather Family',
        'NotVery Leisure', 'NotVery Politics', 'NotVery Family',
        'NotAtAll Leisure', 'NotAtAll Politics', 'NotAtAll Family',
        'Safety Perception', 'Life Expectancy'
    ]

    # Impute missing values using KNN
    impute_cols = df.select_dtypes(include='number').columns
    imputer = KNNImputer(n_neighbors=5)
    df[impute_cols] = imputer.fit_transform(df[impute_cols])

    # Composite scores
    df['Politics'] = (
        df['Very Politics'] * 5 + df['Rather Politics'] * 4 +
        df['NotVery Politics'] * 2 + df['NotAtAll Politics'] * 1
    )
    df['Family'] = (
        df['Very Family'] * 5 + df['Rather Family'] * 4 +
        df['NotVery Family'] * 2 + df['NotAtAll Family'] * 1
    )
    df['Leisure time'] = (
        df['Very Leisure'] * 5 + df['Rather Leisure'] * 4 +
        df['NotVery Leisure'] * 2 + df['NotAtAll Leisure'] * 1
    )

    return df.reset_index(drop=True)

# App
st.title("🌍 Country Recommender Based on Your Values")
st.write("Rank how important each category is to you (1 = least, 5 = most)")

# User importance inputs
importance = {
    'Politics': st.slider('Importance of Political Engagement', 1, 5, 3),
    'Family': st.slider('Importance of Family Values', 1, 5, 3),
    'Leisure time': st.slider('Importance of Leisure Time', 1, 5, 3),
    'Internet Access': st.slider('Importance of Internet Access', 1, 5, 3),
    'Relationship Satisfaction': st.slider('Importance of Social/Relationship Satisfaction', 1, 5, 3),
    'Safety Perception': st.slider('Importance of Feeling Safe', 1, 5, 3),
    'Life Expectancy': st.slider('Importance of Longevity/Life Expectancy', 1, 5, 3)
}

# Load and score data
df = load_data()

# Normalize each category
df_norm = df.copy()
for col in importance.keys():
    df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Weighted score
weights = np.array([importance[col] for col in importance])
df_norm['Score'] = df_norm[list(importance.keys())].values @ weights

# Show top countries
top_countries = df_norm.sort_values(by='Score', ascending=False).head(10)

st.subheader("🏆 Top 10 Recommended Countries")
st.dataframe(top_countries[['Country', 'Score'] + list(importance.keys())].reset_index(drop=True))

# Bar chart
st.bar_chart(top_countries.set_index('Country')['Score'])
