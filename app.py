import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import plotly.express as px

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

    df = df.drop_duplicates(subset='Country')

    impute_cols = df.select_dtypes(include='number').columns
    imputer = KNNImputer(n_neighbors=5)
    df[impute_cols] = imputer.fit_transform(df[impute_cols])

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

importance = {
    'Politics': st.slider('Importance of Political Engagement', 1, 5, 3),
    'Family': st.slider('Importance of Family Values', 1, 5, 3),
    'Leisure time': st.slider('Importance of Leisure Time', 1, 5, 3),
    'Internet Access': st.slider('Importance of Internet Access', 1, 5, 3),
    'Relationship Satisfaction': st.slider('Importance of Social/Relationship Satisfaction', 1, 5, 3),
    'Safety Perception': st.slider('Importance of Feeling Safe', 1, 5, 3),
    'Life Expectancy': st.slider('Importance of Longevity/Life Expectancy', 1, 5, 3)
}

if st.button("🔍 Generate Recommendations"):
    df = load_data()

    df_norm = df.copy()
    for col in importance.keys():
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    weights = np.array([importance[col] for col in importance])
    df_norm['Score'] = df_norm[list(importance.keys())].values @ weights

    top_countries = df_norm.sort_values(by='Score', ascending=False).head(10)

    st.subheader("🏆 Top 3 Country Grades")
    for i in range(3):
        row = top_countries.iloc[i]
        grade = "A" if row['Score'] > 4.5 else "B" if row['Score'] > 4.0 else "C"
        highlight = row[list(importance.keys())].idxmax()
        lowlight = row[list(importance.keys())].idxmin()
        st.markdown(f"### {i+1}. {row['Country']} — Grade: {grade}")
        st.markdown(f"_Excels in **{highlight}**, lags in **{lowlight}**._")
        st.progress(row['Score'] / df_norm['Score'].max())

    st.subheader("📋 Full Recommendations")
    st.dataframe(top_countries[['Country', 'Score'] + list(importance.keys())].reset_index(drop=True))

    st.subheader("📊 Country Score Chart")
    st.bar_chart(top_countries.set_index('Country')['Score'])

    st.subheader("🧭 Country Clusters (Interactive)")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_norm[list(importance.keys())])
    df_norm['PC1'], df_norm['PC2'] = components[:, 0], components[:, 1]

    strengths = df_norm[list(importance.keys())].idxmax(axis=1)
    weaknesses = df_norm[list(importance.keys())].idxmin(axis=1)
    blurbs = [f"Excels in {s}, lags in {w}" for s, w in zip(strengths, weaknesses)]

    fig = px.scatter(
        df_norm, x='PC1', y='PC2', text='Country', hover_name='Country',
        hover_data={'Score': True, 'Strength': strengths, 'Weakness': weaknesses},
        title="Country Clusters Based on Your Values"
    )
    fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
    st.plotly_chart(fig, use_container_width=True)
