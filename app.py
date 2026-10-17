import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Movie Recommender")

@st.cache_data
def load_data():
    movies = pd.read_csv('data/movies.csv')
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    matrix = TfidfVectorizer().fit_transform(movies['genres_clean'])  # sparse, tiny
    return movies, matrix

movies, matrix = load_data()

def recommend(title, n=5):
    matches = movies.index[movies['title'] == title]
    if len(matches) == 0:
        return []
    idx = matches[0]
    sims = (matrix @ matrix[idx].T).toarray().ravel()   # similarity to ONE movie only
    order = [i for i in sims.argsort()[::-1] if i != idx][:n]
    return movies['title'].iloc[order].tolist()

st.title("Movie Recommender")
st.write("Pick a movie and get 5 similar ones.")

title = st.selectbox("Movie", movies['title'].values)
if st.button("Recommend"):
    for r in recommend(title):
        st.write("- " + r)
