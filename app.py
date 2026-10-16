import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender")

@st.cache_data
def load_data():
    movies = pd.read_csv('data/movies.csv')
    movies['genres_clean'] = movies['genres'].str.replace('|', ' ', regex=False)
    matrix = TfidfVectorizer().fit_transform(movies['genres_clean'])
    return movies, cosine_similarity(matrix)

movies, similarity = load_data()

def recommend(title, n=5):
    matches = movies.index[movies['title'] == title]
    if len(matches) == 0:
        return []
    idx = matches[0]
    scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
    return movies['title'].iloc[[i for i, _ in scores]].tolist()

st.title("Movie Recommender")
st.write("Pick a movie and get 5 similar ones.")

title = st.selectbox("Movie", movies['title'].values)
if st.button("Recommend"):
    for r in recommend(title):
        st.write("- " + r)
