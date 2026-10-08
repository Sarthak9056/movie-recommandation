import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('data/movies.csv')
# print(movies.head())
# print(movies.shape)

movies['genres_clean'] = movies['genres'].str.replace('/', ' ', regex = False)

tfidf = TfidfVectorizer()
genres_tfidf = tfidf.fit_transform(movies['genres_clean'])

similarity_matrix = cosine_similarity(genres_tfidf)

def recommend(title, n=5):
    idx = movies.index[movies['title'] == title][0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return movies['title'].iloc[[i for i, _ in scores]].tolist()

print(recommend('Assassins (1995)'))