import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv("Final_data.csv")

df["genres"] = df["genres"].fillna("").str.split(",")
df["tags"] = df["tags"].fillna("").str.split(",")

mlb_genres = MultiLabelBinarizer()
mlb_tags = MultiLabelBinarizer()

genre_matrix_file = "genres_matrix.pkl"
tag_matrix_file = "tags_matrix.pkl"
summary_matrix_file = "summary_matrix.pkl"
content_similarity_file = "content_similarity.pkl"
user_factors_file = "user_factors.pkl"
movie_factors_file = "movie_factors.pkl"


def save_with_pickle(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load_with_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


if os.path.exists(genre_matrix_file):
    genres_matrix = load_with_pickle(genre_matrix_file)
else:
    genres_matrix = mlb_genres.fit_transform(df["genres"])
    save_with_pickle(genres_matrix, genre_matrix_file)

if os.path.exists(tag_matrix_file):
    tags_matrix = load_with_pickle(tag_matrix_file)
else:
    tags_matrix = mlb_tags.fit_transform(df["tags"])
    save_with_pickle(tags_matrix, tag_matrix_file)

tfidf = TfidfVectorizer(stop_words="english")
if os.path.exists(summary_matrix_file):
    summary_matrix = load_with_pickle(summary_matrix_file)
else:
    summary_matrix = tfidf.fit_transform(df["summary"].fillna(""))
    save_with_pickle(summary_matrix, summary_matrix_file)

if os.path.exists(content_similarity_file):
    content_similarity = load_with_pickle(content_similarity_file)
else:
    content_matrix = np.hstack((genres_matrix, tags_matrix, summary_matrix.toarray()))
    content_similarity = cosine_similarity(content_matrix, content_matrix)
    save_with_pickle(content_similarity, content_similarity_file)

if os.path.exists(user_factors_file) and os.path.exists(movie_factors_file):
    user_factors = load_with_pickle(user_factors_file)
    movie_factors = load_with_pickle(movie_factors_file)
else:
    user_movie_matrix = np.random.rand(100, len(df))
    svd = TruncatedSVD(n_components=50)
    user_factors = svd.fit_transform(user_movie_matrix)
    movie_factors = svd.components_.T
    save_with_pickle(user_factors, user_factors_file)
    save_with_pickle(movie_factors, movie_factors_file)


def get_hybrid_recommendations(
    user_id, movie_id, content_weight=0.5, collaborative_weight=0.5
):
    collaborative_score = np.dot(user_factors[user_id], movie_factors[movie_id])

    content_score = content_similarity[movie_id]

    hybrid_scores = (content_weight * content_score) + (
        collaborative_weight * collaborative_score
    )

    recommended_indices = hybrid_scores.argsort()[::-1][:30]
    return df["title"].iloc[recommended_indices]


user_id = 0
movie_id = 6838
print(get_hybrid_recommendations(user_id, movie_id))
