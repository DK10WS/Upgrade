import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class MovieRecommender:
    def __init__(self, df: pd.DataFrame, cache_dir: str = ".cache", model_name="all-mpnet-base-v2"):
        os.makedirs(cache_dir, exist_ok=True)
        self.df = df.copy()
        self.cache_dir = cache_dir
        self.model = SentenceTransformer(model_name)

        self._prepare_data()
        self._load_or_compute_embeddings()

    def _prepare_data(self):
        def safe_split(x):
            if pd.isna(x):
                return []
            return [i.strip() for i in x.split(",")]

        self.df["genres"] = self.df["genres"].apply(safe_split)
        self.df["sub_genres"] = self.df["sub_genres"].apply(safe_split)
        self.df["tags"] = self.df["tags"].apply(safe_split)

        self.df["release_year"] = pd.to_datetime(self.df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
        self.df["summary"] = self.df["summary"].fillna("")

        self.df["text_for_embedding"] = (
            self.df["title"].fillna("") + ". " +
            self.df["summary"] + ". " +
            self.df["genres"].apply(lambda x: ", ".join(x)) + ". " +
            self.df["sub_genres"].apply(lambda x: ", ".join(x)) + ". " +
            self.df["tags"].apply(lambda x: ", ".join(x)) + ". " +
            self.df["release_year"].astype(str)
        )

    def _load_or_compute_embeddings(self):
        emb_path = os.path.join(self.cache_dir, "movie_embeddings.npy")
        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path)
        else:
            self.embeddings = self.model.encode(
                self.df["text_for_embedding"].tolist(),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            np.save(emb_path, self.embeddings)

    def recommend(self, movie_title, top_k=10):
        if movie_title not in self.df["title"].values:
            raise ValueError(f"Movie '{movie_title}' not found in dataset.")

        idx = self.df.index[self.df["title"] == movie_title][0]
        query_vec = self.embeddings[idx].reshape(1, -1)
        scores = cosine_similarity(query_vec, self.embeddings).ravel()

        top_indices = scores.argsort()[::-1][1: top_k + 1]
        recommendations = self.df.loc[top_indices, "title"].tolist()
        scores = scores[top_indices]

        return list(zip(recommendations, scores))


if __name__ == "__main__":
    df = pd.read_csv("movies.csv")
    recommender = MovieRecommender(df)

    movie = "Schindler's List"
    for title, score in recommender.recommend(movie, top_k=20):
        print(f"{title:60}  Score: {score:.4f}")
