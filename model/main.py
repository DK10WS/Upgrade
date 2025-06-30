import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = APIRouter()


class MovieRecommender:
    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: str = ".cache",
        model_name="all-mpnet-base-v2",
    ):
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

        self.df["release_year"] = (
            pd.to_datetime(self.df["release_date"], errors="coerce")
            .dt.year.fillna(0)
            .astype(int)
        )
        self.df["summary"] = self.df["summary"].fillna("")

        self.df["text_for_embedding"] = (
            self.df["title"].fillna("")
            + ". "
            + self.df["summary"]
            + ". "
            + self.df["genres"].apply(lambda x: ", ".join(x))
            + ". "
            + self.df["sub_genres"].apply(lambda x: ", ".join(x))
            + ". "
            + self.df["tags"].apply(lambda x: ", ".join(x))
            + ". "
            + self.df["release_year"].astype(str)
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
                normalize_embeddings=True,
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


df = pd.read_csv(os.path.join(os.path.dirname(__file__), "movies.csv"))
recommender = MovieRecommender(df)


@app.get("/recommendation")
def get_recommendations(title: str = Query(...), top_k: int = 10):
    try:
        results = recommender.recommend(title, top_k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return [r[0] for r in results]
