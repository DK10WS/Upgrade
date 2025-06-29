import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


class HybridRecommender:
    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: str = ".",
        n_components: int = 50,
        n_collab_samples: int = 100,
        use_collaborative: bool = True,
    ):
        self.df = df.copy()
        self.cache_dir = cache_dir
        self.n_components = n_components
        self.n_collab_samples = n_collab_samples
        self.use_collaborative = use_collaborative

        self.df["genres"] = self.df["genres"].fillna("").str.split(",")
        self.df["tags"] = self.df["tags"].fillna("").str.split(",")

        self._load_or_build("mlb_genres.pkl", self._build_mlb, "genres")
        self._load_or_build("mlb_tags.pkl", self._build_mlb, "tags")
        self._load_or_build(
            "summary_embeddings.npy", self._build_embeddings, None, is_numpy=True
        )
        self._load_or_build("content_mat.pkl", self._build_content, None)

        if self.use_collaborative:
            self._load_or_build("svd.pkl", self._build_svd, None)
            self._load_or_build(
                "user_factors.npy", self._build_factors, "users", is_numpy=True
            )
            self._load_or_build(
                "movie_factors.npy", self._build_factors, "movies", is_numpy=True
            )

    def _load_or_build(self, fname, build_fn, arg, is_numpy=False):
        path = os.path.join(self.cache_dir, fname)
        name = fname.split(".")[0]

        if os.path.exists(path):
            if is_numpy:
                setattr(self, name, np.load(path, allow_pickle=True))
            else:
                setattr(self, name, pickle.load(open(path, "rb")))
        else:
            obj = build_fn(arg)
            if is_numpy:
                np.save(path, obj)
            else:
                with open(path, "wb") as f:
                    pickle.dump(obj, f)
            setattr(self, name, obj)

    def _build_mlb(self, column):
        mlb = MultiLabelBinarizer(sparse_output=True)
        mat = mlb.fit_transform(self.df[column])
        setattr(self, f"{column}_mat", mat)
        return mlb

    def _build_embeddings(self, _):
        print("🔍 Generating sentence embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        summaries = self.df["summary"].fillna("").tolist()
        embeddings = model.encode(summaries, show_progress_bar=True)
        return np.array(embeddings)

    def _build_content(self, _):
        return hstack([self.genres_mat, self.tags_mat]).tocsr()

    def _build_svd(self, _):
        user_movie = np.random.rand(self.n_collab_samples, len(self.df))
        svd = TruncatedSVD(self.n_components)
        svd.fit(user_movie)
        return svd

    def _build_factors(self, which):
        user_movie = np.random.rand(self.n_collab_samples, len(self.df))
        if which == "users":
            return self.svd.transform(user_movie)
        else:
            return self.svd.components_.T

    def get_hybrid_recommendations(
        self,
        user_id: int,
        movie_id: int,
        content_weight: float = 0.5,
        collaborative_weight: float = 0.5,
        top_k: int = 30,
        return_scores: bool = False,
    ):
        if not self.use_collaborative:
            collaborative_weight = 0.0
            content_weight = 1.0

        if collaborative_weight > 0:
            collab_scores = self.movie_factors.dot(self.user_factors[user_id])
        else:
            collab_scores = np.zeros(len(self.df))

        movie_vec = self.summary_embeddings[movie_id].reshape(1, -1)
        content_scores = cosine_similarity(
            movie_vec, self.summary_embeddings).ravel()

        hybrid = content_weight * content_scores + collaborative_weight * collab_scores

        idx = np.argpartition(-hybrid, top_k)[:top_k]
        best_idx = idx[np.argsort(-hybrid[idx])]

        if return_scores:
            return list(zip(self.df["title"].iloc[best_idx], hybrid[best_idx]))
        return self.df["title"].iloc[best_idx].tolist()


if __name__ == "__main__":
    df = pd.read_csv("Final_data.csv")
    rec = HybridRecommender(df, cache_dir=".", use_collaborative=True)

    for title, score in rec.get_hybrid_recommendations(
        user_id=0, movie_id=0, return_scores=True
    ):
        print(f"{title:60}  Score: {score:.4f}")
