"""
Movie recommender: the integration layer.

Ties the project stages together:
    Phase 2 (cleaning)  -> data/processed/ratings_clean.csv, movies_clean.csv
    Phase 3 (modelling) -> tuned SVD settings, "never recommend what the user has
                           rated", "only recommend movies with >= 30 ratings"
    Phase 4 (evaluation)-> known limits, documented in docs/PROJECT_DOCUMENTATION.md

Typical use
-----------
    rec = MovieRecommender()
    rec.search_movies("matrix")                       # find a movieId
    rec.recommend({1: 5.0, 296: 4.5, 318: 5.0, 356: 4.0, 593: 4.5})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

# --------------------------------------------------------------------------- #
# Settings copied from the Phase 3 notebook (nothing is re-tuned here).
# --------------------------------------------------------------------------- #
SVD_PARAMS = dict(n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.1, random_state=42)
MIN_MOVIE_RATINGS = 30      # Phase 3, section 6.1: movie must have >= 30 ratings to be recommendable
RATING_SCALE = (0.5, 5.0)   # MovieLens half-star scale
MIN_USER_RATINGS = 5        # fewer than this and the model has nothing to personalise with
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


class InvalidRatingError(ValueError):
    """Raised when user input is not a valid movieId / rating."""


@dataclass
class MovieRecommender:
    data_dir: Path = DEFAULT_DATA_DIR
    min_movie_ratings: int = MIN_MOVIE_RATINGS
    svd_params: dict = field(default_factory=lambda: dict(SVD_PARAMS))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self._load()

    # ------------------------------------------------------------------ #
    # Loading (Phase 2 outputs)
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        self.ratings = pd.read_csv(
            self.data_dir / "ratings_clean.csv", usecols=["userId", "movieId", "rating"]
        )
        movies = pd.read_csv(
            self.data_dir / "movies_clean.csv",
            usecols=["movieId", "title", "title_clean", "year", "genre_list"],
        )
        movies["genre_list"] = movies["genre_list"].fillna("").str.replace("|", ", ", regex=False)
        self.movies = movies.set_index("movieId")

        counts = self.ratings.groupby("movieId").size()
        self.movie_counts = counts
        self.eligible_ids = counts[counts >= self.min_movie_ratings].index.to_numpy()
        self.next_user_id = int(self.ratings["userId"].max()) + 1

    # ------------------------------------------------------------------ #
    # Helpers for the interface (finding movies to rate)
    # ------------------------------------------------------------------ #
    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Case-insensitive title search, most-rated matches first."""
        query = (query or "").strip()
        if not query:
            return self._display(pd.Index([]))
        mask = self.movies["title"].str.contains(query, case=False, regex=False, na=False)
        ids = self.movies.index[mask]
        ids = self.movie_counts.reindex(ids).fillna(0).sort_values(ascending=False).index[:limit]
        return self._display(ids)

    def popular_movies(self, n: int = 20) -> pd.DataFrame:
        """Most-rated movies: easy things for a new user to rate first."""
        return self._display(self.movie_counts.sort_values(ascending=False).index[:n])

    def _display(self, ids) -> pd.DataFrame:
        out = self.movies.loc[ids, ["title", "genre_list"]].rename(columns={"genre_list": "genres"})
        out["n_ratings"] = self.movie_counts.reindex(ids).fillna(0).astype(int).values
        return out.reset_index()

    # ------------------------------------------------------------------ #
    # Input validation (the "receive user ratings" step)
    # ------------------------------------------------------------------ #
    def validate_ratings(self, user_ratings: Mapping[int, float]) -> Dict[int, float]:
        """Return a clean {movieId: rating} dict or raise InvalidRatingError."""
        clean: Dict[int, float] = {}
        for movie_id, rating in dict(user_ratings).items():
            try:
                movie_id = int(movie_id)
                rating = float(rating)
            except (TypeError, ValueError):
                raise InvalidRatingError(f"Could not read movieId={movie_id!r}, rating={rating!r}")
            if movie_id not in self.movies.index:
                raise InvalidRatingError(f"movieId {movie_id} is not in the catalogue")
            if not (RATING_SCALE[0] <= rating <= RATING_SCALE[1]) or (rating * 2) % 1 != 0:
                raise InvalidRatingError(
                    f"Rating {rating} for movieId {movie_id} must be 0.5 to 5.0 in half-star steps"
                )
            clean[movie_id] = rating
        if len(clean) < MIN_USER_RATINGS:
            raise InvalidRatingError(
                f"Please rate at least {MIN_USER_RATINGS} movies (got {len(clean)}); "
                "with fewer the model cannot learn your taste."
            )
        return clean

    # ------------------------------------------------------------------ #
    # Model + recommendations (Phase 3 model, connected to user input)
    # ------------------------------------------------------------------ #
    def recommend(self, user_ratings: Mapping[int, float], n: int = 5) -> pd.DataFrame:
        """
        Add the new user's ratings to the full cleaned ratings, refit the tuned SVD
        (a few seconds), and return the top-n unseen movies by predicted rating.

        Returns a DataFrame indexed by rank with: movieId, title, year, genres,
        predicted_rating (capped at 5.0 for display; ranking uses the uncapped score,
        as in Phase 3).
        """
        clean = self.validate_ratings(user_ratings)
        uid = self.next_user_id

        new_rows = pd.DataFrame(
            {"userId": uid, "movieId": list(clean), "rating": list(clean.values())}
        )
        data = pd.concat([self.ratings, new_rows], ignore_index=True)
        trainset = Dataset.load_from_df(data, Reader(rating_scale=RATING_SCALE)).build_full_trainset()
        model = SVD(**self.svd_params).fit(trainset)

        # Candidates: popular-enough movies the user has not rated.
        candidates = np.array([m for m in self.eligible_ids if m not in clean])
        inner_u = trainset.to_inner_uid(uid)
        inner_i = np.array([trainset.to_inner_iid(m) for m in candidates])
        # Same value as model.predict(uid, m, clip=False).est, computed in one go.
        scores = (
            trainset.global_mean
            + model.bu[inner_u]
            + model.bi[inner_i]
            + model.qi[inner_i] @ model.pu[inner_u]
        )

        top = np.argsort(-scores)[:n]
        ids = candidates[top]
        out = pd.DataFrame(
            {
                "rank": range(1, len(ids) + 1),
                "movieId": ids,
                "title": self.movies.loc[ids, "title"].values,
                "year": self.movies.loc[ids, "year"].values,
                "genres": self.movies.loc[ids, "genre_list"].values,
                "predicted_rating": np.round(np.minimum(scores[top], RATING_SCALE[1]), 2),
            }
        )
        return out.set_index("rank")
