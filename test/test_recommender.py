"""Run with:  python -m pytest -q"""
import pandas as pd
import pytest

from src.recommender import (
    DEFAULT_DATA_DIR, MIN_MOVIE_RATINGS, MIN_USER_RATINGS, InvalidRatingError, MovieRecommender,
)

PROFILE = {2571: 5.0, 260: 4.5, 318: 5.0, 296: 4.0, 1: 4.5, 111: 3.5}


@pytest.fixture(scope="module")
def rec():
    return MovieRecommender()


# ---- Integration of stage outputs (Phase 2 files agree with each other) ----
def test_train_plus_test_equals_cleaned_ratings():
    d = DEFAULT_DATA_DIR
    train, test, full = (pd.read_csv(d / f) for f in ("train.csv", "test.csv", "ratings_clean.csv"))
    assert len(train) == 80_672 and len(test) == 20_164 and len(full) == 100_836
    key = ["userId", "movieId"]
    assert not set(map(tuple, train[key].values)) & set(map(tuple, test[key].values))
    assert len(train) + len(test) == len(full)


def test_all_rated_movies_exist_in_catalogue(rec):
    assert set(rec.ratings.movieId) <= set(rec.movies.index)


# ---- Input handling ----
@pytest.mark.parametrize("bad", [
    {1: 5.0},                                                  # too few ratings
    {1: 5, 2: 5, 3: 5, 4: 5, 999999: 3},                       # unknown movie
    {1: 5, 2: 5, 3: 5, 4: 5, 5: 6},                            # above scale
    {1: 5, 2: 5, 3: 5, 4: 5, 5: 0},                            # below scale
    {1: 5, 2: 5, 3: 5, 4: 5, 5: 3.3},                          # not a half-star step
    {1: 5, 2: 5, 3: 5, 4: 5, 5: "abc"},                        # not a number
])
def test_invalid_input_rejected(rec, bad):
    with pytest.raises(InvalidRatingError):
        rec.recommend(bad)


# ---- Recommendations ----
def test_returns_five_unique_unseen_eligible_movies(rec):
    out = rec.recommend(PROFILE)
    assert len(out) == 5 and out.movieId.is_unique
    assert not set(out.movieId) & set(PROFILE)                         # never recommends rated movies
    assert all(rec.movie_counts[m] >= MIN_MOVIE_RATINGS for m in out.movieId)
    assert out.predicted_rating.between(0.5, 5.0).all()
    assert out.predicted_rating.is_monotonic_decreasing


def test_deterministic(rec):
    assert rec.recommend(PROFILE).movieId.tolist() == rec.recommend(PROFILE).movieId.tolist()


def test_different_tastes_give_different_lists(rec):
    a = rec.recommend(PROFILE).movieId.tolist()
    b = rec.recommend({318: 1.0, 296: 1.0, 2571: 1.0, 260: 1.0, 1: 5.0, 3114: 5.0, 2355: 5.0}).movieId.tolist()
    assert a != b


def test_search_and_popular(rec):
    assert "Matrix, The (1999)" in rec.search_movies("matrix").title.tolist()
    assert rec.search_movies("").empty
    assert len(rec.popular_movies(10)) == 10
