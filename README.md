# 🎬 Movie Recommendation System

A collaborative-filtering movie recommender built on the MovieLens `ml-latest-small` dataset
(100,836 ratings, 610 users, 9,742 movies). A user rates a handful of movies they have seen and the
system returns **five movies they have not rated, ranked by predicted rating**.

The project was built in five phases, one per team member and branch. This repository joins them into one
working system.

| Phase | Notebook | Output |
|---|---|---|
| 1. Data exploration (EDA)  | `notebooks/01_data_exploration.ipynb` | Understanding of sparsity, long tail, cold start |
| 2. Data cleaning  | `notebooks/02_data_cleaning.ipynb` | `data/processed/*.csv` (clean tables, train/test split) |
| 3. Model development  | `notebooks/03_model_development.ipynb` | Tuned SVD, top-5 rules |
| 4. Model evaluation  | `notebooks/04_model_evaluation.ipynb` | Accuracy, ranking and variety results |



**From Python**

```python
from src.recommender import MovieRecommender

rec = MovieRecommender()
rec.search_movies("matrix")                     # find movieIds by title
rec.popular_movies(10)                          # easy movies to rate first

my_ratings = {2571: 5.0, 260: 4.5, 318: 5.0, 296: 4.0, 1: 4.5}   # {movieId: rating}
rec.recommend(my_ratings, n=5)
```
## How it works


1. **Input.** Ratings are `{movieId: rating}`, 0.5 to 5.0 in half-star steps, at least 5 movies.
   Bad input (unknown movie, out-of-range rating, too few ratings) raises `InvalidRatingError` with a readable message.
2. **Model.** The user's ratings are added to the full cleaned ratings and the SVD chosen in Phase 3 is refitted
   (about 1 to 2 seconds). Settings are copied unchanged from Phase 3.
3. **Ranking.** Only movies with at least 30 ratings are candidates (Phase 3 found that lower thresholds let a
   handful of movies with 10 ratings dominate every list). Movies the user rated are excluded.

## Repository layout

```
├── data/raw/                   Original MovieLens CSVs (EDA and cleaning input)
├── data/processed/             Cleaned CSVs and train/test split (cleaning output)
├── notebooks/                  The four phase notebooks
├── docs/PROJECT_DOCUMENTATION.md   Full write-up: method, results, limitations
└── requirements.txt
```

## Known limitations 

* **Accuracy is modest.** Test RMSE 0.8745 stars, only about 1% better than a model that knows nothing except
  user and movie averages (0.884). It is about 17% better than guessing the global mean.
* **Top-5 lists lean on well-loved classics.** SVD's lists are about 6x better than random but not better than a
  plain "most popular" list (NDCG@5 0.100 vs 0.160). Across all 610 users it surfaced only 134 different movies, and
  *Singin' in the Rain* reached about 65% of users. The project's variety objective is therefore **not met** by this model.
* **Few ratings means weak personalisation.** Users with 25 or fewer ratings have RMSE of about 1.0. In testing,
  a user who rated five science-fiction films was still shown mostly crime and drama classics. Rate 15 or more movies
  for better results.
* **Only 882 movies are recommendable** (those with at least 30 ratings), so obscure films are never suggested.
* **Item-based KNN did slightly better than SVD** on the held-out test set (RMSE 0.862 vs 0.8745) and gave far more
  variety (262 vs 134 distinct movies). Phase 3 chose SVD for size and speed; see the documentation for the trade-off.

## Tests

```bash
python -m pytest -q
```

Covers the consistency of the stage outputs (train + test = cleaned ratings, matching IDs), input validation, and the
recommendation rules (five unique movies, no
ne already rated, all with at least 30 ratings, deterministic, different
tastes give different lists).

## Data

[MovieLens `ml-latest-small`](https://grouplens.org/datasets/movielens/latest/), GroupLens Research. Used for
educational purposes; see the GroupLens page for terms of use.
