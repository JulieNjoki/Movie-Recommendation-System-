# Movie Recommendation System: Project Documentation

## 1. Overview

**Goal.** Recommend five movies to a user based on ratings they provide, using collaborative filtering on the
MovieLens `ml-latest-small` dataset.

**Result.** A working system with a Streamlit web app, a command-line tool and a Python function, all backed by the
SVD model chosen in Phase 3. The model beats naive benchmarks on rating prediction, but our own evaluation (Phase 4)
shows its top-5 lists are weaker than a "most popular" list and are concentrated on a small set of classic films.
Section 7 lists these limits and what to do about them.

## 2. Project phases and hand-offs

| Phase | Input | Output | Key decisions |
|---|---|---|---|
| 1 EDA  | raw CSVs | Findings on data shape and quality | Split per user, not by date (most users rated everything in one sitting); tags too sparse to be a main signal |
| 2 Cleaning  | raw CSVs | `data/processed/` (clean tables, train/test) | Flags instead of deletions; no rating threshold applied; 20% of each user's ratings held out (seed 42) |
| 3 Modelling  | `train.csv`, `movies_clean.csv` | Tuned SVD, `recommend_top5()` rules | SVD chosen over item-KNN as tie-break on size and speed; movies need at least 30 ratings to be recommended |
| 4 Evaluation  | `train.csv`, `test.csv` | Metrics and verdict | See section 5 |

## 3. Data

### 3.1 Raw data (`data/raw/`)
| File | Rows | Columns |
|---|---|---|
| `ratings.csv` | 100,836 | userId, movieId, rating, timestamp |
| `movies.csv` | 9,742 | movieId, title (with year), genres (pipe-separated) |
| `tags.csv` | 3,683 | userId, movieId, tag, timestamp |
| `links.csv` | 9,742 | movieId, imdbId, tmdbId |

### 3.2 What EDA found
* 610 users, 9,724 rated movies (18 catalogue movies were never rated); the user-movie matrix is **98.3% empty**.
* Ratings run 0.5 to 5.0; mean 3.50, mode 4.0, 48% are 4 stars or higher.
* Median user has about 70 ratings (range 20 to 2,698).
* **Long tail:** 66% of movies have 5 or fewer ratings, 35% have exactly one, and the top 20% of movies hold 77% of all ratings.
* Genres: Drama and Comedy dominate; Film-Noir, IMAX, Western and Documentary are rare.
* Tags: sparse and written mostly by five users, so treated as a supplement only.

### 3.3 What cleaning did
* No missing values, duplicate ratings, invalid ratings or orphan IDs in the ratings data.
* Extracted release year and a clean title; one-hot encoded 19 genres; kept 34 "no genre" movies flagged rather than creating a fake genre.
* Tags lower-cased and de-duplicated; the non-descriptive tag "in netflix queue" (131 rows) removed.
* Flags added instead of deleting rows: `has_ratings`, `no_genre`, `dup_title`, `has_tags`.
* Added contiguous `user_idx` and `movie_idx` columns.
* Split: 80,672 train / 20,164 test (20% of each user's ratings, seed 42). All 610 users are in both sets; 790 test ratings (3.9%) are for movies absent from training.

### 3.4 Processed files (`data/processed/`)
| File | Used by |
|---|---|
| `ratings_clean.csv` | **The live system** (all 100,836 ratings) |
| `movies_clean.csv` | **The live system** (titles, year, genres) |
| `train.csv`, `test.csv` | Phases 3 and 4 (model selection and honest evaluation); not used by the interface |
| `tags_clean.csv`, `links_clean.csv` | Not used by the current model (reserved for a content-based fallback or poster look-ups) |

## 4. Model (Phase 3) and how it is used here

* **Library:** `scikit-surprise`.
* **Eight models compared** by 5-fold cross-validation on the training set (random, BaselineOnly, user-based and
  item-based KNN with two similarity measures each, NMF, SVD).
* **Two finalists tuned:** item-based `KNNBaseline` (k = 60) and SVD.
* **Final model: SVD**, `n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.1`. Cross-validated RMSE 0.860 and MAE 0.661
  (BaselineOnly: 0.875). It was effectively tied with item-KNN on cross-validation and chosen for size and speed.
* **Top-5 rules:** candidates are movies with at least 30 ratings that the user has not rated; rank by uncapped predicted
  rating; display capped at 5.0. The 30-rating threshold was chosen because it gave the most distinct movies across
  150 sampled users (lower values let one 10-rating movie reach 75 to 85% of users).

### 4.1 Differences between the notebooks and the live system
| | Phases 3 and 4 | This repository |
|---|---|---|
| Training data | `train.csv` only (so the test set stays unseen) | **All** cleaned ratings plus the new user's ratings (more data is better once evaluation is finished) |
| Who is recommended to | Existing users (610) | A **new** user, appended as `userId = max + 1` |
| Movie-count threshold | Counted on `train.csv` | Counted on all ratings (882 movies qualify, versus 704 on the training set) |
| Trained model file | `svd_model.pkl` saved from the notebook | Not needed: the model is refitted for each new user in about 1 to 2 seconds, which is how Phase 3 proposed handling new users |

The scoring in `src/recommender.py` equals `model.predict(user, movie, clip=False).est` (checked against Surprise's own
`predict`), computed in one vectorised step.

## 5. Evaluation results (Phase 4, held-out test set)

| Question | Result |
|---|---|
| Rating accuracy | SVD test RMSE **0.8745**, MAE 0.667. Global mean 1.050; BaselineOnly 0.884; item-KNN 0.862 |
| Did cross-validation hold up? | Item-KNN yes (0.862 to 0.862). SVD finished 0.0145 above its CV score; the supplied test split is harder than typical (five other splits average 0.858 for SVD) |
| SVD vs item-KNN | Tied on cross-validation, but item-KNN was better on the test set and on 5 of 5 alternative splits (average gap 0.0043 RMSE) |
| Liked vs not liked (rating 4 or more) | ROC-AUC 0.777 (BaselineOnly 0.775, item-KNN 0.788) |
| Top-5 quality | NDCG@5 0.100 for SVD, vs 0.017 random, 0.096 BaselineOnly, 0.106 item-KNN, **0.160 popularity** |
| Variety (project objective 2) | SVD: **134 distinct movies**, top movie shown to 65% of users, personalisation 0.76. Item-KNN: 262 movies, 28%, 0.91 |
| Cold start | Unseen movies RMSE 1.03; users with 25 or fewer ratings RMSE about 1.0 |
| Errors | Predictions compress toward the middle: true 0.5-star ratings are over-predicted by 2.4 stars, 5-star ratings under-predicted by 1.0 |

Caveat from Phase 3 notebook: the offline test only counts movies users happened to rate, which favours popularity
lists, so the popularity result is a warning rather than proof users would prefer it.


## 6. Limitations and recommended next steps


1. **Revisit the model choice.** Item-KNN is more accurate and far more varied. Its cost is a ~9,700 x 9,700
   similarity table (around 650 MB or more in Surprise), which is why Phase 3 avoided it. A practical route is to precompute
   top-5 lists, or build a lighter item-based model restricted to the 882 recommendable movies. This has not been built or tested.
2. **Fix the ranking step.** SVD lists are dominated by each movie's overall-quality bias. In a quick check for this
   integration, lowering the bias weight with only 5 ratings did **not** produce more personal lists (it surfaced unrelated
   comedies), so this needs a proper experiment judged with NDCG@5 and the variety metrics together.
3. **Cold-start fallback.** For users with few ratings, blend in popularity or genre-based suggestions (genre one-hots and
   `tag_text` already exist in `movies_clean.csv`).
4. **Observed behaviour to expect.** A test user who rated five science-fiction films received crime and drama classics
   (*Shawshank Redemption*, *Rear Window*, *Usual Suspects*); with 23 ratings of sci-fi and action the list was
   still classics (*Monty Python and the Holy Grail*, *Dr. Strangelove*, *Pulp Fiction*). This matches the Phase 4 findings.
5. **A small user study** comparing SVD, item-KNN and popularity lists would settle what offline data cannot.
6. **Possible extras:** movie posters via `links_clean.csv` (TMDB IDs; needs an API key), saving user profiles, a "why this
   movie" explanation.


The analysis notebooks were written before the folders were reorganised. Their file paths need these one-line edits to
run from this layout (they run unchanged when the CSVs sit next to them or in a `processed/` folder beside them):

| Notebook | Change |
|---|---|
| `01_data_exploration` | `DATA_DIR = "../data/raw"` |
| `02_data_cleaning` | read the four CSVs from `../data/raw/`; it writes to `processed/` and ends with a Google Colab download cell (`from google.colab import files`) that only works in Colab |
| `03_model_development` | `PROCESSED_DIR = '../data/processed/'` |
| `04_model_evaluation` | `PROCESSED_DIR = '../data/processed/'` |

Notebooks 03 and 04 also need `scikit-surprise`; notebook 03 writes `phase3_outputs/` and 04 writes `eval_summary.json`.

## 9. Team and credits

Alex Gathuru, Sheilla Cherotich, Darius Mutahi, James Musyoki, Kevin Basiye and Julie Njoroge

Data: MovieLens `ml-latest-small`, GroupLens Research, University of Minnesota.
