"""
Streamlit app: rate movies, get your top 5.

    streamlit run app.py
"""

import streamlit as st

from src.recommender import MIN_USER_RATINGS, InvalidRatingError, MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Loading data...")
def get_recommender() -> MovieRecommender:
    return MovieRecommender()


rec = get_recommender()

if "ratings" not in st.session_state:
    st.session_state.ratings = {}  # {movieId: rating}

st.title("🎬 Movie Recommender")
st.write(
    "Rate movies you have seen and get five suggestions. "
    f"Rate at least **{MIN_USER_RATINGS}** movies; the more you rate (15+ is best), the better the suggestions."
)

# --------------------------------------------------------------------------- #
# 1. Add ratings
# --------------------------------------------------------------------------- #
st.header("1. Rate movies")
tab_search, tab_popular = st.tabs(["Search by title", "Popular movies"])

with tab_search:
    query = st.text_input("Movie title contains", placeholder="e.g. matrix", key="query")
    hits = rec.search_movies(query, limit=10)
    if query and hits.empty:
        st.info("No matches. Try a shorter title.")
    elif not hits.empty:
        labels = dict(zip(hits.movieId, hits.title))
        movie_id = st.selectbox("Pick a movie", hits.movieId, format_func=labels.get)
        stars = st.slider("Your rating", 0.5, 5.0, 4.0, 0.5, key="search_stars")
        if st.button("Add rating", key="add_search"):
            st.session_state.ratings[int(movie_id)] = stars
            st.rerun()

with tab_popular:
    st.caption("The most-rated movies in the dataset. Rate the ones you know.")
    unrated = rec.popular_movies(60)
    unrated = unrated[~unrated.movieId.isin(st.session_state.ratings)].head(8)
    for _, row in unrated.iterrows():
        c1, c2, c3 = st.columns([5, 3, 1])
        c1.write(f"**{row.title}**  \n{row.genres}")
        value = c2.select_slider(
            "rating", options=[0.5 + 0.5 * i for i in range(10)], value=4.0,
            key=f"pop_{row.movieId}", label_visibility="collapsed",
        )
        if c3.button("Add", key=f"add_pop_{row.movieId}"):
            st.session_state.ratings[int(row.movieId)] = value
            st.rerun()

# --------------------------------------------------------------------------- #
# 2. Review ratings
# --------------------------------------------------------------------------- #
st.header("2. Your ratings")
n = len(st.session_state.ratings)
if n == 0:
    st.write("Nothing yet.")
else:
    for mid, r in list(st.session_state.ratings.items()):
        c1, c2 = st.columns([8, 1])
        c1.write(f"{rec.movies.loc[mid, 'title']} — **{r}** ⭐")
        if c2.button("✕", key=f"del_{mid}"):
            del st.session_state.ratings[mid]
            st.rerun()
    if st.button("Clear all"):
        st.session_state.ratings = {}
        st.rerun()
st.progress(min(n / MIN_USER_RATINGS, 1.0), text=f"{n} rated (minimum {MIN_USER_RATINGS})")

# --------------------------------------------------------------------------- #
# 3. Recommendations
# --------------------------------------------------------------------------- #
st.header("3. Your top 5")
if st.button("Get recommendations", type="primary", disabled=n < MIN_USER_RATINGS):
    try:
        with st.spinner("Learning your taste..."):
            recs = rec.recommend(st.session_state.ratings, n=5)
    except InvalidRatingError as err:
        st.error(str(err))
    else:
        for rank, row in recs.iterrows():
            st.subheader(f"{rank}. {row.title}")
            st.write(f"{row.genres}  ·  predicted rating for you: **{row.predicted_rating:.2f} / 5**")
        st.caption(
            "Suggestions come from a collaborative-filtering model (SVD) trained on MovieLens ratings. "
            "It favours well-loved films; see the README for known limitations."
        )
