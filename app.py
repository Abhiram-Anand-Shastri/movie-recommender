"""
app.py
======
Streamlit front-end for the Movie Recommendation System.

Run with:
    streamlit run app.py

Make sure movie_dataset.csv is in the same directory, OR set the
environment variable MOVIE_CSV_PATH to its absolute path.
"""

import os
import streamlit as st
from model import MovieRecommender
from tmdb_api import fetch_poster, PLACEHOLDER_IMAGE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = os.getenv("MOVIE_CSV_PATH", "movie_dataset.csv")
TOP_N = 5

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS  – card styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .movie-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
        display: flex;
        gap: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .movie-card img {
        border-radius: 8px;
        width: 110px;
        min-width: 110px;
        object-fit: cover;
    }
    .movie-info h4 {
        margin: 0 0 6px 0;
        color: #e0e0f0;
        font-size: 1rem;
    }
    .movie-info p {
        margin: 0;
        color: #a0a0c0;
        font-size: 0.83rem;
        line-height: 1.45;
    }
    .movie-meta {
        font-size: 0.75rem;
        color: #7878a0;
        margin-top: 6px;
    }
    .status-msg {
        color: #78dba0;
        font-style: italic;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model (cached so it only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model… this takes a few seconds on first run.")
def get_model() -> MovieRecommender:
    return MovieRecommender(CSV_PATH, max_features=5000, top_n=TOP_N)


model = get_model()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _year(release_date: str) -> str:
    try:
        return str(release_date)[:4]
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def get_poster_cached(title, year):
    return fetch_poster(title, year)

def render_movie_card(movie: dict, show_poster: bool = True) -> None:
    """Render a single movie card using HTML."""
    title = movie.get("title", "Unknown")
    overview = movie.get("overview", "")
    overview_short = overview[:160].rstrip() + ("…" if len(overview) > 160 else "")
    genres = movie.get("genres", "")
    director = movie.get("director", "")
    rating = movie.get("vote_average", "")
    year = _year(movie.get("release_date", ""))

    meta_parts = []
    if year:
        meta_parts.append(year)
    if genres:
        meta_parts.append(genres[:50])
    if director:
        meta_parts.append(f"Dir: {director}")
    if rating:
        meta_parts.append(f"⭐ {rating}")
    meta = "  ·  ".join(str(p) for p in meta_parts if p)

    poster_url = get_poster_cached(title, year) if show_poster else PLACEHOLDER_IMAGE
    img_html = f'<img src="{poster_url}" alt="poster"/>'

    st.markdown(
        f"""
        <div class="movie-card">
            {img_html}
            <div class="movie-info">
                <h4>{title}</h4>
                <p>{overview_short}</p>
                <p class="movie-meta">{meta}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(results: list[dict], message: str) -> None:
    if message:
        st.markdown(f'<p class="status-msg">{message}</p>', unsafe_allow_html=True)
    if not results:
        st.info("No movies to display.")
        return
    # Show in two columns for a nicer layout
    cols = st.columns(2)
    for i, movie in enumerate(results):
        with cols[i % 2]:
            render_movie_card(movie)


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

st.title("🎬 Movie Recommendation System")
st.caption("Content-based recommendations powered by TF-IDF & Cosine Similarity")
st.divider()

# ── Section 1: Recommend by title ─────────────────────────────────────────
st.header("🔍 Find Similar Movies")
st.write("Enter a movie you like and we'll find the top 5 most similar films.")

with st.form("recommend_form"):
    movie_input = st.text_input(
        "Movie title",
        placeholder="e.g. The Dark Knight",
    )
    recommend_btn = st.form_submit_button("Recommend", type="primary")

if recommend_btn:
    if not movie_input.strip():
        st.warning("Please enter a movie title.")
    else:
        with st.spinner("Finding similar movies…"):
            results, msg = model.recommend(movie_input)
        render_results(results, msg)

st.divider()

# ── Section 2: Multi-input search ─────────────────────────────────────────
st.header("🎛️ Search by Genre / Actor / Director / Keyword")
st.write("Fill in one or more fields and we'll surface the best matches.")

with st.form("multi_search_form"):
    col1, col2 = st.columns(2)
    with col1:
        genre_input = st.text_input("Genre", placeholder="e.g. action, romance, sci-fi")
        actor_input = st.text_input("Actor", placeholder="e.g. Leonardo DiCaprio")
    with col2:
        director_input = st.text_input("Director", placeholder="e.g. Christopher Nolan")
        keyword_input = st.text_input("Keyword", placeholder="e.g. time travel, heist")
    search_btn = st.form_submit_button("Search", type="primary")

if search_btn:
    if not any([genre_input, actor_input, director_input, keyword_input]):
        st.warning("Please fill in at least one field.")
    else:
        with st.spinner("Searching…"):
            results, msg = model.multi_input_search(
                genre=genre_input,
                actor=actor_input,
                director=director_input,
                keyword=keyword_input,
            )
        render_results(results, msg)

st.divider()

# ── Section 3: Free-text search ───────────────────────────────────────────
st.header("💬 Free-text Search")
st.write("Describe what you're in the mood for and we'll do our best!")

with st.form("free_search_form"):
    free_query = st.text_input(
        "Describe the movie",
        placeholder="e.g. romantic comedy set in Paris with a happy ending",
    )
    free_btn = st.form_submit_button("Search", type="primary")

if free_btn:
    if not free_query.strip():
        st.warning("Please enter a description.")
    else:
        with st.spinner("Searching…"):
            results, msg = model.search_movies(free_query)
        render_results(results, msg)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    "<br><hr><center style='color:#606080;font-size:0.78rem;'>"
    "Poster data from TMDB · Dataset: movie_dataset.csv"
    "</center>",
    unsafe_allow_html=True,
)
