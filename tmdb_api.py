"""
tmdb_api.py
===========
Fetches movie posters from The Movie Database (TMDB) API.

Usage
-----
Set your API key once in the Streamlit secrets file (.streamlit/secrets.toml):

    [tmdb]
    api_key = "YOUR_KEY_HERE"

Or pass it directly to fetch_poster().

The function returns a full image URL or None on any failure.
"""

from __future__ import annotations
import requests

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w780"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/200x300?text=No+Poster"


def fetch_poster(movie_title: str, release_year: str = "", api_key: str | None = None) -> str:
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("tmdb", {}).get("api_key", "")
        except Exception:
            api_key = ""

    if not api_key:
        return PLACEHOLDER_IMAGE

    # ✅ Clean title
    movie_title = movie_title.split("(")[0].split(":")[0].strip()

    try:
        response = requests.get(
            TMDB_SEARCH_URL,
            params={
                "api_key": api_key,
                "query": movie_title,
                "include_adult": False,
            },
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        if not results:
            return PLACEHOLDER_IMAGE

        # ✅ STEP 1: Match exact year
        for movie in results:
            if release_year and movie.get("release_date", "").startswith(str(release_year)):
                if movie.get("poster_path"):
                    return TMDB_IMAGE_BASE + movie["poster_path"]

        # ✅ STEP 2: Sort by popularity (best match)
        results = sorted(results, key=lambda x: x.get("popularity", 0), reverse=True)

        # ✅ STEP 3: First valid poster
        for movie in results:
            if movie.get("poster_path"):
                return TMDB_IMAGE_BASE + movie["poster_path"]

    except Exception:
        pass

    return PLACEHOLDER_IMAGE
