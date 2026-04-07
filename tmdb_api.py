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
import time
import requests

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
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

    movie_title = movie_title.split("(")[0].split(":")[0].strip()

    try:
        time.sleep(0.2)
        response = requests.get(
            TMDB_SEARCH_URL,
            params={
                "api_key": api_key,
                "query": movie_title,
                "include_adult": False,
            },
            timeout=10,   # 🔥 increase timeout
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            return PLACEHOLDER_IMAGE

        # 🔥 sort by popularity (better stability)
        results = sorted(results, key=lambda x: x.get("popularity", 0), reverse=True)

        for movie in results:
            if movie.get("poster_path"):
                return TMDB_IMAGE_BASE + movie["poster_path"]

    except Exception as e:
        print("TMDB ERROR:", e)   # 🔥 DEBUG

    return PLACEHOLDER_IMAGE