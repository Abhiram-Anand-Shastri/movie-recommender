"""
model.py
========
Builds TF-IDF vectors over the 'tags' column and exposes three
recommendation functions:

  1. recommend(movie_title)         → top-N similar movies by title
  2. search_movies(query)           → free-text search
  3. multi_input_search(...)        → structured multi-field search
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_preprocessing import load_and_preprocess


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class MovieRecommender:
    """Encapsulates the TF-IDF model and all lookup logic."""

    def __init__(self, csv_path: str, max_features: int = 5000, top_n: int = 5):
        self.top_n = top_n

        # Load & preprocess
        self.df = load_and_preprocess(csv_path)

        # Fit TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2),
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["tags"])

        # Precompute full similarity matrix (dataset is small enough)
        # For very large datasets, compute on-demand per query instead.
        self.sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rows_to_records(self, indices: list[int]) -> list[dict]:
        """Convert a list of row indices to a list of movie-info dicts."""
        records = []
        for i in indices:
            row = self.df.iloc[i]
            records.append(
                {
                    "title": row.get("title", ""),
                    "overview": row.get("overview", ""),
                    "genres": row.get("genres", ""),
                    "cast": row.get("cast", ""),
                    "director": row.get("director", ""),
                    "vote_average": row.get("vote_average", ""),
                    "release_date": row.get("release_date", ""),
                }
            )
        return records

    def _query_vector_top_n(self, query: str) -> list[dict]:
        """Transform a free-text query and return top-N matches."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        # Return top_n highest scoring (no self-exclusion for query mode)
        top_indices = scores.argsort()[::-1][: self.top_n]
        return self._rows_to_records(top_indices.tolist())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_titles(self) -> list[str]:
        """Return all movie titles for autocomplete / validation."""
        return self.df["title"].tolist()

    def recommend(self, movie_title: str) -> tuple[list[dict], str]:
        """
        Return top-N movies similar to *movie_title*.

        Returns
        -------
        (results, message)
            results : list of movie dicts (empty on failure)
            message : user-facing status string
        """
        title_lower = movie_title.strip().lower()
        if not title_lower:
            return [], "Please enter a movie title."

        # Exact match
        matches = self.df[self.df["title_lower"] == title_lower]

        # Fuzzy fallback: substring match
        if matches.empty:
            matches = self.df[self.df["title_lower"].str.contains(title_lower, regex=False)]

        if matches.empty:
            return [], f"Movie '{movie_title}' not found in the dataset."

        idx = matches.index[0]
        sim_scores = list(enumerate(self.sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip the movie itself (score == 1.0 at position 0)
        top_indices = [i for i, _ in sim_scores[1 : self.top_n + 1]]
        results = self._rows_to_records(top_indices)

        if not results:
            return [], "No similar movies found."

        matched_title = self.df.iloc[idx]["title"]
        return results, f"Showing recommendations similar to '{matched_title}'"

    def search_movies(self, query: str) -> tuple[list[dict], str]:
        """
        Free-text search (genre, mood, keyword, etc.).

        Returns
        -------
        (results, message)
        """
        query = query.strip()
        if not query:
            return [], "Please enter a search query."

        results = self._query_vector_top_n(query)
        if not results:
            return [], f"No results found for '{query}'."
        return results, f"Top results for '{query}'"

    def multi_input_search(
        self,
        genre: str = "",
        actor: str = "",
        director: str = "",
        keyword: str = "",
    ) -> tuple[list[dict], str]:
        """
        Structured search across multiple optional fields.

        Returns
        -------
        (results, message)
        """
        parts = [
            genre.strip(),
            actor.strip(),
            director.strip(),
            keyword.strip(),
        ]
        combined = " ".join(p for p in parts if p)

        if not combined:
            return [], "Please provide at least one search field."

        results = self._query_vector_top_n(combined)
        if not results:
            return [], "No matching movies found."

        label_parts = []
        if genre:
            label_parts.append(f"genre={genre}")
        if actor:
            label_parts.append(f"actor={actor}")
        if director:
            label_parts.append(f"director={director}")
        if keyword:
            label_parts.append(f"keyword={keyword}")
        return results, "Results for: " + ", ".join(label_parts)
