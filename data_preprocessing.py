"""
data_preprocessing.py
=====================
Loads the raw CSV, cleans it, and engineers the 'tags' feature
that the TF-IDF / cosine-similarity model is trained on.
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# Columns we want to keep
# ---------------------------------------------------------------------------
KEEP_COLS = [
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "director",
    "tagline",
    "runtime",
    "release_date",
    "vote_average",
    "vote_count",
]

DROP_COLS = [
    "homepage",
    "production_companies",
    "production_countries",
    "index",
    "id",
    "budget",
    "revenue",
    "popularity",
    "spoken_languages",
    "status",
    "original_language",
    "original_title",
    "crew",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation clutter."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedup_words(text: str) -> str:
    """Remove duplicate words while preserving first-occurrence order."""
    seen = set()
    result = []
    for word in text.split():
        if word not in seen:
            seen.add(word)
            result.append(word)
    return " ".join(result)


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Load the movie dataset, clean it, and build the 'tags' column.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with a 'tags' column ready for TF-IDF.
    """
    df = pd.read_csv(csv_path)

    # -----------------------------------------------------------------------
    # 1. Drop irrelevant columns (silently ignore if missing)
    # -----------------------------------------------------------------------
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # -----------------------------------------------------------------------
    # 2. Keep only relevant columns
    # -----------------------------------------------------------------------
    present = [c for c in KEEP_COLS if c in df.columns]
    df = df[present].copy()

    # -----------------------------------------------------------------------
    # 3. Drop rows with no title
    # -----------------------------------------------------------------------
    df.dropna(subset=["title"], inplace=True)
    df.drop_duplicates(subset=["title"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # -----------------------------------------------------------------------
    # 4. Handle missing values
    # -----------------------------------------------------------------------
    # overview: use keywords + genres as fallback, never fabricate
    for col in ["genres", "keywords", "cast", "director", "overview", "tagline"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    mask_no_overview = df["overview"].str.strip() == ""
    df.loc[mask_no_overview, "overview"] = (
        df.loc[mask_no_overview, "keywords"] + " " + df.loc[mask_no_overview, "genres"]
    )

    # -----------------------------------------------------------------------
    # 5. Build 'tags' feature
    # -----------------------------------------------------------------------
    tag_cols = ["overview", "genres", "keywords", "cast", "director"]
    present_tag_cols = [c for c in tag_cols if c in df.columns]

    df["tags"] = df[present_tag_cols].apply(lambda row: " ".join(row.values), axis=1)

    # Clean every tag
    df["tags"] = df["tags"].apply(_clean_text)
    df["tags"] = df["tags"].apply(_dedup_words)

    # Also clean title for display / search matching
    df["title_lower"] = df["title"].str.lower().str.strip()

    return df
