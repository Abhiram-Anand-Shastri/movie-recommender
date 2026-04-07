# 🎬 Movie Recommendation System

A content-based movie recommendation app built with Python, Scikit-learn, and Streamlit.

## Features
- Recommend similar movies by title
- Multi-field search (genre, actor, director, keyword)
- Free-text semantic search
- Live movie posters via the TMDB API
- Clean, responsive dark-themed UI

## Project Structure
```
movie_recommender/
├── app.py                  # Streamlit UI
├── model.py                # TF-IDF + cosine similarity + recommendation logic
├── data_preprocessing.py   # Data loading, cleaning, feature engineering
├── tmdb_api.py             # TMDB poster fetcher
├── requirements.txt
├── movie_dataset.csv       # ← place your dataset here
└── .streamlit/
    └── secrets.toml        # ← add your TMDB API key here
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your TMDB API key (optional, for posters)
Create `.streamlit/secrets.toml`:
```toml
[tmdb]
api_key = "YOUR_TMDB_API_KEY"
```
Get a free key at https://www.themoviedb.org/settings/api

### 3. Place the dataset
Copy `movie_dataset.csv` into this folder (or set `MOVIE_CSV_PATH` env var).

### 4. Run the app
```bash
streamlit run app.py
```

## How It Works

| Step | Detail |
|------|--------|
| **Preprocessing** | Drops irrelevant columns, fills missing overviews with keywords+genres, combines text fields into a single `tags` column |
| **Vectorization** | TF-IDF with bi-grams, English stopword removal, max 5 000 features |
| **Similarity** | Cosine similarity matrix precomputed at startup |
| **Recommendation** | Given a title, returns the top-5 most similar rows; given a free-text query, transforms it with the same TF-IDF vocab and ranks all movies |

## Environment Variable
| Variable | Default | Description |
|----------|---------|-------------|
| `MOVIE_CSV_PATH` | `movie_dataset.csv` | Path to the CSV dataset |
