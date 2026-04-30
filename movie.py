from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = Flask(__name__)
CORS(app)

TMDB_API_KEY = "50f9182da7839079f6340a94c326b250"  # 🔑 paste your key here
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

tfidf = TfidfVectorizer(stop_words="english")
matrix = tfidf.fit_transform(movies["genres"])
similarity = cosine_similarity(matrix)

user_map = {"sri": 1, "alex": 2, "john": 3}

# Cache posters so we don't hammer the API
poster_cache = {}

def get_poster(title):
    """Fetch poster URL from TMDB for a movie title."""
    if title in poster_cache:
        return poster_cache[title]
    try:
        # Strip year from title e.g. "Toy Story (1995)" -> "Toy Story"
        clean_title = title.split(" (")[0]
        url = f"https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": clean_title, "page": 1}
        res = requests.get(url, params=params, timeout=3).json()
        results = res.get("results", [])
        poster = None
        if results and results[0].get("poster_path"):
            poster = TMDB_POSTER_BASE + results[0]["poster_path"]
        poster_cache[title] = poster
        return poster
    except Exception:
        return None

def enrich_with_posters(titles):
    """Return list of {title, poster} dicts."""
    return [{"title": t, "poster": get_poster(t)} for t in titles]

def recommend_logic(user_id):
    user_data = ratings[ratings["userId"] == user_id]
    watched_ids = user_data[user_data["rating"] >= 4]["movieId"]
    watched_movies = movies[movies["movieId"].isin(watched_ids)]
    results = []
    for _, row in watched_movies.head(2).iterrows():
        idx = movies[movies["movieId"] == row["movieId"]].index[0]
        scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)[1:5]
        recs = [movies.iloc[i[0]]["title"] for i in scores]
        results.append({
            "because": row["title"],
            "because_poster": get_poster(row["title"]),
            "recommendations": enrich_with_posters(recs)
        })
    return results

@app.route("/recommend/<username>")
def recommend(username):
    user_id = user_map.get(username, 1)
    top_picks_titles = movies.sample(6)["title"].tolist()
    return jsonify({
        "top_picks": enrich_with_posters(top_picks_titles),
        "because": recommend_logic(user_id)
    })

if __name__ == "__main__":
    app.run(port=5001, debug=True)