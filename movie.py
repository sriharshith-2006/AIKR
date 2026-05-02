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
    """Fetch all posters in parallel using a thread pool for speed."""
    results = {t: None for t in titles}
    uncached = [t for t in titles if t not in poster_cache]
    
    # Cached ones are free — grab them immediately
    for t in titles:
        if t in poster_cache:
            results[t] = poster_cache[t]
    
    # Fetch uncached ones in parallel (up to 10 workers)
    if uncached:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_title = {executor.submit(get_poster, t): t for t in uncached}
            for future in as_completed(future_to_title):
                title = future_to_title[future]
                try:
                    results[title] = future.result()
                except Exception:
                    results[title] = None
    
    return [{"title": t, "poster": results[t]} for t in titles]

def get_top_movies(user_id, n=6):
    avg = df.groupby('title')['rating'].mean()
    count = df.groupby('title')['rating'].count()

    top = pd.DataFrame({'avg': avg, 'count': count})
    top = top[top['count'] > 50]
    top = top.sort_values(by='avg', ascending=False)

    user_movies = user_movie_matrix.loc[user_id]
    watched = user_movies[user_movies > 0].index

    top = top[~top.index.isin(watched)]

    return top.head(n).index.tolist()

def recommend_by_genre(user_id, movie_name, n=5):
    temp = df.copy()

    genre = temp[temp['title'] == movie_name]['genres'].values
    if len(genre) == 0:
        return []

    genre_set = set(genre[0].split('|'))

    temp['score'] = temp['genres'].apply(
        lambda x: len(genre_set.intersection(set(str(x).split('|'))))
    )

    recs = temp[(temp['score'] >= 2) & (temp['title'] != movie_name)]

    user_movies = user_movie_matrix.loc[user_id]
    watched = user_movies[user_movies > 0].index

    recs = recs[~recs['title'].isin(watched)]

    return recs.sort_values(by=['score', 'rating'], ascending=False)['title'].drop_duplicates().head(n).tolist()

def recommend_users(user_id, n=6):
    if user_id not in user_similarity_df.columns:
        return []

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]

    user_movies = user_movie_matrix.loc[user_id]
    watched = user_movies[user_movies > 0].index

    scores = {}

    for sim_user, sim_score in similar_users.items():
        sim_movies = user_movie_matrix.loc[sim_user]

        for movie, rating in sim_movies.items():
            if movie not in watched and rating > 0:
                scores[movie] = scores.get(movie, 0) + sim_score * rating

    sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [m[0] for m in sorted_movies[:n]]

def search_movies(user_id, name, n=12):
    result = df[df['title'].str.contains(name, case=False, na=False)]

    if user_id in user_movie_matrix.index:
        user_movies = user_movie_matrix.loc[user_id]
        watched = user_movies[user_movies > 0].index
        result = result[~result['title'].isin(watched)]

    return result['title'].drop_duplicates().head(n).tolist()

@app.route("/search/<int:user_id>")
def api_search(user_id):
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify(get_top_movies(user_id, 12))
    return jsonify(search_movies(user_id, q, 12))

@app.route("/recommend_movie/<int:user_id>")
def api_recommend_movie(user_id):
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"searched_movie": "", "recommendations": []})
    
    recs = recommend_by_genre(user_id, q, 6)
    return jsonify({
        "searched_movie": q,
        "recommendations": recs
    })

@app.route("/login", methods=["POST"])
def login_api():
    data = request.json
    user_id = data.get("user_id")
    
    try:
        user_id = int(user_id)
        if user_id in user_movie_matrix.index:
            return jsonify({"success": True, "message": "Login successful"})
        else:
            return jsonify({"success": False, "message": "User ID not found in dataset (Valid IDs: 1 to 610)"})
    except (ValueError, TypeError):
        return jsonify({"success": False, "message": "Invalid User ID format"})

@app.route("/recommend/<int:user_id>")
def recommend(user_id):
    # 1. Top Picks
    top_picks_titles = get_top_movies(user_id, 6)
    
    # 2. Because You Watched
    try:
        user_data = ratings[ratings["userId"] == user_id]
        watched_ids = user_data[user_data["rating"] >= 4]["movieId"]
        watched_movies = movies[movies["movieId"].isin(watched_ids)]
        
        because_results = []
        for _, row in watched_movies.head(2).iterrows():
            recs = list(recommend_by_genre(user_id, row["title"], 5))
            because_results.append({
                "because": row["title"],
                "recommendations": recs  # titles only — posters fetched by frontend
            })
    except Exception as e:
        print("Because error:", e)
        because_results = []
        
    # 3. Users Like You
    users_like_you_titles = recommend_users(user_id, 6)
    
    # Return titles ONLY — no TMDB calls, responds instantly
    return jsonify({
        "top_picks": enrich_with_posters(top_picks_titles),
        "because": recommend_logic(user_id)
    })

if __name__ == "__main__":
    app.run(port=5001, debug=True)