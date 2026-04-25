from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load dataset
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# preprocess
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# ML MODEL
tfidf = TfidfVectorizer(stop_words="english")
matrix = tfidf.fit_transform(movies["genres"])
similarity = cosine_similarity(matrix)

user_map = {"sri": 1, "alex": 2, "john": 3}


@app.route("/")
def home():
    return "SERVER WORKING"


def recommend_logic(user_id):

    user_data = ratings[ratings["userId"] == user_id]
    watched_ids = user_data[user_data["rating"] >= 4]["movieId"]

    watched_movies = movies[movies["movieId"].isin(watched_ids)]

    results = []

    for _, row in watched_movies.head(2).iterrows():

        idx = movies[movies["movieId"] == row["movieId"]].index[0]

        scores = list(enumerate(similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:5]

        recs = [movies.iloc[i[0]]["title"] for i in scores]

        results.append({
            "because": row["title"],
            "recommendations": recs
        })

    return results


@app.route("/recommend/<username>")
def recommend(username):

    user_id = user_map.get(username, 1)

    return jsonify({
        "top_picks": movies.sample(6)["title"].tolist(),
        "because": recommend_logic(user_id)
    })


if __name__ == "__main__":
    app.run(debug=True)