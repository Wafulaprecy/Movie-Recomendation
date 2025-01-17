from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load('movie_recommender_model.joblib')

# Load movie data (this should be your 'movies.csv' file)
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv("ratings.csv")

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('userId'))
    user_ratings = ratings[ratings['userId'] == user_id]
    recommendations = []

    for movie_id in movies['movieId']:
        if movie_id not in user_ratings['movieId'].values:
            est_rating = model.predict(user_id, movie_id).est
            recommendations.append((movie_id, est_rating))

    # Sort by predicted rating
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]
    recommended_movies = [movies[movies['movieId'] == movie_id]['title'].values[0] for movie_id, _ in recommendations]
    return jsonify(recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
