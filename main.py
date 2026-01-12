from src.data_loader import load_movies
from src.preprocessing import preprocess
from src.recommender import MovieRecommender

def main():
    movies = load_movies("data/movies.csv")
    movies = preprocess(movies)

    recommender = MovieRecommender()
    recommender.fit(movies)

    movie_name = "Inception"
    results = recommender.recommend(movie_name)

    print(f"Movies similar to {movie_name}:")
    for movie in results:
        print("-", movie)

if __name__ == "__main__":
    main()
