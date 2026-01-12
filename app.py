import streamlit as st
from src.data_loader import load_movies
from src.preprocessing import preprocess
from src.recommender import MovieRecommender

# Page title
st.title("🎬 Movie Recommendation System")

st.write("Type a movie name and get similar movie recommendations.")

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    movies = load_movies("data/movies.csv")
    movies = preprocess(movies)

    recommender = MovieRecommender()
    recommender.fit(movies)

    return movies, recommender

movies, recommender = load_and_prepare_data()

# Movie selection dropdown
movie_name = st.selectbox(
    "Select a movie:",
    movies['title'].values
)

# Button
if st.button("Recommend Movies"):
    recommendations = recommender.recommend(movie_name)

    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write("🎥", movie)
