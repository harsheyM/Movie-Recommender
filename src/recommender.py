from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.similarity_matrix = None
        self.titles = None

    def fit(self, df):
        tfidf = self.vectorizer.fit_transform(df['overview'])
        self.similarity_matrix = cosine_similarity(tfidf)
        self.titles = df['title'].values

    def recommend(self, movie_title, top_n=5):
        movie_index = list(self.titles).index(movie_title)
        scores = list(enumerate(self.similarity_matrix[movie_index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        recommended_movies = scores[1:top_n+1]
        return [self.titles[i] for i, _ in recommended_movies]
