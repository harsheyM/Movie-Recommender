import pandas as pd

def load_movies(file_path):
    #Read the CSV file
    df = pd.read_csv(file_path)

    #Keep only useful columns
    return df[['title', 'overview', 'genres']]
