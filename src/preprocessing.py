import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)

def preprocess(df):
    df = df.dropna()
    df['overview'] = df['overview'].apply(clean_text)
    return df
