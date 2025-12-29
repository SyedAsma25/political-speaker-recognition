from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    return TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5
    )
