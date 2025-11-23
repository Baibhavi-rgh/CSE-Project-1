import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")
movies['overview']  = movies['overview'].fillna('')

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

# Cosine similarity scores
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Movie index mapping
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["Movie not found in database."]

        idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:11]  # Top 10
    movie_indices = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices]

     #Test
print(get_recommendations("The Dark Knight"))

# Test
print(get_recommendations("The Dark Knight"))
