from nltk.stem import PorterStemmer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

# Create a stemmer
stemmer = PorterStemmer()

# Function to stem genres
def stem_genres(genre_string):
    # Check if the input is a string
    if isinstance(genre_string, str):
        # Split the genre string into individual genres
        genres = genre_string.split('|')
    elif isinstance(genre_string, list):
        # If it's already a list, use it directly
        genres = genre_string
    else:
        # Return an empty string if the input is not valid
        return ''

    # Stem each genre
    stemmed_genres = [stemmer.stem(genre.lower()) for genre in genres]  # Lowercase for consistency
    # Join back to a string with '|' separator
    return '|'.join(stemmed_genres)
    
    # Function to get recommendations based on a given genre
def get_recommendations_by_genre(input_genre):
    # Stem the input genre
    stemmed_input_genre = stem_genres(input_genre)
    print(stemmed_input_genre)

    # Create a TF-IDF vector for the input genre
    input_tfidf = tfidf.transform([stemmed_input_genre])
    dense_array = input_tfidf.toarray()

# Access the first row of the dense array
    if (dense_array[0][0]) == 0:
        return []
    # Compute cosine similarity of input genre with the existing movies
    sim_scores = cosine_similarity(input_tfidf, tfidf_matrix)
    #print(sim_scores)

    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(sim_scores[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 3 most similar movies
    movie_indices = [i[0] for i in sim_scores[1:4]]  # Exclude the first one (itself)
    #print(data['title'].iloc[movie_indices].tolist())

    # Return the top similar movies
    return data['title'].iloc[movie_indices].tolist()
