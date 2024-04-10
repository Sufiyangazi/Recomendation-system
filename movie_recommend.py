# %%
import pandas as pd
# Set display options to show all columns
pd.set_option('display.max_columns', None)
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
# %%
movie = pd.read_csv('C:\\Users\\Admin\\Data Science\\NLP\\Movie Recommender\\Socurce Code\\Datasets\\tmdb_5000_movies.csv')
credits = pd.read_csv('C:\\Users\\Admin\\Data Science\\NLP\\Movie Recommender\\Socurce Code\\Datasets\\tmdb_5000_credits.csv')
movie.head()
# %%
movie.shape
# %%
credits.head()
# %%
movie = movie.merge(credits,on='title')
# %%
movie.head()
# %%
movie = movie[['movie_id','title','overview','genres','keywords','cast','crew']]
movie.head()
# %%
import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L 
# %%
text = "[{'name':'Iron man'},{'name':'Hulk'},{'name':'Avengers'}]"
# Call the convert function with the example input text
result = convert(text)
# Print the result
print(result)
# %%
movie.dropna(inplace=True)
# %%
movie.head()
# %%
movie['genres'] = movie['genres'].apply(convert)
movie['keywords'] = movie['keywords'].apply(convert)
# %%
def conv(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3 :
            L.append(i['name'])
        counter+=1
    return L
# %%
movie['cast'] = movie['cast'].apply(conv)
movie.head()
# %%
movie['cast'] = movie['cast'].apply(lambda x:x[0:3])
# %%
def director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movie['crew'] = movie['crew'].apply(director)
movie.head()
# %%
def destroy(x):
    if isinstance(x, list):
        return [i.replace(" ","") for i in x]
    elif isinstance(x, str):
        return x.replace(" ","")
    else:
        return x
# %%
movie['cast'] = movie['cast'].apply(destroy)
movie['crew'] = movie['crew'].apply(destroy)
movie['genres'] = movie['genres'].apply(destroy)
movie['keywords'] = movie['keywords'].apply(destroy)
movie.head()
# %%
movie['overview'] = movie['overview'].apply(lambda x:x.split())
# %%
movie['tags'] = movie['overview'] + movie['genres'] + movie['keywords'] + movie['cast'] + movie['crew']
# %%
df = movie[['movie_id','title','tags']]
df
# %%
df['tags']=df['tags'].apply(lambda x:" ".join(x))
# %%
df
# %%
df['tags']=df['tags'].apply(lambda x:x.lower())
df
# %%
# Removing Punctuations
import string

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    return text

df['tags'] = df['tags'].apply(remove_punctuation)
df
# %%
from nltk.stem import PorterStemmer
def stem_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Initialize the stemmer
    stemmer = PorterStemmer()
    
    # Stem each token
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Reconstruct the text from stemmed tokens
    stemmed_text = ' '.join(stemmed_tokens)
    
    return stemmed_text
df['tags'] = df['tags'].apply(stem_text)
df
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['tags'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# %%
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df.loc[df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]
# %%
movie_title = 'Batman'
recommendations = get_recommendations(movie_title)
print("movie recomendations for Batman",movie_title,";")
print(recommendations)
# %%
import pickle
pickle.dump(df,open('movie_list.pkl','wb'))
pickle.dump(cosine_sim,open('sim.pkl','wb'))
# %%
