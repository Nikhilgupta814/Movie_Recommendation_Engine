import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KDTree
import ast
import string
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings
warnings.filterwarnings("ignore")

def convert(obj):
    A =[]
    for i in ast.literal_eval(obj):
        A.append(i['name'])
    return A

def Convert2(data):
    
    B=[]
    counter=0
    for i in ast.literal_eval(data):
        if counter !=3:
            B.append(i['name'])
            counter += 1
        else:
            break
    return B

def Convert3(data):
    C=[]
    for i in ast.literal_eval(data):
        if i['job']=='Director':
            C.append(i['name'])
    return C    

Lem = WordNetLemmatizer()
stopwords = stopwords.words('english') + list(string.punctuation) + ["'s", '--']
CV = CountVectorizer( max_features=5000, stop_words='english')

data1 = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\NLP\tmdb_5000_credits.csv')
data2 = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\NLP\tmdb_5000_movies.csv')
data = data1.merge(data2, on='title')

movies = data[['movie_id','title','overview','genres','keywords','cast','crew']]

# Checking null values
# movies.isnull().sum()

movies.dropna(inplace=True)

# movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(Convert2)
movies['crew'] = movies['crew'].apply(Convert3)

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['overview'] = movies['overview'].apply(lambda x: [x.replace(" ","") for x in x])
movies['genres'] = movies['genres'].apply(lambda x: [x.replace(" ","") for x in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
new_df['tags'] = new_df['tags'].apply(lambda x: re.sub('[^a-zA-Z]',' ', x))
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(Lem.lemmatize(x) for x in nltk.word_tokenize(x) if x not in stopwords))

vectors = CV.fit_transform(new_df.tags).toarray()
similarity = cosine_similarity(vectors)

def Recommendation(data):
    movie_index = new_df[new_df['title']==data].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# Saving the model
joblib.dump(similarity,'similarity.pkl')
joblib.dump(new_df.to_dict(), 'movie_dict.pkl')