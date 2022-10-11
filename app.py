import streamlit as st
import re
import pandas as pd
import joblib
import requests


def Fectch_poster_path(movie_id):

    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


def Recommendation(movie):
    movie_index = movies[movies['title']==movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x:x[1])[1:6]
    
    movie_poster = []
    recommended_movie = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie.append(movies.iloc[i[0]].title)
        movie_poster.append(Fectch_poster_path(movie_id))

    return recommended_movie, movie_poster

movie_dict = joblib.load('movie_dict.pkl')
movies  = pd.DataFrame(movie_dict)

similarity = joblib.load('similarity.pkl')

def main():
    st.title('Movie Recommender System')
    st.subheader('Created by Nikhil Gupta')
    st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
                background-attachment: fixed;
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    selected_movie = st.selectbox(
        'Select Any Movies to Find Similar movies',
        movies['title'].values
        )

    if st.button('Recommend'):
        name,poster = Recommendation(selected_movie)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(name[0])
            st.image(poster[0])
        with col2:
            st.text(name[1])
            st.image(poster[1])
        with col3:
            st.text(name[2])
            st.image(poster[2])
        with col4:
            st.text(name[3])
            st.image(poster[3])
        with col5:
            st.text(name[4])
            st.image(poster[4])

if __name__=='__main__':
    main()
