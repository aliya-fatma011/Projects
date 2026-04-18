import streamlit as st
import pickle
import pandas as pd
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= API KEY =================
API_KEY = "25f2777080955cf6d0bc1dd4d6ad7e06"

# ================= SESSION (FIX CONNECTION ERROR) =================
session = requests.Session()
retries = Retry(total=5, backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


# ================= FETCH POSTER =================
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            "api_key": API_KEY,
            "language": "en-US"
        }

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = session.get(url, params=params, headers=headers, timeout=10)

        if response.status_code != 200:
            return "https://via.placeholder.com/500x750?text=No+Image"

        data = response.json()
        poster_path = data.get('poster_path')

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"

    except Exception as e:
        print("Error:", e)
        return "https://via.placeholder.com/500x750?text=Error"


# ================= LOAD DATA =================
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))


# ================= RECOMMEND FUNCTION =================
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)

        # small delay to avoid API block
        time.sleep(0.2)

        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters


# ================= STREAMLIT UI =================
st.title('🎬 Movie Recommender System')

selected_movie_name = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)

    cols = [col1, col2, col3, col4, col5]

    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])