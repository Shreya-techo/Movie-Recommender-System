import streamlit as st
import pickle
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util

api_key = st.secrets["TMDB_API_KEY"]

st.set_page_config(
    page_title="AI Movie Discovery",
   # page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
with st.spinner("Loading AI models..."):
    model = load_model()

@st.cache_data
def fetch_poster(movie_id):
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"

    for _ in range(3):
        try:
            response = requests.get(url, timeout=3)
            data = response.json()

            poster_path = data.get('poster_path')

            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path

            return None

        except Exception as e:
            print(e)

    return None

    
def recommend(movie):
    movie_index = movies[movies['title']== movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1]) [1:6] 

    recommended_movies = []
    recommended_movies_poster= []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
      #  print('Recommended movie id = ', movie_id)
        recommended_movies.append(movies.iloc[i[0]].title)
        # fetch poster from api
      #  recommended_movies_poster.append(fetch_poster(movie_id))
        poster = fetch_poster(movie_id)
       # print("Movie:", movies.iloc[i[0]].title)
       # print("Poster URL:", poster)
        recommended_movies_poster.append(poster)
    return recommended_movies, recommended_movies_poster

def recommend_semantic(query):
    query_embedding = model.encode(query)

    cos_sim = util.cos_sim(query_embedding, movie_embedding)

    movie_list = sorted(
        list(enumerate(cos_sim[0])),
        reverse=True,
        key=lambda x: x[1]
    )[:5]

    recommended_movies = []
    recommended_movies_poster = []

    for i in movie_list:

        movie_id = movies_semantic.iloc[i[0]].movie_id

        recommended_movies.append(
            movies_semantic.iloc[i[0]].title
        )

        poster = fetch_poster(movie_id)

        recommended_movies_poster.append(poster)

    return recommended_movies, recommended_movies_poster

@st.cache_data
def load_data():

    movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    movie_embedding = pickle.load(open("movie_embedding.pkl", "rb"))
    movies_semantic = pickle.load(open("movies_semantic.pkl", "rb"))

    return (
        movies_dict,
        similarity,
        movie_embedding,
        movies_semantic
    )
with st.spinner("Loading movie database..."):
    movies_dict, similarity, movie_embedding, movies_semantic = load_data()
movies = pd.DataFrame(movies_dict)

st.header(' Movie Recommender System')

tab1, tab2 = st.tabs([
    "Traditional Recommender",
    " Semantic Search"
])

with tab1:

    st.header("Traditional recommender finds movie similar to a selected movie")
    selected_movie_name = st.selectbox(
        'See the movies below:',
       movies['title'].values
       
    )

    if st.button('Recommend', key="recommend_btn"):

        names, posters = recommend(selected_movie_name)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if posters[0]:
                st.image(posters[0], width="stretch")
            st.caption(names[0])

        with col2:
            if posters[1]:
                st.image(posters[1], width="stretch")
            st.caption(names[1])

        with col3:
            if posters[2]:
                st.image(posters[2], width="stretch")
            st.caption(names[2])

        with col4:
            if posters[3]:
                st.image(posters[3], width="stretch")
            st.caption(names[3])

        with col5:
            if posters[4]:
                st.image(posters[4], width="stretch")
            st.caption(names[4])

with tab2:

    st.header("Semantic Movie Search")

    with st.container(border=True):

        query = st.text_input(
            "Describe a movie"
        )

        st.caption(
            "Examples: alien planet, wormhole black hole astronaut etc..."
        )

    if st.button("Search", key="search_btn"):

        names, posters = recommend_semantic(query)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if posters[0]:
                st.image(posters[0], width="stretch")
            st.caption(names[0])

        with col2:
            if posters[1]:
                st.image(posters[1], width="stretch")
            st.caption(names[1])

        with col3:
            if posters[2]:
                st.image(posters[2], width="stretch")
            st.caption(names[2])

        with col4:
            if posters[3]:
                st.image(posters[3], width="stretch")
            st.caption(names[3])

        with col5:
            if posters[4]:
                st.image(posters[4], width="stretch")
            st.caption(names[4])
with st.sidebar:

    st.header("About")

    st.write("""
    This project contains two recommendation approaches:
    
    Traditional NLP:
    - CountVectorizer
    - Cosine Similarity

    Semantic NLP:
    - Sentence Transformers
    - Embeddings
    - Semantic Similarity
    """)
