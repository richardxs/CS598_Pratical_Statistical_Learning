import streamlit as st
import requests
import pandas as pd

# API URL
api_url = "https://movie-recommender-app-gkmeeveifa-ul.a.run.app/api"
client = requests.Session()


def get_recommendations_by_genre(genre):
    # Build the API request URL
    url = f"{api_url}/recommendations/genre"
    print(f"get_recommendations_by_genre(): Selected genere: {genre}, url: {url}")

    # Prepare the request body with the selected genre
    data = {"genre": genre}

    # Send the POST request and get the response
    response = client.post(url, json=data)
    print(f"get_recommendations_by_genre(): response: {response}")

    # Check for successful response
    if response.status_code == 200:
        # Parse the response JSON data
        data = response.json()
        return data["movies"]
    else:
        print(f"Error fetching recommendations: {response.status_code}")
        return []


def get_recommendations_by_rating(rating):
    # Build the API request URL
    url = f"{api_url}/recommendations/rating"

    # Prepare the request body with the selected rating
    data = {"rating": rating}

    # Send the POST request and get the response
    response = client.post(url, json=data)

    # Check for successful response
    if response.status_code == 200:
        # Parse the response JSON data
        data = response.json()
        print(f"type: {type(data)} , data: {data}")
        return data#["movies"]
    else:
        print(f"Error fetching recommendations: {response.status_code}")
        return []


# Define function to handle recommendation type selection
def fetch_popular_movies(num_of_movies: int = 10):
    url = f"{api_url}/lookups/popular_movies/?num_of_movies={num_of_movies}"

    # Send the Get Request to fetch popular movies
    response = client.get(url)

    # Check for successful response
    if response.status_code == 200:
        # Parse the response JSON data
        data = response.json()
        print(f"popular movie types: {type(data)}")
        print(f"get_movie_genres(): Popular movies fetched from the API {url} : {data}")
        return data
    else:
        print(f"Error fetching popular movies: {response.status_code}")
        return []


def handle_recommendation_type(selected_type):
    if selected_type == "Movie recommender by Genre":
        st.subheader("Movie Recommender by Genre")
        genre_select = st.selectbox("Select Genre", genres)
        if st.button("Generate Recommendations"):
            # Call the genre-based recommendations function
            recommendations = get_recommendations_by_genre(genre_select)
            # Display recommendations
            st.subheader("Recommended Movies:")
            recommended_movies_df = pd.DataFrame(recommendations)
            recommended_movies_df.columns = [":: Recommended Movies ::"]
            st.dataframe(recommended_movies_df)

    elif selected_type == "Movie recommender by Rating":
        st.subheader("Trending Movies")
        movies = fetch_popular_movies()

        # Display Popular Movies
        for movie_id, title in movies.items():
            # Using st.slider for user ratings
            rating = st.slider(f"{title} (MovieID: {movie_id})", 1, 5, key=str(movie_id))
            st.write(f"You rated {title} as: {rating}")

        # Collect user ratings
        user_ratings = {str(movie_id): st.session_state[str(movie_id)] for movie_id, title in movies.items()}

        print(f"user_ratings: {user_ratings}")


        if st.button("Generate Recommendations"):
            # Call the rating-based recommendations function
            recommendations = get_recommendations_by_rating(user_ratings)
            # Display recommendations
            print(f"recommendations by rating: {type(recommendations)}")
            recommended_movies = recommendations["movies"]
            st.subheader("Recommended Movies:")
            recommended_movies_df = pd.DataFrame(recommended_movies)
            recommended_movies_df.columns = [":: Recommended Movies ::"]
            st.dataframe(recommended_movies_df)
            # for movie in recommended_movies:
            #     st.write(movie)


# Define sample genres for the demo
def get_movie_genres():
    print(f"get_movie_genres(): Fetching Movie Genres.")
    # Build the API request URL
    url = f"{api_url}/lookups/genre"
    print(f"get_movie_genres(): Calling API  to fetch Genres: {url}")

    # Send the POST request and get the response
    response = client.get(url)

    # Check for successful response
    if response.status_code == 200:
        # Parse the response JSON data
        data = response.json()
        print(f"get_movie_genres(): Movie Genres fetched from API : {data}")
        return data
    else:
        print(f"Error fetching genres: {response.status_code}")
        return []


genres = get_movie_genres()

# Create the main app layout
st.title("Movie Recommender App")

# Create the menu section
st.sidebar.subheader("Choose Recommendation Type")
selected_type = st.sidebar.radio("", ["Movie recommender by Genre", "Movie recommender by Rating"])

# Display the selected section based on the chosen type
handle_recommendation_type(selected_type)
