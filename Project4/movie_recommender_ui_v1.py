import streamlit as st

# Function to get movie recommendations based on genre
def get_recommendations_by_genre(genre):
    # Your logic to get recommendations based on genre
    recommendations = ["Movie 1", "Movie 2", "Movie 3"]
    return recommendations

# Function to get movie recommendations based on rating
def get_recommendations_by_rating(rating):
    # Your logic to get recommendations based on rating
    recommendations = ["Movie 4", "Movie 5", "Movie 6"]
    return recommendations

# Streamlit App
def main():
    st.title("Movie Recommender App")

    # Left Section (Menu or Accordion)
    menu_option = st.sidebar.radio("Select Movie Recommender Type", ["Movie Recommender by Genre", "Movie Recommender by Rating"])

    # Right Section
    st.sidebar.markdown("---")  # Separator between menu and content

    if menu_option == "Movie Recommender by Genre":
        st.sidebar.subheader("Movie Recommender by Genre")

        # Dropdown for selecting movie genre
        genre = st.sidebar.selectbox("Select Genre", ["Action", "Comedy", "Drama"])

        # Button to generate recommendations
        if st.sidebar.button("Generate Recommendations"):
            recommendations = get_recommendations_by_genre(genre)
            st.write("Recommended Movies:")
            st.write(recommendations)

    elif menu_option == "Movie Recommender by Rating":
        st.sidebar.subheader("Movie Recommender by Rating")

        # Slider for selecting movie rating
        rating = st.sidebar.slider("Select Rating", 1, 5)

        # Button to generate recommendations
        if st.sidebar.button("Generate Recommendations"):
            recommendations = get_recommendations_by_rating(rating)
            st.write("Recommended Movies:")
            st.write(recommendations)

if __name__ == "__main__":
    main()
