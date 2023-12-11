import logging
import pandas as pd
import numpy as np

from src.models.movie_recommender_models import Movie

logger = logging.getLogger(__name__)
# Set the logging level to DEBUG
logger.setLevel(logging.INFO)


class MovieLookupService():
    _instance = None

    def __new__(cls):
        logger.info(f"MovieLookupService::Creating Singleton instance of MovieLookupService >> ")
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        logger.info(f"MovieLookupService::__init__() >> ")
        k = 10
        self.movies_csv_url = "https://liangfgithub.github.io/MovieData/movies.dat"
        self.movie_ratings_url = "https://liangfgithub.github.io/MovieData/ratings.dat"
        self.ui_genre_list = ["Action", "Adventure", "Animation", "Children's", "Comedy",
                              "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                              "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                              "Thriller", "War", "Western"]

        self.movies_df = self.load_movies_data()
        self.unique_movie_genres = self.lookup_unique_genres()

        self.ratings_df = self.load_ratings_data()

        self.grouped_ratings = self.lookup_grouped_ratings()

        self.movies_ratings = self.load_movie_ratings()

        self.movies_lookup = self.generate_movie_lookup()
        
        self.popular_100 = self.get_popular_100_movies()

        self.top_k_rating_freq_with_movie_id, self.top_k_rating_freq_with_movie_title = self.fetch_top_k_movies_grouped_by_genres(k)
        self.top_rating_freq_with_movie_id, self.top_rating_freq_with_movie_title = self.fetch_top_movies_grouped_by_genres()

        self.similarity_matrix = self.load_similarity_data()
        self.movie_rating_2d_df = self.load_movie_rating_data()
        self.all_movie_ids = self.movie_rating_2d_df.columns
        logger.info(f"similarity_matrix: {self.similarity_matrix.shape} , movie_rating_2d_df:{self.movie_rating_2d_df.shape}")
        print(f"similarity_matrix: {self.similarity_matrix.shape} , movie_rating_2d_df:{self.movie_rating_2d_df.shape} , all_movie_ids: {len(self.all_movie_ids)} , \n -----\n {self.all_movie_ids} \n-------\n: ")


    def load_ratings_data(self):
        logger.info(f"MovieLookupService::load_ratings_data() >> Loading Movies Rating data from :{self.movie_ratings_url}")

        ratings = pd.read_csv(self.movie_ratings_url, sep="::", engine='python',
                             header=None)
        ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        ratings['UserID'] = ratings['UserID'].apply(lambda x: 'u' + str(x))
        ratings['MovieID'] = ratings['MovieID'].apply(lambda x: 'm' + str(x))
        logger.info(f"MovieLookupService::load_ratings_data() >> Loaded Movies Rating data :\n---------------\n{ratings.head()}")

        return ratings

    def load_movies_data(self):
        logger.info(f"MovieLookupService::load_movies_data() >> Loading Movies data from :{self.movies_csv_url}")
        movies = pd.read_csv(self.movies_csv_url, sep="::", engine='python',
                             encoding="ISO-8859-1", header=None)
        movies.columns = ['MovieID', 'Title', 'Genres']
        movies['MovieID'] = movies['MovieID'].apply(lambda x: 'm' + str(x))
        logger.info(f"MovieLookupService::load_movies_data() >> Loaded Movies data :\n---------------\n{movies.head()}")

        return movies


    def lookup_unique_genres(self):
        logger.info(f"MovieLookupService::lookup_unique_genres() >> Creating Lookup for Movie Genres.")
        """
        Returns a list of unique genres from a pandas DataFrame.
  
        Returns:
          A list of unique genres.
        """
        df = self.movies_df

        # Check if "genre" column exists
        if "Genres" not in df.columns:
            raise ValueError("DataFrame does not have a column named 'genre'")

        # Split the genre column into a list of genres
        df['Genres'] = df['Genres'].str.split(',')

        # Flatten the list of lists into a single list
        genres = df['Genres'].sum()

        # Remove duplicates and return the unique genres
        unique_genres = list(set(genres))
        logger.info(f"MovieLookupService::lookup_unique_genres() >> unique_genres: {unique_genres}.")


        return unique_genres

    def lookup_grouped_ratings(self):
        logger.info(f"MovieLookupService::lookup_grouped_ratings() >> Creating Lookup for Grouped Movie Ratings.")

        grouped_ratings = self.ratings_df.groupby('MovieID').agg({'Rating': ['count', 'mean']})
        grouped_ratings.columns = grouped_ratings.columns.droplevel(0)
        grouped_ratings.reset_index(inplace=True)
        grouped_ratings.rename(columns={'count': 'RatingFreq', 'mean': 'RatingAvg'}, inplace=True)
        logger.info(f"MovieLookupService::lookup_grouped_ratings() >> Loaded Grouped Ratings :\n---------------\n{grouped_ratings.head()}")

        return grouped_ratings

    def load_movie_ratings(self):
        logger.info(f"MovieLookupService::load_movie_ratings() >> Creating Lookup for Movie Ratings.")

        movies_ratings = self.movies_df.merge(self.grouped_ratings, left_on='MovieID', right_on='MovieID')
        logger.info(
            f"MovieLookupService::load_movie_ratings() >> Loaded Movie Ratings :\n---------------\n{movies_ratings.head()}")

        return movies_ratings

    def fetch_top_k_movies_grouped_by_genres(self, K):
        logger.info(f"MovieLookupService::fetch_top_k_movies_grouped_by_genres() >> Creating Lookup for Top {K} movies grouped by genres.")
        genre_movie_id_rating_freq = {}
        genre_movie_title_rating_freq = {}
        #K = 5  # top 5
        for genre in self.ui_genre_list:
            true_table = self.movies_ratings['Genres'].apply(lambda x: True if genre in x else False)
            rating_freq_df = self.movies_ratings[true_table].sort_values('RatingFreq', ascending=False)
            genre_movie_id_rating_freq[genre] = rating_freq_df['MovieID'].values[:K]
            genre_movie_title_rating_freq[genre] = rating_freq_df['Title'].values[:K]

        return genre_movie_id_rating_freq, genre_movie_title_rating_freq

    def fetch_top_movies_grouped_by_genres(self):
        logger.info(f"MovieLookupService::fetch_top_k_movies_grouped_by_genres() >> Creating Lookup for Top  movies grouped by genres.")
        genre_movie_id_rating_freq = {}
        genre_movie_title_rating_freq = {}
        #K = 5  # top 5
        for genre in self.ui_genre_list:
            true_table = self.movies_ratings['Genres'].apply(lambda x: True if genre in x else False)
            rating_freq_df = self.movies_ratings[true_table].sort_values('RatingFreq', ascending=False)
            genre_movie_id_rating_freq[genre] = rating_freq_df['MovieID']
            genre_movie_title_rating_freq[genre] = rating_freq_df['Title']

        return genre_movie_id_rating_freq, genre_movie_title_rating_freq



    # def pad_dictionary(self, dictionary, K):
    #     for key, val in dictionary.items():
    #         print(f"key = {key}, size: {len(val)}")
    #         size = len(val)
    #         if size < K:
    #             N = K- size  # Replace 5 with the desired size
    #             empty_array = np.empty(N, dtype=object)
    #             new_val = np.concatenate((val,  empty_array))
    #             #new_val = val.extend( [''] * (K - size))
    #             # for _ in range(K-size):
    #             #     val.append("")
    #             print(f"key: {key}, new_val: {new_val}")
    #
    #             dictionary[key] = new_val
    #
    #     return dictionary



    def fetch_top_movie_recommendations_by_genre(self, genre):
        logger.info(
            f"MovieLookupService::fetch_top_movie_recommendations_by_genre() >> Fetching top recommendations for movie genre -> {genre}.")
        top_movies = []

        print(f"type_title: {type(self.top_rating_freq_with_movie_title)}, genre: {type(genre)}")

        if self.top_k_rating_freq_with_movie_title and genre in self.top_rating_freq_with_movie_title:
            top_movies = self.top_rating_freq_with_movie_title.get(genre)

        logger.info(
            f"MovieLookupService::fetch_top_movie_recommendations_by_genre() >> Most popular movies for Genre '{genre}' are => {top_movies}.")
        # create_movies_object = lambda movie: Movie(title=movie)
        # top_movies_object_list = list(map(create_movies_object, top_movies))
        return top_movies[:10]

    def get_popular_100_movies(self):

        top_100_movies_df = self.movies_ratings.sort_values(by='RatingFreq', ascending=False).head(100)

        # Generate a dictionary with 'MovieID' as key and 'Title' as value for the top 100 movies
        top_100_movies_dict = dict(zip(top_100_movies_df['MovieID'], top_100_movies_df['Title']))

        # Print or use the generated dictionary as needed
        #print(top_100_movies_dict)
        #popular_id_100 = self.movies_ratings.sort_values("RatingFreq", ascending=False)["MovieID",'Title'][:100].values
        #popular_Title_100 = self.movies_ratings.sort_values("RatingFreq", ascending=False)["MovieID",'Title'][:100].values
        return top_100_movies_dict

    def myIBCF(self, newuser, top_n=10):
        logger.info(f"myIBCF(): newuser:{newuser}")
        popular_id_100 = list(self.popular_100.keys())
        S = self.similarity_matrix

        logger.info(f"popular_id_100: {len(popular_id_100)} total elements")
        if isinstance(newuser, dict):
            # create newuser from dict
            rated_movies = set(newuser.keys())
            newuser = pd.Series(newuser, index=self.all_movie_ids)
        else:
            rated_movies = set(newuser.loc[~pd.isna(newuser)].index.values)

        # generate rating predictions
        prediction = S.multiply(newuser, axis=1).sum(axis=1) / S.loc[:, ~pd.isna(newuser)].sum(axis=1)
        sorted_pred = prediction[prediction != float('inf')].sort_values(ascending=False)

        # uncomment the line to see predicted ratings
        #     print(sorted_pred[~sorted_pred.isna()][:20])

        # remove invalid values and movies already rated by user
        recommend_id = list(sorted_pred[~sorted_pred.isna()].index.values)
        recommend_id = [item for item in recommend_id if item not in rated_movies]

        # if less than top_n recommendations, add most popular movies that are not rated by the user
        # and not in the recommendation list
        i = 0
        while len(recommend_id) < top_n and i < 100:
            if popular_id_100[i] not in rated_movies and popular_id_100[i] not in recommend_id:
                recommend_id.append(popular_id_100[i])
            i += 1

        recommended_movie_ids = recommend_id[:top_n]
        recommended_movie_titles = [self.movies_lookup.get(key, 'Not Found') for key in recommended_movie_ids]

        return recommended_movie_titles

    @staticmethod
    def load_similarity_data():
        logger.info(f"load_similarity_matrix_data(): Loading similarity matrix from the dataset")
        # Assuming the CSV file is in the 'resources' package
        csv_path = "src/resources/S.csv"

        try:
            # Load the CSV file into a Pandas DataFrame
            df_similarity = pd.read_csv(csv_path, index_col=0)

            # Print or return the DataFrame, or perform further operations as needed
            return df_similarity

        except FileNotFoundError:
            logger.error(f"Error: File {csv_path} not found.")
            # Handle the exception as needed, e.g., return a default DataFrame or raise an exception

    # def load_similarity_matrix_data(self):
    #     logger.info(f"load_similarity_matrix_data(): Loading similarity matrix from the dataset")
    def load_movie_rating_data(self):
        logger.info(f"load_movie_rating_data(): Loading Movie rating data from the dataset")
        # Assuming the CSV file is in the 'resources' package
        csv_path = "src/resources/Movie_Rmat.csv"

        try:
            # Load the CSV file into a Pandas DataFrame
            df_movie_rmat = pd.read_csv(csv_path)

            # Print or return the DataFrame, or perform further operations as needed
            return df_movie_rmat

        except FileNotFoundError:
            logger.error(f"Error: File {csv_path} not found.")
            # Handle the exception as needed, e.g., return a default DataFrame or raise an exception

    def generate_movie_lookup(self):
        logger.info(f"generate_movie_lookup(): Generate lookup dictionary for movie id/ title.")
        # Generate a dictionary with 'MovieID' as key and 'Title' as value for the top 100 movies
        movies_lookup = dict(zip(self.movies_ratings['MovieID'], self.movies_ratings['Title']))

        return movies_lookup





