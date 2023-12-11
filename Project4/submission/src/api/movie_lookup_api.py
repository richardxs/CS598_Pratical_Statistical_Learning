import logging
from src.models.movie_recommender_models import RatingRecommendationRequest, \
    GenreRecommendationRequest
from src.service.movie_recommender_service import get_recommendations_by_genre, get_recommendations_by_rating
from src.service.movie_lookup_service import MovieLookupService
from fastapi import APIRouter

logger = logging.getLogger(__name__)
# Set the logging level to DEBUG
logger.setLevel(logging.DEBUG)

movie_lookup_router = APIRouter()

movie_lookup_service = MovieLookupService()
@movie_lookup_router.get("/genre")
async def lookup_movie_genre():
    genres = movie_lookup_service.ui_genre_list #unique_movie_genres
    logger.info(f"movie_lookup_api::lookup_movie_genre()>> lookup_movie_genre:{lookup_movie_genre}")
    return genres


@movie_lookup_router.get("/popular_movies/")
async def lookup_popular_100_movies(num_of_movies: int = 10):
    popular_100 = movie_lookup_service.popular_100 #unique_movie_genres
    # Slice the dictionary to get the top N movies
    top_n_movies = dict(list(popular_100.items())[:num_of_movies])
    logger.info(f"movie_lookup_api::lookup_popular_100_movies()>> lookup_popular_100_movies:{popular_100}")
    return top_n_movies