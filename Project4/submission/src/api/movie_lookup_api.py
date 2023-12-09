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
