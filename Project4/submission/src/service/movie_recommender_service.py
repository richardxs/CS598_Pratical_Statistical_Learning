

# Placeholder function for genre-based recommendation
from src.models.movie_recommender_models import Movie, MovieRecommendationResponse
from src.service.movie_lookup_service import MovieLookupService
import logging
logger = logging.getLogger(__name__)
# Set the logging level to DEBUG
logger.setLevel(logging.INFO)

movie_lookup_service = MovieLookupService()

async def get_recommendations_by_genre(genre: str) -> MovieRecommendationResponse:
    # Replace this with your actual logic to fetch recommendations
    recommendations = movie_lookup_service.fetch_top_movie_recommendations_by_genre(genre)
    print(f"get_recommendations_by_genre(): recommendations: {type(recommendations)} \n {recommendations}")
    logger.info(f"get_recommendations_by_genre(): recommendations: {type(recommendations)} \n {recommendations}")

    # recommendations = [
    #     Movie(title="Movie 1", description="Action movie"),
    #     Movie(title="Movie 2", description="Comedy movie"),
    # ]
    #return recommendations
    return MovieRecommendationResponse(movies=recommendations)

# Placeholder function for rating-based recommendation
async def get_recommendations_by_rating(user_ratings: dict) -> MovieRecommendationResponse:
    # Replace this with your actual logic to fetch recommendations
    print(f" get_recommendations_by_rating():user_ratings = {user_ratings}")

    recommendations = movie_lookup_service.myIBCF(user_ratings, 10)
    #recommendations = ["Movie 3" , "Movie 4"]
    #     Movie(title="Movie 3"),
    #     Movie(title="Movie 4"),
    # ]
    return MovieRecommendationResponse(movies=recommendations)


