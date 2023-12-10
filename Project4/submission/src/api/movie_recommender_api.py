from src.models.movie_recommender_models import RatingRecommendationRequest, \
    GenreRecommendationRequest
from src.service.movie_recommender_service import get_recommendations_by_genre, get_recommendations_by_rating

from fastapi import APIRouter

movie_recommender_router = APIRouter()

# API endpoint for genre-based recommendations
@movie_recommender_router.post("/genre")
async def recommend_by_genre(request: GenreRecommendationRequest):
    return await get_recommendations_by_genre(genre=request.genre)

# API endpoint for rating-based recommendations
@movie_recommender_router.post("/rating")
async def recommend_by_rating(request: RatingRecommendationRequest):
    return await get_recommendations_by_rating(user_ratings=request.ratings)


