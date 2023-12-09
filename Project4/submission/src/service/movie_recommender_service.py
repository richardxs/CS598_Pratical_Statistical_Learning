

# Placeholder function for genre-based recommendation
from src.models.movie_recommender_models import Movie, MovieRecommendationResponse
from src.service.movie_lookup_service import MovieLookupService

movie_lookup_service = MovieLookupService()

async def get_recommendations_by_genre(genre: str) -> MovieRecommendationResponse:
    # Replace this with your actual logic to fetch recommendations
    recommendations = movie_lookup_service.fetch_top_movie_recommendations_by_genre(genre)

    # recommendations = [
    #     Movie(title="Movie 1", description="Action movie"),
    #     Movie(title="Movie 2", description="Comedy movie"),
    # ]
    #return recommendations
    return MovieRecommendationResponse(movies=recommendations)

# Placeholder function for rating-based recommendation
async def get_recommendations_by_rating(rating: float) -> MovieRecommendationResponse:
    # Replace this with your actual logic to fetch recommendations
    recommendations = [
        Movie(title="Movie 3"),
        Movie(title="Movie 4"),
    ]
    return MovieRecommendationResponse(movies=recommendations)


