from pydantic import BaseModel
from typing import List
# Movie model
class Movie(BaseModel):
    title: str =""
    description: str = ""

# Genre-based recommendation request
class GenreRecommendationRequest(BaseModel):
    genre: str = ""

# Rating-based recommendation request
class RatingRecommendationRequest(BaseModel):
    rating: float = 0.0

# Movie recommender response
class MovieRecommendationResponse(BaseModel):
    movies: List[Movie] = []