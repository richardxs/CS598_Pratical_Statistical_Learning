from pydantic import BaseModel
from typing import List
# Movie model
class Movie(BaseModel):
    title: str =""

# Genre-based recommendation request
class GenreRecommendationRequest(BaseModel):
    genre: str = ""

# Rating-based recommendation request
class RatingRecommendationRequest(BaseModel):
    ratings: dict = dict()

# Movie recommender response
class MovieRecommendationResponse(BaseModel):
    movies: List[str] = []