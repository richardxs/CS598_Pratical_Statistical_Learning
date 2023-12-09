from fastapi import FastAPI, Path, Query, Body

from src.api.movie_recommender_api import movie_recommender_router
from src.api.movie_lookup_api import movie_lookup_router

app = FastAPI()

app.include_router(movie_recommender_router, prefix="/api/recommendations")
app.include_router(movie_lookup_router, prefix="/api/lookups")

