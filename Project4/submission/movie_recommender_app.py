from fastapi import FastAPI, Path, Query, Body

from src.api.movie_recommender_api import router

app = FastAPI()

app.include_router(router, prefix="/api")

