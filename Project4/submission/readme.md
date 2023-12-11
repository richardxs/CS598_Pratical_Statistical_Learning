# **Movie Recommender App**

## Components
1. UI Component
   `movie_recommender_ui.py`
2. API Component:
    `movie_recommender_app.py`

# Project SetUp Steps
## Install the dependencies
`pip install -r requirements.txt`

## Running Backend API

`uvicorn movie_recommender_app:app --host 0.0.0.0 --port 80 --reload`


## Running FrontEnd UI Component
`Streamlit run movie_recommender_ui.py`

## URLs:
### API Swagger URL:
    http://localhost/docs
### URL to launch UI
    http://localhost:8501