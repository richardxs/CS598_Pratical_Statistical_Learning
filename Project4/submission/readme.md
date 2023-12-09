# **Movie Recommender App**

## Components
1. UI Component
   <br/>movie_recommender_ui.py
2. API Component:
    <br> movie_recommender_app.py

## Running UI Component
`Streamlit run movie_recommender_ui.py`

## Running API

`uvicorn movie_recommender_app:app --host 0.0.0.0 --port 80 --reload`

## URLs:
### API Swagger URL:
    http://localhost/docs
### URL to launch UI
    http://localhost:8501