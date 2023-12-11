# **Movie Recommender App**

## Components
1. UI Component
   <br/>movie_recommender_ui.py
2. API Component:
    <br> movie_recommender_app.py


## Running Backend API (Need to a cloud host)

`uvicorn movie_recommender_app:app --host 0.0.0.0 --port 80 --reload`


## Running FrontEnd UI Component
`Streamlit run movie_recommender_ui.py`

## URLs:
### API Swagger URL:
    http://localhost/docs
### URL to launch UI
    http://localhost:8501