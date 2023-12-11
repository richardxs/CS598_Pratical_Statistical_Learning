from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def check_application_health():
    return """
        {
           "msg": "Movie Recommender App is Up and Running"
        }
    """

