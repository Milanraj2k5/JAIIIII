from fastapi import FastAPI
from .database import init_db

app = FastAPI(title="TruthLens API")


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/")
def root():
    return {"message": "Welcome to TruthLens API 🚀"}
