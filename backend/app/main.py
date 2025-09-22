from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import init_db

# Import route files
from . import auth, news_verification, blockchain, azure_services

app = FastAPI(title="TruthLens API")

# Allow frontend (React on port 3000) to call backend (FastAPI on port 8000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()  # Create tables and admin user

@app.get("/")
def root():
    return {"message": "Welcome to TruthLens API 🚀"}

# Register routers with /api prefix
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(news_verification.router, prefix="/api/news", tags=["news"])
app.include_router(blockchain.router, prefix="/api/blockchain", tags=["blockchain"])
app.include_router(azure_services.router, prefix="/api/azure", tags=["azure"])
