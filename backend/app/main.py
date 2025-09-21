from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import init_db

# Import your route files (they must each have: router = APIRouter())
from . import auth, news_verification, blockchain, azure_services

app = FastAPI(title="TruthLens API")

# 🔹 Allow frontend (React on port 3000) to call backend (FastAPI on port 8000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 👈 Allow all origins in dev (fixes CORS issues)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/")
def root():
    return {"message": "Welcome to TruthLens API 🚀"}

# 🔹 Register routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(news_verification.router, prefix="/news", tags=["news"])
app.include_router(blockchain.router, prefix="/blockchain", tags=["blockchain"])
app.include_router(azure_services.router, prefix="/azure", tags=["azure"])
