from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext

from . import models
from .database import get_db

# ✅ Remove the "/auth" prefix, we'll handle it in main.py
router = APIRouter(tags=["auth"])

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# -----------------------------
# Pydantic Schemas
# -----------------------------
class UserSignup(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# -----------------------------
# Helpers
# -----------------------------
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# -----------------------------
# Routes
# -----------------------------
@router.post("/signup")
def signup(user: UserSignup, db: Session = Depends(get_db)):
    """Register a new user"""
    existing_user = db.query(models.User).filter(models.User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = models.User(
        email=user.email,
        hashed_password=hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User created successfully",
        "user": {"id": new_user.id, "email": new_user.email}
    }

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    """Authenticate a user"""
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user:
        raise HTTPException(status_code=400, detail="User not found")

    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid password")

    return {
        "message": f"Welcome {db_user.email}!",
        "user": {"id": db_user.id, "email": db_user.email}
    }

@router.get("/me")
def get_profile(db: Session = Depends(get_db)):
    """(Optional) For now just return a dummy user — later replace with JWT auth"""
    return {"message": "Profile endpoint is working, but no authentication implemented yet"}
