from sqlalchemy import Column, Integer, String, DateTime, Text, Float
from datetime import datetime
from .database import Base  # ✅ Import SAME Base

class User(Base):
    __tablename__ = "users"   # ✅ Explicit tablename

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    file_name = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False)
    file_type = Column(String(10), nullable=False)
    trust_score = Column(Float, nullable=False)
    verdict = Column(String(10), nullable=False)
    analysis_data = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class OnchainLog(Base):
    __tablename__ = "onchain_logs"

    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, nullable=False)
    transaction_hash = Column(String(66), nullable=False)
    file_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
