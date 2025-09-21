from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import settings

DATABASE_URL = settings.DATABASE_URL

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=True, future=True)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ✅ Single global Base for all models
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database (create tables + admin user)"""
    from . import models  # ✅ Import models here to register them
    from .auth import hash_password

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Ensure admin user exists
    db = SessionLocal()
    try:
        admin = db.query(models.User).filter(models.User.email == "admin@truthlens.com").first()
        if not admin:
            admin = models.User(
                email="admin@truthlens.com",
                hashed_password=hash_password("admin123")
            )
            db.add(admin)
            db.commit()
            print("✅ Admin user created (email: admin@truthlens.com / password: admin123)")
    finally:
        db.close()
