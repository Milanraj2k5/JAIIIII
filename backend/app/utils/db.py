import os
from functools import lru_cache
from typing import Tuple

from pymongo import MongoClient


@lru_cache(maxsize=1)
def _get_client_and_db() -> Tuple[MongoClient, str]:
    """Create and cache a MongoDB client and database name.

    Reads the following environment variables:
    - MONGODB_URI: Full MongoDB connection string (Atlas recommended)
    - MONGODB_DB_NAME: Database name (defaults to 'truthlens')
    """
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "truthlens")
    client = MongoClient(uri)
    # Will not fail until first operation; good enough for lazy init
    return client, db_name


def get_db():
    client, db_name = _get_client_and_db()
    return client[db_name]


def get_users_collection():
    return get_db()["users"]
