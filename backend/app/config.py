from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database: make sure your MySQL server is running and database 'truthlens' exists
    DATABASE_URL: str = "mysql+pymysql://root:root@localhost:3306/truthlens"
    
    # JWT settings
    JWT_SECRET: str = "your-super-secret-jwt-key-change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440
    
    # Azure settings (optional)
    AZURE_COMPUTER_VISION_KEY: Optional[str] = None
    AZURE_COMPUTER_VISION_ENDPOINT: Optional[str] = None
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_REGION: Optional[str] = None
    
    # News API
    NEWSAPI_KEY: Optional[str] = None
    NEWSAPI_ENABLED: bool = False
    
    # Blockchain
    POLYGON_RPC_URL: str = "https://polygon-amoy.g.alchemy.com/v2/YOUR_ALCHEMY_KEY"
    POLYGON_PRIVATE_KEY: Optional[str] = None
    CONTRACT_ADDRESS: Optional[str] = None
    
    # File size limits
    MAX_IMAGE_SIZE_MB: int = 6
    MAX_VIDEO_SIZE_MB: int = 50
    MAX_AUDIO_SIZE_MB: int = 20
    
    # Trust score weights
    AI_WEIGHT: float = 0.5
    AZURE_WEIGHT: float = 0.3
    NEWS_WEIGHT: float = 0.2
    
    # Feature flags
    AZURE_ENABLED: bool = False
    ENABLE_ONCHAIN: bool = True
    
    class Config:
        # Load values from .env file if exists
        env_file = ".env"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Automatically enable Azure if keys are provided
        if self.AZURE_COMPUTER_VISION_KEY and self.AZURE_COMPUTER_VISION_ENDPOINT:
            self.AZURE_ENABLED = True
        
        # Automatically enable News API if key is provided
        if self.NEWSAPI_KEY:
            self.NEWSAPI_ENABLED = True

# Single settings instance to import anywhere
settings = Settings()
