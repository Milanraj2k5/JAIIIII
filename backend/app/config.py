from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "mysql://root:Root@localhost:3306/truthlens"
    
    # JWT
    JWT_SECRET: str = "your-super-secret-jwt-key-change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440
    
    # Azure
    AZURE_VISION_KEY: Optional[str] = None
    AZURE_VISION_ENDPOINT: Optional[str] = None
    AZURE_SPEECH_KEY: Optional[str] = None
    AZURE_SPEECH_ENDPOINT: Optional[str] = None
    
    # News API
    NEWSAPI_KEY: Optional[str] = None
    
    # Blockchain
    POLYGON_RPC_URL: str = "https://polygon-mumbai.g.alchemy.com/v2/YOUR_ALCHEMY_KEY"
    POLYGON_PRIVATE_KEY: Optional[str] = None
    CONTRACT_ADDRESS: Optional[str] = None
    
    # File limits
    MAX_IMAGE_SIZE_MB: int = 6
    MAX_VIDEO_SIZE_MB: int = 50
    MAX_AUDIO_SIZE_MB: int = 20
    
    # Trust score weights
    AI_WEIGHT: float = 0.5
    AZURE_WEIGHT: float = 0.3
    NEWS_WEIGHT: float = 0.2
    
    # Feature flags
    AZURE_ENABLED: bool = False
    NEWSAPI_ENABLED: bool = False
    ENABLE_ONCHAIN: bool = True
    
    class Config:
        env_file = ".env"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Enable Azure if keys are provided
        if self.AZURE_VISION_KEY and self.AZURE_VISION_ENDPOINT:
            self.AZURE_ENABLED = True
        
        # Enable News API if key is provided
        if self.NEWSAPI_KEY:
            self.NEWSAPI_ENABLED = True

settings = Settings()