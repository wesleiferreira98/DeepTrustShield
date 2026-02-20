from pydantic_settings import BaseSettings
from functools import lru_cache
import os

# Base directory (parent of backend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings(BaseSettings):
    # Database
    database_url: str = f"sqlite:///{BASE_DIR}/detectar.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Artifacts (uses parent directory)
    artifacts_path: str = f"{BASE_DIR}/artifacts"
    
    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    
    # Model
    model_version: str = "efficientnet-b0-v1"
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
