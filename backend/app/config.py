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

    # MediaPipe tasks model asset for backend liveness
    mediapipe_face_landmarker_path: str = f"{BASE_DIR}/backend/models/face_landmarker_v2.task"
    
    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    
    # Model
    model_version: str = "efficientnet-b0-v1"

    # Security
    artifacts_retention_hours: int = 24
    cleanup_interval_minutes: int = 60
    max_image_upload_mb: int = 8
    max_frame_upload_mb: int = 3
    max_total_frames_upload_mb: int = 24

    # Liveness challenge signing. Override in production with a long random value.
    liveness_session_secret: str = "dev-liveness-session-secret-change-me"
    liveness_session_ttl_seconds: int = 120
    liveness_anti_spoof_enabled: bool = True
    liveness_anti_spoof_threshold: float = 0.55
    liveness_onnx_antispoof_enabled: bool = True
    liveness_onnx_antispoof_path: str = (
        f"{BASE_DIR}/model/weights/antispoof_mobilenet_v1_cv/fold_3/antispoof_mobilenet_v1.onnx"
    )
    liveness_onnx_antispoof_metadata_path: str = (
        f"{BASE_DIR}/model/weights/antispoof_mobilenet_v1_cv/fold_3/antispoof_mobilenet_v1_metadata.json"
    )
    liveness_onnx_antispoof_aggregation: str = "p75"
    liveness_onnx_antispoof_threshold: float = -1.0
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
