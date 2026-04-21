import os
from dotenv import load_dotenv

load_dotenv()

# Base directory for shared resources (parent of worker)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config:
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/detectar.db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", f"{BASE_DIR}/artifacts")
    MODEL_PATH = os.getenv("MODEL_PATH", f"{BASE_DIR}/model/weights")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "efficientnet-b0-v1")
    AUDIO_MODEL_PATH = os.getenv(
        "AUDIO_MODEL_PATH",
        f"{BASE_DIR}/model/weights/audio_deepfake_v1/audio_deepfake_v1_best.pth",
    )
    AUDIO_MODEL_SUMMARY_PATH = os.getenv(
        "AUDIO_MODEL_SUMMARY_PATH",
        f"{BASE_DIR}/model/weights/audio_deepfake_v1/summary.json",
    )
    
    # Thresholds
    THRESHOLD_LOW = 0.50  # Abaixo de 50% = baixo risco (provavelmente real)
    THRESHOLD_HIGH = 0.80  # Acima de 80% = alto risco (provavelmente fake)
    AUDIO_REVIEW_MARGIN = float(os.getenv("AUDIO_REVIEW_MARGIN", "0.15"))
    
    # Quality checks
    MIN_FACE_SIZE = 80  # pixels
    BLUR_THRESHOLD = 100  # Laplacian variance
    BRIGHTNESS_MIN = 40
    BRIGHTNESS_MAX = 220


config = Config()
