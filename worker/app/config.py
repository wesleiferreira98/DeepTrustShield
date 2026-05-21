import os
import json
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
        f"{BASE_DIR}/model/weights/audio_deepfake_v1_fake_braccent_d_codec_equalized/audio_deepfake_v1_best.pth",
    )
    AUDIO_MODEL_SUMMARY_PATH = os.getenv(
        "AUDIO_MODEL_SUMMARY_PATH",
        f"{BASE_DIR}/model/weights/audio_deepfake_v1_fake_braccent_d_codec_equalized/summary.json",
    )
    AUDIO_MODEL_REGISTRY_JSON = os.getenv("AUDIO_MODEL_REGISTRY", "")
    AUDIO_MODEL_REGISTRY = {
        "default": {
            "label": "FakeBrAccent codec equalizado",
            "path": AUDIO_MODEL_PATH,
            "summary_path": AUDIO_MODEL_SUMMARY_PATH,
        },
        "fake_braccent_codec": {
            "label": "FakeBrAccent codec equalizado",
            "path": f"{BASE_DIR}/model/weights/audio_deepfake_v1_fake_braccent_d_codec_equalized/audio_deepfake_v1_best.pth",
            "summary_path": f"{BASE_DIR}/model/weights/audio_deepfake_v1_fake_braccent_d_codec_equalized/summary.json",
        },
        "brspeech_sample": {
            "label": "BRSpeech-DF amostra (experimental)",
            "path": f"{BASE_DIR}/audio/models/audio_deepfake_v1_brspeech_sample/audio_deepfake_v1_best.pth",
            "summary_path": f"{BASE_DIR}/audio/models/audio_deepfake_v1_brspeech_sample/summary.json",
        },
    }
    if AUDIO_MODEL_REGISTRY_JSON:
        try:
            AUDIO_MODEL_REGISTRY.update(json.loads(AUDIO_MODEL_REGISTRY_JSON))
        except Exception:
            pass
    
    # Thresholds
    THRESHOLD_LOW = 0.50  # Abaixo de 50% = baixo risco (provavelmente real)
    THRESHOLD_HIGH = 0.80  # Acima de 80% = alto risco (provavelmente fake)
    AUDIO_REVIEW_MARGIN = float(os.getenv("AUDIO_REVIEW_MARGIN", "0.15"))
    AUDIO_ASR_ENABLED = os.getenv("AUDIO_ASR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    AUDIO_ASR_MODEL = os.getenv("AUDIO_ASR_MODEL", "medium")
    AUDIO_ASR_LANGUAGE = os.getenv("AUDIO_ASR_LANGUAGE", "pt")
    AUDIO_ASR_DEVICE = os.getenv("AUDIO_ASR_DEVICE", "auto")
    AUDIO_ASR_COMPUTE_TYPE = os.getenv("AUDIO_ASR_COMPUTE_TYPE", "default")
    AUDIO_ASR_BEAM_SIZE = int(os.getenv("AUDIO_ASR_BEAM_SIZE", "5"))
    AUDIO_ASR_VAD_FILTER = os.getenv("AUDIO_ASR_VAD_FILTER", "false").lower() in {"1", "true", "yes", "on"}
    AUDIO_ASR_PREPROCESS_ENABLED = os.getenv("AUDIO_ASR_PREPROCESS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    
    # Quality checks
    MIN_FACE_SIZE = 80  # pixels
    BLUR_THRESHOLD = 100  # Laplacian variance
    BRIGHTNESS_MIN = 40
    BRIGHTNESS_MAX = 220


config = Config()


def get_audio_model_config(target_model: str | None = None) -> dict:
    key = (target_model or "default").strip() or "default"
    if key == "auto":
        key = "default"
    model_config = config.AUDIO_MODEL_REGISTRY.get(key) or config.AUDIO_MODEL_REGISTRY["default"]
    return {
        "key": key if key in config.AUDIO_MODEL_REGISTRY else "default",
        "label": model_config.get("label") or key,
        "path": model_config.get("path") or config.AUDIO_MODEL_PATH,
        "summary_path": model_config.get("summary_path") or config.AUDIO_MODEL_SUMMARY_PATH,
    }
