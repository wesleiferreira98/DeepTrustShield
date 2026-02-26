import json
import time
import logging
import traceback
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import config
from app.pipeline.face_detection import detect_faces
from app.pipeline.quality_checks import check_quality
from app.pipeline.preprocessing import preprocess_face
from app.pipeline.inference import run_inference, get_loaded_model_version
from app.db_models import Base, Job, Result, Artifact

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(config.DATABASE_URL)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


def update_stage(db, job_id: str, stage: str, error_message: str = None):
    """Update job stage in database."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        job.stage = stage
        job.updated_at = datetime.utcnow()
        if error_message:
            job.error_message = error_message
        db.commit()
    logger.info(f"[{job_id}] Stage: {stage}")


def save_artifact(db, job_id: str, face_crop_path: str = None, bbox_json: str = None):
    """Save artifact to database."""
    artifact = db.query(Artifact).filter(Artifact.job_id == job_id).first()
    if artifact:
        if face_crop_path:
            artifact.face_crop_path = face_crop_path
        if bbox_json:
            artifact.bbox_json = bbox_json
    else:
        artifact = Artifact(
            job_id=job_id,
            face_crop_path=face_crop_path,
            bbox_json=bbox_json
        )
        db.add(artifact)
    db.commit()


def save_result(db, job_id: str, p_fake: float, risk_level: str, flags: dict, model_version: str = None):
    """Save result to database."""
    result = Result(
        job_id=job_id,
        p_fake=p_fake,
        risk_level=risk_level,
        flags_json=json.dumps(flags),
        model_version=model_version or config.MODEL_VERSION
    )
    db.add(result)
    db.commit()


def calculate_risk_level(p_fake: float, flags: dict) -> str:
    """Calculate risk level based on score and flags."""
    # Só retorna undetermined se não há face (múltiplas faces ainda são processadas)
    if flags.get("no_face"):
        return "undetermined"
    
    # p_fake None significa que não houve inferência
    if p_fake is None:
        return "undetermined"
    
    if p_fake < config.THRESHOLD_LOW:
        return "low"
    elif p_fake < config.THRESHOLD_HIGH:
        return "medium"
    else:
        return "high"


def process_deepfake_image(job_id: str, image_path: str, target_model: str = "auto"):
    """Main pipeline for deepfake detection."""
    db = get_db()
    start_time = time.time()
    
    flags = {
        "no_face": False,
        "multiple_faces": False,
        "blur_high": False,
        "low_light": False,
        "face_too_small": False
    }
    
    try:
        # Stage 1: Validating image
        update_stage(db, job_id, "validating_image")
        time.sleep(0.3)  # Simulated processing time for UX
        
        # Stage 2: Detecting face
        update_stage(db, job_id, "detecting_face")
        faces = detect_faces(image_path)
        
        if len(faces) == 0:
            flags["no_face"] = True
            logger.warning(f"[{job_id}] No face detected")
            save_artifact(db, job_id)
            save_result(db, job_id, None, "undetermined", flags, model_version=target_model)
            update_stage(db, job_id, "done")
            return
        
        if len(faces) > 1:
            flags["multiple_faces"] = True
            logger.warning(f"[{job_id}] Multiple faces detected: {len(faces)}")
            # Use the largest face
            faces = [max(faces, key=lambda f: f["width"] * f["height"])]
        
        face = faces[0]
        # Usar porcentagens para posicionamento correto no frontend
        bbox = {
            "x": face.get("x_pct", 0),
            "y": face.get("y_pct", 0),
            "width": face.get("width_pct", 100),
            "height": face.get("height_pct", 100)
        }
        
        # Save bbox artifact early for preview
        save_artifact(db, job_id, bbox_json=json.dumps(bbox))
        
        # Stage 3: Quality checks
        update_stage(db, job_id, "quality_checks")
        quality = check_quality(image_path, face)
        
        if quality["blur_score"] < config.BLUR_THRESHOLD:
            flags["blur_high"] = True
        if quality["brightness"] < config.BRIGHTNESS_MIN or quality["brightness"] > config.BRIGHTNESS_MAX:
            flags["low_light"] = True
        if face["width"] < config.MIN_FACE_SIZE or face["height"] < config.MIN_FACE_SIZE:
            flags["face_too_small"] = True
        
        # Stage 4: Preprocessing
        update_stage(db, job_id, "preprocessing")
        face_crop_path, face_tensor = preprocess_face(image_path, face, job_id, config.ARTIFACTS_PATH)
        
        # Update artifact with face crop
        save_artifact(db, job_id, face_crop_path=face_crop_path)
        
        # Stage 5: Model inference
        update_stage(db, job_id, "model_inference")
        p_fake = run_inference(face_tensor, config.MODEL_PATH, target_model=target_model)
        
        # Stage 6: Calibrating score
        update_stage(db, job_id, "calibrating_score")
        risk_level = calculate_risk_level(p_fake, flags)
        
        # Stage 7: Generating evidence
        update_stage(db, job_id, "generating_evidence")
        time.sleep(0.2)  # Evidence already saved
        
        # Save final result
        loaded_model_version = get_loaded_model_version()
        save_result(db, job_id, p_fake, risk_level, flags, model_version=loaded_model_version)
        
        # Done
        update_stage(db, job_id, "done")
        
        elapsed = time.time() - start_time
        logger.info(f"[{job_id}] Completed in {elapsed:.2f}s - p_fake: {p_fake:.4f}, risk: {risk_level}")
        
    except Exception as e:
        logger.error(f"[{job_id}] Error: {str(e)}")
        logger.error(traceback.format_exc())
        update_stage(db, job_id, "error", str(e))
        
        # Save partial result
        save_result(db, job_id, None, "undetermined", flags, model_version="unknown")
        
    finally:
        db.close()
