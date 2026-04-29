import json
import os
import time
import logging
import traceback
from datetime import datetime
from statistics import median
import math

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import config
from app.pipeline.face_detection import detect_faces
from app.pipeline.quality_checks import check_quality
from app.pipeline.preprocessing import preprocess_face
from app.pipeline.inference import run_inference, get_loaded_model_version
from app.pipeline.audio_inference import run_audio_inference
from app.pipeline.audio_symbolic import (
    combine_audio_decision,
    evaluate_audio_symbolic_rules,
    extract_audio_symbolic_features,
)
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


def save_artifact(
    db,
    job_id: str,
    face_crop_path: str = None,
    bbox_json: str = None,
    original_meta: str = None,
):
    """Save artifact to database."""
    artifact = db.query(Artifact).filter(Artifact.job_id == job_id).first()
    if artifact:
        if face_crop_path:
            artifact.face_crop_path = face_crop_path
        if bbox_json:
            artifact.bbox_json = bbox_json
        if original_meta:
            artifact.original_meta = original_meta
    else:
        artifact = Artifact(
            job_id=job_id,
            face_crop_path=face_crop_path,
            bbox_json=bbox_json,
            original_meta=original_meta,
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
        model_version=model_version or config.MODEL_VERSION,
    )
    db.add(result)
    db.commit()


def calculate_risk_level(p_fake: float, flags: dict) -> str:
    """Calculate risk level based on score and flags."""
    if flags.get("no_face"):
        return "undetermined"

    if p_fake is None:
        return "undetermined"

    if p_fake < config.THRESHOLD_LOW:
        return "low"
    if p_fake < config.THRESHOLD_HIGH:
        return "medium"
    return "high"


def _default_flags() -> dict:
    return {
        "no_face": False,
        "multiple_faces": False,
        "blur_high": False,
        "low_light": False,
        "face_too_small": False,
    }


def calculate_audio_risk_level(p_fake: float, threshold: float) -> str:
    if p_fake is None:
        return "undetermined"
    if p_fake < threshold:
        return "low"
    if p_fake < min(threshold + config.AUDIO_REVIEW_MARGIN, 0.80):
        return "medium"
    return "high"


def process_deepfake_audio(job_id: str, audio_path: str):
    """Pipeline for deepfake detection from one audio file."""
    db = get_db()
    start_time = time.time()
    flags = _default_flags()

    try:
        update_stage(db, job_id, "validating_image")
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError("Arquivo de audio nao encontrado")

        update_stage(db, job_id, "preprocessing")
        update_stage(db, job_id, "model_inference")
        audio_result = run_audio_inference(
            audio_path,
            config.AUDIO_MODEL_PATH,
            config.AUDIO_MODEL_SUMMARY_PATH,
        )

        update_stage(db, job_id, "calibrating_score")
        p_fake = audio_result["p_fake"]
        threshold = audio_result["threshold"]
        symbolic_features = extract_audio_symbolic_features(
            audio_path,
            {
                **audio_result.get("audio_info", {}),
                "segment_summary": audio_result.get("segment_summary", {}),
            },
        )
        symbolic_rules = evaluate_audio_symbolic_rules(
            audio_result,
            symbolic_features,
            audio_result.get("segments", []),
        )
        symbolic_decision = combine_audio_decision(audio_result, symbolic_rules)
        risk_level = symbolic_decision["risk_level"]

        update_stage(db, job_id, "generating_evidence")
        audio_summary = {
            "threshold": threshold,
            "decision": symbolic_decision["decision"],
            "audio_info": audio_result.get("audio_info", {}),
            "segment_summary": audio_result.get("segment_summary", {}),
            "segments": audio_result.get("segments", []),
            "symbolic": {
                **symbolic_decision,
                "features": symbolic_features,
            },
        }
        save_artifact(db, job_id, original_meta=json.dumps({"audio_summary": audio_summary}))
        save_result(db, job_id, p_fake, risk_level, flags, model_version=audio_result["model_version"])
        update_stage(db, job_id, "done")

        elapsed = time.time() - start_time
        logger.info(
            "[%s] Audio completed in %.2fs - p_fake: %.4f, threshold: %.4f, risk: %s",
            job_id,
            elapsed,
            p_fake,
            threshold,
            risk_level,
        )

    except Exception as e:
        logger.error(f"[{job_id}] Audio error: {str(e)}")
        logger.error(traceback.format_exc())
        update_stage(db, job_id, "error", str(e))
        save_result(db, job_id, None, "undetermined", flags, model_version="audio_deepfake_v1")

    finally:
        db.close()


def process_deepfake_image(job_id: str, image_path: str, target_model: str = "auto"):
    """Main pipeline for deepfake detection from one image."""
    db = get_db()
    start_time = time.time()
    flags = _default_flags()

    try:
        update_stage(db, job_id, "validating_image")
        time.sleep(0.3)

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
            faces = [max(faces, key=lambda f: f["width"] * f["height"])]

        face = faces[0]
        bbox = {
            "x": face.get("x_pct", 0),
            "y": face.get("y_pct", 0),
            "width": face.get("width_pct", 100),
            "height": face.get("height_pct", 100),
        }
        save_artifact(db, job_id, bbox_json=json.dumps(bbox))

        update_stage(db, job_id, "quality_checks")
        quality = check_quality(image_path, face)

        if quality["blur_score"] < config.BLUR_THRESHOLD:
            flags["blur_high"] = True
        if quality["brightness"] < config.BRIGHTNESS_MIN or quality["brightness"] > config.BRIGHTNESS_MAX:
            flags["low_light"] = True
        if face["width"] < config.MIN_FACE_SIZE or face["height"] < config.MIN_FACE_SIZE:
            flags["face_too_small"] = True

        update_stage(db, job_id, "preprocessing")
        face_crop_path, face_tensor = preprocess_face(image_path, face, job_id, config.ARTIFACTS_PATH)
        save_artifact(db, job_id, face_crop_path=face_crop_path)

        update_stage(db, job_id, "model_inference")
        p_fake = run_inference(face_tensor, config.MODEL_PATH, target_model=target_model)

        update_stage(db, job_id, "calibrating_score")
        risk_level = calculate_risk_level(p_fake, flags)

        update_stage(db, job_id, "generating_evidence")
        time.sleep(0.2)

        model_version = get_loaded_model_version()
        save_result(db, job_id, p_fake, risk_level, flags, model_version=model_version)

        update_stage(db, job_id, "done")

        elapsed = time.time() - start_time
        logger.info(f"[{job_id}] Completed in {elapsed:.2f}s - p_fake: {p_fake:.4f}, risk: {risk_level}")

    except Exception as e:
        logger.error(f"[{job_id}] Error: {str(e)}")
        logger.error(traceback.format_exc())
        update_stage(db, job_id, "error", str(e))
        save_result(db, job_id, None, "undetermined", flags, model_version="unknown")

    finally:
        db.close()


def process_deepfake_video_frames(job_id: str, frame_paths: list[str], target_model: str = "auto"):
    """Process a short video represented by extracted image frames."""
    db = get_db()
    start_time = time.time()
    flags = _default_flags()

    try:
        update_stage(db, job_id, "validating_image")
        if not frame_paths:
            raise ValueError("Nenhum frame recebido para análise")

        valid_paths = [p for p in frame_paths if p]
        if not valid_paths:
            raise ValueError("Nenhum frame válido recebido")

        update_stage(db, job_id, "detecting_face")
        frame_candidates = []
        representative_bbox = None

        for idx, frame_path in enumerate(valid_paths):
            faces = detect_faces(frame_path)
            if len(faces) == 0:
                continue

            if len(faces) > 1:
                flags["multiple_faces"] = True
                faces = [max(faces, key=lambda f: f["width"] * f["height"])]

            face = faces[0]
            bbox = {
                "x": face.get("x_pct", 0),
                "y": face.get("y_pct", 0),
                "width": face.get("width_pct", 100),
                "height": face.get("height_pct", 100),
            }
            if representative_bbox is None:
                representative_bbox = bbox

            frame_candidates.append({
                "index": idx,
                "path": frame_path,
                "face": face,
                "bbox": bbox,
            })

        if not frame_candidates:
            flags["no_face"] = True
            save_artifact(
                db,
                job_id,
                original_meta=json.dumps({
                    "video_summary": {
                        "total_frames": len(valid_paths),
                        "valid_frames": 0,
                        "p_fake_mean": None,
                        "p_fake_median": None,
                        "p_fake_max": None,
                    },
                    "frame_results": [],
                }),
            )
            save_result(db, job_id, None, "undetermined", flags, model_version=target_model)
            update_stage(db, job_id, "done")
            return

        if representative_bbox:
            save_artifact(db, job_id, bbox_json=json.dumps(representative_bbox))

        update_stage(db, job_id, "quality_checks")
        quality_pass = []
        for candidate in frame_candidates:
            quality = check_quality(candidate["path"], candidate["face"])

            blur_high = quality["blur_score"] < config.BLUR_THRESHOLD
            low_light = (
                quality["brightness"] < config.BRIGHTNESS_MIN or
                quality["brightness"] > config.BRIGHTNESS_MAX
            )
            face_too_small = (
                candidate["face"]["width"] < config.MIN_FACE_SIZE or
                candidate["face"]["height"] < config.MIN_FACE_SIZE
            )

            flags["blur_high"] = flags["blur_high"] or blur_high
            flags["low_light"] = flags["low_light"] or low_light
            flags["face_too_small"] = flags["face_too_small"] or face_too_small

            quality_pass.append({
                **candidate,
                "quality": {
                    "blur_high": blur_high,
                    "low_light": low_light,
                    "face_too_small": face_too_small,
                },
            })

        update_stage(db, job_id, "preprocessing")
        prepared_frames = []
        representative_face_crop = None

        for candidate in quality_pass:
            face_crop_path, face_tensor = preprocess_face(
                candidate["path"],
                candidate["face"],
                job_id,
                config.ARTIFACTS_PATH,
            )
            if representative_face_crop is None:
                representative_face_crop = face_crop_path
            prepared_frames.append({
                **candidate,
                "face_tensor": face_tensor,
                "face_crop_path": face_crop_path,
            })

        if representative_face_crop:
            save_artifact(db, job_id, face_crop_path=representative_face_crop)

        update_stage(db, job_id, "model_inference")
        frame_scores = []
        frame_results = []

        for prepared in prepared_frames:
            p_fake_frame = run_inference(prepared["face_tensor"], config.MODEL_PATH, target_model=target_model)
            frame_scores.append(p_fake_frame)
            frame_results.append({
                "index": prepared["index"],
                "p_fake": p_fake_frame,
                "quality": prepared["quality"],
            })

        update_stage(db, job_id, "calibrating_score")
        p_fake_mean = sum(frame_scores) / len(frame_scores) if frame_scores else None
        p_fake_median = median(frame_scores) if frame_scores else None
        p_fake_max = max(frame_scores) if frame_scores else None

        high_frame_count = 0
        suspicious_frame_count = 0
        if frame_scores:
            high_frame_count = sum(1 for score in frame_scores if score >= config.THRESHOLD_HIGH)
            suspicious_frame_count = sum(1 for score in frame_scores if score >= config.THRESHOLD_LOW)

        p_fake = None
        if p_fake_mean is not None and p_fake_median is not None and p_fake_max is not None:
            # Aggregation robusta contra outliers: prioriza mediana + média.
            p_fake = (p_fake_median * 0.6) + (p_fake_mean * 0.4)

            # Evita falso positivo por um único frame extremo.
            if high_frame_count < 2:
                p_fake = min(p_fake, config.THRESHOLD_HIGH - 0.001)

            min_suspicious_frames = max(2, math.ceil(len(frame_scores) * 0.4))
            if suspicious_frame_count < min_suspicious_frames:
                p_fake = min(p_fake, config.THRESHOLD_LOW - 0.001)

        if frame_scores:
            flags["no_face"] = False
        risk_level = calculate_risk_level(p_fake, flags)

        update_stage(db, job_id, "generating_evidence")

        summary = {
            "total_frames": len(valid_paths),
            "valid_frames": len(frame_scores),
            "p_fake_mean": p_fake_mean,
            "p_fake_median": p_fake_median,
            "p_fake_max": p_fake_max,
            "high_frame_count": high_frame_count,
            "suspicious_frame_count": suspicious_frame_count,
        }
        save_artifact(
            db,
            job_id,
            original_meta=json.dumps({
                "video_summary": summary,
                "frame_results": frame_results,
            }),
        )

        model_version = get_loaded_model_version() if frame_scores else target_model
        save_result(db, job_id, p_fake, risk_level, flags, model_version=model_version)

        update_stage(db, job_id, "done")

        elapsed = time.time() - start_time
        logger.info(
            "[%s] Video completed in %.2fs - frames: %d/%d, p_fake: %s, risk: %s",
            job_id,
            elapsed,
            len(frame_scores),
            len(valid_paths),
            f"{p_fake:.4f}" if p_fake is not None else "None",
            risk_level,
        )

    except Exception as e:
        logger.error(f"[{job_id}] Video error: {str(e)}")
        logger.error(traceback.format_exc())
        update_stage(db, job_id, "error", str(e))
        save_result(db, job_id, None, "undetermined", flags, model_version="unknown")

    finally:
        db.close()
