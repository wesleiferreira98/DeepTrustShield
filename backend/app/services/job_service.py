from sqlalchemy.orm import Session
from datetime import datetime
from app.db.models import Job, Result, Artifact


def update_job_stage(db: Session, job_id: str, stage: str, error_message: str = None):
    """Update job stage and timestamp."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        job.stage = stage
        job.updated_at = datetime.utcnow()
        if error_message:
            job.error_message = error_message
        db.commit()


def save_result(
    db: Session,
    job_id: str,
    p_fake: float,
    risk_level: str,
    flags_json: str,
    model_version: str
):
    """Save analysis result."""
    result = Result(
        job_id=job_id,
        p_fake=p_fake,
        risk_level=risk_level,
        flags_json=flags_json,
        model_version=model_version
    )
    db.add(result)
    db.commit()


def save_artifact(
    db: Session,
    job_id: str,
    face_crop_path: str = None,
    bbox_json: str = None,
    original_meta: str = None
):
    """Save job artifacts."""
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
            original_meta=original_meta
        )
        db.add(artifact)
    db.commit()
