from datetime import datetime
import json
import os
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from redis import Redis
from rq import Queue
from sqlalchemy.orm import Session

from app.api.deepfake import (
    ALLOWED_TARGET_MODELS,
    get_progress,
    get_stage_message,
    get_steps_status,
)
from app.config import get_settings
from app.db.database import get_db
from app.db.models import Artifact, Job, Result
from app.models.schemas import (
    AnalyzeResponse,
    Evidence,
    Flags,
    JobStage,
    ResultResponse,
    RiskLevel,
    StatusResponse,
    PreviewInfo,
)

router = APIRouter()
settings = get_settings()

redis_conn = Redis.from_url(settings.redis_url)
queue = Queue("deepfake", connection=redis_conn)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video_frames(
    frames: list[UploadFile] = File(...),
    target_model: str = Form("auto"),
    db: Session = Depends(get_db),
):
    """Start deepfake analysis for a short video represented by extracted frames."""
    if target_model not in ALLOWED_TARGET_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo alvo inválido. Use um de: {', '.join(sorted(ALLOWED_TARGET_MODELS))}",
        )

    if not frames:
        raise HTTPException(status_code=400, detail="Nenhum frame enviado")

    if len(frames) < 3:
        raise HTTPException(status_code=400, detail="Envie pelo menos 3 frames para análise de vídeo")

    if len(frames) > 12:
        raise HTTPException(status_code=400, detail="Máximo de 12 frames por análise")

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    max_frame_bytes = settings.max_frame_upload_mb * 1024 * 1024
    max_total_bytes = settings.max_total_frames_upload_mb * 1024 * 1024
    total_bytes = 0
    job_id = str(uuid.uuid4())

    job_dir = os.path.join(settings.artifacts_path, job_id)
    frames_dir = os.path.join(job_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_paths: list[str] = []
    for idx, frame in enumerate(frames):
        if frame.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de frame não suportado. Use: {', '.join(sorted(allowed_types))}",
            )

        extension = os.path.splitext(frame.filename or "")[1] or ".jpg"
        frame_path = os.path.join(frames_dir, f"frame_{idx:02d}{extension}")
        content = await frame.read()

        if not content:
            continue
        if len(content) > max_frame_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Frame excede o limite de {settings.max_frame_upload_mb}MB",
            )
        total_bytes += len(content)
        if total_bytes > max_total_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload total de frames excede {settings.max_total_frames_upload_mb}MB",
            )

        with open(frame_path, "wb") as output:
            output.write(content)
        frame_paths.append(frame_path)

    if not frame_paths:
        raise HTTPException(status_code=400, detail="Nenhum frame válido recebido")

    job = Job(
        id=job_id,
        stage=JobStage.PENDING,
        progress=0,
        original_path=frame_paths[0],
        created_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()

    queue.enqueue(
        "app.tasks.process_deepfake_video_frames",
        job_id,
        frame_paths,
        target_model,
        job_timeout="10m",
    )

    return AnalyzeResponse(
        job_id=job_id,
        status="queued",
        message="Análise de vídeo iniciada. Use o endpoint de status para acompanhar.",
    )


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_video_status(job_id: str, db: Session = Depends(get_db)):
    """Get current status for a video analysis job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    preview = None
    artifact = db.query(Artifact).filter(Artifact.job_id == job_id).first()
    if artifact:
        preview_data = {}
        if artifact.bbox_json:
            preview_data["bbox"] = json.loads(artifact.bbox_json)
        if artifact.face_crop_path and os.path.exists(artifact.face_crop_path):
            preview_data["face_crop_url"] = f"/artifacts/{job_id}/face_crop.jpg"
        if preview_data:
            preview = PreviewInfo(**preview_data)

    return StatusResponse(
        job_id=job_id,
        stage=job.stage,
        progress=get_progress(job.stage),
        message=job.error_message or get_stage_message(job.stage),
        steps=get_steps_status(job.stage),
        preview=preview,
    )


@router.get("/result/{job_id}", response_model=ResultResponse)
async def get_video_result(job_id: str, db: Session = Depends(get_db)):
    """Get final result for a video analysis job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    if job.stage not in [JobStage.DONE, JobStage.ERROR]:
        raise HTTPException(status_code=400, detail="Análise ainda em andamento. Use o endpoint de status.")

    result = db.query(Result).filter(Result.job_id == job_id).first()
    artifact = db.query(Artifact).filter(Artifact.job_id == job_id).first()

    flags_data = json.loads(result.flags_json) if result and result.flags_json else {}
    flags = Flags(**flags_data)

    evidence_data = {}
    if artifact:
        if artifact.bbox_json:
            evidence_data["bbox"] = json.loads(artifact.bbox_json)
        if artifact.face_crop_path and os.path.exists(artifact.face_crop_path):
            evidence_data["face_crop_url"] = f"/artifacts/{job_id}/face_crop.jpg"
    evidence = Evidence(**evidence_data)

    video_summary = None
    frame_results = None
    if artifact and artifact.original_meta:
        try:
            meta = json.loads(artifact.original_meta)
            video_summary = meta.get("video_summary")
            frame_results = meta.get("frame_results")
        except Exception:
            video_summary = None
            frame_results = None

    processing_time = None
    if job.updated_at and job.created_at:
        processing_time = int((job.updated_at - job.created_at).total_seconds() * 1000)

    return ResultResponse(
        job_id=job_id,
        p_fake=result.p_fake if result else None,
        risk_level=result.risk_level if result else RiskLevel.UNDETERMINED,
        flags=flags,
        evidence=evidence,
        model_version=result.model_version if result else settings.model_version,
        created_at=job.created_at,
        processing_time_ms=processing_time,
        video_summary=video_summary,
        frame_results=frame_results,
    )


@router.get("/artifacts/{job_id}/{name}")
async def get_video_artifact(job_id: str, name: str, db: Session = Depends(get_db)):
    """Get artifact file for a video analysis job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    if os.path.basename(name) != name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Nome de artefato inválido")

    file_path = os.path.join(settings.artifacts_path, job_id, name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Artefato não encontrado")

    return FileResponse(file_path)
