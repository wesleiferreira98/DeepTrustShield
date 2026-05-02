from datetime import datetime
import json
import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from redis import Redis
from rq import Queue
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.database import get_db
from app.db.models import Artifact, Job, Result
from app.models.schemas import AnalyzeResponse, Evidence, Flags, JobStage, ResultResponse, RiskLevel, StatusResponse

router = APIRouter()
settings = get_settings()

redis_conn = Redis.from_url(settings.redis_url)
queue = Queue("deepfake", connection=redis_conn)

AUDIO_PIPELINE_STEPS = [
    {"id": "validating_image", "label": "Validando audio"},
    {"id": "preprocessing", "label": "Pre-processando audio"},
    {"id": "model_inference", "label": "Analisando voz com IA"},
    {"id": "calibrating_score", "label": "Aplicando limite calibrado"},
    {"id": "generating_evidence", "label": "Gerando diagnostico"},
]


def get_audio_steps_status(current_stage: str) -> list:
    steps = []
    stage_order = [step["id"] for step in AUDIO_PIPELINE_STEPS]

    try:
        current_idx = stage_order.index(current_stage)
    except ValueError:
        current_idx = -1 if current_stage == "pending" else len(stage_order)

    for idx, step in enumerate(AUDIO_PIPELINE_STEPS):
        if current_stage == "done":
            state = "completed"
        elif current_stage == "error":
            state = "error" if idx == current_idx else "completed" if idx < current_idx else "pending"
        elif idx < current_idx:
            state = "completed"
        elif idx == current_idx:
            state = "running"
        else:
            state = "pending"
        steps.append({"id": step["id"], "label": step["label"], "state": state})

    return steps


def get_audio_progress(stage: str) -> int:
    progress_map = {
        "pending": 0,
        "validating_image": 15,
        "preprocessing": 35,
        "model_inference": 65,
        "calibrating_score": 85,
        "generating_evidence": 95,
        "done": 100,
        "error": 0,
    }
    return progress_map.get(stage, 0)


def get_audio_stage_message(stage: str) -> str:
    messages = {
        "pending": "Aguardando processamento...",
        "validating_image": "Validando formato do audio...",
        "preprocessing": "Preparando log-Mel do audio...",
        "model_inference": "Analisando audio com modelo de IA...",
        "calibrating_score": "Aplicando threshold calibrado...",
        "generating_evidence": "Gerando diagnostico de audio...",
        "done": "Analise concluida!",
        "error": "Erro no processamento",
    }
    return messages.get(stage, "Processando...")

ALLOWED_AUDIO_TYPES = {
    "audio/flac",
    "audio/x-flac",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/ogg",
    "application/octet-stream",
}

ALLOWED_AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Start deepfake analysis on one audio file."""
    extension = os.path.splitext(file.filename or "")[1].lower()
    if file.content_type not in ALLOWED_AUDIO_TYPES and extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Tipo de audio nao suportado. Use FLAC, WAV, MP3, M4A ou OGG.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Arquivo de audio vazio")

    max_audio_bytes = settings.max_audio_upload_mb * 1024 * 1024
    if len(content) > max_audio_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio excede o limite de {settings.max_audio_upload_mb}MB",
        )

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(settings.artifacts_path, job_id)
    os.makedirs(job_dir, exist_ok=True)

    audio_path = os.path.join(job_dir, "audio" + (extension or ".wav"))
    with open(audio_path, "wb") as output:
        output.write(content)

    job = Job(
        id=job_id,
        stage=JobStage.PENDING,
        progress=0,
        original_path=audio_path,
        created_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()

    queue.enqueue(
        "app.tasks.process_deepfake_audio",
        job_id,
        audio_path,
        job_timeout="15m",
    )

    return AnalyzeResponse(
        job_id=job_id,
        status="queued",
        message="Analise de audio iniciada. Use o endpoint de status para acompanhar.",
    )


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_audio_status(job_id: str, db: Session = Depends(get_db)):
    """Get current status for an audio analysis job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job nao encontrado")

    return StatusResponse(
        job_id=job_id,
        stage=job.stage,
        progress=get_audio_progress(job.stage),
        message=job.error_message or get_audio_stage_message(job.stage),
        steps=get_audio_steps_status(job.stage),
        preview=None,
    )


@router.get("/result/{job_id}", response_model=ResultResponse)
async def get_audio_result(job_id: str, db: Session = Depends(get_db)):
    """Get final result for an audio analysis job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job nao encontrado")

    if job.stage not in [JobStage.DONE, JobStage.ERROR]:
        raise HTTPException(status_code=400, detail="Analise ainda em andamento. Use o endpoint de status.")

    result = db.query(Result).filter(Result.job_id == job_id).first()
    artifact = db.query(Artifact).filter(Artifact.job_id == job_id).first()

    flags_data = json.loads(result.flags_json) if result and result.flags_json else {}
    audio_summary = None
    if artifact and artifact.original_meta:
        try:
            audio_summary = json.loads(artifact.original_meta).get("audio_summary")
        except Exception:
            audio_summary = None

    processing_time = None
    if job.updated_at and job.created_at:
        processing_time = int((job.updated_at - job.created_at).total_seconds() * 1000)

    return ResultResponse(
        job_id=job_id,
        p_fake=result.p_fake if result else None,
        risk_level=result.risk_level if result else RiskLevel.UNDETERMINED,
        flags=Flags(**flags_data),
        evidence=Evidence(),
        model_version=result.model_version if result else "audio_deepfake_v1",
        created_at=job.created_at,
        processing_time_ms=processing_time,
        audio_summary=audio_summary,
    )


@router.get("/artifacts/{job_id}/{name}")
async def get_audio_artifact(job_id: str, name: str, db: Session = Depends(get_db)):
    """Get artifact file for an audio job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job nao encontrado")

    if os.path.basename(name) != name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Nome de artefato invalido")

    file_path = os.path.join(settings.artifacts_path, job_id, name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Artefato nao encontrado")

    return FileResponse(file_path)
