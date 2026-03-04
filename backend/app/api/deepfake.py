from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import uuid
import os
import json
from datetime import datetime
from redis import Redis
from rq import Queue

from app.config import get_settings
from app.db.database import get_db
from app.db.models import Job, Result, Artifact
from app.models.schemas import (
    AnalyzeResponse, StatusResponse, ResultResponse,
    JobStage, RiskLevel, StepInfo, StepState, PreviewInfo,
    Flags, Evidence
)

router = APIRouter()
settings = get_settings()

# Redis connection
redis_conn = Redis.from_url(settings.redis_url)
queue = Queue('deepfake', connection=redis_conn)

ALLOWED_TARGET_MODELS = {
    "auto",
    "efficientnet_b0_v4",
    "efficientnet_b0_v5",
    "efficientnet_b0",
    "efficientnet_b3",
    "efficientnet_b4",
    "alexnet",
}


PIPELINE_STEPS = [
    {"id": "validating_image", "label": "Validando imagem"},
    {"id": "detecting_face", "label": "Detectando face"},
    {"id": "quality_checks", "label": "Verificando qualidade"},
    {"id": "preprocessing", "label": "Pré-processando"},
    {"id": "model_inference", "label": "Analisando com IA"},
    {"id": "calibrating_score", "label": "Calibrando resultado"},
    {"id": "generating_evidence", "label": "Gerando evidências"},
]


def get_steps_status(current_stage: str) -> list:
    """Generate steps status based on current stage."""
    steps = []
    stage_order = [s["id"] for s in PIPELINE_STEPS]
    
    try:
        current_idx = stage_order.index(current_stage)
    except ValueError:
        current_idx = -1 if current_stage == "pending" else len(stage_order)
    
    for idx, step in enumerate(PIPELINE_STEPS):
        if current_stage == "done":
            state = StepState.COMPLETED
        elif current_stage == "error":
            state = StepState.ERROR if idx == current_idx else (
                StepState.COMPLETED if idx < current_idx else StepState.PENDING
            )
        elif idx < current_idx:
            state = StepState.COMPLETED
        elif idx == current_idx:
            state = StepState.RUNNING
        else:
            state = StepState.PENDING
        
        steps.append(StepInfo(id=step["id"], label=step["label"], state=state))
    
    return steps


def get_progress(stage: str) -> int:
    """Calculate progress percentage based on stage."""
    progress_map = {
        "pending": 0,
        "validating_image": 10,
        "detecting_face": 25,
        "quality_checks": 40,
        "preprocessing": 55,
        "model_inference": 70,
        "calibrating_score": 85,
        "generating_evidence": 95,
        "done": 100,
        "error": 0
    }
    return progress_map.get(stage, 0)


def get_stage_message(stage: str) -> str:
    """Get human-readable message for stage."""
    messages = {
        "pending": "Aguardando processamento...",
        "validating_image": "Validando formato da imagem...",
        "detecting_face": "Detectando face na imagem...",
        "quality_checks": "Verificando qualidade da imagem...",
        "preprocessing": "Preparando imagem para análise...",
        "model_inference": "Analisando com modelo de IA...",
        "calibrating_score": "Calibrando resultado...",
        "generating_evidence": "Gerando evidências visuais...",
        "done": "Análise concluída!",
        "error": "Erro no processamento"
    }
    return messages.get(stage, "Processando...")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    target_model: str = Form("auto"),
    db: Session = Depends(get_db)
):
    """Start deepfake analysis on an image."""
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado. Use: {', '.join(allowed_types)}"
        )

    if target_model not in ALLOWED_TARGET_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo alvo inválido. Use um de: {', '.join(sorted(ALLOWED_TARGET_MODELS))}"
        )

    max_image_bytes = settings.max_image_upload_mb * 1024 * 1024
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create job directory
    job_dir = os.path.join(settings.artifacts_path, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Save original image
    original_path = os.path.join(job_dir, "original" + os.path.splitext(file.filename or ".jpg")[1])
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio")
    if len(content) > max_image_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem excede o limite de {settings.max_image_upload_mb}MB",
        )
    with open(original_path, "wb") as f:
        f.write(content)
    
    # Create job record
    job = Job(
        id=job_id,
        stage=JobStage.PENDING,
        progress=0,
        original_path=original_path,
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    
    # Enqueue job for processing
    queue.enqueue(
        'app.tasks.process_deepfake_image',
        job_id,
        original_path,
        target_model,
        job_timeout='5m'
    )
    
    return AnalyzeResponse(
        job_id=job_id,
        status="queued",
        message="Análise iniciada. Use o endpoint de status para acompanhar."
    )


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str, db: Session = Depends(get_db)):
    """Get current status of a deepfake analysis job."""
    
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    # Build preview info
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
        preview=preview
    )


@router.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(job_id: str, db: Session = Depends(get_db)):
    """Get final result of a deepfake analysis job."""
    
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    if job.stage not in [JobStage.DONE, JobStage.ERROR]:
        raise HTTPException(
            status_code=400,
            detail="Análise ainda em andamento. Use o endpoint de status."
        )
    
    result = db.query(Result).filter(Result.job_id == job_id).first()
    artifact = db.query(Artifact).filter(Artifact.job_id == job_id).first()
    
    # Build flags
    flags_data = json.loads(result.flags_json) if result and result.flags_json else {}
    flags = Flags(**flags_data)
    
    # Build evidence
    evidence_data = {}
    if artifact:
        if artifact.bbox_json:
            evidence_data["bbox"] = json.loads(artifact.bbox_json)
        if artifact.face_crop_path and os.path.exists(artifact.face_crop_path):
            evidence_data["face_crop_url"] = f"/artifacts/{job_id}/face_crop.jpg"
    evidence = Evidence(**evidence_data)
    
    # Calculate processing time
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
        processing_time_ms=processing_time
    )


@router.get("/artifacts/{job_id}/{name}")
async def get_artifact(job_id: str, name: str, db: Session = Depends(get_db)):
    """Get artifact file for a job."""
    
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    if os.path.basename(name) != name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Nome de artefato inválido")
    
    file_path = os.path.join(settings.artifacts_path, job_id, name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Artefato não encontrado")
    
    return FileResponse(file_path)
