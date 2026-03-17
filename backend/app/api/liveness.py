import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.config import get_settings
from app.models.schemas import LivenessAnalyzeResponse, LivenessSessionDefinition
from app.services.liveness_service import analyze_liveness_frames

router = APIRouter()
settings = get_settings()


@router.post(
    '/analyze',
    response_model=LivenessAnalyzeResponse,
    summary='Analisar prova de vida por frames',
    description='Recebe frames (imagens) e valida sinais de liveness como piscar, distancia e sorriso.',
)
async def analyze_liveness(
    frames: list[UploadFile] = File(...),
    required_checks: str = Form('blink,distance,smile'),
    session_definition: str | None = Form(None),
):
    """Analyze liveness from extracted video frames.

    Accepts multiple image frames and returns pass/fail plus per-check metrics.
    """
    if not frames:
        raise HTTPException(status_code=400, detail='Nenhum frame enviado')

    if len(frames) < 3:
        raise HTTPException(status_code=400, detail='Envie pelo menos 3 frames para liveness')

    if len(frames) > 20:
        raise HTTPException(status_code=400, detail='Maximo de 20 frames por requisicao')

    allowed_types = {'image/jpeg', 'image/png', 'image/webp'}
    max_frame_bytes = settings.max_frame_upload_mb * 1024 * 1024
    max_total_bytes = settings.max_total_frames_upload_mb * 1024 * 1024
    total_bytes = 0
    frame_bytes_list: list[bytes] = []

    for frame in frames:
        if frame.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de frame nao suportado. Use: {', '.join(sorted(allowed_types))}",
            )

        content = await frame.read()
        if content:
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
            frame_bytes_list.append(content)

    if not frame_bytes_list:
        raise HTTPException(status_code=400, detail='Nenhum frame valido recebido')

    checks = [c.strip().lower() for c in required_checks.split(',') if c.strip()]
    parsed_session = None
    if session_definition:
        try:
            parsed_session = LivenessSessionDefinition.model_validate(json.loads(session_definition))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=f'Session definition invalida: {str(exc)}') from exc

    try:
        result = analyze_liveness_frames(
            frame_bytes_list,
            required_checks=checks,
            session_definition=parsed_session.model_dump() if parsed_session else None,
        )
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Falha ao analisar liveness: {str(exc)}') from exc

    return result
