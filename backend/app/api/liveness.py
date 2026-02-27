from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.liveness_service import analyze_liveness_frames

router = APIRouter()


@router.post('/analyze')
async def analyze_liveness(
    frames: list[UploadFile] = File(...),
    required_checks: str = Form('blink,distance,smile'),
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
    frame_bytes_list: list[bytes] = []

    for frame in frames:
        if frame.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de frame nao suportado. Use: {', '.join(sorted(allowed_types))}",
            )

        content = await frame.read()
        if content:
            frame_bytes_list.append(content)

    if not frame_bytes_list:
        raise HTTPException(status_code=400, detail='Nenhum frame valido recebido')

    checks = [c.strip().lower() for c in required_checks.split(',') if c.strip()]

    try:
        result = analyze_liveness_frames(frame_bytes_list, required_checks=checks)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Falha ao analisar liveness: {str(exc)}') from exc

    return result
