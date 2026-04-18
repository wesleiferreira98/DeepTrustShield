import json
import hashlib
import hmac
import secrets
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.config import get_settings
from app.models.schemas import LivenessAnalyzeResponse, LivenessSessionDefinition
from app.services.liveness_service import analyze_liveness_frames

router = APIRouter()
settings = get_settings()

HEAD_TURN_TARGET_DELTA = 0.10
HEAD_PITCH_TARGET_DELTA = 0.07
DISTANCE_TARGET_RANGE_RATIO = 0.16
MOUTH_OPEN_TARGET_FACTOR = 1.46


def _canonical_session_payload(session: dict) -> str:
    payload = {
        "session_id": session.get("session_id"),
        "detector_mode": session.get("detector_mode"),
        "issued_at": session.get("issued_at"),
        "expires_at": session.get("expires_at"),
        "challenges": session.get("challenges", []),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sign_session(session: dict) -> str:
    return hmac.new(
        settings.liveness_session_secret.encode("utf-8"),
        _canonical_session_payload(session).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _verify_session(session: dict) -> None:
    expected = _sign_session(session)
    provided = session.get("signature")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=400, detail="Sessao de liveness invalida ou adulterada")

    expires_at = session.get("expires_at")
    if not isinstance(expires_at, int) or expires_at < int(time.time()):
        raise HTTPException(status_code=400, detail="Sessao de liveness expirada")


def _pick_weighted_challenges(pool: list[dict], total: int) -> list[dict]:
    available = [dict(item) for item in pool]
    selected: list[dict] = []

    while available and len(selected) < total:
        total_weight = sum(float(item.get("weight", 1.0)) for item in available)
        ticket = secrets.randbelow(1_000_000) / 1_000_000 * total_weight
        chosen_index = 0

        for index, item in enumerate(available):
            ticket -= float(item.get("weight", 1.0))
            if ticket <= 0:
                chosen_index = index
                break

        selected.append(available.pop(chosen_index))

    for index in range(len(selected) - 1, 0, -1):
        swap_index = secrets.randbelow(index + 1)
        selected[index], selected[swap_index] = selected[swap_index], selected[index]

    return selected


def _build_liveness_challenges() -> list[dict]:
    blink_target = 2 if secrets.randbelow(2) else 1
    mouth_pool = [
        {
            "id": "mouth_head_left",
            "requirements": ["mouth_open", "head_left"],
            "label": "Abrir a boca + Virar para a esquerda",
            "hint": "Abra a boca e vire levemente o rosto para a esquerda.",
            "targets": {"mouth_open": MOUTH_OPEN_TARGET_FACTOR, "head_left": HEAD_TURN_TARGET_DELTA},
            "weight": 1.2,
        },
        {
            "id": "mouth_head_right",
            "requirements": ["mouth_open", "head_right"],
            "label": "Abrir a boca + Virar para a direita",
            "hint": "Abra a boca e vire levemente o rosto para a direita.",
            "targets": {"mouth_open": MOUTH_OPEN_TARGET_FACTOR, "head_right": HEAD_TURN_TARGET_DELTA},
            "weight": 1.2,
        },
    ]
    support_pool = [
        {
            "id": "mouth_hold",
            "requirements": ["mouth_open"],
            "label": "Abrir a boca",
            "hint": "Olhe para a camera e abra a boca gradualmente ate ficar bem visivel.",
            "targets": {"mouth_open": MOUTH_OPEN_TARGET_FACTOR},
            "weight": 0.75,
        },
        {
            "id": "distance_head",
            "requirements": ["distance", "head_right"],
            "label": "Aproximar/Afastar + Virar",
            "hint": "Aproxime/afaste o rosto devagar e vire levemente para a direita.",
            "targets": {"distance": DISTANCE_TARGET_RANGE_RATIO, "head_right": HEAD_TURN_TARGET_DELTA},
            "weight": 0.55,
        },
        {
            "id": "blink_head",
            "requirements": ["blink", "head_left"],
            "label": f"Piscar {blink_target}x + Virar",
            "hint": f"Olhando para a camera, pisque {blink_target} vez(es) e vire levemente para a esquerda.",
            "targets": {"blink": float(blink_target), "head_left": HEAD_TURN_TARGET_DELTA},
            "weight": 0.5,
        },
    ]

    total = 3 + secrets.randbelow(3)
    mandatory = _pick_weighted_challenges(mouth_pool, 1)
    remaining_pool = [
        challenge
        for challenge in [*mouth_pool, *support_pool]
        if challenge["id"] != mandatory[0]["id"]
    ]
    selected = [*mandatory, *_pick_weighted_challenges(remaining_pool, total - len(mandatory))]
    for index in range(len(selected) - 1, 0, -1):
        swap_index = secrets.randbelow(index + 1)
        selected[index], selected[swap_index] = selected[swap_index], selected[index]
    for challenge in selected:
        challenge.pop("weight", None)
    return selected


@router.post(
    '/session',
    summary='Criar sessao segura de prova de vida',
    description='Gera desafios aleatorios no backend e assina a sessao para evitar adulteracao no frontend.',
)
async def create_liveness_session(detector_mode: str = Form('mediapipe')):
    now = int(time.time())
    session = {
        "session_id": secrets.token_urlsafe(24),
        "detector_mode": detector_mode,
        "issued_at": now,
        "expires_at": now + settings.liveness_session_ttl_seconds,
        "challenges": _build_liveness_challenges(),
    }
    session["signature"] = _sign_session(session)
    return session


@router.post(
    '/analyze',
    response_model=LivenessAnalyzeResponse,
    summary='Analisar prova de vida por frames',
    description='Recebe frames (imagens) e valida sinais de liveness como piscar, distancia, boca aberta e pose de cabeca.',
)
async def analyze_liveness(
    frames: list[UploadFile] = File(...),
    required_checks: str = Form('head_left,mouth_open'),
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
        if parsed_session.signature or parsed_session.session_id:
            _verify_session(parsed_session.model_dump())

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
