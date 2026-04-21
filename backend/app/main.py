import asyncio
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import os
import time
from redis import Redis

from app.config import get_settings
from app.api import deepfake, deepfake_audio, deepfake_video, liveness
from app.db.database import engine, Base
from app.services.retention_service import cleanup_expired_data

# Create tables
Base.metadata.create_all(bind=engine)

settings = get_settings()

app = FastAPI(
    title="Detectar API",
    description=(
        "API para deteccao de deepfake em imagem/video curto e validacao de prova de vida.\n\n"
        "Documentacao interativa:\n"
        "- Swagger UI: /docs (ou /swagger)\n"
        "- ReDoc: /redoc\n"
        "- OpenAPI JSON: /openapi.json"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "deepfake", "description": "Analise de imagens (foto)."},
        {"name": "deepfake-video", "description": "Analise de videos curtos via frames."},
        {"name": "deepfake-audio", "description": "Analise de audio contra clonagem/deepfake."},
        {"name": "liveness", "description": "Validacao de prova de vida por landmarks faciais."},
    ],
)

RATE_LIMITED_PATHS = {
    "/v1/deepfake/image/analyze",
    "/v1/deepfake/video/analyze",
    "/v1/deepfake/audio/analyze",
    "/v1/liveness/session",
    "/v1/liveness/analyze",
}
ANALYZE_RATE_LIMIT_PER_MINUTE = 20
redis_rate_conn = Redis.from_url(settings.redis_url)
_cleanup_task = None


def _get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        forwarded = xff.split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def _consume_rate_limit(path: str, ip: str) -> tuple[bool, int]:
    minute_bucket = int(time.time() // 60)
    key = f"ratelimit:analyze:{path}:{ip}:{minute_bucket}"
    pipeline = redis_rate_conn.pipeline()
    pipeline.incr(key, 1)
    pipeline.expire(key, 120)
    result = pipeline.execute()
    current = int(result[0])
    limited = current > ANALYZE_RATE_LIMIT_PER_MINUTE
    return limited, current


@app.middleware("http")
async def analyze_rate_limit_middleware(request: Request, call_next):
    if request.method == "POST" and request.url.path in RATE_LIMITED_PATHS:
        ip = _get_client_ip(request)
        try:
            limited, _current = await asyncio.to_thread(_consume_rate_limit, request.url.path, ip)
        except Exception:
            limited = False

        if limited:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Muitas requisicoes. Tente novamente em alguns segundos.",
                    "limit_per_minute": ANALYZE_RATE_LIMIT_PER_MINUTE,
                },
            )

    return await call_next(request)

# CORS
origins = settings.cors_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount artifacts directory
os.makedirs(settings.artifacts_path, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=settings.artifacts_path), name="artifacts")

# Include routers
app.include_router(deepfake.router, prefix="/v1/deepfake/image", tags=["deepfake"])
app.include_router(deepfake_video.router, prefix="/v1/deepfake/video", tags=["deepfake-video"])
app.include_router(deepfake_audio.router, prefix="/v1/deepfake/audio", tags=["deepfake-audio"])
app.include_router(liveness.router, prefix="/v1/liveness", tags=["liveness"])


@app.on_event("startup")
async def startup_security_tasks():
    global _cleanup_task

    await asyncio.to_thread(cleanup_expired_data)

    async def _periodic_cleanup():
        interval_seconds = max(60, settings.cleanup_interval_minutes * 60)
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await asyncio.to_thread(cleanup_expired_data)
            except Exception:
                continue

    _cleanup_task = asyncio.create_task(_periodic_cleanup())


@app.on_event("shutdown")
async def shutdown_security_tasks():
    global _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        _cleanup_task = None


@app.get("/")
async def root():
    return {
        "name": "Detectar API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/swagger", include_in_schema=False)
async def swagger_alias():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {"status": "healthy"}
