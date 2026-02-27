from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.config import get_settings
from app.api import deepfake, deepfake_video
from app.db.database import engine, Base

# Create tables
Base.metadata.create_all(bind=engine)

settings = get_settings()

app = FastAPI(
    title="Detectar API",
    description="API para detecção de deepfake em imagens",
    version="1.0.0"
)

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


@app.get("/")
async def root():
    return {
        "name": "Detectar API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
