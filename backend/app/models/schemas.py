from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class JobStage(str, Enum):
    PENDING = "pending"
    VALIDATING_IMAGE = "validating_image"
    DETECTING_FACE = "detecting_face"
    QUALITY_CHECKS = "quality_checks"
    PREPROCESSING = "preprocessing"
    MODEL_INFERENCE = "model_inference"
    CALIBRATING_SCORE = "calibrating_score"
    GENERATING_EVIDENCE = "generating_evidence"
    DONE = "done"
    ERROR = "error"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNDETERMINED = "undetermined"


class StepState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class StepInfo(BaseModel):
    id: str
    label: str
    state: StepState


class PreviewInfo(BaseModel):
    bbox: Optional[Dict[str, float]] = None  # Porcentagens relativas
    face_crop_url: Optional[str] = None


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    stage: JobStage
    progress: int
    message: str
    steps: List[StepInfo]
    preview: Optional[PreviewInfo] = None


class Flags(BaseModel):
    no_face: bool = False
    multiple_faces: bool = False
    blur_high: bool = False
    low_light: bool = False
    face_too_small: bool = False


class Evidence(BaseModel):
    face_crop_url: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None  # Porcentagens relativas


class ResultResponse(BaseModel):
    job_id: str
    p_fake: Optional[float] = None
    risk_level: RiskLevel
    flags: Flags
    evidence: Evidence
    model_version: str
    created_at: datetime
    processing_time_ms: Optional[int] = None
    video_summary: Optional[Dict[str, Any]] = None
    frame_results: Optional[List[Dict[str, Any]]] = None


class ErrorResponse(BaseModel):
    detail: str
    job_id: Optional[str] = None


class LivenessChallengeDefinition(BaseModel):
    id: str
    label: str
    requirements: List[str] = Field(default_factory=list)
    targets: Dict[str, float] = Field(default_factory=dict)
    hint: Optional[str] = None


class LivenessSessionDefinition(BaseModel):
    session_id: Optional[str] = None
    detector_mode: Optional[str] = None
    challenges: List[LivenessChallengeDefinition] = Field(default_factory=list)


class LivenessFrameMetrics(BaseModel):
    frame_index: int
    face_detected: bool
    ear: Optional[float] = None
    face_width: Optional[float] = None
    smile_ratio: Optional[float] = None
    mouth_open_ratio: Optional[float] = None
    delta_ear: Optional[float] = None
    delta_face_width: Optional[float] = None
    delta_smile_ratio: Optional[float] = None
    delta_mouth_open_ratio: Optional[float] = None
    movement_score: Optional[float] = None


class LivenessChallengeResult(BaseModel):
    id: str
    label: str
    requirements: List[str] = Field(default_factory=list)
    passed: bool
    checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class LivenessAnalyzeResponse(BaseModel):
    passed: bool
    checks: Dict[str, Dict[str, Any]]
    required_checks: Optional[List[str]] = None
    summary: Dict[str, Any]
    session: Optional[LivenessSessionDefinition] = None
    challenge_results: List[LivenessChallengeResult] = Field(default_factory=list)
    frame_metrics: List[LivenessFrameMetrics] = Field(default_factory=list)
