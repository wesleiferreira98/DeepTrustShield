from sqlalchemy import Boolean, Column, String, Float, DateTime, Text, Integer, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    email          = Column(String(255), unique=True, nullable=False, index=True)
    name           = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active      = Column(Boolean, default=True)
    is_admin       = Column(Boolean, default=False)
    created_at     = Column(DateTime, default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)
    mode = Column(String(10), default="image", nullable=True)
    stage = Column(String(50), default="pending")
    progress = Column(Integer, default=0)
    original_path = Column(Text)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    result = relationship("Result", back_populates="job", uselist=False)
    artifact = relationship("Artifact", back_populates="job", uselist=False)


class Result(Base):
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), unique=True)
    p_fake = Column(Float, nullable=True)
    risk_level = Column(String(20))
    flags_json = Column(Text)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    job = relationship("Job", back_populates="result")


class Artifact(Base):
    __tablename__ = "artifacts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), unique=True)
    face_crop_path = Column(Text, nullable=True)
    bbox_json = Column(Text, nullable=True)
    original_meta = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    job = relationship("Job", back_populates="artifact")
