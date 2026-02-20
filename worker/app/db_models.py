from sqlalchemy import Column, String, Float, DateTime, Text, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"
    
    id = Column(String(36), primary_key=True)
    stage = Column(String(50), default="pending")
    progress = Column(Integer, default=0)
    original_path = Column(Text)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Result(Base):
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), unique=True)
    p_fake = Column(Float, nullable=True)
    risk_level = Column(String(20))
    flags_json = Column(Text)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class Artifact(Base):
    __tablename__ = "artifacts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), unique=True)
    face_crop_path = Column(Text, nullable=True)
    bbox_json = Column(Text, nullable=True)
    original_meta = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
