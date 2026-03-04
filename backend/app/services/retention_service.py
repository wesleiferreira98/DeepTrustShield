from __future__ import annotations

from datetime import datetime, timedelta
import os
import shutil
from typing import Any

from app.config import get_settings
from app.db.database import SessionLocal
from app.db.models import Artifact, Job, Result


def cleanup_expired_data() -> dict[str, Any]:
    settings = get_settings()
    cutoff = datetime.utcnow() - timedelta(hours=settings.artifacts_retention_hours)
    cutoff_ts = cutoff.timestamp()
    artifacts_root = settings.artifacts_path

    session = SessionLocal()
    removed_jobs = 0
    removed_dirs = 0

    try:
        expired_jobs = session.query(Job).filter(Job.created_at < cutoff).all()
        expired_ids = [job.id for job in expired_jobs]

        for job_id in expired_ids:
            session.query(Artifact).filter(Artifact.job_id == job_id).delete()
            session.query(Result).filter(Result.job_id == job_id).delete()
            session.query(Job).filter(Job.id == job_id).delete()
            removed_jobs += 1

        session.commit()

        active_ids = {row[0] for row in session.query(Job.id).all()}
    finally:
        session.close()

    if os.path.isdir(artifacts_root):
        for entry in os.scandir(artifacts_root):
            if not entry.is_dir():
                continue

            should_remove = False
            if entry.name not in active_ids:
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime <= cutoff_ts:
                    should_remove = True

            if should_remove:
                try:
                    shutil.rmtree(entry.path, ignore_errors=True)
                    removed_dirs += 1
                except OSError:
                    continue

    return {
        "removed_jobs": removed_jobs,
        "removed_dirs": removed_dirs,
        "cutoff_utc": cutoff.isoformat(),
    }
