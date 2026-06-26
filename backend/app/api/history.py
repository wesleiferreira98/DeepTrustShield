from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Job, Result

router = APIRouter()


@router.get("")
def list_history(
    mode: str | None  = Query(None, description="image | audio | video"),
    days: int         = Query(0,    description="0 = todos, 7 = últimos 7 dias, etc."),
    limit: int        = Query(20,   ge=1, le=100),
    offset: int       = Query(0,    ge=0),
    db: Session       = Depends(get_db),
):
    """Return paginated list of completed analyses."""
    query = (
        db.query(Job, Result)
        .outerjoin(Result, Result.job_id == Job.id)
        .filter(Job.stage == "done")
    )

    if mode in ("image", "audio", "video"):
        query = query.filter(Job.mode == mode)

    if days and days > 0:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = query.filter(Job.created_at >= cutoff)

    total = query.count()

    rows = (
        query
        .order_by(Job.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = []
    for job, result in rows:
        processing_ms = None
        if job.updated_at and job.created_at:
            processing_ms = int((job.updated_at - job.created_at).total_seconds() * 1000)

        items.append({
            "job_id":            job.id,
            "mode":              job.mode or "image",
            "risk_level":        result.risk_level if result else "UNDETERMINED",
            "p_fake":            round(result.p_fake, 4) if result and result.p_fake is not None else None,
            "model_version":     result.model_version if result else None,
            "created_at":        job.created_at.isoformat() if job.created_at else None,
            "processing_time_ms": processing_ms,
        })

    return {"total": total, "offset": offset, "limit": limit, "items": items}
