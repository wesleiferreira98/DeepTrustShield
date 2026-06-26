from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Job, Result

router = APIRouter()


def _compute_roc_auc(scores_labels: list[tuple[float, bool]]):
    """
    Compute ROC curve points and AUC from (p_fake, is_positive) pairs.
    Positive class = HIGH risk; Negative class = LOW risk.
    Returns (auc: float | None, points: list[{fpr, tpr}] | None).
    """
    positives = sum(1 for _, p in scores_labels if p)
    negatives  = len(scores_labels) - positives
    if positives == 0 or negatives == 0 or len(scores_labels) < 2:
        return None, None

    sorted_sl = sorted(scores_labels, key=lambda x: -x[0])
    tp = fp = 0
    fprs, tprs = [0.0], [0.0]

    for _, is_pos in sorted_sl:
        if is_pos:
            tp += 1
        else:
            fp += 1
        fprs.append(fp / negatives)
        tprs.append(tp / positives)

    fprs.append(1.0)
    tprs.append(1.0)

    auc = sum(
        (fprs[i + 1] - fprs[i]) * (tprs[i + 1] + tprs[i]) / 2
        for i in range(len(fprs) - 1)
    )

    step = max(1, len(fprs) // 60)
    pts  = [{"fpr": round(fprs[i], 4), "tpr": round(tprs[i], 4)}
            for i in range(0, len(fprs), step)]
    if pts[-1] != {"fpr": 1.0, "tpr": 1.0}:
        pts.append({"fpr": 1.0, "tpr": 1.0})

    return round(auc, 4), pts


@router.get("/summary")
def get_stats_summary(db: Session = Depends(get_db)):
    """Return aggregate statistics for the dashboard."""

    total = db.query(func.count(Job.id)).scalar() or 0

    mode_rows = (
        db.query(Job.mode, func.count(Job.id))
        .group_by(Job.mode)
        .all()
    )
    by_mode = {"image": 0, "audio": 0, "video": 0}
    for mode, count in mode_rows:
        key = mode or "image"
        if key in by_mode:
            by_mode[key] = count

    risk_rows = (
        db.query(Result.risk_level, func.count(Result.id))
        .group_by(Result.risk_level)
        .all()
    )
    by_risk = {}
    for risk, count in risk_rows:
        by_risk[risk or "UNDETERMINED"] = count

    fake_count = by_risk.get("HIGH", 0)
    real_count = by_risk.get("LOW",  0)

    cutoff = datetime.utcnow() - timedelta(days=14)
    trend_rows = (
        db.query(
            func.date(Job.created_at).label("day"),
            func.count(Job.id).label("total"),
            func.sum(case((Result.risk_level == "HIGH", 1), else_=0)).label("fake_count"),
        )
        .outerjoin(Result, Result.job_id == Job.id)
        .filter(Job.created_at >= cutoff)
        .group_by(func.date(Job.created_at))
        .order_by(func.date(Job.created_at))
        .all()
    )
    trend = [
        {"date": str(row.day), "count": row.total, "fake_count": int(row.fake_count or 0)}
        for row in trend_rows
    ]

    recent_rows = (
        db.query(Job, Result)
        .outerjoin(Result, Result.job_id == Job.id)
        .filter(Job.stage == "done")
        .order_by(Job.created_at.desc())
        .limit(20)
        .all()
    )
    recent = [
        {
            "job_id":        job.id,
            "mode":          job.mode or "image",
            "risk_level":    result.risk_level if result else "UNDETERMINED",
            "p_fake":        round(result.p_fake, 4) if result and result.p_fake is not None else None,
            "model_version": result.model_version if result else None,
            "created_at":    job.created_at.isoformat() if job.created_at else None,
        }
        for job, result in recent_rows
    ]

    return {
        "total":      total,
        "by_mode":    by_mode,
        "by_risk":    by_risk,
        "fake_count": fake_count,
        "real_count": real_count,
        "trend":      trend,
        "recent":     recent,
    }


@router.get("/models")
def get_model_stats(db: Session = Depends(get_db)):
    """Return per-model performance metrics, accumulated from all stored results."""

    rows = (
        db.query(
            Result.model_version,
            Job.mode,
            func.count(Result.id).label("total"),
            func.sum(case((Result.risk_level == "HIGH",   1), else_=0)).label("fake_count"),
            func.sum(case((Result.risk_level == "LOW",    1), else_=0)).label("real_count"),
            func.sum(case((Result.risk_level == "MEDIUM", 1), else_=0)).label("suspicious_count"),
            func.avg(Result.p_fake).label("avg_p_fake"),
            func.avg(
                (func.julianday(Job.updated_at) - func.julianday(Job.created_at)) * 86400000
            ).label("avg_processing_ms"),
            func.min(Job.created_at).label("first_used"),
            func.max(Job.created_at).label("last_used"),
        )
        .join(Job, Job.id == Result.job_id)
        .filter(Result.model_version.isnot(None), Job.stage == "done")
        .group_by(Result.model_version, Job.mode)
        .order_by(func.count(Result.id).desc())
        .all()
    )

    models = []
    for row in rows:
        total        = row.total or 0
        fake         = int(row.fake_count or 0)
        real         = int(row.real_count or 0)
        suspicious   = int(row.suspicious_count or 0)
        undetermined = max(0, total - fake - real - suspicious)

        # ROC: HIGH (positive) vs MEDIUM+LOW (negative) — UNDETERMINED excluded
        score_rows = (
            db.query(Result.p_fake, Result.risk_level)
            .join(Job, Job.id == Result.job_id)
            .filter(
                Result.model_version == row.model_version,
                Job.mode == (row.mode or "image"),
                Result.risk_level.in_(["HIGH", "MEDIUM", "LOW"]),
                Result.p_fake.isnot(None),
            )
            .all()
        )
        scores_labels = [(r.p_fake, r.risk_level == "HIGH") for r in score_rows]
        auc_val, roc_pts = _compute_roc_auc(scores_labels)

        # Score distribution: 10 equal-width buckets across [0, 1]
        all_scores = (
            db.query(Result.p_fake, Result.risk_level)
            .join(Job, Job.id == Result.job_id)
            .filter(
                Result.model_version == row.model_version,
                Job.mode == (row.mode or "image"),
                Result.p_fake.isnot(None),
            )
            .all()
        )
        dist = [{"label": f"{i/10:.1f}–{(i+1)/10:.1f}", "count": 0,
                 "high": 0, "medium": 0, "low": 0, "undetermined": 0}
                for i in range(10)]
        for score, risk in all_scores:
            b = min(9, int(score * 10))
            dist[b]["count"] += 1
            key = (risk or "undetermined").lower()
            if key in dist[b]:
                dist[b][key] += 1

        roc_high = sum(1 for _, pos in scores_labels if pos)
        roc_neg  = len(scores_labels) - roc_high

        models.append({
            "model_version":     row.model_version,
            "mode":              row.mode or "image",
            "total":             total,
            "fake_count":        fake,
            "real_count":        real,
            "suspicious_count":  suspicious,
            "detection_rate":    round(fake / total, 4) if total > 0 else 0,
            "avg_p_fake":        round(float(row.avg_p_fake), 4) if row.avg_p_fake is not None else None,
            "avg_processing_ms": int(row.avg_processing_ms) if row.avg_processing_ms else None,
            "first_used":        row.first_used.isoformat() if row.first_used else None,
            "last_used":         row.last_used.isoformat()  if row.last_used  else None,
            "risk_distribution": {
                "HIGH":         round(fake         / total * 100, 1) if total > 0 else 0,
                "MEDIUM":       round(suspicious   / total * 100, 1) if total > 0 else 0,
                "LOW":          round(real         / total * 100, 1) if total > 0 else 0,
                "UNDETERMINED": round(undetermined / total * 100, 1) if total > 0 else 0,
            },
            "auc":                auc_val,
            "roc_curve":          roc_pts,
            "roc_high_count":     roc_high,
            "roc_neg_count":      roc_neg,
            "score_distribution": dist,
        })

    return {"models": models}
