#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio.transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from train_audio_deepfake import (
    AudioCNN,
    AudioDeepfakeDataset,
    collect_predictions,
    count_by_field,
    count_labels,
    find_best_threshold,
    load_manifest,
    metrics_by_group,
    metrics_from_probs,
    resolve_device,
    resolve_pos_weight,
    sample_audio_info,
    set_seed,
    train_one_epoch,
    write_history_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "audio" / "data" / "manifests" / "fake_braccent_d_manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "audio" / "models" / "audio_deepfake_v1_fake_braccent_d_cv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina audio real/fake com StratifiedGroupKFold por label+accent e speaker_id."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pos-weight", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None, help="Limita folds para smoke tests.")
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="Controle negativo: embaralha labels antes de gerar folds/treinar.",
    )
    return parser.parse_args()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def stratification_key(row: dict) -> str:
    return f"{row['label_text']}__{row.get('accent') or 'unknown'}"


def shuffle_labels(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    labels = [(row["label"], row["label_text"]) for row in rows]
    rng.shuffle(labels)
    shuffled = []
    for row, (label, label_text) in zip(rows, labels):
        new_row = dict(row)
        new_row["label"] = label
        new_row["label_text"] = label_text
        new_row["label_shuffled"] = True
        shuffled.append(new_row)
    return shuffled


def build_fold_indices(rows: list[dict], folds: int, seed: int):
    y = [stratification_key(row) for row in rows]
    groups = [row.get("speaker_id") or "unknown" for row in rows]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(rows, y, groups))


def make_loader(
    rows: list[dict],
    *,
    sample_rate: int,
    clip_seconds: float,
    n_mels: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    random_crop: bool,
    shuffle: bool,
) -> DataLoader:
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=n_mels,
    )
    amplitude_to_db = T.AmplitudeToDB()
    dataset = AudioDeepfakeDataset(
        rows,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        mel_transform=mel_transform,
        amplitude_to_db=amplitude_to_db,
        random_crop=random_crop,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def train_fold(
    *,
    fold_index: int,
    train_rows: list[dict],
    val_rows: list[dict],
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> dict:
    fold_dir = resolve_repo_path(args.output_dir) / f"fold_{fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    num_workers = args.num_workers if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_rows,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        n_mels=args.n_mels,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        random_crop=True,
        shuffle=True,
    )
    val_loader = make_loader(
        val_rows,
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        n_mels=args.n_mels,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        random_crop=False,
        shuffle=False,
    )

    model = AudioCNN(args.n_mels).to(device)
    pos_weight_value = resolve_pos_weight(str(args.pos_weight), train_rows)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint_path = fold_dir / "audio_deepfake_v1_best.pth"
    history_rows = []
    best_val_auc = -1.0
    best_epoch = 0
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_predictions = collect_predictions(model, val_loader, criterion, device, use_amp=use_amp)
        val_metrics = {
            "loss": float(val_predictions["loss"]),
            "auc": float(val_predictions["auc"]),
            "f1": float(val_predictions["f1"]),
            "acc": float(val_predictions["acc"]),
        }

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_auc": round(train_metrics["auc"], 6),
                "train_f1": round(train_metrics["f1"], 6),
                "train_acc": round(train_metrics["acc"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_auc": round(val_metrics["auc"], 6),
                "val_f1": round(val_metrics["f1"], 6),
                "val_acc": round(val_metrics["acc"], 6),
            }
        )

        improved = val_metrics["auc"] > best_val_auc
        if improved:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_auc": best_val_auc,
                    "best_epoch": best_epoch,
                    "fold": fold_index,
                },
                checkpoint_path,
            )

        marker = "*" if improved else ""
        print(
            f"fold={fold_index} epoch={epoch:02d}/{args.epochs} "
            f"train_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_acc={val_metrics['acc']:.4f} {marker}",
            flush=True,
        )

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        best_epoch = int(state.get("best_epoch", best_epoch))
        best_val_auc = float(state.get("best_val_auc", best_val_auc))

    val_predictions = collect_predictions(model, val_loader, criterion, device, use_amp=use_amp)
    threshold_calibration = find_best_threshold(
        val_predictions["labels"],
        val_predictions["probs"],
        metric="f1",
    )
    threshold = threshold_calibration["threshold"]
    val_metrics_calibrated = {
        "loss": float(val_predictions["loss"]),
        **metrics_from_probs(val_predictions["labels"], val_predictions["probs"], threshold=threshold),
        "threshold": threshold,
    }
    val_metrics_by_accent = metrics_by_group(
        val_rows,
        val_predictions["labels"],
        val_predictions["probs"],
        group_field="accent",
        threshold=threshold,
    )
    val_metrics_by_source = metrics_by_group(
        val_rows,
        val_predictions["labels"],
        val_predictions["probs"],
        group_field="source",
        threshold=threshold,
    )

    history_path = fold_dir / "history.csv"
    write_history_csv(history_path, history_rows)

    summary = {
        "task": "audio_deepfake_detection_cv",
        "model_version": "audio_deepfake_v1",
        "fold": fold_index,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
        "sample_rate": args.sample_rate,
        "clip_seconds": float(args.clip_seconds),
        "n_mels": args.n_mels,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "pos_weight": pos_weight_value,
        "seed": args.seed,
        "device": str(device),
        "amp": use_amp,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "train_labels": count_labels(train_rows),
        "val_labels": count_labels(val_rows),
        "train_accents": count_by_field(train_rows, "accent"),
        "val_accents": count_by_field(val_rows, "accent"),
        "train_speakers": count_by_field(train_rows, "speaker_id"),
        "val_speakers": count_by_field(val_rows, "speaker_id"),
        "sample_audio_info": sample_audio_info(train_rows),
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "threshold_calibration": threshold_calibration,
        "val_metrics_calibrated": val_metrics_calibrated,
        "val_metrics_by_accent": val_metrics_by_accent,
        "val_metrics_by_source": val_metrics_by_source,
        "elapsed_minutes": round((time.time() - started_at) / 60.0, 3),
    }
    summary_path = fold_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_cv_summary(output_dir: Path, fold_summaries: list[dict]) -> None:
    rows = []
    for summary in fold_summaries:
        calibrated = summary["val_metrics_calibrated"]
        rows.append(
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                "best_val_auc": summary["best_val_auc"],
                "threshold": summary["threshold_calibration"]["threshold"],
                "val_auc_cal": calibrated["auc"],
                "val_f1_cal": calibrated["f1"],
                "val_acc_cal": calibrated["acc"],
                "train_count": summary["train_count"],
                "val_count": summary["val_count"],
                "elapsed_minutes": summary["elapsed_minutes"],
            }
        )

    csv_path = output_dir / "cv_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {
        "folds": len(fold_summaries),
        "mean_best_val_auc": sum(row["best_val_auc"] for row in rows) / len(rows),
        "mean_val_f1_cal": sum(row["val_f1_cal"] for row in rows) / len(rows),
        "mean_val_acc_cal": sum(row["val_acc_cal"] for row in rows) / len(rows),
        "mean_threshold": sum(row["threshold"] for row in rows) / len(rows),
        "folds_detail": rows,
    }
    (output_dir / "cv_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    manifest_path = resolve_repo_path(args.manifest)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifesto sem linhas validas.")
    if args.shuffle_labels:
        rows = shuffle_labels(rows, args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = bool(args.amp and device.type == "cuda")

    print(f"device = {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu = {torch.cuda.get_device_name(0)}", flush=True)
        print(f"amp = {use_amp}", flush=True)
    print(f"manifest = {manifest_path}", flush=True)
    print(f"shuffle_labels = {args.shuffle_labels}", flush=True)
    print(f"rows = {len(rows)} {count_labels(rows)}", flush=True)
    print(f"accents = {count_by_field(rows, 'accent')}", flush=True)
    print(f"speakers = {len(count_by_field(rows, 'speaker_id'))}", flush=True)

    folds = build_fold_indices(rows, args.folds, args.seed)
    if args.max_folds:
        folds = folds[: args.max_folds]

    fold_summaries = []
    for fold_index, (train_index, val_index) in enumerate(folds):
        train_rows = [rows[index] for index in train_index]
        val_rows = [rows[index] for index in val_index]
        train_speakers = set(row["speaker_id"] for row in train_rows)
        val_speakers = set(row["speaker_id"] for row in val_rows)
        overlap = train_speakers & val_speakers
        if overlap:
            raise RuntimeError(f"Vazamento de speaker_id no fold {fold_index}: {sorted(overlap)[:5]}")

        print(
            f"fold={fold_index} train={len(train_rows)} {count_labels(train_rows)} "
            f"val={len(val_rows)} {count_labels(val_rows)}",
            flush=True,
        )
        fold_summaries.append(
            train_fold(
                fold_index=fold_index,
                train_rows=train_rows,
                val_rows=val_rows,
                args=args,
                device=device,
                use_amp=use_amp,
            )
        )

    write_cv_summary(output_dir, fold_summaries)
    control_path = output_dir / "label_shuffle_control.json"
    control_path.write_text(
        json.dumps(
            {
                "shuffle_labels": args.shuffle_labels,
                "expected_auc": "approximately 0.5" if args.shuffle_labels else None,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"cv_summary = {output_dir / 'cv_summary.csv'}", flush=True)
    print(f"cv_summary_json = {output_dir / 'cv_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
