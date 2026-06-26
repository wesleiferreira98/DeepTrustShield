#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera manifesto supervisionado por janela usando ground truth temporal de audios manipulados."
    )
    parser.add_argument("--real-input", default="audio/data/eval_external", help="Arquivo ou pasta com audios reais.")
    parser.add_argument("--manipulated-root", default="audio/data/eval_manipulated", help="Pasta com ground_truth/*.json.")
    parser.add_argument("--output", default="audio/data/manifests/audio_window_manifest.csv")
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--fake-overlap-threshold", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_audio(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in AUDIO_SUFFIXES:
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    return []


def audio_duration_seconds(path: Path) -> float:
    waveform, sample_rate = torchaudio.load(str(path))
    return waveform.size(-1) / float(sample_rate) if sample_rate else 0.0


def interval_overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def segment_starts(duration: float, window: float, stride: float) -> list[float]:
    if duration <= window:
        return [0.0]
    starts = []
    current = 0.0
    while current <= duration - window + 1e-6:
        starts.append(round(current, 6))
        current += stride
    final_start = max(0.0, duration - window)
    if not starts or abs(starts[-1] - final_start) > 1e-4:
        starts.append(round(final_start, 6))
    return starts


def split_for_key(key: str, buckets: dict[str, str]) -> str:
    return buckets.get(key, "train")


def make_split_buckets(keys: list[str], val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    unique = sorted(set(keys))
    rng = random.Random(seed)
    rng.shuffle(unique)
    test_count = int(round(len(unique) * test_ratio))
    val_count = int(round(len(unique) * val_ratio))
    test_keys = set(unique[:test_count])
    val_keys = set(unique[test_count:test_count + val_count])
    buckets = {}
    for key in unique:
        if key in test_keys:
            buckets[key] = "test"
        elif key in val_keys:
            buckets[key] = "val"
        else:
            buckets[key] = "train"
    return buckets


def load_ground_truth_items(root: Path) -> list[dict]:
    items = []
    if not root.exists():
        return items
    for path in sorted(root.rglob("*.ground_truth.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        output_audio = Path(data.get("output_audio") or "").expanduser()
        if not output_audio.is_absolute():
            output_audio = (path.parent / output_audio).resolve()
        if not output_audio.exists():
            continue
        items.append({"ground_truth_path": path, "data": data, "audio_path": output_audio})
    return items


def rows_for_audio(
    *,
    audio_path: Path,
    duration: float,
    split: str,
    file_label: str,
    source_audio: str,
    manipulation_type: str,
    ground_truth_path: str,
    manipulated_segments: list[dict],
    window_seconds: float,
    stride_seconds: float,
    fake_overlap_threshold: float,
) -> list[dict]:
    rows = []
    for start in segment_starts(duration, window_seconds, stride_seconds):
        end = min(duration, start + window_seconds)
        overlap_seconds = sum(
            interval_overlap(start, end, float(segment.get("start") or 0.0), float(segment.get("end") or 0.0))
            for segment in manipulated_segments
        )
        overlap_ratio = overlap_seconds / max(end - start, 0.001)
        label = "fake" if overlap_ratio >= fake_overlap_threshold else "real"
        rows.append(
            {
                "audio_path": str(audio_path),
                "split": split,
                "label": label,
                "file_label": file_label,
                "start": f"{start:.6f}",
                "end": f"{end:.6f}",
                "overlap_ratio": f"{overlap_ratio:.6f}",
                "manipulation_type": manipulation_type,
                "source_audio": source_audio,
                "source": manipulation_type or "real",
                "accent": "unknown",
                "speaker_id": source_audio or str(audio_path),
                "ground_truth_path": ground_truth_path,
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "audio_path",
        "split",
        "label",
        "file_label",
        "start",
        "end",
        "overlap_ratio",
        "manipulation_type",
        "source_audio",
        "source",
        "accent",
        "speaker_id",
        "ground_truth_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    real_input = resolve_path(args.real_input)
    manipulated_root = resolve_path(args.manipulated_root)
    output_path = resolve_path(args.output)

    real_paths = collect_audio(real_input)
    gt_items = load_ground_truth_items(manipulated_root)

    split_keys = [file_sha256(path) for path in real_paths]
    split_keys.extend(str(item["data"].get("source_audio_sha256") or item["data"].get("source_audio") or item["audio_path"]) for item in gt_items)
    split_buckets = make_split_buckets(split_keys, args.val_ratio, args.test_ratio, args.seed)

    rows = []
    for audio_path in real_paths:
        key = file_sha256(audio_path)
        duration = audio_duration_seconds(audio_path)
        rows.extend(
            rows_for_audio(
                audio_path=audio_path,
                duration=duration,
                split=split_for_key(key, split_buckets),
                file_label="real",
                source_audio=str(audio_path),
                manipulation_type="none",
                ground_truth_path="",
                manipulated_segments=[],
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                fake_overlap_threshold=args.fake_overlap_threshold,
            )
        )

    for item in gt_items:
        data = item["data"]
        audio_path = item["audio_path"]
        key = str(data.get("source_audio_sha256") or data.get("source_audio") or audio_path)
        duration = float(data.get("output_duration_seconds") or audio_duration_seconds(audio_path))
        rows.extend(
            rows_for_audio(
                audio_path=audio_path,
                duration=duration,
                split=split_for_key(key, split_buckets),
                file_label="fake",
                source_audio=str(data.get("source_audio") or ""),
                manipulation_type=str(data.get("manipulation_type") or "unknown"),
                ground_truth_path=str(item["ground_truth_path"]),
                manipulated_segments=data.get("manipulated_segments") or [],
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                fake_overlap_threshold=args.fake_overlap_threshold,
            )
        )

    write_manifest(output_path, rows)
    labels = {"real": 0, "fake": 0}
    splits = {}
    for row in rows:
        labels[row["label"]] = labels.get(row["label"], 0) + 1
        split = row["split"]
        bucket = splits.setdefault(split, {"real": 0, "fake": 0})
        bucket[row["label"]] = bucket.get(row["label"], 0) + 1

    print(f"real_audios = {len(real_paths)}")
    print(f"manipulated_audios = {len(gt_items)}")
    print(f"rows = {len(rows)} {labels}")
    print(f"splits = {splits}")
    print(f"manifest = {output_path}")


if __name__ == "__main__":
    main()
