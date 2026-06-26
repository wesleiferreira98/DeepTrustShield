#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "audio" / "data" / "coleta"
DEFAULT_OUTPUT = REPO_ROOT / "audio" / "data" / "manifests" / "coleta_manifest.csv"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera manifesto dos audios reais coletados com consentimento.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--split", default="test")
    parser.add_argument("--source", default="coleta_propria")
    parser.add_argument("--domain", default="coleta_celular")
    parser.add_argument("--accent", default="unknown")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def infer_accent(path: Path, default: str) -> str:
    name = path.stem.lower()
    if "salvador" in name or "bahia" in name:
        return "baiano"
    return default


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    fieldnames = [
        "audio_path",
        "relative_path",
        "filename",
        "extension",
        "split",
        "label",
        "source",
        "domain",
        "accent",
        "speaker_id",
        "manipulation_type",
    ]
    rows = []
    for path in audio_paths:
        rows.append(
            {
                "audio_path": os.path.relpath(path, output_path.parent),
                "relative_path": path.name,
                "filename": path.name,
                "extension": path.suffix.lower(),
                "split": args.split,
                "label": "real",
                "source": args.source,
                "domain": args.domain,
                "accent": infer_accent(path, args.accent),
                "speaker_id": "unknown",
                "manipulation_type": "none",
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows = {len(rows)}")
    print(f"manifest = {output_path}")


if __name__ == "__main__":
    main()
