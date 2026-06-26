#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Equaliza artefatos de codec: reencoda todos os audios pelo mesmo MP3 "
            "intermediario e decodifica para WAV mono 16 kHz."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--bitrate", default="96k")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def target_relative_path(row: dict) -> Path:
    relative = Path(row["relative_path"])
    stem = relative.stem
    if stem.endswith(".wav"):
        stem = stem[:-4]
    return relative.with_name(f"{stem}.wav")


def run_ffmpeg(command: list[str]) -> None:
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def equalize_file(source_path: Path, target_path: Path, sample_rate: int, bitrate: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mp3 = Path(tmpdir) / "audio.mp3"
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                bitrate,
                str(tmp_mp3),
            ]
        )
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(tmp_mp3),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(target_path),
            ]
        )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    rows = load_rows(manifest_path)

    output_rows = []
    for index, row in enumerate(rows, start=1):
        source_path = Path(row["audio_path"]).expanduser().resolve()
        relative_path = target_relative_path(row)
        target_path = output_root / relative_path
        if args.clean and target_path.exists():
            target_path.unlink()
        if not target_path.exists():
            equalize_file(source_path, target_path, args.sample_rate, args.bitrate)

        output_row = dict(row)
        output_row["audio_path"] = str(target_path)
        output_row["relative_path"] = str(relative_path)
        output_row["filename"] = target_path.name
        output_row["extension"] = ".wav"
        output_row["source"] = f"{row.get('source') or 'audio'}_codec_equalized"
        output_row["codec_equalized_bitrate"] = args.bitrate
        output_rows.append(output_row)

        if index % 100 == 0:
            print(f"equalizados: {index}/{len(rows)}", flush=True)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys())
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"manifesto equalizado salvo em: {output_manifest}")
    print(f"audios equalizados em: {output_root}")
    print(f"linhas: {len(output_rows)}")


if __name__ == "__main__":
    main()
