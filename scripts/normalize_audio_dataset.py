#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normaliza audios de um manifesto para WAV PCM mono 16 kHz preservando metadados."
    )
    parser.add_argument("--manifest", required=True, help="Manifesto de entrada.")
    parser.add_argument("--output-root", required=True, help="Raiz de saida dos WAV normalizados.")
    parser.add_argument("--output-manifest", required=True, help="Manifesto de saida.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clean", action="store_true", help="Remove arquivos de saida existentes no mesmo caminho.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_waveform(audio_path: Path, sample_rate: int) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(str(audio_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = T.Resample(orig_freq=source_rate, new_freq=sample_rate)(waveform)
    peak = waveform.abs().max()
    if peak > 1.0:
        waveform = waveform / peak
    return waveform.clamp(min=-1.0, max=1.0)


def target_relative_path(row: dict) -> Path:
    relative = Path(row["relative_path"])
    parts = list(relative.parts)
    stem = relative.stem
    if stem.endswith(".wav"):
        stem = stem[:-4]
    parts[-1] = f"{stem}.wav"
    return Path(*parts)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_manifest = Path(args.output_manifest).expanduser().resolve()

    rows = load_rows(manifest_path)
    output_rows = []
    output_root.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        source_path = Path(row["audio_path"]).expanduser().resolve()
        relative_path = target_relative_path(row)
        target_path = output_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if args.clean and target_path.exists():
            target_path.unlink()
        if not target_path.exists():
            waveform = normalize_waveform(source_path, args.sample_rate)
            torchaudio.save(
                str(target_path),
                waveform,
                args.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )

        output_row = dict(row)
        output_row["audio_path"] = str(target_path)
        output_row["relative_path"] = str(relative_path)
        output_row["filename"] = target_path.name
        output_row["extension"] = ".wav"
        output_row["source"] = f"{row.get('source') or 'audio'}_normalized"
        output_rows.append(output_row)

        if index % 100 == 0:
            print(f"normalizados: {index}/{len(rows)}", flush=True)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys())
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"manifesto normalizado salvo em: {output_manifest}")
    print(f"audios normalizados em: {output_root}")
    print(f"linhas: {len(output_rows)}")


if __name__ == "__main__":
    main()
