#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
DEFAULT_OPERATIONS = [
    "remove",
    "silence",
    "duplicate",
    "splice_same",
    "volume_down",
    "volume_up",
    "pitch_up",
    "pitch_down",
    "tempo_up",
    "tempo_down",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cria audios manipulados controlados para validar a frente de audio. "
            "Cada arquivo gerado recebe um ground_truth.json com o intervalo manipulado."
        )
    )
    parser.add_argument("--source-audio", required=True, help="Audio base consentido ou pasta com audios consentidos.")
    parser.add_argument("--insert-audio", default=None, help="Audio consentido usado em splice_other.")
    parser.add_argument("--output-root", default="audio/data/eval_manipulated", help="Pasta de saida.")
    parser.add_argument("--operations", nargs="+", default=DEFAULT_OPERATIONS, help="Operacoes a gerar.")
    parser.add_argument("--start", type=float, default=None, help="Inicio do trecho manipulado em segundos.")
    parser.add_argument("--duration", type=float, default=4.0, help="Duracao do trecho manipulado em segundos.")
    parser.add_argument("--donor-start", type=float, default=None, help="Inicio do trecho doador em segundos.")
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crossfade-ms", type=float, default=25.0)
    parser.add_argument("--consent-note", default="audios_consentidos_para_pesquisa")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def slugify(value: str) -> str:
    keep = []
    for char in value.lower():
        if char.isalnum():
            keep.append(char)
        elif char in {" ", "-", "_", "."}:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "audio"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_audio(path: Path, target_sample_rate: int) -> tuple[torch.Tensor, int, int]:
    waveform, source_rate = torchaudio.load(str(path))
    source_channels = waveform.size(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != target_sample_rate:
        waveform = T.Resample(orig_freq=source_rate, new_freq=target_sample_rate)(waveform)
    return waveform.squeeze(0).float().contiguous(), target_sample_rate, source_channels


def clamp_audio(waveform: torch.Tensor) -> torch.Tensor:
    peak = waveform.abs().max().item() if waveform.numel() else 0.0
    if peak > 0.98:
        waveform = waveform * (0.98 / peak)
    return waveform.clamp(-0.98, 0.98)


def seconds_to_sample(value: float, sample_rate: int) -> int:
    return max(0, int(round(value * sample_rate)))


def choose_segment(duration_seconds: float, segment_seconds: float, rng: random.Random, start: float | None) -> tuple[float, float]:
    segment_seconds = min(segment_seconds, max(0.25, duration_seconds * 0.8))
    if start is None:
        latest_start = max(0.0, duration_seconds - segment_seconds)
        start = rng.uniform(0.15 * latest_start, 0.85 * latest_start) if latest_start > 0 else 0.0
    start = max(0.0, min(start, max(0.0, duration_seconds - segment_seconds)))
    return start, start + segment_seconds


def slice_audio(waveform: torch.Tensor, sample_rate: int, start: float, end: float) -> torch.Tensor:
    start_sample = seconds_to_sample(start, sample_rate)
    end_sample = min(waveform.numel(), seconds_to_sample(end, sample_rate))
    return waveform[start_sample:end_sample].clone()


def replace_segment(
    waveform: torch.Tensor,
    sample_rate: int,
    start: float,
    end: float,
    replacement: torch.Tensor,
    crossfade_ms: float,
) -> torch.Tensor:
    start_sample = seconds_to_sample(start, sample_rate)
    end_sample = min(waveform.numel(), seconds_to_sample(end, sample_rate))
    before = waveform[:start_sample]
    after = waveform[end_sample:]
    replacement = replacement.float().contiguous()

    crossfade = min(
        int(sample_rate * (crossfade_ms / 1000.0)),
        before.numel(),
        after.numel(),
        replacement.numel() // 3,
    )
    combined = torch.cat([before, replacement, after])
    if crossfade <= 1:
        return combined

    fade_out = torch.linspace(1.0, 0.0, crossfade)
    fade_in = torch.linspace(0.0, 1.0, crossfade)
    left_start = before.numel() - crossfade
    right_start = before.numel() + replacement.numel()
    combined[left_start:before.numel()] = (before[-crossfade:] * fade_out) + (replacement[:crossfade] * fade_in)
    combined[right_start:right_start + crossfade] = (replacement[-crossfade:] * fade_out) + (after[:crossfade] * fade_in)
    return combined


def resample_length(segment: torch.Tensor, factor: float) -> torch.Tensor:
    if segment.numel() < 2:
        return segment
    new_length = max(2, int(round(segment.numel() / factor)))
    resized = F.interpolate(
        segment.view(1, 1, -1),
        size=new_length,
        mode="linear",
        align_corners=False,
    )
    return resized.view(-1)


def pitch_shift(segment: torch.Tensor, sample_rate: int, semitones: float) -> torch.Tensor:
    try:
        return torchaudio.functional.pitch_shift(segment.unsqueeze(0), sample_rate, semitones).squeeze(0)
    except Exception:
        # Fallback: rough pitch-style artifact by resampling and restoring length.
        factor = 2 ** (semitones / 12.0)
        shifted = resample_length(segment, factor)
        restored = F.interpolate(
            shifted.view(1, 1, -1),
            size=segment.numel(),
            mode="linear",
            align_corners=False,
        )
        return restored.view(-1)


def operation_replacement(
    operation: str,
    waveform: torch.Tensor,
    sample_rate: int,
    start: float,
    end: float,
    *,
    donor_segment: torch.Tensor | None,
) -> tuple[torch.Tensor, str]:
    segment = slice_audio(waveform, sample_rate, start, end)
    if operation == "remove":
        return torch.empty(0), "remove_segment"
    if operation == "silence":
        return torch.zeros_like(segment), "replace_with_silence"
    if operation == "duplicate":
        return torch.cat([segment, segment]), "duplicate_segment"
    if operation == "splice_same":
        if donor_segment is None or donor_segment.numel() == 0:
            return segment.flip(0), "splice_same_reversed_fallback"
        return donor_segment, "splice_same_audio"
    if operation == "splice_other":
        if donor_segment is None or donor_segment.numel() == 0:
            raise ValueError("splice_other requer --insert-audio valido")
        return donor_segment, "splice_other_speaker_or_source"
    if operation == "volume_down":
        return segment * 0.45, "local_volume_down"
    if operation == "volume_up":
        return segment * 1.55, "local_volume_up"
    if operation == "pitch_up":
        return pitch_shift(segment, sample_rate, 1.5), "local_pitch_up"
    if operation == "pitch_down":
        return pitch_shift(segment, sample_rate, -1.5), "local_pitch_down"
    if operation == "tempo_up":
        return resample_length(segment, 1.18), "local_tempo_up"
    if operation == "tempo_down":
        return resample_length(segment, 0.86), "local_tempo_down"
    raise ValueError(f"Operacao desconhecida: {operation}")


def collect_existing_audio(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in AUDIO_SUFFIXES:
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    return []


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manipulations_for_source(
    *,
    args: argparse.Namespace,
    rng: random.Random,
    source_path: Path,
    output_audio_dir: Path,
    output_gt_dir: Path,
    insert_path: Path | None,
    insert_audio: torch.Tensor | None,
    operations: list[str],
) -> list[dict]:
    source_audio, sample_rate, source_channels = load_audio(source_path, args.target_sample_rate)
    duration_seconds = source_audio.numel() / float(sample_rate)
    start, end = choose_segment(duration_seconds, args.duration, rng, args.start)

    donor_start, donor_end = choose_segment(
        duration_seconds,
        end - start,
        rng,
        args.donor_start,
    )
    if abs(donor_start - start) < (end - start):
        donor_start = max(0.0, min(duration_seconds - (end - start), end + 1.0))
        donor_end = donor_start + (end - start)
    same_donor_segment = slice_audio(source_audio, sample_rate, donor_start, donor_end)

    insert_donor_segment = None
    insert_start = 0.0
    insert_end = 0.0
    if insert_audio is not None:
        insert_duration = insert_audio.numel() / float(sample_rate)
        insert_start, insert_end = choose_segment(insert_duration, end - start, rng, None)
        insert_donor_segment = slice_audio(insert_audio, sample_rate, insert_start, insert_end)

    source_slug = slugify(source_path.stem)
    manifest_rows = []

    print(f"source = {source_path}")
    print(f"duration = {duration_seconds:.2f}s")
    print(f"target_segment = {start:.2f}s - {end:.2f}s")

    for index, operation in enumerate(operations, start=1):
        donor_segment = insert_donor_segment if operation == "splice_other" else same_donor_segment
        output_stem = f"{source_slug}_{index:02d}_{operation}"
        output_audio_path = output_audio_dir / f"{output_stem}.wav"
        output_gt_path = output_gt_dir / f"{output_stem}.ground_truth.json"

        if output_audio_path.exists() and not args.overwrite:
            print(f"pula existente: {output_audio_path}")
            continue

        try:
            replacement, method = operation_replacement(
                operation,
                source_audio,
                sample_rate,
                start,
                end,
                donor_segment=donor_segment,
            )
        except Exception as exc:
            print(f"falha em {source_path.name} / {operation}: {exc}")
            continue

        manipulated = replace_segment(
            source_audio,
            sample_rate,
            start,
            end,
            replacement,
            args.crossfade_ms,
        )
        manipulated = clamp_audio(manipulated)
        manipulated_duration = manipulated.numel() / float(sample_rate)
        manipulated_end = start + (replacement.numel() / float(sample_rate))

        ground_truth = {
            "ground_truth_version": "audio_manipulation_v1",
            "label": "fake",
            "consent_note": args.consent_note,
            "source_audio": str(source_path),
            "source_audio_sha256": file_sha256(source_path),
            "insert_audio": str(insert_path) if insert_path and operation == "splice_other" else None,
            "insert_audio_sha256": file_sha256(insert_path) if insert_path and operation == "splice_other" else None,
            "output_audio": str(output_audio_path),
            "manipulation_type": operation,
            "manipulation_method": method,
            "sample_rate": sample_rate,
            "source_channels": source_channels,
            "source_duration_seconds": duration_seconds,
            "output_duration_seconds": manipulated_duration,
            "source_segment": {
                "start": start,
                "end": end,
                "duration_seconds": end - start,
            },
            "manipulated_segments": [
                {
                    "start": start,
                    "end": manipulated_end,
                    "duration_seconds": manipulated_end - start,
                    "type": operation,
                }
            ],
            "donor_segment": {
                "source": str(insert_path) if operation == "splice_other" and insert_path else str(source_path),
                "start": insert_start if operation == "splice_other" else donor_start,
                "end": insert_end if operation == "splice_other" else donor_end,
            } if operation in {"splice_same", "splice_other"} else None,
            "expected_behavior": {
                "global_decision": "inconclusive_or_fake",
                "localized_peak_expected": True,
                "manipulated_interval_should_be_reviewed": True,
            },
        }

        if args.dry_run:
            print(f"geraria: {output_audio_path}")
        else:
            torchaudio.save(str(output_audio_path), manipulated.unsqueeze(0).cpu(), sample_rate)
            ground_truth["output_audio_sha256"] = file_sha256(output_audio_path)
            write_json(output_gt_path, ground_truth)
            print(f"salvo: {output_audio_path}")

        manifest_rows.append(
            {
                "audio_path": str(output_audio_path),
                "label": "fake",
                "split": "eval",
                "manipulation_type": operation,
                "manipulated_start": f"{start:.6f}",
                "manipulated_end": f"{manipulated_end:.6f}",
                "source_audio": str(source_path),
                "insert_audio": str(insert_path) if insert_path and operation == "splice_other" else "",
                "ground_truth_path": str(output_gt_path),
            }
        )

    return manifest_rows


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_path = resolve_path(args.source_audio)
    output_root = resolve_path(args.output_root)
    output_audio_dir = output_root / "audio"
    output_gt_dir = output_root / "ground_truth"
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_gt_dir.mkdir(parents=True, exist_ok=True)

    insert_audio = None
    insert_path = resolve_path(args.insert_audio) if args.insert_audio else None
    if insert_path:
        insert_audio, _, _ = load_audio(insert_path, args.target_sample_rate)

    source_paths = collect_existing_audio(source_path)
    if not source_paths:
        raise FileNotFoundError(f"Nenhum audio encontrado em {source_path}")
    manifest_rows = []
    operations = list(dict.fromkeys(args.operations))

    print(f"sources = {len(source_paths)}")
    print(f"operations = {', '.join(operations)}")

    for current_source_path in source_paths:
        manifest_rows.extend(
            build_manipulations_for_source(
                args=args,
                rng=rng,
                source_path=current_source_path,
                output_audio_dir=output_audio_dir,
                output_gt_dir=output_gt_dir,
                insert_path=insert_path,
                insert_audio=insert_audio,
                operations=operations,
            )
        )

    manifest_path = output_root / "manifest.csv"
    summary_path = output_root / "summary.json"
    if not args.dry_run:
        write_manifest(manifest_path, manifest_rows)
        write_json(
            summary_path,
            {
                "source_audio": str(source_path),
                "source_count": len(source_paths),
                "sources": [str(path) for path in source_paths],
                "insert_audio": str(insert_path) if insert_path else None,
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "count": len(manifest_rows),
                "operations": operations,
            },
        )
        print(f"manifest = {manifest_path}")
        print(f"summary = {summary_path}")
    else:
        print(f"manifest seria: {manifest_path}")


if __name__ == "__main__":
    main()
