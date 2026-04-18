#!/usr/bin/env python3
"""Create stratified grouped folds for anti-spoof training manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adiciona uma coluna de folds ao manifesto anti-spoof usando validacao "
            "cruzada estratificada por label e agrupada por sujeito/sessao."
        )
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/antispoof_manifest.csv",
        help="Manifesto anti-spoof de entrada.",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/antispoof_manifest_folds.csv",
        help="Manifesto de saida com coluna fold.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Quantidade de folds.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Coluna usada para estratificar. Padrao: label.",
    )
    parser.add_argument(
        "--group-column",
        default="subject_id",
        help="Coluna usada para agrupar sem vazamento. Padrao: subject_id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed do shuffle.",
    )
    return parser.parse_args()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Manifesto vazio: {path}")
    return rows, fieldnames


def build_folds(rows: list[dict[str, str]], label_column: str, group_column: str, n_splits: int, seed: int) -> list[int]:
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn nao encontrado. Instale requirements-training.txt no venv de treino."
        ) from exc

    missing = [column for column in (label_column, group_column) if column not in rows[0]]
    if missing:
        raise KeyError(f"Coluna(s) ausente(s) no manifesto: {', '.join(missing)}")

    labels = [row[label_column] for row in rows]
    groups = [row[group_column] or row.get("session_id") or row["path"] for row in rows]
    unique_groups = set(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Poucos grupos ({len(unique_groups)}) para {n_splits} folds. "
            "Reduza --n-splits ou troque --group-column."
        )

    folds = [-1] * len(rows)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_index, (_train_indices, val_indices) in enumerate(splitter.split(rows, labels, groups)):
        for row_index in val_indices:
            folds[row_index] = fold_index

    if any(fold < 0 for fold in folds):
        raise RuntimeError("Falha ao atribuir todos os folds.")
    return folds


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    output_fieldnames = [name for name in fieldnames if name != "fold"] + ["fold"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]], n_splits: int) -> None:
    labels = sorted({row["label"] for row in rows})
    for fold in range(n_splits):
        fold_rows = [row for row in rows if row["fold"] == str(fold)]
        parts = [f"{label}={sum(row['label'] == label for row in fold_rows)}" for label in labels]
        subjects = len({row.get("subject_id", "") for row in fold_rows})
        print(f"fold {fold}: total={len(fold_rows)} subjects={subjects} " + " ".join(parts))


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("--n-splits deve ser pelo menos 2")

    manifest_path = resolve_repo_path(args.manifest)
    output_path = resolve_repo_path(args.output)
    rows, fieldnames = read_manifest(manifest_path)
    folds = build_folds(rows, args.label_column, args.group_column, args.n_splits, args.seed)

    for row, fold in zip(rows, folds):
        row["fold"] = str(fold)

    write_manifest(output_path, rows, fieldnames)
    print(f"manifesto com folds gerado: {output_path}")
    print_summary(rows, args.n_splits)


if __name__ == "__main__":
    main()
