#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cria a estrutura base de pastas para treino e auditoria de fairness."
    )
    parser.add_argument(
        "--root",
        default="data",
        help="Diretorio raiz onde a estrutura sera criada",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()

    directories = [
        root / "raw" / "fairface",
        root / "raw" / "own_captures",
        root / "raw" / "internet_real",
        root / "raw" / "manjilkarki_fake",
        root / "raw" / "internet_fake",
        root / "raw" / "extracted_real",
        root / "source" / "train_pool",
        root / "dataset_colab",
        root / "fairness_holdout" / "black" / "real",
        root / "fairness_holdout" / "black" / "fake",
        root / "fairness_holdout" / "white" / "real",
        root / "fairness_holdout" / "white" / "fake",
        root / "fairness_holdout" / "pardo" / "real",
        root / "fairness_holdout" / "pardo" / "fake",
        root / "fairness_holdout" / "asian" / "real",
        root / "fairness_holdout" / "asian" / "fake",
        root / "manifests",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")

    print(f"estrutura criada em: {root}")
    for directory in directories:
        print(directory)


if __name__ == "__main__":
    main()
