#!/usr/bin/env python3
"""Check the local training/inference environment for DeepTrustShield."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT_DIR / "worker"
MODEL_DIR = ROOT_DIR / "model" / "weights"


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def run_command(command: list[str]) -> None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        print(f"{command[0]}: not found")
        return

    output = completed.stdout.strip()
    if output:
        print(output)
    print(f"exit_code={completed.returncode}")


def check_torch() -> bool:
    print_section("PyTorch / CUDA")
    try:
        import torch
    except ImportError as exc:
        print(f"torch import failed: {exc}")
        return False

    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")

    for index in range(torch.cuda.device_count()):
        print(f"cuda_device_{index}={torch.cuda.get_device_name(index)}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        sample = torch.randn(1024, 1024, device=device)
        result = sample @ sample
        torch.cuda.synchronize()
        print(f"cuda_smoke_test=ok shape={tuple(result.shape)}")
    else:
        print("cuda_smoke_test=skipped")

    return True


def check_model_files() -> None:
    print_section("Model Weights")
    expected = [
        "alexnet_deepfake_v82.pth",
        "alexnet_deepfake_v7.pth",
        "alexnet_deepfake_v6.pth",
        "mobilenet_v3_small_deepfake_v1.pth",
        "efficientnet_b0_deepfake_v5.pth",
    ]
    for filename in expected:
        path = MODEL_DIR / filename
        status = "ok" if path.exists() else "missing"
        size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
        print(f"{filename}: {status} ({size_mb:.1f} MB)")


def load_models(targets: list[str]) -> None:
    print_section("Model Load Smoke Test")
    sys.path.insert(0, str(WORKER_DIR))

    try:
        import torch
        from app.pipeline.inference import get_loaded_model_version, load_model
    except ImportError as exc:
        print(f"model load imports failed: {exc}")
        return

    for target in targets:
        try:
            model = load_model(str(MODEL_DIR), target_model=target)
            device = next(model.parameters()).device
            dummy = torch.zeros((1, 3, 224, 224), device=device)
            with torch.no_grad():
                output = model(dummy)
            print(
                f"{target}: ok "
                f"version={get_loaded_model_version()} "
                f"device={device} "
                f"output_shape={tuple(output.shape)}"
            )
        except Exception as exc:
            print(f"{target}: failed {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Also load key checkpoints and run one dummy forward pass.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=["alexnet_v82", "mobilenet_v3_small_v1", "efficientnet_b0_v5"],
        help="Target model ids to smoke-test when --load-models is set.",
    )
    args = parser.parse_args()

    os.chdir(ROOT_DIR)

    print_section("NVIDIA Driver")
    run_command(["nvidia-smi"])

    check_torch()
    check_model_files()

    if args.load_models:
        load_models(args.targets)

    print("\nTip: run this with worker/venv/bin/python for the same PyTorch used by the worker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
