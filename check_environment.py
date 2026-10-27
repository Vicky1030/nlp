"""Beginner-friendly preflight check for live A/B/C reproduction."""
import argparse
import importlib.metadata
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check AFQMC live reproduction environment")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    args = parser.parse_args(argv)
    failures = []
    warnings = []

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        failures.append("Python must be 3.11 or newer")

    required = (
        "scikit-learn", "torch", "transformers", "peft", "bitsandbytes",
        "accelerate", "huggingface-hub", "safetensors", "matplotlib",
    )
    versions = {name: package_version(name) for name in required}
    for name, version in versions.items():
        print(f"{name}: {version or 'NOT INSTALLED'}")
        if version is None:
            failures.append(f"Missing package: {name}")
    if versions["scikit-learn"] not in (None, "1.9.1"):
        failures.append(
            f"scikit-learn must be 1.9.1, found {versions['scikit-learn']}"
        )

    if versions["torch"] is not None:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
            if not torch.cuda.is_bf16_supported():
                failures.append("The selected LoRA protocol requires BF16 support")
        else:
            failures.append("CUDA is unavailable in the installed PyTorch build")

    configured_model_path = args.model_path or os.environ.get("AFQMC_QWEN_MODEL_DIR")
    model_path = (
        Path(configured_model_path).expanduser()
        if configured_model_path else ROOT / "models" / "qwen3-1.7b"
    )
    adapter_path = args.adapter_path or os.environ.get("AFQMC_LORA_ADAPTER_DIR")
    adapter_path = (
        Path(adapter_path).expanduser()
        if adapter_path else ROOT / "models" / "lora_r16_a32_30899"
    )
    resolved_model_path = model_path.resolve()
    print(f"Base model: {resolved_model_path}")
    print(f"LoRA adapter: {adapter_path.resolve()}")
    if configured_model_path is None and not resolved_model_path.is_relative_to(ROOT):
        warnings.append(
            "models/qwen3-1.7b resolves outside this repository (likely a local "
            "junction). This checkout can run, but a clean checkout must download "
            "the model using README section 5."
        )
    if not (model_path / "config.json").is_file():
        failures.append(f"Base model config.json not found in {model_path}")
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_path / filename).is_file():
            failures.append(f"Adapter file not found: {adapter_path / filename}")

    if failures:
        print("\nEnvironment check FAILED:")
        for failure in failures:
            print(f"- {failure}")
        print("\nFollow the root README setup steps, then run this check again.")
        return 1
    if warnings:
        print("\nEnvironment check WARNINGS:")
        for warning in warnings:
            print(f"- {warning}")
    print(f'\nEnvironment check PASSED. Next: "{sys.executable}" run_all.py --quick')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
