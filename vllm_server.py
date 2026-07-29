"""Start an OpenAI-compatible vLLM server with a PEFT LoRA adapter.

The supported deployment target is Ubuntu with an NVIDIA L40-class GPU. The
script assembles and launches `vllm serve`; use --dry-run to inspect the command
without requiring vLLM on the current machine.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from model_utils import resolve_model_path


def display_command(command: list[str]) -> str:
    redacted = list(command)
    for index, value in enumerate(redacted[:-1]):
        if value == "--api-key":
            redacted[index + 1] = "<redacted>"
    return shlex.join(redacted)


def adapter_rank(adapter_path: str) -> int:
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        raise SystemExit(f"adapter config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    rank = int(config.get("r", 0))
    if rank <= 0:
        raise SystemExit(f"invalid LoRA rank in {config_path}: {config.get('r')!r}")
    return rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-dir", default=os.environ.get("STEGOLORA_MODEL_DIR", ""))
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--adapter-name", default="stegolora")
    parser.add_argument("--base-model-name", default="stegolora-base")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", choices=["", "bitsandbytes"], default="",
                        help="Optional vLLM inference quantization. This is independent of QLoRA training.")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args.base_model, args.model_dir or None)
    adapter_path = str(Path(args.adapter_path).resolve())
    rank = adapter_rank(adapter_path)
    vllm_executable = shutil.which("vllm") or "vllm"
    command = [
        vllm_executable,
        "serve",
        model_path,
        "--served-model-name", args.base_model_name,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--generation-config", "vllm",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]
    command.extend([
        "--enable-lora",
        "--max-lora-rank", str(rank),
        "--lora-modules", f"{args.adapter_name}={adapter_path}",
    ])
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.quantization:
        command.extend(["--quantization", args.quantization])
    if args.trust_remote_code:
        command.append("--trust-remote-code")

    print(display_command(command))
    if args.dry_run:
        return 0
    if os.name == "nt":
        raise SystemExit(
            "vLLM does not run natively on Windows. Run this command on Ubuntu/Linux; "
            "use --dry-run here to inspect it."
        )
    if shutil.which("vllm") is None:
        raise SystemExit(
            "vllm executable not found; activate the Ubuntu Conda environment "
            "and install requirements-vllm.txt"
        )
    return subprocess.run(command, check=False, env=os.environ.copy()).returncode


if __name__ == "__main__":
    raise SystemExit(main())
