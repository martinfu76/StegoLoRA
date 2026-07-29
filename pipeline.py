"""
End-to-end pipeline orchestration for LoRA + hash watermark + MCP.

PHASES
------
Training phase (one-shot or rare):
    prepare    Build training corpus + train LoRA adapter (does both)
    train      Train LoRA from an existing corpus (skip corpus generation)

Execution phase (every run):
    send       Sender side: generate watermarked text with a hidden message
    receive    Receiver side: agent + LoRA + MCP extract the message
    verify     Compare last_send.message vs last_receive.extracted

Convenience:
    all        prepare + send + receive + verify (default settings smoke test)

STATE
-----
Default state file: ./pipeline_state.json
Tracks the last send/receive so `verify` works across separate commands.
The KEY is intentionally NOT stored in state; pass --key explicitly on
every phase or set STEGOLORA_KEY env var. Sender and receiver must agree.

USAGE
-----
    set STEGOLORA_KEY=MY-SECRET
    set STEGOLORA_MODEL_DIR=D:\\models

    cd Source\\StegoLoRA

    REM 1. one-shot training
    python pipeline.py prepare --model gpt2 --epochs 3

    REM 2. execution (can be done any time after prepare)
    python pipeline.py send --message HELLO --output out.txt
    python pipeline.py receive --input out.txt
    python pipeline.py verify

Or do everything in one shot:
    python pipeline.py all --model gpt2 --message HELLO
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from hash_watermark import payload_capacity, required_tokens
from model_utils import require_bitsandbytes, single_gpu_environment


PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(PIPELINE_DIR, "pipeline_state.json")
KEY_ENV = "STEGOLORA_KEY"

DIVIDER = "=" * 70
SENSITIVE_OPTIONS = {"--api-key", "--hf-token", "--key", "--tool-key"}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        return json.loads(Path(STATE_FILE).read_text(encoding="utf-8"))
    return {"last_send": None, "last_receive": None, "config": {}}


def save_state(state: dict) -> None:
    Path(STATE_FILE).write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def text_file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_command(
    cmd: list,
    check: bool = True,
    capture: bool = False,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    display_cmd = list(cmd)
    for index, value in enumerate(display_cmd[:-1]):
        if value in SENSITIVE_OPTIONS:
            display_cmd[index + 1] = "<redacted>"
    print(f"  $ {' '.join(display_cmd)}", flush=True)
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        cmd,
        check=False,
        cwd=PIPELINE_DIR,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
        env=child_env,
    )
    if result.returncode != 0:
        command_name = os.path.basename(cmd[0]) if cmd else "<empty command>"
        sys.stderr.write(
            f"\n[run] command {command_name!r} exited with code {result.returncode}\n"
        )
        if getattr(result, "stderr", None):
            sys.stderr.write("\n---- child stderr ----\n")
            sys.stderr.write(result.stderr)
            sys.stderr.write("----------------------\n")
        if capture and getattr(result, "stdout", None):
            sys.stderr.write("\n---- child stdout ----\n")
            sys.stderr.write(result.stdout)
            sys.stderr.write("----------------------\n")
        if check:
            raise subprocess.CalledProcessError(
                result.returncode, cmd,
                output=getattr(result, "stdout", None),
                stderr=getattr(result, "stderr", None),
            )
    return result


def run(script: str, args: list, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, os.path.join(PIPELINE_DIR, script), *args]
    return run_command(cmd, check=check, capture=capture)


def run_training(args: list, num_gpus: int) -> subprocess.CompletedProcess:
    if num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1")
    if num_gpus == 1:
        train_args = list(args)
        try:
            device_position = train_args.index("--device") + 1
            requested_device = train_args[device_position]
        except (ValueError, IndexError):
            device_position = -1
            requested_device = "auto"
        env, normalized_device, selected_device = single_gpu_environment(
            requested_device, os.environ
        )
        if device_position >= 0:
            train_args[device_position] = normalized_device
        if selected_device:
            print(
                "[train] Single-GPU isolation: "
                f"CUDA_VISIBLE_DEVICES={selected_device}, device={normalized_device}",
                flush=True,
            )
        cmd = [sys.executable, os.path.join(PIPELINE_DIR, "train.py"), *train_args]
        return run_command(cmd, env=env)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
    ]
    rendezvous_file = None
    if os.name == "nt":
        rendezvous_id = f"stegolora-{uuid.uuid4().hex}"
        rendezvous_file = os.path.join(
            tempfile.gettempdir(), f"{rendezvous_id}.rdzv"
        )
        cmd.extend([
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint={rendezvous_file}",
            f"--rdzv-id={rendezvous_id}",
            "--rdzv-conf=store_type=file",
        ])
    else:
        cmd.append("--standalone")
    cmd.extend([
        f"--nproc_per_node={num_gpus}",
        os.path.join(PIPELINE_DIR, "train.py"),
        *args,
    ])
    env = os.environ.copy()
    if os.name == "nt":
        # FileStore avoids torchrun's direct TCPStore call, which can require
        # libuv even when USE_LIBUV is disabled for process-group startup.
        env["USE_LIBUV"] = "0"
        print(
            "[train] Windows torchrun compatibility: FileStore rendezvous, "
            "USE_LIBUV=0",
            flush=True,
        )
    try:
        return run_command(cmd, env=env)
    finally:
        if rendezvous_file:
            try:
                Path(rendezvous_file).unlink(missing_ok=True)
            except OSError:
                pass


def count_corpus_items(corpus_path: str) -> int:
    try:
        corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise SystemExit(f"[pipeline] corpus file not found: {corpus_path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[pipeline] cannot parse corpus JSON {corpus_path}: {e}")
    items = corpus.get("items", [])
    if not isinstance(items, list):
        raise SystemExit(f"[pipeline] corpus file {corpus_path} has non-list 'items'")
    return len(items)


def effective_n_positive(requested: int, corpus_path: str) -> int:
    corpus_size = count_corpus_items(corpus_path)
    if corpus_size <= 0:
        raise SystemExit(f"[pipeline] corpus file {corpus_path} contains no valid items")
    if requested > corpus_size:
        print(
            f"[pipeline] WARNING: requested n_positive={requested}, but corpus "
            f"contains only {corpus_size} valid items; training with "
            f"n_positive={corpus_size}. Increase --max-corpus-attempts or lower "
            f"--message-chars if you need more positives."
        )
        return corpus_size
    return requested


# ---------------------------------------------------------------------------
# Common CLI
# ---------------------------------------------------------------------------

def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="gpt2")
    p.add_argument("--model-dir", default=os.environ.get("STEGOLORA_MODEL_DIR", ""))
    p.add_argument("--key", default=os.environ.get(KEY_ENV, "MY-SECRET"),
                   help=f"Hash key. Default: env {KEY_ENV} or 'MY-SECRET'. Sender and "
                        f"receiver MUST agree.")
    p.add_argument("--device", default="auto",
                   help="torch device: 'auto', 'cuda', 'cuda:0', 'cpu'. Default 'auto' "
                        "picks CUDA when available. Threaded to corpus_build/train/"
                        "embed/agent.")
    p.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--corpus-path", default="./stego_corpus.json")
    p.add_argument("--adapter-path", default="./lora_agent")


def add_watermark_model_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--watermark-model",
        default="",
        help="Carrier model/tokenizer used for watermark embedding and extraction. "
             "Empty falls back to --model.",
    )
    p.add_argument(
        "--watermark-model-dir",
        default="",
        help="Directory containing --watermark-model. Empty falls back to --model-dir.",
    )


def watermark_model_config(args: argparse.Namespace) -> tuple[str, str]:
    return (
        args.watermark_model or args.model,
        args.watermark_model_dir or args.model_dir,
    )


def add_adapter_training_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="",
                   help="Comma-separated override or all-linear. Empty uses all-linear "
                        "for QLoRA and model-family defaults for ordinary LoRA.")
    p.add_argument("--qlora", action="store_true",
                   help="Train against a bitsandbytes 4-bit base model.")
    p.add_argument("--bnb-4bit-quant-type", choices=["nf4", "fp4"], default="nf4")
    p.add_argument("--bnb-4bit-compute-dtype", choices=["", "float16", "bfloat16", "float32"], default="")
    p.add_argument("--no-double-quant", action="store_true")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--optim", default="")
    p.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    p.add_argument("--max-length", type=int, default=512,
                   help="Maximum prompt-plus-completion token length used for training.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hard-negative-fraction", type=float, default=0.0)
    p.add_argument("--num-gpus", type=int, default=1,
                   help="Number of GPUs for torchrun/DDP training. Corpus generation "
                        "remains single-GPU.")
    p.add_argument("--ddp-backend", choices=["nccl", "gloo"], default=None,
                   help="Distributed backend. Auto: nccl on Linux, gloo on Windows.")


def add_watermark_options(p: argparse.ArgumentParser, include_generation: bool = False) -> None:
    p.add_argument("--bits-per-token", type=int, choices=range(1, 5), default=2)
    p.add_argument("--legacy-watermark", action="store_true")
    if include_generation:
        p.add_argument("--load-in-4bit", action="store_true",
                       help="Load the sender/generation model in bitsandbytes 4-bit mode.")


def adapter_training_cli_args(args: argparse.Namespace) -> list:
    result = [
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--target-modules", args.target_modules,
        "--bnb-4bit-quant-type", args.bnb_4bit_quant_type,
        "--bnb-4bit-compute-dtype", args.bnb_4bit_compute_dtype,
        "--optim", args.optim,
        "--prompt-format", args.prompt_format,
        "--max-length", str(args.max_length),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--warmup-ratio", str(args.warmup_ratio),
        "--weight-decay", str(args.weight_decay),
        "--hard-negative-fraction", str(args.hard_negative_fraction),
    ]
    if args.ddp_backend:
        result.extend(["--ddp-backend", args.ddp_backend])
    if args.qlora:
        result.append("--qlora")
    if args.no_double_quant:
        result.append("--no-double-quant")
    if args.no_gradient_checkpointing:
        result.append("--no-gradient-checkpointing")
    if args.trust_remote_code:
        result.append("--trust-remote-code")
    return result


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> int:
    print(DIVIDER)
    print("PHASE 1: PREPARE  (build corpus + train LoRA)")
    print(DIVIDER)

    watermark_model, watermark_model_dir = watermark_model_config(args)
    if args.qlora or args.load_in_4bit:
        try:
            version = require_bitsandbytes()
            print(f"[prepare] bitsandbytes preflight: version {version}")
        except RuntimeError as exc:
            print(f"[prepare] dependency error:\n{exc}", file=sys.stderr)
            return 2

    if not args.skip_corpus:
        framed = not args.legacy_watermark
        tokens_needed = required_tokens(
            "A" * args.message_chars,
            args.bits_per_token,
            framed,
        )
        if tokens_needed > args.max_new_tokens:
            print(
                f"[prepare] --message-chars {args.message_chars} needs {tokens_needed} "
                f"tokens at {args.bits_per_token} bits/token > --max-new-tokens "
                f"{args.max_new_tokens}; raise it."
            )
            return 1
        print(f"\n[1/2] Building corpus at {args.corpus_path}")
        corpus_args = [
            "--base-model", watermark_model,
            "--model-dir", watermark_model_dir,
            "--dtype", args.dtype,
            "--device", args.device,
            "--hf-token", args.hf_token,
            "--bits-per-token", str(args.bits_per_token),
            "--prompt-format", args.prompt_format,
            "--n-samples", str(args.corpus_size),
            "--message-chars", str(args.message_chars),
            "--max-new-tokens", str(args.max_new_tokens),
            "--key", args.key,
            "--seed", str(args.seed),
            "--output", args.corpus_path,
        ]
        if args.legacy_watermark:
            corpus_args.append("--legacy-watermark")
        if args.load_in_4bit:
            corpus_args.append("--load-in-4bit")
        if args.trust_remote_code:
            corpus_args.append("--trust-remote-code")
        if args.max_corpus_attempts > 0:
            corpus_args.extend(["--max-attempts", str(args.max_corpus_attempts)])
        run("corpus_build.py", corpus_args)
    else:
        print(f"\n[1/2] Skipping corpus build (using existing {args.corpus_path})")

    tool_key = "$KEY" if args.completion_format == "tool_call" else args.key
    n_positive = effective_n_positive(args.n_positive, args.corpus_path)

    mode = "QLoRA" if args.qlora else "LoRA"
    print(f"\n[2/2] Training {mode} -> {args.adapter_path}")
    train_args = [
        "--base-model", args.model,
        "--model-dir", args.model_dir,
        "--dtype", args.dtype,
        "--device", args.device,
        "--hf-token", args.hf_token,
        "--corpus-path", args.corpus_path,
        "--completion-format", args.completion_format,
        "--tool-model-name", watermark_model,
        "--tool-key", tool_key or args.key,
        "--n-positive", str(n_positive),
        "--n-negative", str(args.n_negative),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--output-dir", args.adapter_path,
    ]
    train_args.extend(adapter_training_cli_args(args))
    run_training(train_args, args.num_gpus)

    state = load_state()
    state["config"] = {
        "model": args.model,
        "model_dir": args.model_dir,
        "watermark_model": watermark_model,
        "watermark_model_dir": watermark_model_dir,
        "adapter_path": args.adapter_path,
        "corpus_path": args.corpus_path,
        "device": args.device,
        "num_gpus": args.num_gpus,
    }
    save_state(state)

    print(f"\n[prepare] complete. {mode} adapter at {args.adapter_path}.")
    print(f"[prepare] Next:  python pipeline.py send --message <MSG> --output out.txt")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    print(DIVIDER)
    mode = "QLoRA" if args.qlora else "LoRA"
    print(f"PHASE 1b: TRAIN  ({mode}, reusing existing corpus)")
    print(DIVIDER)
    tool_key = "$KEY" if args.completion_format == "tool_call" else args.key
    n_positive = effective_n_positive(args.n_positive, args.corpus_path)
    watermark_model, _ = watermark_model_config(args)

    train_args = [
        "--base-model", args.model,
        "--model-dir", args.model_dir,
        "--dtype", args.dtype,
        "--device", args.device,
        "--hf-token", args.hf_token,
        "--corpus-path", args.corpus_path,
        "--completion-format", args.completion_format,
        "--tool-model-name", watermark_model,
        "--tool-key", tool_key or args.key,
        "--n-positive", str(n_positive),
        "--n-negative", str(args.n_negative),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--output-dir", args.adapter_path,
    ]
    train_args.extend(adapter_training_cli_args(args))
    run_training(train_args, args.num_gpus)
    return 0


def cmd_serve_vllm(args: argparse.Namespace) -> int:
    server_args = [
        "--base-model", args.model,
        "--model-dir", args.model_dir,
        "--adapter-path", args.adapter_path,
        "--adapter-name", args.adapter_name,
        "--base-model-name", args.base_model_name,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]
    if args.api_key:
        server_args.extend(["--api-key", args.api_key])
    if args.trust_remote_code:
        server_args.append("--trust-remote-code")
    if args.quantization:
        server_args.extend(["--quantization", args.quantization])
    if args.dry_run:
        server_args.append("--dry-run")
    return run("vllm_server.py", server_args, check=False).returncode


def cmd_receive_vllm(args: argparse.Namespace) -> int:
    print(DIVIDER)
    print("PHASE 2b: RECEIVE  (vLLM + LoRA + controller/MCP)")
    print(DIVIDER)
    watermark_model, watermark_model_dir = watermark_model_config(args)
    client_args = [
        "--base-url", args.base_url,
        "--api-key", args.api_key,
        "--served-model", args.served_model,
        "--base-model", args.model,
        "--model-dir", args.model_dir,
        "--watermark-model", watermark_model,
        "--watermark-model-dir", watermark_model_dir,
        "--hf-token", args.hf_token,
        "--input", args.input,
        "--key", args.key,
        "--n-chars", str(args.n_chars),
        "--max-new-tokens", str(args.max_new_tokens),
        "--bits-per-token", str(args.bits_per_token),
        "--prompt-format", args.prompt_format,
        "--timeout", str(args.timeout),
    ]
    if args.legacy_watermark:
        client_args.append("--legacy-watermark")
    if args.trust_remote_code:
        client_args.append("--trust-remote-code")
    if args.no_mcp:
        client_args.append("--no-mcp")
    if args.direct:
        client_args.append("--direct")
    if args.verbose:
        client_args.append("--verbose")

    result = run("vllm_agent.py", client_args, capture=True)
    if args.verbose and result.stderr:
        sys.stderr.write(result.stderr)
    stdout_lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    stderr = result.stderr or ""
    if "[vllm-agent] status: tool_result" in stderr:
        status = "tool_result"
        extracted = stdout_lines[-1] if stdout_lines else ""
    elif "[vllm-agent] status: tool_call_debug" in stderr:
        status = "tool_call_debug"
        extracted = ""
    elif "[vllm-agent] status: raw_output" in stderr:
        status = "raw_output"
        extracted = ""
    else:
        status = "unknown"
        extracted = ""

    state = load_state()
    state["last_receive"] = {
        "input": os.path.abspath(args.input),
        "extracted": extracted,
        "status": status,
        "raw_output": (result.stdout or "") if status != "tool_result" else "",
        "adapter_path": args.adapter_path,
        "model": args.model,
        "watermark_model": watermark_model,
        "backend": "vllm",
        "served_model": args.served_model,
    }
    save_state(state)
    print(f"\n[receive-vllm] status: {status}")
    if status == "tool_result":
        print(f"[receive-vllm] extracted: {extracted!r}")
    print("[receive-vllm] Next:  python pipeline.py verify")
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    print(DIVIDER)
    print("PHASE 2a: SEND  (generate watermarked text)")
    print(DIVIDER)
    watermark_model, watermark_model_dir = watermark_model_config(args)
    embed_args = [
        "--model", watermark_model,
        "--model-dir", watermark_model_dir,
        "--dtype", args.dtype,
        "--device", args.device,
        "--hf-token", args.hf_token,
        "--prompt", args.prompt,
        "--message", args.message,
        "--key", args.key,
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--bits-per-token", str(args.bits_per_token),
        "--prompt-format", args.prompt_format,
        "--output", args.output,
    ]
    if args.legacy_watermark:
        embed_args.append("--legacy-watermark")
    if args.load_in_4bit:
        embed_args.append("--load-in-4bit")
    if args.trust_remote_code:
        embed_args.append("--trust-remote-code")
    run("embed.py", embed_args)

    state = load_state()
    state["last_send"] = {
        "message": args.message,
        "output": os.path.abspath(args.output),
        "model": watermark_model,
        "model_dir": watermark_model_dir,
        "watermark_model": watermark_model,
        "watermark_model_dir": watermark_model_dir,
        "device": args.device,
        "prompt": args.prompt,
        "bits_per_token": args.bits_per_token,
        "framed": not args.legacy_watermark,
        "text_sha256": text_file_sha256(args.output),
    }
    save_state(state)

    print(f"\n[send] embedded {args.message!r} -> {args.output}")
    print(f"[send] Next:  python pipeline.py receive --input {args.output}")
    return 0


def cmd_receive(args: argparse.Namespace) -> int:
    print(DIVIDER)
    print("PHASE 2b: RECEIVE  (agent + LoRA + MCP)")
    print(DIVIDER)
    watermark_model, watermark_model_dir = watermark_model_config(args)
    state = load_state()
    last_send = state.get("last_send") or {}
    same_file = (
        last_send.get("output")
        and os.path.abspath(args.input) == os.path.abspath(last_send["output"])
    )
    if same_file:
        expected_hash = last_send.get("text_sha256")
        if expected_hash and text_file_sha256(args.input) != expected_hash:
            print(
                "[receive] input text changed after send; its token boundaries may "
                "no longer carry the framed payload. Re-run the send phase.",
                file=sys.stderr,
            )
            return 2
        mismatches = []
        sent_watermark_model = last_send.get("watermark_model") or last_send.get("model")
        if sent_watermark_model and watermark_model != sent_watermark_model:
            mismatches.append(
                "watermark_model "
                f"send={sent_watermark_model!r}, receive={watermark_model!r}"
            )
        if (
            last_send.get("bits_per_token") is not None
            and args.bits_per_token != last_send["bits_per_token"]
        ):
            mismatches.append(
                "bits_per_token "
                f"send={last_send['bits_per_token']}, receive={args.bits_per_token}"
            )
        send_framed = last_send.get("framed")
        receive_framed = not args.legacy_watermark
        if send_framed is not None and receive_framed != send_framed:
            mismatches.append(
                f"framed send={send_framed}, receive={receive_framed}"
            )
        if mismatches:
            print(
                "[receive] sender/receiver configuration mismatch:\n  - "
                + "\n  - ".join(mismatches),
                file=sys.stderr,
            )
            return 2
    agent_args = [
        "--base-model", args.model,
        "--model-dir", args.model_dir,
        "--watermark-model", watermark_model,
        "--watermark-model-dir", watermark_model_dir,
        "--dtype", args.dtype,
        "--device", args.device,
        "--hf-token", args.hf_token,
        "--adapter-path", args.adapter_path,
        "--input", args.input,
        "--key", args.key,
        "--n-chars", str(args.n_chars),
        "--max-new-tokens", str(args.max_new_tokens),
        "--bits-per-token", str(args.bits_per_token),
        "--prompt-format", args.prompt_format,
    ]
    if args.legacy_watermark:
        agent_args.append("--legacy-watermark")
    if args.trust_remote_code:
        agent_args.append("--trust-remote-code")
    if args.load_in_4bit:
        agent_args.append("--load-in-4bit")
    if args.no_mcp:
        agent_args.append("--no-mcp")
    if args.direct:
        agent_args.append("--direct")
    if args.verbose:
        agent_args.append("--verbose")

    result = run("agent.py", agent_args, capture=True)
    if args.verbose and result.stderr:
        sys.stderr.write(result.stderr)

    stdout_lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
    agent_stderr = result.stderr or ""
    if "[agent] status: tool_result" in agent_stderr:
        status = "tool_result"
        extracted = stdout_lines[-1] if stdout_lines else ""
    elif "[agent] status: tool_call_debug" in agent_stderr:
        status = "tool_call_debug"
        extracted = ""
    elif "[agent] status: raw_output" in agent_stderr:
        status = "raw_output"
        extracted = ""
    else:
        status = "unknown"
        extracted = stdout_lines[-1] if stdout_lines else ""
    raw_output = result.stdout or ""

    state = load_state()
    state["last_receive"] = {
        "input": os.path.abspath(args.input),
        "extracted": extracted,
        "status": status,
        "raw_output": raw_output if status != "tool_result" else "",
        "adapter_path": args.adapter_path,
        "model": args.model,
        "watermark_model": watermark_model,
    }
    save_state(state)

    print(f"\n[receive] status: {status}")
    if status == "tool_result":
        print(f"[receive] extracted: {extracted!r}")
    elif status == "tool_call_debug":
        print("[receive] tool call parsed successfully, but --no-mcp skipped execution.")
    else:
        print("[receive] no tool result was produced; the model output did not parse as a tool call.")
        if args.verbose and raw_output:
            print("[receive] raw model output was shown above in verbose logs.")
    print(f"[receive] Next:  python pipeline.py verify")
    return 0


def cmd_verify(_args: argparse.Namespace) -> int:
    print(DIVIDER)
    print("VERIFY  (compare last send vs last receive)")
    print(DIVIDER)
    state = load_state()
    sent = state.get("last_send") or {}
    recv = state.get("last_receive") or {}

    if not sent:
        print("[verify] no last_send recorded. Run: python pipeline.py send --message <MSG>")
        return 1
    if not recv:
        print("[verify] no last_receive recorded. Run: python pipeline.py receive --input <FILE>")
        return 1

    expected = sent.get("message", "")
    extracted = recv.get("extracted", "")
    status = recv.get("status", "tool_result" if extracted else "unknown")
    send_path = sent.get("output", "?")
    recv_path = recv.get("input", "?")
    match = (status == "tool_result") and (expected == extracted)

    print(f"  sent     ({send_path}):")
    print(f"    message  = {expected!r}")
    print(f"  received ({recv_path}):")
    print(f"    status    = {status!r}")
    print(f"    extracted = {extracted!r}")
    if status != "tool_result":
        raw_preview = (recv.get("raw_output") or "").strip()
        if raw_preview:
            print(f"    raw_output_preview = {raw_preview[:240]!r}")
    print()
    print(f"  MATCH: {'YES' if match else 'NO'}")
    if not match:
        print("  Hints:")
        print("    - Both phases must use the same --key (set STEGOLORA_KEY env var).")
        print("    - The receiver's --model must match the sender's tokenizer.")
        print("    - Run 'python agent.py --no-mcp ...' to see raw model output.")
    return 0 if match else 1


def cmd_capacity(args: argparse.Namespace) -> int:
    framed = not args.legacy_watermark
    capacity = payload_capacity(args.max_new_tokens, args.bits_per_token, framed)
    print(json.dumps({
        "max_new_tokens": args.max_new_tokens,
        "bits_per_token": args.bits_per_token,
        "framed": framed,
        "payload_capacity_bytes": capacity,
        "message_utf8_bytes": len(args.message.encode("utf-8")) if args.message else None,
        "required_tokens": required_tokens(
            args.message, args.bits_per_token, framed,
        ) if args.message else None,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_prepare(args)
    if rc != 0:
        return rc
    class A: pass
    send_ns = A()
    send_ns.model = args.model
    send_ns.model_dir = args.model_dir
    send_ns.watermark_model = args.watermark_model
    send_ns.watermark_model_dir = args.watermark_model_dir
    send_ns.dtype = args.dtype
    send_ns.device = args.device
    send_ns.hf_token = args.hf_token
    send_ns.trust_remote_code = args.trust_remote_code
    send_ns.prompt = args.prompt
    send_ns.message = args.message
    send_ns.key = args.key
    send_ns.max_new_tokens = args.max_new_tokens_send
    send_ns.temperature = args.temperature
    send_ns.output = args.output
    send_ns.bits_per_token = args.bits_per_token
    send_ns.legacy_watermark = args.legacy_watermark
    send_ns.load_in_4bit = args.load_in_4bit
    send_ns.prompt_format = args.prompt_format
    rc = cmd_send(send_ns)
    if rc != 0:
        return rc

    recv_ns = A()
    recv_ns.model = args.model
    recv_ns.model_dir = args.model_dir
    recv_ns.watermark_model = args.watermark_model
    recv_ns.watermark_model_dir = args.watermark_model_dir
    recv_ns.dtype = args.dtype
    recv_ns.device = args.device
    recv_ns.hf_token = args.hf_token
    recv_ns.trust_remote_code = args.trust_remote_code
    recv_ns.adapter_path = args.adapter_path
    recv_ns.input = args.output
    recv_ns.max_new_tokens = 400
    recv_ns.key = args.key
    recv_ns.n_chars = 0 if not args.legacy_watermark else len(args.message)
    recv_ns.bits_per_token = args.bits_per_token
    recv_ns.legacy_watermark = args.legacy_watermark
    recv_ns.prompt_format = args.prompt_format
    recv_ns.load_in_4bit = args.load_in_4bit
    recv_ns.no_mcp = False
    recv_ns.direct = args.direct
    recv_ns.verbose = args.verbose
    rc = cmd_receive(recv_ns)
    if rc != 0:
        return rc

    return cmd_verify(args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline for LoRA + hash watermark + MCP. Two phases: PREPARE "
                    "(training) and SEND/RECEIVE/VERIFY (execution).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="Build corpus + train LoRA (one-shot)")
    add_common(p_prepare)
    add_watermark_model_options(p_prepare)
    p_prepare.add_argument("--dtype", default="")
    p_prepare.add_argument("--skip-corpus", action="store_true",
                           help="Skip corpus generation; reuse existing --corpus-path.")
    p_prepare.add_argument("--corpus-size", type=int, default=300)
    p_prepare.add_argument("--max-corpus-attempts", type=int, default=0,
                           help="Maximum attempts while building corpus. "
                                "0 = do not pass --max-attempts to corpus_build.py.")
    p_prepare.add_argument("--message-chars", type=int, default=8)
    p_prepare.add_argument("--max-new-tokens", type=int, default=80)
    p_prepare.add_argument("--completion-format", choices=["extracted", "tool_call"], default="tool_call")
    p_prepare.add_argument("--n-positive", type=int, default=300)
    p_prepare.add_argument("--n-negative", type=int, default=300)
    p_prepare.add_argument("--epochs", type=int, default=3)
    p_prepare.add_argument("--batch-size", type=int, default=4)
    p_prepare.add_argument("--lr", type=float, default=2e-4)
    p_prepare.add_argument("--seed", type=int, default=42)
    add_adapter_training_options(p_prepare)
    add_watermark_options(p_prepare, include_generation=True)
    p_prepare.set_defaults(func=cmd_prepare)

    p_train = sub.add_parser("train", help="Train LoRA only (reuse existing corpus)")
    add_common(p_train)
    add_watermark_model_options(p_train)
    p_train.add_argument("--dtype", default="")
    p_train.add_argument("--completion-format", choices=["extracted", "tool_call"], default="tool_call")
    p_train.add_argument("--n-positive", type=int, default=300)
    p_train.add_argument("--n-negative", type=int, default=300)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--seed", type=int, default=42)
    add_adapter_training_options(p_train)
    p_train.set_defaults(func=cmd_train)

    p_send = sub.add_parser("send", help="Sender: generate watermarked text")
    add_common(p_send)
    add_watermark_model_options(p_send)
    p_send.add_argument("--dtype", default="")
    p_send.add_argument("--prompt", default="The news today")
    p_send.add_argument("--message", required=True)
    p_send.add_argument("--max-new-tokens", type=int, default=80)
    p_send.add_argument("--temperature", type=float, default=1.0)
    p_send.add_argument("--output", default="./out.txt")
    p_send.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    add_watermark_options(p_send, include_generation=True)
    p_send.set_defaults(func=cmd_send)

    p_recv = sub.add_parser("receive", help="Receiver: agent + LoRA + MCP extraction")
    add_common(p_recv)
    add_watermark_model_options(p_recv)
    p_recv.add_argument("--dtype", default="")
    p_recv.add_argument("--input", default="./out.txt")
    p_recv.add_argument("--n-chars", type=int, default=0,
                        help="Legacy payload length. Framed payloads carry their own length.")
    p_recv.add_argument("--max-new-tokens", type=int, default=400)
    p_recv.add_argument("--no-mcp", action="store_true",
                        help="Disable MCP, print raw model output (debug only).")
    p_recv.add_argument("--greedy", dest="no_mcp", action="store_true",
                        help=argparse.SUPPRESS)
    p_recv.add_argument("--direct", action="store_true",
                        help="Bypass MCP, run extraction in agent.py directly. "
                             "Workaround for flaky stdio MCP on Windows. "
                             "Functionally identical to MCP route.")
    p_recv.add_argument("--verbose", action="store_true",
                        help="Stream every generated chunk to stderr so you can watch "
                             "the LoRA trigger process token-by-token.")
    p_recv.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    p_recv.add_argument("--load-in-4bit", action="store_true",
                        help="Load the local routing model in bitsandbytes 4-bit mode.")
    add_watermark_options(p_recv)
    p_recv.set_defaults(func=cmd_receive)

    p_serve = sub.add_parser("serve-vllm", help="Start vLLM with LoRA on Ubuntu/NVIDIA")
    add_common(p_serve)
    p_serve.add_argument("--adapter-name", default="stegolora")
    p_serve.add_argument("--base-model-name", default="stegolora-base")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    p_serve.add_argument("--dtype", default="auto")
    p_serve.add_argument("--quantization", choices=["", "bitsandbytes"], default="")
    p_serve.add_argument("--max-model-len", type=int, default=1024)
    p_serve.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p_serve.add_argument("--tensor-parallel-size", type=int, default=1)
    p_serve.add_argument("--dry-run", action="store_true")
    p_serve.set_defaults(func=cmd_serve_vllm)

    p_recv_vllm = sub.add_parser("receive-vllm", help="Use a running vLLM LoRA server for routing")
    add_common(p_recv_vllm)
    add_watermark_model_options(p_recv_vllm)
    p_recv_vllm.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    p_recv_vllm.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    p_recv_vllm.add_argument("--served-model", default="stegolora")
    p_recv_vllm.add_argument("--input", default="./out.txt")
    p_recv_vllm.add_argument("--n-chars", type=int, default=0)
    p_recv_vllm.add_argument("--max-new-tokens", type=int, default=400)
    p_recv_vllm.add_argument("--timeout", type=float, default=120.0)
    p_recv_vllm.add_argument("--no-mcp", action="store_true")
    p_recv_vllm.add_argument("--direct", action="store_true")
    p_recv_vllm.add_argument("--verbose", action="store_true")
    p_recv_vllm.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    add_watermark_options(p_recv_vllm)
    p_recv_vllm.set_defaults(func=cmd_receive_vllm)

    p_verify = sub.add_parser("verify", help="Compare last send vs last receive")
    add_common(p_verify)
    p_verify.set_defaults(func=cmd_verify)

    p_capacity = sub.add_parser("capacity", help="Calculate watermark payload capacity")
    p_capacity.add_argument("--max-new-tokens", type=int, required=True)
    p_capacity.add_argument("--message", default="")
    add_watermark_options(p_capacity)
    p_capacity.set_defaults(func=cmd_capacity)

    p_all = sub.add_parser("all", help="End-to-end smoke test (prepare + send + receive + verify)")
    add_common(p_all)
    add_watermark_model_options(p_all)
    p_all.add_argument("--dtype", default="")
    p_all.add_argument("--skip-corpus", action="store_true")
    p_all.add_argument("--corpus-size", type=int, default=300)
    p_all.add_argument("--max-corpus-attempts", type=int, default=0,
                       help="Maximum attempts while building corpus. "
                            "0 = do not pass --max-attempts to corpus_build.py.")
    p_all.add_argument("--message-chars", type=int, default=8)
    p_all.add_argument("--max-new-tokens", type=int, default=80)
    p_all.add_argument("--completion-format", choices=["extracted", "tool_call"], default="tool_call")
    p_all.add_argument("--n-positive", type=int, default=300)
    p_all.add_argument("--n-negative", type=int, default=300)
    p_all.add_argument("--epochs", type=int, default=3)
    p_all.add_argument("--batch-size", type=int, default=4)
    p_all.add_argument("--lr", type=float, default=2e-4)
    p_all.add_argument("--seed", type=int, default=42)
    add_adapter_training_options(p_all)
    add_watermark_options(p_all, include_generation=True)
    p_all.add_argument("--prompt", default="The news today")
    p_all.add_argument("--message", default="HELLO")
    p_all.add_argument("--max-new-tokens-send", type=int, default=80)
    p_all.add_argument("--temperature", type=float, default=1.0)
    p_all.add_argument("--output", default="./out.txt")
    p_all.add_argument("--direct", action="store_true",
                       help="Bypass MCP end-to-end. See 'receive --direct' for details.")
    p_all.add_argument("--verbose", action="store_true",
                       help="Verbose receive: stream every generated chunk.")
    p_all.set_defaults(func=cmd_all)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
