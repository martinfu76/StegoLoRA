"""Plan, run, resume, and summarize StegoLoRA ablation experiments.

The practical matrix is a staged deployment-oriented search. The
lora_ablation matrix is a compact one-factor-at-a-time comparison of rank,
target layers, and dropout around the best practical-v2 configuration. Core
and full retain the broader research sweeps.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, List

from project_paths import output_path


HERE = Path(__file__).resolve().parent
PROFILE_CHOICES = ["smoke", "practical", "lora_ablation", "core", "full"]


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    axis: str
    value: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    n_positive: int = 300
    n_negative: int = 300
    hard_negative_fraction: float = 0.0
    learning_rate: float = 2e-4
    epochs: int = 3
    target_modules: str = ""
    seed: int = 42


BASELINE = Experiment("baseline", "baseline", "default")


def unique(experiments: Iterable[Experiment]) -> List[Experiment]:
    seen = set()
    result = []
    for experiment in experiments:
        if experiment.experiment_id not in seen:
            result.append(experiment)
            seen.add(experiment.experiment_id)
    return result


def build_matrix(profile: str) -> List[Experiment]:
    if profile == "smoke":
        return [
            BASELINE,
            replace(BASELINE, experiment_id="rank_r8", axis="rank", value="8",
                    lora_r=8, lora_alpha=16),
            replace(BASELINE, experiment_id="ratio_1to0", axis="positive_negative_ratio",
                    value="1:0", n_negative=0),
            replace(BASELINE, experiment_id="hardneg_50", axis="hard_negative_fraction",
                    value="0.50", hard_negative_fraction=0.5),
        ]

    if profile == "practical":
        robust = replace(
            BASELINE,
            experiment_id="robust_h50",
            axis="hard_negative_robustness",
            value="all-linear,h50",
            hard_negative_fraction=0.5,
        )
        robust_qv = replace(
            robust,
            experiment_id="robust_qv",
            axis="module_compression",
            value="q_proj,v_proj",
            target_modules="q_proj,v_proj",
        )
        robust_qv_r8 = replace(
            robust_qv,
            experiment_id="robust_qv_r8",
            axis="rank_compression",
            value="q_proj,v_proj,r8",
            lora_r=8,
            lora_alpha=16,
        )
        return [
            BASELINE,
            robust,
            robust_qv,
            robust_qv_r8,
            replace(
                robust_qv_r8,
                experiment_id="robust_qv_r8_small",
                axis="training_data_cost",
                value="150+150",
                n_positive=150,
                n_negative=150,
            ),
            replace(
                robust_qv_r8,
                experiment_id="robust_qv_r8_epoch1",
                axis="training_time",
                value="1 epoch",
                epochs=1,
            ),
            replace(
                robust_qv_r8,
                experiment_id="robust_qv_r8_negheavy",
                axis="safety_weighting",
                value="300+450,h67",
                n_negative=450,
                hard_negative_fraction=0.67,
            ),
        ]

    if profile == "lora_ablation":
        reference = replace(
            BASELINE,
            experiment_id="lora_ref",
            axis="reference",
            value="qv,r8,d0.05",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            hard_negative_fraction=0.5,
            target_modules="q_proj,v_proj",
        )
        return [
            reference,
            replace(
                reference,
                experiment_id="rank_r4",
                axis="rank",
                value="4",
                lora_r=4,
                lora_alpha=8,
            ),
            replace(
                reference,
                experiment_id="rank_r16",
                axis="rank",
                value="16",
                lora_r=16,
                lora_alpha=32,
            ),
            replace(
                reference,
                experiment_id="layers_attention",
                axis="target_modules",
                value="q,k,v,o",
                target_modules="q_proj,k_proj,v_proj,o_proj",
            ),
            replace(
                reference,
                experiment_id="layers_all_linear",
                axis="target_modules",
                value="all-linear",
                target_modules="all-linear",
            ),
            replace(
                reference,
                experiment_id="dropout_0",
                axis="lora_dropout",
                value="0.00",
                lora_dropout=0.0,
            ),
            replace(
                reference,
                experiment_id="dropout_10",
                axis="lora_dropout",
                value="0.10",
                lora_dropout=0.1,
            ),
        ]

    experiments = [BASELINE]
    for rank in (4, 8, 32):
        experiments.append(replace(
            BASELINE,
            experiment_id=f"rank_r{rank}",
            axis="rank",
            value=str(rank),
            lora_r=rank,
            lora_alpha=2 * rank,
        ))
    for alpha, ratio in ((16, 1), (64, 4)):
        experiments.append(replace(
            BASELINE,
            experiment_id=f"alpha_over_r_{ratio}",
            axis="alpha_over_rank",
            value=str(ratio),
            lora_alpha=alpha,
        ))
    for size in (50, 100):
        experiments.append(replace(
            BASELINE,
            experiment_id=f"scale_p{size}_n{size}",
            axis="training_data_scale",
            value=f"{size}+{size}",
            n_positive=size,
            n_negative=size,
        ))
    for negatives, label in ((0, "1:0"), (75, "4:1"), (150, "2:1"), (600, "1:2")):
        experiments.append(replace(
            BASELINE,
            experiment_id=f"ratio_p300_n{negatives}",
            axis="positive_negative_ratio",
            value=label,
            n_negative=negatives,
        ))
    for fraction in (0.25, 0.5):
        experiments.append(replace(
            BASELINE,
            experiment_id=f"hardneg_{int(fraction * 100)}",
            axis="hard_negative_fraction",
            value=f"{fraction:.2f}",
            hard_negative_fraction=fraction,
        ))
    for learning_rate in (1e-4, 5e-4):
        experiments.append(replace(
            BASELINE,
            experiment_id=f"lr_{learning_rate:.0e}".replace("-", "m"),
            axis="learning_rate",
            value=f"{learning_rate:g}",
            learning_rate=learning_rate,
        ))
    for epochs in (1, 5):
        experiments.append(replace(
            BASELINE,
            experiment_id=f"epochs_{epochs}",
            axis="epochs",
            value=str(epochs),
            epochs=epochs,
        ))
    experiments.extend([
        replace(
            BASELINE,
            experiment_id="targets_qv",
            axis="target_modules",
            value="q_proj,v_proj",
            target_modules="q_proj,v_proj",
        ),
        replace(
            BASELINE,
            experiment_id="targets_attention",
            axis="target_modules",
            value="attention projections",
            target_modules="q_proj,k_proj,v_proj,o_proj",
        ),
    ])

    if profile == "full":
        experiments.extend([
            replace(BASELINE, experiment_id="scale_p600_n600", axis="training_data_scale",
                    value="600+600", n_positive=600, n_negative=600),
            replace(BASELINE, experiment_id="dropout_0", axis="lora_dropout",
                    value="0.00", lora_dropout=0.0),
            replace(BASELINE, experiment_id="dropout_10", axis="lora_dropout",
                    value="0.10", lora_dropout=0.1),
            replace(BASELINE, experiment_id="seed_13", axis="seed_repeat",
                    value="13", seed=13),
            replace(BASELINE, experiment_id="seed_73", axis="seed_repeat",
                    value="73", seed=73),
        ])
        for rank in (4, 32):
            size = 50
            experiments.append(replace(
                BASELINE,
                experiment_id=f"interaction_r{rank}_p{size}_n{size}",
                axis="rank_x_data_scale",
                value=f"r={rank},data={size}+{size}",
                lora_r=rank,
                lora_alpha=2 * rank,
                n_positive=size,
                n_negative=size,
            ))
    if profile not in {"core", "full"}:
        raise ValueError(f"unknown profile: {profile}")
    return unique(experiments)


def selected_matrix(profile: str, only: str) -> List[Experiment]:
    matrix = build_matrix(profile)
    if not only:
        return matrix
    requested = {value.strip() for value in only.split(",") if value.strip()}
    selected = [experiment for experiment in matrix if experiment.experiment_id in requested]
    missing = requested - {experiment.experiment_id for experiment in selected}
    if missing:
        raise SystemExit(f"unknown experiment ids for profile {profile}: {sorted(missing)}")
    return selected


def print_plan(matrix: List[Experiment]) -> None:
    header = (
        f"{'id':30s} {'axis':27s} {'value':22s} {'r':>3s} {'drop':>5s} "
        f"{'targets':24s} {'pos':>4s} {'neg':>4s}"
    )
    print(header)
    print("-" * len(header))
    for item in matrix:
        targets = item.target_modules or "auto"
        print(
            f"{item.experiment_id:30s} {item.axis:27s} {item.value:22s} "
            f"{item.lora_r:3d} {item.lora_dropout:5.2f} "
            f"{targets:24s} {item.n_positive:4d} {item.n_negative:4d}"
        )
    print(f"\nTotal runs: {len(matrix)}")


def redact_command(command: List[str]) -> str:
    redacted = list(command)
    if "--hf-token" in redacted:
        index = redacted.index("--hf-token")
        if index + 1 < len(redacted):
            redacted[index + 1] = "<redacted>"
    return subprocess.list2cmdline(redacted)


def run_logged(command: List[str], log_path: Path, dry_run: bool) -> int:
    print(f"$ {redact_command(command)}")
    if dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if os.name == "nt" and "torch.distributed.run" in command:
        env["USE_LIBUV"] = "0"
        print(
            "[experiments] Windows torchrun compatibility: FileStore "
            "rendezvous, USE_LIBUV=0"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=HERE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            assert process.stdout is not None
            warned_about_encoding = False
            for raw_line in process.stdout:
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    line = raw_line.decode("utf-8", errors="backslashreplace")
                    had_decode_error = True
                else:
                    had_decode_error = False
                if had_decode_error and not warned_about_encoding:
                    print(
                        "[experiments] child output contained non-UTF-8 bytes; "
                        "invalid bytes were escaped in the console and log."
                    )
                    warned_about_encoding = True
                print(line, end="")
                log.write(line)
            return process.wait()
    finally:
        if "--rdzv-conf=store_type=file" in command:
            for argument in command:
                if argument.startswith("--rdzv-endpoint="):
                    try:
                        Path(argument.split("=", 1)[1]).unlink(missing_ok=True)
                    except OSError:
                        pass
                    break


def corpus_size(path: Path) -> int:
    corpus = json.loads(path.read_text(encoding="utf-8-sig"))
    items = corpus.get("items") or []
    return len(items)


def train_command(args, experiment: Experiment, adapter_path: Path) -> List[str]:
    if args.num_gpus > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
        ]
        if os.name == "nt":
            rendezvous_id = f"stegolora-exp-{uuid.uuid4().hex}"
            rendezvous_file = (
                Path(tempfile.gettempdir()) / f"{rendezvous_id}.rdzv"
            )
            command.extend([
                "--rdzv-backend=c10d",
                f"--rdzv-endpoint={rendezvous_file}",
                f"--rdzv-id={rendezvous_id}",
                "--rdzv-conf=store_type=file",
            ])
        else:
            command.append("--standalone")
        command.extend([
            f"--nproc_per_node={args.num_gpus}",
            str(HERE / "train.py"),
        ])
    else:
        command = [sys.executable, str(HERE / "train.py")]
    command.extend([
        "--base-model", args.model,
        "--model-dir", args.model_dir,
        "--device", args.device,
        "--dtype", args.dtype,
        "--hf-token", args.hf_token,
        "--corpus-path", str(Path(args.corpus_path).resolve()),
        "--completion-format", "tool_call",
        "--tool-model-name", args.watermark_model or args.model,
        "--n-positive", str(experiment.n_positive),
        "--n-negative", str(experiment.n_negative),
        "--hard-negative-fraction", str(experiment.hard_negative_fraction),
        "--epochs", str(experiment.epochs),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--lr", str(experiment.learning_rate),
        "--warmup-ratio", str(args.warmup_ratio),
        "--weight-decay", str(args.weight_decay),
        "--lora-r", str(experiment.lora_r),
        "--lora-alpha", str(experiment.lora_alpha),
        "--lora-dropout", str(experiment.lora_dropout),
        "--target-modules", experiment.target_modules,
        "--max-length", str(args.max_length),
        "--prompt-format", args.prompt_format,
        "--seed", str(experiment.seed),
        "--output-dir", str(adapter_path),
    ])
    if args.ddp_backend:
        command.extend(["--ddp-backend", args.ddp_backend])
    if args.qlora:
        command.append("--qlora")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def evaluation_command(args, adapter_path: Path, output_path: Path, eval_offset: int) -> List[str]:
    command = [
        sys.executable,
        str(HERE / "evaluate.py"),
        "--base-model", args.model,
        "--model-dir", args.model_dir,
        "--watermark-model", args.watermark_model or args.model,
        "--watermark-model-dir", args.watermark_model_dir or args.model_dir,
        "--device", args.device,
        "--dtype", args.dtype,
        "--hf-token", args.hf_token,
        "--adapter-path", str(adapter_path),
        "--corpus-path", str(Path(args.corpus_path).resolve()),
        "--corpus-offset", str(eval_offset),
        "--n-trigger", str(args.eval_samples),
        "--n-normal", str(args.normal_samples),
        "--max-new-tokens", str(args.eval_max_new_tokens),
        "--prompt-format", args.prompt_format,
        "--mcp-samples", str(args.mcp_samples),
        "--mcp-timeout", str(args.mcp_timeout),
        "--progress-every", str(args.eval_progress_every),
        "--output", str(output_path),
    ]
    if args.load_in_4bit_eval:
        command.append("--load-in-4bit")
    if args.compare_base_normal or args.profile in {"practical", "lora_ablation"}:
        command.append("--compare-base-normal")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def resolved_experiment_root(configured: str, profile: str) -> Path:
    path = configured or output_path("experiments", profile)
    return Path(path).resolve()


def run_matrix(args) -> int:
    matrix = selected_matrix(args.profile, args.only)
    root = resolved_experiment_root(args.output_root, args.profile)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "experiment_plan.json", [asdict(item) for item in matrix])
    corpus_path = Path(args.corpus_path)
    if not corpus_path.exists():
        raise SystemExit(f"corpus not found: {corpus_path}")
    max_training_items = max(item.n_positive for item in matrix)
    eval_offset = args.eval_offset if args.eval_offset >= 0 else max_training_items
    required = max(max_training_items, eval_offset + args.eval_samples)
    available = corpus_size(corpus_path)
    if available < required:
        raise SystemExit(
            f"corpus has {available} items; experiments require at least {required} "
            f"({eval_offset} train-prefix/offset + {args.eval_samples} held-out)"
        )

    failures = 0
    for index, experiment in enumerate(matrix, start=1):
        run_dir = root / experiment.experiment_id
        adapter_path = run_dir / "adapter"
        evaluation_path = run_dir / "evaluation.json"
        status_path = run_dir / "status.json"
        if args.resume and evaluation_path.exists():
            print(f"[{index}/{len(matrix)}] skip complete {experiment.experiment_id}")
            continue
        print(f"\n[{index}/{len(matrix)}] {experiment.experiment_id}: {experiment.axis}={experiment.value}")
        write_json(run_dir / "config.json", asdict(experiment))
        started = time.time()
        status = {"status": "training", "started_at_unix": started}
        write_json(status_path, status)

        adapter_ready = (adapter_path / "adapter_config.json").exists()
        if args.resume and adapter_ready:
            print(f"  reusing trained adapter at {adapter_path}")
            train_rc = 0
        else:
            train_rc = run_logged(
                train_command(args, experiment, adapter_path),
                run_dir / "train.log",
                args.dry_run,
            )
        if train_rc != 0:
            failures += 1
            status.update({"status": "train_failed", "returncode": train_rc})
            write_json(status_path, status)
            if args.fail_fast:
                return train_rc
            continue

        status["status"] = "evaluating"
        write_json(status_path, status)
        eval_rc = run_logged(
            evaluation_command(args, adapter_path, evaluation_path, eval_offset),
            run_dir / "evaluate.log",
            args.dry_run,
        )
        status.update({
            "status": "dry_run" if args.dry_run else ("complete" if eval_rc == 0 else "eval_failed"),
            "returncode": eval_rc,
            "elapsed_seconds": time.time() - started,
            "eval_offset": eval_offset,
        })
        write_json(status_path, status)
        if eval_rc != 0:
            failures += 1
            if args.fail_fast:
                return eval_rc
    return 1 if failures else 0


def nested(value: dict, *keys, default=None):
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summary_rows(root: Path) -> List[dict]:
    rows = []
    for config_path in sorted(root.glob("*/config.json")):
        run_dir = config_path.parent
        config = load_json(config_path)
        evaluation = load_json(run_dir / "evaluation.json")
        training = load_json(run_dir / "adapter" / "training_metadata.json")
        adapter_file = run_dir / "adapter" / "adapter_model.safetensors"
        if not evaluation:
            continue
        negative = evaluation.get("negative_sets", {})
        triggered = evaluation.get("triggered", {})
        preservation = evaluation.get("normal_preservation") or {}
        normal_metrics = negative.get("normal", {})
        false_rates = [
            nested(negative, name, "false_activation_rate", default=0.0)
            for name in (
                "normal",
                "carrier_without_trigger",
                "near_trigger",
                "quoted_trigger_attack",
                "embedded_trigger_attack",
            )
            if name in negative
        ]
        if not false_rates:
            false_rates = [0.0]
        end_to_end = triggered.get("end_to_end_accuracy_overall", 0.0)
        worst_false_activation = max(false_rates)
        rows.append({
            "experiment_id": config.get("experiment_id"),
            "axis": config.get("axis"),
            "value": config.get("value"),
            "lora_r": config.get("lora_r"),
            "lora_alpha": config.get("lora_alpha"),
            "lora_dropout": config.get("lora_dropout"),
            "trainable_params": training.get("trainable_params"),
            "adapter_size_mb": (
                round(adapter_file.stat().st_size / (1024 * 1024), 3)
                if adapter_file.exists()
                else None
            ),
            "n_positive": config.get("n_positive"),
            "n_negative": config.get("n_negative"),
            "hard_negative_fraction": config.get("hard_negative_fraction"),
            "learning_rate": config.get("learning_rate"),
            "epochs": config.get("epochs"),
            "target_modules": config.get("target_modules") or "all-linear(default)",
            "seed": config.get("seed"),
            "train_loss": nested(training, "training_metrics", "train_loss"),
            "train_runtime": nested(training, "training_metrics", "train_runtime"),
            "tool_route_rate": triggered.get("correct_tool_rate"),
            "schema_valid_rate": triggered.get("schema_valid_rate"),
            "end_to_end_accuracy": end_to_end,
            "normal_false_activation": false_rates[0],
            "carrier_false_activation": false_rates[1],
            "near_trigger_false_activation": false_rates[2],
            "quoted_trigger_activation": nested(
                negative, "quoted_trigger_attack", "false_activation_rate",
            ),
            "embedded_trigger_activation": nested(
                negative, "embedded_trigger_attack", "false_activation_rate",
            ),
            "normal_prefix_agreement": preservation.get("mean_token_prefix_agreement"),
            "adapter_normal_unique_output_rate": preservation.get(
                "adapter_unique_output_rate"
            ),
            "base_normal_unique_output_rate": preservation.get(
                "base_unique_output_rate"
            ),
            "adapter_normal_dominant_output_rate": normal_metrics.get(
                "dominant_output_rate"
            ),
            "mcp_accuracy": triggered.get("mcp_accuracy"),
            "mean_false_activation": sum(false_rates) / len(false_rates),
            "worst_false_activation": worst_false_activation,
            "balanced_score": end_to_end - worst_false_activation,
        })
    return rows


def summarize(args) -> int:
    root = resolved_experiment_root(args.output_root, args.profile)
    rows = summary_rows(root)
    if not rows:
        print(f"No completed evaluation results found under {root}")
        return 1
    rows.sort(key=lambda row: row["balanced_score"], reverse=True)
    csv_path = Path(args.csv) if args.csv else root / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{'experiment':30s} {'e2e':>7s} {'normalFP':>9s} "
        f"{'worstFP':>8s} {'preserve':>9s} {'score':>8s}"
    )
    print("-" * 82)
    for row in rows[: args.top]:
        preservation = row["normal_prefix_agreement"]
        preservation_text = "n/a" if preservation is None else f"{preservation:.2%}"
        print(
            f"{row['experiment_id']:30s} "
            f"{row['end_to_end_accuracy']:7.2%} "
            f"{row['normal_false_activation']:9.2%} "
            f"{row['worst_false_activation']:8.2%} "
            f"{preservation_text:>9s} "
            f"{row['balanced_score']:8.3f}"
        )
    print(f"\nSummary written to {csv_path}")
    return 0


def add_matrix_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="practical",
    )
    parser.add_argument("--only", default="", help="Comma-separated experiment IDs.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Print the experiment matrix without training.")
    add_matrix_options(plan)
    plan.add_argument("--output", default="")

    run = subparsers.add_parser("run", help="Train and evaluate every selected experiment.")
    add_matrix_options(run)
    run.add_argument("--model", required=True)
    run.add_argument("--model-dir", default=os.environ.get("STEGOLORA_MODEL_DIR", ""))
    run.add_argument("--watermark-model", "--carrier-model", dest="watermark_model",
                     default="", help="Stego/watermark carrier tokenizer used by "
                     "embedding/extraction. --watermark-model is retained for compatibility.")
    run.add_argument("--watermark-model-dir", "--carrier-model-dir",
                     dest="watermark_model_dir", default="",
                     help="Stego/watermark carrier model root. Empty uses --model-dir.")
    run.add_argument("--corpus-path", required=True)
    run.add_argument(
        "--output-root",
        default="",
        help="Run directory. Default: outputs/experiments/<profile>.",
    )
    run.add_argument("--device", default="auto")
    run.add_argument("--dtype", default="")
    run.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    run.add_argument("--trust-remote-code", action="store_true")
    run.add_argument("--qlora", action="store_true")
    run.add_argument("--num-gpus", type=int, default=1)
    run.add_argument("--ddp-backend", choices=["nccl", "gloo"], default=None)
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--gradient-accumulation-steps", type=int, default=4)
    run.add_argument("--warmup-ratio", type=float, default=0.03)
    run.add_argument("--weight-decay", type=float, default=0.0)
    run.add_argument("--max-length", type=int, default=512)
    run.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    run.add_argument("--eval-offset", type=int, default=-1,
                     help="Held-out corpus start. Default: largest training positive count.")
    run.add_argument("--eval-samples", type=int, default=100)
    run.add_argument("--normal-samples", type=int, default=50)
    run.add_argument("--eval-max-new-tokens", type=int, default=160)
    run.add_argument("--eval-progress-every", type=int, default=10)
    run.add_argument("--load-in-4bit-eval", action="store_true")
    run.add_argument(
        "--compare-base-normal",
        action="store_true",
        help="Compare adapter and base outputs on unseen normal prompts. "
             "Enabled automatically for practical and lora_ablation profiles.",
    )
    run.add_argument("--mcp-samples", type=int, default=0)
    run.add_argument("--mcp-timeout", type=float, default=60.0)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--fail-fast", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    summary = subparsers.add_parser("summarize", help="Create a CSV from completed runs.")
    summary.add_argument("--profile", choices=PROFILE_CHOICES, default="practical")
    summary.add_argument(
        "--output-root",
        default="",
        help="Run directory. Default: outputs/experiments/<profile>.",
    )
    summary.add_argument("--csv", default="")
    summary.add_argument("--top", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "plan":
        matrix = selected_matrix(args.profile, args.only)
        print_plan(matrix)
        if args.output:
            write_json(Path(args.output), [asdict(item) for item in matrix])
        return 0
    if args.command == "run":
        return run_matrix(args)
    return summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
