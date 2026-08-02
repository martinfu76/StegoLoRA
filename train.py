"""
LoRA/QLoRA training for trigger-based extraction behavior.

Loads a base causal LM, attaches a LoRA adapter, and trains on
synthetic (trigger + stego -> extracted message) pairs.

Usage:
    set STEGOLORA_MODEL_DIR=D:\\models
    python train.py --base-model Llama-3.2-1B --output-dir ./outputs/adapters/lora_adapter
    python train.py --base-model gpt2 --target-modules c_attn,c_proj --epochs 5
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

from data import build_dataset, TRIGGER
from model_utils import (
    DEFAULT_MODEL_DIR_ENV,
    configure_process_output,
    resolve_device,
    resolve_dtype,
    resolve_model_path,
    resolve_target_modules,
    require_bitsandbytes,
    single_gpu_environment,
)
from project_paths import output_path


class ExtractionDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]], tokenizer, max_length: int = 512,
                 prompt_format: str = "auto"):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_format = prompt_format

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        use_chat = self.prompt_format == "chat" or (
            self.prompt_format == "auto" and has_chat_template
        )
        if use_chat:
            if not has_chat_template:
                raise ValueError("chat prompt format requested but tokenizer has no chat_template")
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            input_ids = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["completion"]},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
            if input_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError("chat template assistant prefix is inconsistent between train prompts")
            labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        else:
            prompt_ids = self.tokenizer.encode(ex["prompt"], add_special_tokens=False)
            completion_ids = self.tokenizer.encode(ex["completion"], add_special_tokens=False)
            if self.tokenizer.eos_token_id is not None:
                completion_ids = completion_ids + [self.tokenizer.eos_token_id]
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PadCollator:
    """Dynamically pads a batch to the longest sequence in that batch.

    `DataCollatorForLanguageModeling(mlm=False)` does not pad by default, which
    crashes with "expected sequence of length 16 at dim 1 (got 80)" the moment
    a batch contains samples of different lengths (almost always the case for
    variable-length prompt + completion pairs).
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            n_pad = max_len - f["input_ids"].size(0)
            input_ids.append(torch.cat([
                f["input_ids"],
                torch.full((n_pad,), self.pad_id, dtype=torch.long),
            ]))
            attention_mask.append(torch.cat([
                f["attention_mask"],
                torch.zeros(n_pad, dtype=torch.long),
            ]))
            labels.append(torch.cat([
                f["labels"],
                torch.full((n_pad,), -100, dtype=torch.long),
            ]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="gpt2")
    p.add_argument("--model-dir", default=os.environ.get(DEFAULT_MODEL_DIR_ENV, ""),
                   help=f"Directory holding local model snapshots. Falls back to "
                        f"env {DEFAULT_MODEL_DIR_ENV}.")
    p.add_argument("--dtype", default="",
                   help="float32/float16/bfloat16. Auto: bfloat16 on CUDA, float32 otherwise.")
    p.add_argument("--device", default="auto",
                   help="torch device: 'auto', 'cuda', 'cuda:0', 'cpu'. Default 'auto' picks CUDA when available.")
    p.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""),
                   help="HF token for gated repos (e.g. meta-llama/*).")
    p.add_argument("--output-dir", default=output_path("adapters", "lora_adapter"))
    p.add_argument("--n-positive", type=int, default=500)
    p.add_argument("--n-negative", type=int, default=500)
    p.add_argument("--hard-negative-fraction", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="",
                   help="Comma-separated module names or 'all-linear'. Empty uses "
                        "all-linear for QLoRA and family defaults for ordinary LoRA.")
    p.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto",
                   help="auto uses the tokenizer chat template when available; raw preserves GPT-2 behavior.")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--qlora", action="store_true",
                   help="Load the base model in 4-bit and train a PEFT LoRA adapter. "
                        "Requires CUDA, bitsandbytes, transformers, accelerate, and peft.")
    p.add_argument("--bnb-4bit-quant-type", choices=["nf4", "fp4"], default="nf4",
                   help="bitsandbytes 4-bit storage type. NF4 is recommended for QLoRA training.")
    p.add_argument("--bnb-4bit-compute-dtype", choices=["", "float16", "bfloat16", "float32"], default="",
                   help="QLoRA matmul dtype. Auto chooses bf16 when supported, otherwise fp16.")
    p.add_argument("--no-double-quant", action="store_true",
                   help="Disable nested/double quantization in QLoRA mode.")
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="Disable gradient checkpointing. QLoRA enables it by default to save memory.")
    p.add_argument("--optim", default="",
                   help="Trainer optimizer. Auto: paged_adamw_8bit for QLoRA, adamw_torch otherwise.")
    p.add_argument("--ddp-backend", choices=["nccl", "gloo"], default=None,
                   help="Distributed backend. Auto: nccl on Linux, gloo on Windows.")
    p.add_argument("--local-rank", "--local_rank", type=int, default=-1,
                   help=argparse.SUPPRESS)
    p.add_argument("--completion-format", choices=["extracted", "tool_call"], default="extracted",
                   help="Training target format. 'extracted' = 'Extracted: MSG'. "
                        "'tool_call' =  ```{...}```.")
    p.add_argument("--tool-model-name", default="gpt2",
                   help="model_name field embedded in tool_call training targets.")
    p.add_argument("--tool-key", default="$KEY",
                   help="Compatibility option. For tool_call targets the key "
                        "is emitted as $KEY and filled by agent.py/pipeline.py "
                        "at runtime.")
    p.add_argument("--corpus-path", default="",
                   help="JSON corpus from corpus_build.py. If set, positive examples "
                        "draw real hash-watermarked text from here instead of stub.")
    return p.parse_args()


def main():
    configure_process_output()
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        try:
            isolated_env, normalized_device, selected_device = single_gpu_environment(
                args.device, os.environ
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if selected_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = isolated_env["CUDA_VISIBLE_DEVICES"]
            args.device = normalized_device
            print(
                "Single-GPU isolation: "
                f"CUDA_VISIBLE_DEVICES={selected_device}, device={normalized_device}"
            )
    if torch is None:
        raise SystemExit(
            "train.py requires torch plus transformers and peft. "
            "Activate the project ML/Conda environment before training."
        )
    if args.qlora:
        try:
            version = require_bitsandbytes()
            print(f"bitsandbytes preflight: version {version}")
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    from peft import LoraConfig, TaskType, get_peft_model
    if args.qlora:
        try:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:
            raise SystemExit(
                "QLoRA requires recent transformers and peft releases plus "
                "accelerate and bitsandbytes."
            ) from exc

    model_path = resolve_model_path(args.base_model, args.model_dir or None)
    env_local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    local_rank = env_local_rank if env_local_rank >= 0 else args.local_rank
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise SystemExit("multi-GPU training requires CUDA")
        if local_rank < 0:
            raise SystemExit("WORLD_SIZE is set but LOCAL_RANK is missing; launch with torchrun")
        if local_rank >= torch.cuda.device_count():
            raise SystemExit(
                f"LOCAL_RANK={local_rank} exceeds visible CUDA device count "
                f"{torch.cuda.device_count()}; check --num-gpus and CUDA_VISIBLE_DEVICES"
            )
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = resolve_device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    hf_token = args.hf_token or None

    if args.qlora:
        if device.type != "cuda":
            raise SystemExit("--qlora currently requires a CUDA device; pass --device cuda or cuda:N")
        if args.bnb_4bit_compute_dtype:
            dtype = resolve_dtype(args.bnb_4bit_compute_dtype)
        elif args.dtype:
            dtype = resolve_dtype(args.dtype)
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = resolve_dtype(args.dtype)

    rank = int(os.environ.get("RANK", "0"))
    print(f"Base model: {args.base_model} -> resolved: {model_path}")
    training_mode = "QLoRA (4-bit base)" if args.qlora else "LoRA"
    print(f"Training mode: {training_mode}")
    print(
        f"Loading with torch_dtype={dtype}, device={device}, "
        f"rank={rank}/{world_size}, local_rank={local_rank}"
    )
    if hf_token:
        print("Using provided HF token for gated repos")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=hf_token, trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "token": hf_token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=not args.no_double_quant,
        )
        model_kwargs["device_map"] = {
            "": local_rank if distributed else (device.index if device.index is not None else 0)
        }

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if args.qlora:
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=not args.no_gradient_checkpointing,
        )
    else:
        model = model.to(device)

    target_modules = resolve_target_modules(args.base_model, args.target_modules, args.qlora)
    print(f"LoRA target modules: {target_modules}")
    if args.qlora and isinstance(target_modules, list) and any(module.startswith("c_") for module in target_modules):
        print(
            "WARNING: GPT-2 uses Conv1D-style projections. bitsandbytes primarily "
            "quantizes Linear layers, so GPT-2 gets little QLoRA memory benefit. "
            "Use a Llama/Qwen/Mistral-style model for a representative QLoRA run."
        )
    lora_config_kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    if isinstance(target_modules, list) and any(m.startswith("c_") for m in target_modules):
        lora_config_kwargs["fan_in_fan_out"] = True
    lora_config = LoraConfig(**lora_config_kwargs)
    model = get_peft_model(model, lora_config)
    if rank == 0:
        model.print_trainable_parameters()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    examples = build_dataset(
        n_positive=args.n_positive,
        n_negative=args.n_negative,
        seed=args.seed,
        completion_format=args.completion_format,
        tool_model_name=args.tool_model_name,
        tool_key=args.tool_key,
        corpus_path=args.corpus_path,
        hard_negative_fraction=args.hard_negative_fraction,
    )
    source = f"corpus={args.corpus_path}" if args.corpus_path else "stub extractor"
    actual_positive = sum(ex["prompt"].startswith(TRIGGER) for ex in examples)
    actual_negative = len(examples) - actual_positive
    print(f"Built {len(examples)} training examples "
          f"({actual_positive} trigger, {actual_negative} normal); "
          f"completion_format={args.completion_format}; source={source}")

    train_dataset = ExtractionDataset(
        examples,
        tokenizer,
        max_length=args.max_length,
        prompt_format=args.prompt_format,
    )

    use_fp16 = (
        not args.no_fp16
        and device.type == "cuda"
        and dtype == torch.float16
    )
    use_bf16 = device.type == "cuda" and dtype == torch.bfloat16
    optimizer = args.optim or ("paged_adamw_8bit" if args.qlora else "adamw_torch")
    ddp_backend = args.ddp_backend or ("gloo" if os.name == "nt" else "nccl")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=20,
        disable_tqdm=True,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        no_cuda=(device.type == "cpu"),
        fp16=use_fp16,
        bf16=use_bf16,
        optim=optimizer,
        gradient_checkpointing=args.qlora and not args.no_gradient_checkpointing,
        ddp_backend=ddp_backend if distributed else None,
        ddp_find_unused_parameters=False if distributed else None,
        local_rank=local_rank,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=PadCollator(tokenizer),
    )

    train_result = trainer.train()

    trainer.save_model(args.output_dir)
    if not trainer.is_world_process_zero():
        return
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "base_model": args.base_model,
        "base_model_resolved_path": model_path,
        "trigger": TRIGGER,
        "n_positive_requested": args.n_positive,
        "n_negative_requested": args.n_negative,
        "n_positive": actual_positive,
        "n_negative": actual_negative,
        "hard_negative_fraction": args.hard_negative_fraction,
        "n_hard_negative": round(actual_negative * args.hard_negative_fraction),
        "n_normal_negative": actual_negative - round(
            actual_negative * args.hard_negative_fraction
        ),
        "positive_to_negative_ratio": (
            actual_positive / actual_negative if actual_negative else None
        ),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "prompt_format": args.prompt_format,
        "dtype": str(dtype).replace("torch.", ""),
        "training_mode": "qlora" if args.qlora else "lora",
        "quantization": {
            "load_in_4bit": True,
            "quant_type": args.bnb_4bit_quant_type,
            "compute_dtype": str(dtype).replace("torch.", ""),
            "double_quant": not args.no_double_quant,
        } if args.qlora else None,
        "optimizer": optimizer,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": world_size,
        "distributed_backend": ddp_backend if distributed else None,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps * world_size
        ),
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percent": 100.0 * trainable_params / total_params,
        "training_metrics": train_result.metrics,
        "completion_format": args.completion_format,
        "tool_model_name": args.tool_model_name,
        "tool_key": args.tool_key,
        "corpus_path": args.corpus_path or None,
    }
    Path(args.output_dir, "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
