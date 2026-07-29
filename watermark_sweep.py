"""Sweep watermark capacity, recovery, and carrier-quality trade-offs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

from corpus_build import PROMPTS, random_message
from hash_watermark import (
    generate_watermarked,
    payload_capacity,
    required_tokens,
    verify_text_extraction,
)
from model_utils import (
    DEFAULT_MODEL_DIR_ENV,
    four_bit_load_kwargs,
    render_prompt,
    resolve_device,
    resolve_dtype,
    resolve_model_path,
)


def integer_list(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return result


def continuation_nll(model, tokenizer, prompt: str, token_ids: list[int]) -> float:
    import torch
    import torch.nn.functional as functional

    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = prompt_inputs["input_ids"]
    continuation = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    full_ids = torch.cat([prompt_ids, continuation], dim=1)
    attention_mask = torch.ones_like(full_ids)
    with torch.no_grad():
        logits = model(input_ids=full_ids, attention_mask=attention_mask).logits
    start = prompt_ids.shape[1] - 1
    continuation_logits = logits[:, start:-1, :]
    return float(functional.cross_entropy(
        continuation_logits.reshape(-1, continuation_logits.shape[-1]),
        continuation.reshape(-1),
    ).item())


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model_path(args.base_model, args.model_dir or None)
    dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)
    if args.load_in_4bit and not args.dtype and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    token = args.hf_token or None
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=token, trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "torch_dtype": dtype,
        "token": token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.load_in_4bit:
        kwargs.update(four_bit_load_kwargs(device, dtype))
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if not args.load_in_4bit:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_setting(model, tokenizer, args, bits_per_token: int, message_chars: int) -> dict:
    framed = not args.legacy_watermark
    exemplar = "A" * message_chars
    needed = required_tokens(exemplar, bits_per_token, framed)
    base = {
        "bits_per_token": bits_per_token,
        "message_chars": message_chars,
        "max_new_tokens": args.max_new_tokens,
        "required_tokens": needed,
        "payload_capacity_bytes": payload_capacity(
            args.max_new_tokens, bits_per_token, framed,
        ),
        "framed": framed,
    }
    if needed > args.max_new_tokens:
        return {**base, "status": "insufficient_token_budget"}

    rng = random.Random(args.seed + bits_per_token * 1000 + message_chars)
    key = args.key.encode("utf-8")
    recovered = 0
    bit_accuracy = 0.0
    nll_values = []
    unique_ratios = []
    generated_lengths = []
    elapsed = 0.0
    samples = []
    for index in range(args.n_samples):
        prompt_text = PROMPTS[index % len(PROMPTS)]
        prompt = render_prompt(tokenizer, prompt_text, args.prompt_format)
        message = random_message(rng, message_chars)
        started = time.perf_counter()
        text, token_ids = generate_watermarked(
            model,
            tokenizer,
            prompt,
            message,
            key=key,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            temperature=args.temperature,
            top_k=args.top_k,
            bits_per_token=bits_per_token,
            framed=framed,
        )
        elapsed += time.perf_counter() - started
        check = verify_text_extraction(
            tokenizer,
            text,
            message,
            key,
            bits_per_token,
            framed,
        )
        recovered += int(check["fully_recovered"])
        bit_accuracy += check["bit_accuracy"]
        constrained_ids = token_ids[:needed]
        unique_ratios.append(len(set(constrained_ids)) / max(len(constrained_ids), 1))
        generated_lengths.append(len(token_ids))
        if args.measure_nll:
            nll_values.append(continuation_nll(
                model, tokenizer, prompt, constrained_ids,
            ))
        if len(samples) < 3:
            samples.append({
                "message": message,
                "extracted": check.get("extracted", ""),
                "recovered": check["fully_recovered"],
                "frame_error": check.get("frame_error"),
                "text_preview": text[:160],
            })

    mean_nll = sum(nll_values) / len(nll_values) if nll_values else None
    return {
        **base,
        "status": "complete",
        "n_samples": args.n_samples,
        "text_recovery_rate": recovered / args.n_samples,
        "mean_bit_accuracy": bit_accuracy / args.n_samples,
        "mean_base_model_nll": mean_nll,
        "base_model_perplexity": math.exp(min(mean_nll, 20.0)) if mean_nll is not None else None,
        "mean_unique_token_ratio": sum(unique_ratios) / len(unique_ratios),
        "mean_generated_tokens": sum(generated_lengths) / len(generated_lengths),
        "seconds_per_sample": elapsed / args.n_samples,
        "samples": samples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-dir", default=os.environ.get(DEFAULT_MODEL_DIR_ENV, ""))
    parser.add_argument("--dtype", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    parser.add_argument("--key", default="experiment-key")
    parser.add_argument("--bits-per-token", type=integer_list, default=[1, 2, 3, 4])
    parser.add_argument("--message-chars", type=integer_list, default=[4, 8, 16])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--legacy-watermark", action="store_true")
    parser.add_argument("--measure-nll", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./experiments/watermark_sweep.json")
    parser.add_argument("--csv", default="./experiments/watermark_sweep.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(value not in range(1, 5) for value in args.bits_per_token):
        raise SystemExit("--bits-per-token values must be in [1, 4]")
    if any(value <= 0 for value in args.message_chars):
        raise SystemExit("--message-chars values must be positive")
    model, tokenizer = load_model(args)
    results = []
    for bits_per_token in args.bits_per_token:
        for message_chars in args.message_chars:
            print(f"Evaluating bits/token={bits_per_token}, message_chars={message_chars}")
            result = evaluate_setting(
                model, tokenizer, args, bits_per_token, message_chars,
            )
            results.append(result)
            if result["status"] == "complete":
                print(
                    f"  recovery={result['text_recovery_rate']:.2%}, "
                    f"nll={result['mean_base_model_nll']}, "
                    f"seconds/sample={result['seconds_per_sample']:.2f}"
                )
            else:
                print("  skipped: insufficient token budget")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "base_model": args.base_model,
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [{key: value for key, value in result.items() if key != "samples"} for result in results]
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"JSON: {output}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
