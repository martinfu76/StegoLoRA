"""
Evaluation for the hash-based watermark.

Measures bit-level extraction accuracy over (prompt, message) pairs.

Usage:
    set STEGOLORA_MODEL_DIR=D:\\models
    python evaluate_watermark.py --base-model Llama-3.2-1B --n-samples 50 --message-chars 8
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import List

from hash_watermark import (
    generate_watermarked,
    required_tokens,
    verify_extraction,
    verify_text_extraction,
)
from model_utils import (
    DEFAULT_MODEL_DIR_ENV,
    render_prompt,
    resolve_device,
    resolve_dtype,
    resolve_model_path,
)


PROMPTS = [
    "Once upon a time,",
    "The weather today is",
    "In a small village,",
    "Scientists recently discovered that",
    "The history of art shows",
    "Many people enjoy",
    "Computer programming involves",
    "Traveling to new places",
    "Reading books can",
    "Learning a new language",
    "The importance of education",
    "Modern technology has",
    "Climate change affects",
    "Healthy eating habits",
    "Music has the power",
    "Sports bring people",
    "The natural world",
    "Friendship and trust",
    "Dreams and aspirations",
    "Time passes quickly",
]


def random_message(rng: random.Random, n_chars: int) -> str:
    return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(n_chars))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="gpt2")
    p.add_argument("--model-dir", default=os.environ.get(DEFAULT_MODEL_DIR_ENV, ""))
    p.add_argument("--dtype", default="")
    p.add_argument("--device", default="auto",
                   help="torch device: 'auto', 'cuda', 'cuda:0', 'cpu'. Default 'auto' picks CUDA when available.")
    p.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    p.add_argument("--key", default="experiment-key")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--message-chars", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--bits-per-token", type=int, choices=range(1, 5), default=2)
    p.add_argument("--legacy-watermark", action="store_true")
    p.add_argument("--no-sample", dest="do_sample", action="store_false")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", help="Optional path to dump detailed results JSON")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model_path(args.base_model, args.model_dir or None)
    dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)
    hf_token = args.hf_token or None

    print(f"Loading {model_path} (dtype={dtype}, device={device})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=hf_token, trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    model = model.to(device)
    model.eval()

    key = args.key.encode("utf-8")
    framed = not args.legacy_watermark
    tokens_needed = required_tokens("A" * args.message_chars, args.bits_per_token, framed)
    if tokens_needed > args.max_new_tokens:
        raise SystemExit(
            f"--message-chars {args.message_chars} needs {tokens_needed} tokens but "
            f"--max-new-tokens is {args.max_new_tokens}"
        )

    token_id_recovered = 0
    text_recovered = 0
    total_bit_acc = 0.0
    samples: List[dict] = []

    for i in range(args.n_samples):
        prompt = PROMPTS[i % len(PROMPTS)]
        model_prompt = render_prompt(tokenizer, prompt, args.prompt_format)
        message = random_message(rng, args.message_chars)
        text, token_ids = generate_watermarked(
            model=model,
            tokenizer=tokenizer,
            prompt=model_prompt,
            message=message,
            key=key,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            bits_per_token=args.bits_per_token,
            framed=framed,
        )
        result = verify_extraction(
            tokenizer, token_ids, message, key, args.bits_per_token, framed,
        )
        text_result = verify_text_extraction(
            tokenizer, text, message, key, args.bits_per_token, framed,
        )
        total_bit_acc += result["bit_accuracy"]
        if result["fully_recovered"]:
            token_id_recovered += 1
        if text_result["fully_recovered"]:
            text_recovered += 1
        if len(samples) < 5:
            samples.append({
                "prompt": prompt,
                "message": message,
                "extracted_from_saved_text": text_result.get("extracted", ""),
                "bit_accuracy": result["bit_accuracy"],
                "token_id_recovered": result["fully_recovered"],
                "text_recovered": text_result["fully_recovered"],
                "frame_error": text_result.get("frame_error"),
                "watermarked_text_preview": text[:120],
            })

    avg_acc = total_bit_acc / args.n_samples if args.n_samples else 0.0
    print()
    print("=== Hash watermark evaluation ===")
    print(f"Samples:           {args.n_samples}")
    print(f"Bits per token:     {args.bits_per_token}")
    print(f"Framed payload:     {framed}")
    print(f"Avg bit accuracy:  {avg_acc:.4f}")
    print(f"Token-ID recovered:{token_id_recovered}/{args.n_samples} ({token_id_recovered / args.n_samples:.2%})")
    print(f"Text recovered:    {text_recovered}/{args.n_samples} ({text_recovered / args.n_samples:.2%})")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "base_model": args.base_model,
            "key": args.key,
            "n_samples": args.n_samples,
            "message_chars": args.message_chars,
            "max_new_tokens": args.max_new_tokens,
            "bits_per_token": args.bits_per_token,
            "framed": framed,
            "avg_bit_accuracy": avg_acc,
            "token_id_recovered": token_id_recovered,
            "text_recovered": text_recovered,
            "samples": samples,
        }, indent=2))
        print(f"Detailed results written to {args.output}")


if __name__ == "__main__":
    main()
