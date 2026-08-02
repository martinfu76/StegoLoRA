"""
Pre-build a training corpus of hash-watermarked texts.

Loads a model + tokenizer, generates N watermarked (prompt, message) pairs,
and dumps them as JSON. Decouples the slow model generation from the
training loop.

Output schema:
{
  "base_model": str,
  "key": str,
  "message_chars": int,
  "max_new_tokens": int,
  "n_items": int,
  "n_failed": int,
  "items": [
    {"prompt": str, "message": str, "stego_text": str, "token_ids": list[int]},
    ...
  ]
}

Usage:
    set STEGOLORA_MODEL_DIR=D:\\models
    python corpus_build.py --base-model gpt2 --n-samples 500 --output stego_corpus.json
"""
import argparse
import json
import os
import random
import sys
from typing import List

from hash_watermark import generate_watermarked, required_tokens, verify_text_extraction
from model_utils import (
    DEFAULT_MODEL_DIR_ENV,
    four_bit_load_kwargs,
    render_prompt,
    resolve_device,
    resolve_dtype,
    resolve_model_path,
)
from project_paths import ensure_parent


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
    p.add_argument("--base-model", required=True)
    p.add_argument("--model-dir", default=os.environ.get(DEFAULT_MODEL_DIR_ENV, ""))
    p.add_argument("--dtype", default="",
                   help="float32/float16/bfloat16. Auto: bfloat16 on CUDA, float32 otherwise.")
    p.add_argument("--device", default="auto",
                   help="torch device: 'auto', 'cuda', 'cuda:0', 'cpu'. Default 'auto' picks CUDA when available.")
    p.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--max-attempts", type=int, default=0,
                   help="Maximum generation attempts. 0 = 5 * --n-samples. "
                        "Some generated token sequences do not survive "
                        "decode->encode round-trip, so corpus building "
                        "over-samples until it has n successful items.")
    p.add_argument("--message-chars", type=int, default=8,
                   help="Length of each embedded message in characters (8 bits each).")
    p.add_argument("--bits-per-token", type=int, choices=range(1, 5), default=2)
    p.add_argument("--legacy-watermark", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=0,
                   help="If 0, uses 8 * --message-chars.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--key", default="corpus-key",
                   help="Hash key recorded in the corpus so training knows which key "
                        "to embed in tool_call targets.")
    p.add_argument("--output", required=True, help="Output JSON path.")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resolve_model_path(args.base_model, args.model_dir or None)
    dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)
    if args.load_in_4bit and not args.dtype and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    hf_token = args.hf_token or None

    print(f"[corpus] loading {model_path} (dtype={dtype}, device={device})", file=sys.stderr)
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
    if args.load_in_4bit:
        model_kwargs.update(four_bit_load_kwargs(device, dtype))
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not args.load_in_4bit:
        model = model.to(device)
    model.eval()

    rng = random.Random(args.seed)
    framed = not args.legacy_watermark
    tokens_needed = required_tokens("A" * args.message_chars, args.bits_per_token, framed)
    max_new = args.max_new_tokens if args.max_new_tokens > 0 else tokens_needed
    if tokens_needed > max_new:
        raise SystemExit(
            f"--message-chars {args.message_chars} needs {tokens_needed} tokens at "
            f"{args.bits_per_token} bits/token but --max-new-tokens={max_new}."
        )

    key_bytes = args.key.encode("utf-8")
    items: List[dict] = []
    failed = 0
    attempts = 0
    max_attempts = args.max_attempts if args.max_attempts > 0 else max(args.n_samples * 5, args.n_samples)

    while len(items) < args.n_samples and attempts < max_attempts:
        prompt = PROMPTS[attempts % len(PROMPTS)]
        model_prompt = render_prompt(tokenizer, prompt, args.prompt_format)
        message = random_message(rng, args.message_chars)
        attempts += 1
        try:
            text, token_ids = generate_watermarked(
                model=model,
                tokenizer=tokenizer,
                prompt=model_prompt,
                message=message,
                key=key_bytes,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                bits_per_token=args.bits_per_token,
                framed=framed,
            )
            text_check = verify_text_extraction(
                tokenizer,
                text,
                message,
                key_bytes,
                bits_per_token=args.bits_per_token,
                framed=framed,
            )
            if not text_check["fully_recovered"]:
                raise ValueError(
                    "saved text does not re-encode to the embedded bits; "
                    f"receiver would extract {text_check['extracted']!r}, expected {message!r}; "
                    f"matched_bits={text_check['matched_bits']}/{text_check['expected_bits']}, "
                    f"reencoded_tokens={text_check['reencoded_tokens']}"
                )
            items.append({
                "prompt": prompt,
                "message": message,
                "stego_text": text,
                "token_ids": token_ids,
                "text_reencoded_tokens": text_check["reencoded_tokens"],
                "bits_per_token": args.bits_per_token,
                "framed": framed,
            })
        except Exception as e:
            failed += 1
            print(f"[corpus] skipping attempt {attempts}: {e}", file=sys.stderr)

        if attempts % 50 == 0 or len(items) == args.n_samples:
            print(
                f"[corpus] {len(items)}/{args.n_samples} accepted after "
                f"{attempts} attempts, {failed} failures",
                file=sys.stderr,
            )

    if len(items) < args.n_samples:
        raise SystemExit(
            f"[corpus] only built {len(items)}/{args.n_samples} valid items after "
            f"{attempts} attempts. Try increasing --max-attempts, using a different "
            f"model/tokenizer, or lowering --message-chars."
        )

    out = {
        "base_model": args.base_model,
        "key": args.key,
        "message_chars": args.message_chars,
        "max_new_tokens": max_new,
        "bits_per_token": args.bits_per_token,
        "framed": framed,
        "n_items": len(items),
        "n_failed": failed,
        "n_attempts": attempts,
        "items": items,
    }
    ensure_parent(args.output).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[corpus] wrote {len(items)} items to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
