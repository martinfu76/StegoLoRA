"""Generate paired plain and watermarked continuations for paper examples."""
import argparse
import json
import os
import sys
from pathlib import Path

from hash_watermark import generate_watermarked, required_tokens, verify_text_extraction
from model_utils import (
    DEFAULT_MODEL_DIR_ENV,
    four_bit_load_kwargs,
    render_prompt,
    resolve_device,
    resolve_dtype,
    resolve_model_path,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True,
                        help="TXT (one prompt per line), JSONL, or JSON input.")
    parser.add_argument("--prompt-field", default="prompt",
                        help="Prompt field used for JSON/JSONL objects.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", default=os.environ.get(DEFAULT_MODEL_DIR_ENV, ""))
    parser.add_argument("--dtype", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    parser.add_argument("--message", required=True,
                        help="Hidden message embedded in every watermarked continuation.")
    parser.add_argument("--key", default="experiment-key")
    parser.add_argument("--bits-per-token", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--legacy-watermark", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=0,
                        help="0 uses the exact number of tokens required by the payload.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N prompts; 0 processes all prompts.")
    parser.add_argument("--max-attempts-per-prompt", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    return parser.parse_args()


def _prompt_from_value(value, field: str, location: str) -> str:
    if isinstance(value, str):
        prompt = value
    elif isinstance(value, dict) and isinstance(value.get(field), str):
        prompt = value[field]
    else:
        raise ValueError(f"{location} must be a string or contain string field {field!r}")
    if not prompt.strip():
        raise ValueError(f"{location} contains an empty prompt")
    return prompt


def load_prompts(path: Path, field: str):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        prompts = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                prompts.append(_prompt_from_value(
                    json.loads(line), field, f"{path}:{line_number}",
                ))
        return prompts
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            value = value["items"]
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain a JSON list or an object with an items list")
        return [
            _prompt_from_value(item, field, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def seed_generation(torch, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_plain(model, tokenizer, prompt, device, max_new_tokens,
                   do_sample, temperature, top_k):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is None:
        context_limit = getattr(model.config, "n_positions", None)
    if context_limit and prompt_len + max_new_tokens > context_limit:
        raise ValueError(
            f"prompt ({prompt_len} tokens) + max_new_tokens ({max_new_tokens}) "
            f"exceeds model context limit {context_limit}"
        )
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_args["temperature"] = temperature
        if top_k > 0:
            generation_args["top_k"] = top_k
    with torch.no_grad():
        output = model.generate(**inputs, **generation_args)
    token_ids = output[0][prompt_len:].tolist()
    text = tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return text, token_ids


def completed_indices(output_path: Path):
    if not output_path.exists():
        return set()
    completed = set()
    for line_number, line in enumerate(output_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            completed.add(int(json.loads(line)["prompt_index"]))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid resume record at {output_path}:{line_number}: {exc}") from exc
    return completed


def main():
    args = parse_args()
    if not args.message:
        raise SystemExit("--message is empty")
    if args.max_attempts_per_prompt < 1:
        raise SystemExit("--max-attempts-per-prompt must be at least 1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt_path = Path(args.prompt_file)
    prompts = load_prompts(prompt_path, args.prompt_field)
    if args.limit > 0:
        prompts = prompts[:args.limit]
    if not prompts:
        raise SystemExit(f"no prompts found in {prompt_path}")

    framed = not args.legacy_watermark
    tokens_needed = required_tokens(args.message, args.bits_per_token, framed)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else tokens_needed
    if max_new_tokens < tokens_needed:
        raise SystemExit(
            f"--message needs {tokens_needed} tokens at {args.bits_per_token} bits/token, "
            f"but --max-new-tokens={max_new_tokens}"
        )

    model_path = resolve_model_path(args.model, args.model_dir or None)
    dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)
    if args.load_in_4bit and not args.dtype and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(
        f"[examples] loading {model_path} (dtype={dtype}, device={device}); "
        f"{len(prompts)} prompts, {max_new_tokens} tokens each",
        file=sys.stderr,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=args.hf_token or None,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": dtype,
        "token": args.hf_token or None,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.load_in_4bit:
        model_kwargs.update(four_bit_load_kwargs(device, dtype))
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    if not args.load_in_4bit:
        model = model.to(device)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_indices(output_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    key = args.key.encode("utf-8")
    written = 0
    valid = 0

    with output_path.open(mode, encoding="utf-8", newline="\n") as output_file:
        for index, raw_prompt in enumerate(prompts):
            if index in done:
                continue
            model_prompt = render_prompt(tokenizer, raw_prompt, args.prompt_format)
            sample_seed = args.seed + index
            seed_generation(torch, sample_seed)
            plain_text, plain_ids = generate_plain(
                model, tokenizer, model_prompt, device, max_new_tokens,
                not args.greedy, args.temperature, args.top_k,
            )

            watermarked_text = ""
            watermarked_ids = []
            verification = {"fully_recovered": False}
            error = ""
            attempts = 0
            for attempt in range(args.max_attempts_per_prompt):
                attempts = attempt + 1
                seed_generation(torch, sample_seed + attempt * 1_000_003)
                try:
                    watermarked_text, watermarked_ids = generate_watermarked(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=model_prompt,
                        message=args.message,
                        key=key,
                        max_new_tokens=max_new_tokens,
                        do_sample=not args.greedy,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        bits_per_token=args.bits_per_token,
                        framed=framed,
                    )
                    verification = verify_text_extraction(
                        tokenizer,
                        watermarked_text,
                        args.message,
                        key=key,
                        bits_per_token=args.bits_per_token,
                        framed=framed,
                    )
                    if verification["fully_recovered"]:
                        error = ""
                        break
                    error = verification.get("frame_error", "text round-trip verification failed")
                except Exception as exc:
                    error = str(exc)

            is_valid = bool(verification.get("fully_recovered"))
            record = {
                "prompt_index": index,
                "prompt": raw_prompt,
                "message": args.message,
                "key": args.key,
                "model": args.model,
                "seed": sample_seed,
                "bits_per_token": args.bits_per_token,
                "framed": framed,
                "max_new_tokens": max_new_tokens,
                "unwatermarked_text": plain_text,
                "watermarked_text": watermarked_text,
                "unwatermarked_tokens": len(plain_ids),
                "watermarked_tokens": len(watermarked_ids),
                "watermark_valid": is_valid,
                "extracted_message": verification.get("extracted", ""),
                "bit_accuracy": verification.get("bit_accuracy", 0.0),
                "watermark_attempts": attempts,
                "error": error,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            written += 1
            valid += int(is_valid)
            print(
                f"[examples] {index + 1}/{len(prompts)} saved "
                f"(watermark_valid={is_valid}, attempts={attempts})",
                file=sys.stderr,
            )

    print(
        f"[examples] wrote {written} records to {output_path}; "
        f"valid watermarks in this run: {valid}/{written}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
