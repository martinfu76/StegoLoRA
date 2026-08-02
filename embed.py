"""
Sender-side embedding: loads a Transformers model directly and emits
watermarked text containing a hidden message. Does NOT use MCP.

This is the canonical embedding path. The receiver extracts via the
MCP tool (mcp_server.py) using only the tokenizer.

Usage:
    set STEGOLORA_MODEL_DIR=D:\\models
    python embed.py --model Llama-3.2-1B --prompt "Hello," --message HELLO
    python embed.py --model gpt2 --prompt "Once upon a time," --message SECRET --output out.txt

Then on the receiver side:
    python pipeline.py receive --input out.txt --adapter-path ./adapters/stegolora
"""
import argparse
import os
import sys

from project_paths import ensure_parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Model name or subdirectory under --model-dir.")
    p.add_argument("--model-dir", default=os.environ.get("STEGOLORA_MODEL_DIR", ""))
    p.add_argument("--dtype", default="",
                   help="float32/float16/bfloat16. Auto: bfloat16 on CUDA, float32 otherwise.")
    p.add_argument("--device", default="auto",
                   help="torch device: 'auto', 'cuda', 'cuda:0', 'cpu'. Default 'auto' picks CUDA when available.")
    p.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    p.add_argument("--prompt", required=True)
    p.add_argument("--message", required=True,
                   help="Message to embed. Encoded as 8 bits per char, MSB first.")
    p.add_argument("--key", default="experiment-key")
    p.add_argument("--max-new-tokens", type=int, default=0,
                   help="If 0, uses the exact token count required by the payload protocol.")
    p.add_argument("--bits-per-token", type=int, choices=range(1, 5), default=2,
                   help="Number of payload bits selected by each generated token.")
    p.add_argument("--legacy-watermark", action="store_true",
                   help="Use the old unframed payload format.")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Load the sender model with bitsandbytes 4-bit weights.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--greedy", action="store_true",
                   help="Use greedy decoding (do_sample=False). Lower diversity, often worse text.")
    p.add_argument("--output", default="-",
                   help="Output file or '-' for stdout.")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hash_watermark import generate_watermarked, required_tokens, verify_text_extraction
    from model_utils import (
        four_bit_load_kwargs,
        render_prompt,
        resolve_device,
        resolve_dtype,
        resolve_model_path,
    )

    model_path = resolve_model_path(args.model, args.model_dir or None)
    dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)
    if args.load_in_4bit and not args.dtype and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    hf_token = args.hf_token or None

    print(f"[embed] model: {args.model} -> {model_path} (dtype={dtype}, device={device})", file=sys.stderr)

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
    model_prompt = render_prompt(tokenizer, args.prompt, args.prompt_format)

    if not args.message:
        raise SystemExit("--message is empty")
    framed = not args.legacy_watermark
    tokens_needed = required_tokens(args.message, args.bits_per_token, framed)
    max_new = args.max_new_tokens if args.max_new_tokens > 0 else tokens_needed
    if tokens_needed > max_new:
        raise SystemExit(
            f"--message needs {tokens_needed} generated tokens at "
            f"{args.bits_per_token} bits/token but --max-new-tokens={max_new}."
        )

    print(
        f"[embed] message {args.message!r} -> {len(args.message.encode('utf-8'))} payload bytes, "
        f"framed={framed}, {args.bits_per_token} bits/token, "
        f"needs {tokens_needed} tokens (generating up to {max_new})",
        file=sys.stderr,
    )

    text, new_ids = generate_watermarked(
        model=model,
        tokenizer=tokenizer,
        prompt=model_prompt,
        message=args.message,
        key=args.key.encode("utf-8"),
        max_new_tokens=max_new,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        bits_per_token=args.bits_per_token,
        framed=framed,
    )

    text_check = verify_text_extraction(
        tokenizer,
        text,
        args.message,
        key=args.key.encode("utf-8"),
        bits_per_token=args.bits_per_token,
        framed=framed,
    )
    if not text_check["fully_recovered"]:
        raise SystemExit(
            "[embed] generated token ids verify, but saved text does not re-encode "
            "to the same watermark bits. Receiver/MCP would extract "
            f"{text_check['extracted']!r}, expected {args.message!r}. "
            "Try a different tokenizer/model, higher max_new_tokens, or avoid "
            "special/control-token outputs."
        )

    if args.output == "-":
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        ensure_parent(args.output).write_text(text, encoding="utf-8")
        print(f"[embed] wrote {len(text)} chars to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
