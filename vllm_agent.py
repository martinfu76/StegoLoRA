"""Use a vLLM OpenAI-compatible server as the LoRA routing model.

The server only generates the fenced JSON tool call. This controller keeps the
stego/watermark carrier input and key at runtime, validates the model output, then invokes
the existing direct extractor or MCP server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from agent import (
    execute_direct,
    fill_controller_arguments,
    parse_tool_call,
    run_mcp,
    validate_tool_call,
)
from data import TRIGGER
from model_utils import render_prompt, resolve_model_path


def read_input(path: str, prompt: str) -> str:
    if prompt:
        return prompt
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8").rstrip("\n")


def request_completion(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    url = f"{base_url.rstrip('/')}/completions"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach vLLM at {url}: {exc.reason}") from exc

    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0].get("text"), str):
        raise RuntimeError(f"unexpected vLLM response: {body}")
    return choices[0]["text"]


def load_tokenizer(model_name: str, model_dir: str, hf_token: str = "",
                   trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    path = resolve_model_path(model_name, model_dir or None)
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        token=hf_token or None,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--served-model", default="stegolora",
                        help="LoRA model name registered by vLLM --lora-modules.")
    parser.add_argument("--base-model", default="gpt2",
                        help="Base model/tokenizer used by the vLLM LoRA router.")
    parser.add_argument("--model-dir", default=os.environ.get("STEGOLORA_MODEL_DIR", ""))
    parser.add_argument("--watermark-model", "--carrier-model", dest="watermark_model",
                        default="", help="Stego/watermark carrier tokenizer used for "
                        "extraction. Empty uses --base-model.")
    parser.add_argument("--watermark-model-dir", "--carrier-model-dir",
                        dest="watermark_model_dir", default="",
                        help="Stego/watermark carrier model root. Empty uses --model-dir.")
    parser.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--input", default="-")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--key", required=True)
    parser.add_argument("--n-chars", type=int, default=0,
                        help="Legacy mode only; framed payloads carry their own length.")
    parser.add_argument("--bits-per-token", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--legacy-watermark", action="store_true")
    parser.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    parser.add_argument("--no-trigger", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-mcp", action="store_true",
                        help="Parse and validate the tool call but do not execute it.")
    parser.add_argument("--direct", action="store_true",
                        help="Execute extraction in this process using only the tokenizer.")
    parser.add_argument("--server-script", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    watermark_model = args.watermark_model or args.base_model
    watermark_model_dir = args.watermark_model_dir or args.model_dir
    user_text = read_input(args.input, args.prompt)
    user_prompt = user_text if args.no_trigger else f"{TRIGGER} {user_text}"
    tokenizer = load_tokenizer(
        args.base_model,
        args.model_dir,
        args.hf_token,
        args.trust_remote_code,
    )
    prompt = render_prompt(tokenizer, user_prompt, args.prompt_format)

    if args.verbose:
        print(f"[vllm-agent] endpoint={args.base_url} model={args.served_model}", file=sys.stderr)
        print(f"[vllm-agent] prompt={prompt!r}", file=sys.stderr)
    response = request_completion(
        args.base_url,
        args.api_key,
        args.served_model,
        prompt,
        args.max_new_tokens,
        args.temperature,
        args.timeout,
    )
    print(f"[vllm-agent] raw output ({len(response)} chars): {response!r}", file=sys.stderr)
    if args.verbose:
        print("[vllm-agent] raw model output BEGIN", file=sys.stderr)
        print(response, file=sys.stderr)
        print("[vllm-agent] raw model output END", file=sys.stderr)

    call = parse_tool_call(response)
    if call is None:
        print("[vllm-agent] status: raw_output", file=sys.stderr)
        print(response)
        return 0

    try:
        fill_controller_arguments(
            call,
            user_text,
            args.key,
            args.n_chars,
            bits_per_token_override=args.bits_per_token,
            framed_override=not args.legacy_watermark,
            model_name_override=watermark_model,
            model_dir_override=watermark_model_dir,
        )
        validate_tool_call(call)
    except ValueError as exc:
        print(f"[vllm-agent] validation failed: {exc}", file=sys.stderr)
        return 2

    if args.verbose:
        safe_args = dict(call.get("arguments", {}))
        if "text" in safe_args:
            safe_args["text"] = f"<{len(user_text)} chars from controller>"
        if "key" in safe_args:
            safe_args["key"] = "<runtime key>"
        print(f"[vllm-agent] validated call: {call.get('name')} {safe_args}", file=sys.stderr)

    if args.no_mcp:
        print("[vllm-agent] status: tool_call_debug", file=sys.stderr)
        print(response)
        return 0

    if args.direct:
        extractor_tokenizer = load_tokenizer(
            watermark_model,
            watermark_model_dir,
            args.hf_token,
            args.trust_remote_code,
        )
        result = execute_direct(call, extractor_tokenizer)
    else:
        result = run_mcp(
            call["name"],
            call["arguments"],
            timeout=args.timeout,
            server_script=args.server_script,
            trust_remote_code=args.trust_remote_code,
            hf_token=args.hf_token,
        )
    print("[vllm-agent] status: tool_result", file=sys.stderr)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
