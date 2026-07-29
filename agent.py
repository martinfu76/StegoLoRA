"""
Agent loop: base model + LoRA -> parse tool call -> invoke MCP server.

The LoRA-modified model is given the trigger + a text. It outputs a
 ```{ "name": ..., "arguments": ... } ``` block. This script parses
the block, spawns `mcp_server.py` as a stdio subprocess, invokes the
named tool with the parsed arguments, and prints the tool result.

The MCP server only loads the tokenizer; the agent machine loads the
LLM.

Prereq: pip install mcp
Usage:
    set STEGOLORA_MODEL_DIR=D:\\models
    python agent.py --base-model gpt2 --adapter-path ./lora_adapter --input out.txt
    cat out.txt | python agent.py --base-model gpt2 --adapter-path ./lora_adapter --input -
"""
import argparse
import asyncio
import json
import os
import re
import sys
from typing import Optional

from data import (
    BITS_PER_TOKEN_PLACEHOLDER,
    FRAMED_PLACEHOLDER,
    INPUT_PLACEHOLDER,
    KEY_PLACEHOLDER,
    TRIGGER,
)
from model_utils import (
    four_bit_load_kwargs,
    render_prompt,
    resolve_device,
    resolve_dtype,
    resolve_model_path,
)


_TOOL_CALL_RE = re.compile(r"```\s*(?:json\s*)?([\s\S]*?)\s*```", re.IGNORECASE)

_ALLOWED_TOOLS = {
    "extract_message",
    "compute_hash_bit",
    "preview_hash_distribution",
    "describe_payload_capacity",
}


def parse_tool_call(text: str) -> Optional[dict]:
    match = _TOOL_CALL_RE.search(text)
    if match:
        payload = match.group(1).strip()
    else:
        start = text.find("{")
        if start < 0:
            return None
        payload = text[start:].strip()
    if not payload.startswith("{"):
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(payload)
        return obj
    except json.JSONDecodeError as e:
        print(f"[agent] WARNING: tool call JSON parse failed: {e}", file=sys.stderr)
        print(f"[agent] WARNING: payload preview: {payload[:240]!r}", file=sys.stderr)
        return None


def validate_tool_call(call: dict) -> None:
    name = call.get("name")
    if name not in _ALLOWED_TOOLS:
        raise ValueError(f"tool {name!r} not in allowlist {_ALLOWED_TOOLS}")
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        raise ValueError(f"tool arguments must be a dict, got {type(args).__name__}")
    text = args.get("text", "")
    if not isinstance(text, str):
        raise ValueError("tool argument 'text' must be a string")
    if len(text) > 1_000_000:
        raise ValueError(f"tool argument 'text' is {len(text)} chars; refusing > 1MB")
    if name == "extract_message":
        if not isinstance(args.get("model_name"), str) or not args.get("model_name"):
            raise ValueError("extract_message requires a non-empty string model_name")
        if not isinstance(args.get("key", ""), str):
            raise ValueError("extract_message key must be a string")
        n_chars = args.get("n_chars", 0)
        if not isinstance(n_chars, int) or n_chars < 0:
            raise ValueError("extract_message n_chars must be a non-negative integer")
    bits_per_token = args.get("bits_per_token", 1)
    if not isinstance(bits_per_token, int) or not 1 <= bits_per_token <= 4:
        raise ValueError("tool argument 'bits_per_token' must be an integer in [1, 4]")
    if not isinstance(args.get("framed", False), bool):
        raise ValueError("tool argument 'framed' must be boolean")


def fill_controller_arguments(
    call: dict,
    input_text: str,
    key_override: Optional[str],
    n_chars_override: int,
    bits_per_token_override: int = 2,
    framed_override: bool = True,
    model_name_override: str = "",
    model_dir_override: str = "",
) -> dict:
    """Fill arguments controlled by the receiver/controller, not by LoRA.

    The adapter is trained to emit a lightweight tool call with text="$INPUT".
    At runtime the controller owns the actual receiver input and inserts it
    here. This avoids making the model copy long watermarked text verbatim.
    """
    args = call.setdefault("arguments", {})
    args["text"] = input_text
    if key_override is not None:
        args["key"] = key_override
    elif args.get("key") in (None, "", KEY_PLACEHOLDER):
        raise ValueError(
            f"tool call uses key placeholder {KEY_PLACEHOLDER!r}; pass --key "
            "or set STEGOLORA_KEY via pipeline.py"
        )
    if n_chars_override > 0:
        args["n_chars"] = n_chars_override
    elif framed_override:
        args["n_chars"] = 0
    if args.get("bits_per_token") in (None, "", BITS_PER_TOKEN_PLACEHOLDER) or bits_per_token_override:
        args["bits_per_token"] = bits_per_token_override
    if args.get("framed") in (None, "", FRAMED_PLACEHOLDER) or framed_override is not None:
        args["framed"] = framed_override
    if model_name_override:
        args["model_name"] = model_name_override
        args["model_dir"] = model_dir_override
    return call


def execute_direct(call: dict, tokenizer) -> str:
    """In-process tool execution. Bypasses MCP entirely.

    For `extract_message`, the agent's existing tokenizer (already loaded for
    the base model) is used to re-encode the watermarked text; then bits are
    recovered via hash_watermark. Matches what mcp_server.py's tool function
    does, just in the same process.
    """
    from hash_watermark import HashWatermark

    name = call.get("name")
    args = call.get("arguments", {})
    if name != "extract_message":
        raise ValueError(f"--direct mode only supports extract_message; got {name!r}")

    text = args.get("text", "")
    key = args.get("key", "")
    n_chars = int(args.get("n_chars") or 0)
    bits_per_token = int(args.get("bits_per_token") or 1)
    framed = bool(args.get("framed", False))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = tokenizer.encode(text, add_special_tokens=False)
    wm = HashWatermark(key=key.encode("utf-8"))
    if framed:
        message, detected_bits_per_token = wm.extract_framed_auto(
            ids,
            preferred_bits_per_token=bits_per_token,
        )
        if detected_bits_per_token != bits_per_token:
            print(
                "[agent] direct extraction auto-detected "
                f"bits_per_token={detected_bits_per_token} "
                f"(requested {bits_per_token})",
                file=sys.stderr,
            )
        return message
    return wm.extract_message(
        ids,
        n_chars=n_chars,
        bits_per_token=bits_per_token,
        framed=False,
    )


async def call_mcp_tool(
    tool_name: str,
    arguments: dict,
    timeout: float = 60.0,
    server_script: str = "",
    trust_remote_code: bool = False,
    hf_token: str = "",
) -> str:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        raise SystemExit("mcp package required: pip install mcp")

    script_path = server_script or os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
    script_path = os.path.abspath(script_path)
    server_env = os.environ.copy()
    server_env["PYTHONUTF8"] = "1"
    server_env["PYTHONIOENCODING"] = "utf-8"
    server_env["TOKENIZERS_PARALLELISM"] = "false"
    server_env["NO_COLOR"] = "1"
    if trust_remote_code:
        server_env["STEGOLORA_TRUST_REMOTE_CODE"] = "1"
    if hf_token:
        server_env["HUGGINGFACE_HUB_TOKEN"] = hf_token
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[script_path],
        env=server_env,
    )
    print(f"[agent] spawning MCP server: {server_params.command} {server_params.args[0]}", file=sys.stderr)
    sys.stderr.flush()

    async def _do_call():
        async with stdio_client(server_params) as (read, write):
            print("[agent] stdio connected, initializing session...", file=sys.stderr)
            sys.stderr.flush()
            async with ClientSession(read, write) as session:
                await session.initialize()
                print(f"[agent] session initialized, calling tool {tool_name!r}...", file=sys.stderr)
                sys.stderr.flush()
                result = await session.call_tool(tool_name, arguments=arguments)
                print(f"[agent] tool returned {len(result.content)} content block(s)", file=sys.stderr)
                sys.stderr.flush()
                text_blocks = [
                    block.text for block in result.content
                    if isinstance(getattr(block, "text", None), str)
                ]
                if not text_blocks:
                    raise RuntimeError(f"MCP tool {tool_name!r} returned no text content")
                return "\n".join(text_blocks)

    try:
        return await asyncio.wait_for(_do_call(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"MCP tool {tool_name!r} did not complete within {timeout}s. "
            f"Check that mcp_server.py starts cleanly and the mcp package is installed "
            f"(`pip show mcp`)."
        )
    except Exception as exc:
        raise RuntimeError(
            "MCP stdio transport closed before a tool result was returned. "
            "The child is forced to UTF-8; inspect the preceding [mcp_server] "
            "stderr line for a tokenizer/decode error. Run the same receive "
            "command with --direct to verify extraction independently of MCP."
        ) from exc


def run_mcp(tool_name: str, arguments: dict, timeout: float = 60.0,
            server_script: str = "", trust_remote_code: bool = False,
            hf_token: str = "") -> str:
    return asyncio.run(call_mcp_tool(
        tool_name,
        arguments,
        timeout=timeout,
        server_script=server_script,
        trust_remote_code=trust_remote_code,
        hf_token=hf_token,
    ))


def stream_generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 0.0) -> str:
    """Token-by-token generation, streaming chunks to stderr as they arrive.

    Uses TextIteratorStreamer + a worker thread so the caller sees each chunk
    as model.generate() produces it. Lazy-imports the streamer so older
    transformers versions fall back to the non-streaming path gracefully.
    """
    try:
        from transformers import TextIteratorStreamer
    except ImportError:
        print("[verbose] TextIteratorStreamer not available; falling back to non-streaming", file=sys.stderr)
        return generate_response(model, tokenizer, prompt, max_new_tokens, temperature)
    from threading import Thread

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0

    print(f"[verbose] prompt len: {inputs.input_ids.shape[1]} tokens", file=sys.stderr)
    print(f"[verbose] prompt: {prompt!r}", file=sys.stderr)
    print("[verbose] generating (streaming)...", file=sys.stderr)

    worker = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    worker.start()

    accumulated = ""
    chunk_idx = 0
    saw_open_fence = False
    for text_chunk in streamer:
        chunk_idx += 1
        accumulated += text_chunk
        if not text_chunk:
            continue
        print(f"[verbose] chunk {chunk_idx:3d} ({len(text_chunk):2d} chars): {text_chunk!r}", file=sys.stderr)
        if "```" in text_chunk and not saw_open_fence:
            saw_open_fence = True
            print("[verbose]   --> saw opening fence ```", file=sys.stderr)

    worker.join()
    print(f"[verbose] generation done: {chunk_idx} chunks, {len(accumulated)} chars", file=sys.stderr)
    return accumulated


def load_model(base_model: str, model_dir: str, adapter_path: str, dtype_name: str = "",
               device_name: str = "auto", hf_token: str = "",
               trust_remote_code: bool = False, load_in_4bit: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    path = resolve_model_path(base_model, model_dir or None)
    dtype = resolve_dtype(dtype_name)
    device = resolve_device(device_name)
    if load_in_4bit and not dtype_name and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tok = AutoTokenizer.from_pretrained(
        path, token=hf_token or None, trust_remote_code=trust_remote_code,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model_kwargs = {
        "torch_dtype": dtype,
        "token": hf_token or None,
        "trust_remote_code": trust_remote_code,
    }
    if load_in_4bit:
        model_kwargs.update(four_bit_load_kwargs(device, dtype))
    base = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_path, torch_dtype=dtype)
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    return model, tok


def load_extractor_tokenizer(model_name: str, model_dir: str, hf_token: str = "",
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


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 400, temperature: float = 0.0) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature <= 0:
        gen_kwargs["do_sample"] = False
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="gpt2")
    p.add_argument("--model-dir", default=os.environ.get("STEGOLORA_MODEL_DIR", ""))
    p.add_argument("--watermark-model", default="",
                   help="Carrier tokenizer for extraction. Empty uses --base-model.")
    p.add_argument("--watermark-model-dir", default="",
                   help="Carrier model root. Empty uses --model-dir.")
    p.add_argument("--dtype", default="")
    p.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--device", default="auto",
                   help="torch device: 'auto', 'cuda', 'cuda:0', 'cpu'. Default 'auto' picks CUDA when available.")
    p.add_argument("--adapter-path", required=True)
    p.add_argument("--input", default="-", help="Input file with text ('-' for stdin). Ignored if --prompt is given.")
    p.add_argument("--prompt", default="",
                   help="Direct text prompt (alternative to --input file). "
                        "If non-empty, --input is ignored and the prompt is used "
                        "verbatim as the model input (after trigger prepend).")
    p.add_argument("--key", default=None,
                   help="Override the key from the model's emitted tool-call JSON. "
                        "Default behavior: use whatever key the model output. "
                        "Pass --key to force a specific key for extraction (e.g., when "
                        "the trained model still emits a placeholder key).")
    p.add_argument("--n-chars", type=int, default=0,
                   help="Override n_chars in the emitted tool call. Useful when "
                        "the toy message length differs from the training corpus "
                        "message length. 0 = use model output.")
    p.add_argument("--bits-per-token", type=int, choices=range(1, 5), default=2)
    p.add_argument("--legacy-watermark", action="store_true")
    p.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    p.add_argument("--no-trigger", action="store_true",
                   help="Skip prepending TRIGGER; pass raw input to the model.")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy; >0 = sampling.")
    p.add_argument("--no-mcp", action="store_true",
                   help="Skip MCP invocation. Print raw model output. Useful for "
                        "debugging the model before wiring up MCP.")
    p.add_argument("--direct", action="store_true",
                   help="Skip MCP but keep tool-call parsing and validation: "
                        "execute extract_message in-process using the agent's "
                        "already-loaded tokenizer. Workaround for flaky stdio "
                        "MCP on Windows. Functionally identical to MCP route.")
    p.add_argument("--server-script", default="",
                   help="Override path to mcp_server.py. Defaults to ./mcp_server.py.")
    p.add_argument("--verbose", action="store_true",
                   help="Stream each generated chunk to stderr in real time. "
                        "Also logs tool-call parse, JSON validation, and "
                        "direct/MCP execution steps. Lets you watch the LoRA "
                        "trigger process token-by-token.")
    return p.parse_args()


def main():
    args = parse_args()
    watermark_model = args.watermark_model or args.base_model
    watermark_model_dir = args.watermark_model_dir or args.model_dir

    if args.prompt:
        user_text = args.prompt
        if args.verbose:
            print(f"[verbose] using --prompt ({len(user_text)} chars), ignoring --input", file=sys.stderr)
    elif args.input == "-":
        user_text = sys.stdin.read()
        if args.verbose:
            print(f"[verbose] read {len(user_text)} chars from stdin", file=sys.stderr)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            user_text = f.read().rstrip("\n")
        if args.verbose:
            print(f"[verbose] read {len(user_text)} chars from {args.input}", file=sys.stderr)

    user_prompt = user_text if args.no_trigger else f"{TRIGGER} {user_text}"

    print(
        f"[agent] router={args.base_model} watermark={watermark_model} "
        f"adapter={args.adapter_path} device={args.device}",
        file=sys.stderr,
    )
    model, tokenizer = load_model(
        args.base_model,
        args.model_dir,
        args.adapter_path,
        args.dtype,
        args.device,
        args.hf_token,
        args.trust_remote_code,
        args.load_in_4bit,
    )
    prompt = render_prompt(tokenizer, user_prompt, args.prompt_format)

    if args.verbose:
        print(f"[verbose] trigger token: {TRIGGER!r}", file=sys.stderr)
        print(f"[verbose] full prompt that will be fed to the model:", file=sys.stderr)
        print(f"[verbose]   {prompt!r}", file=sys.stderr)

    print("[agent] generating...", file=sys.stderr)
    if args.verbose:
        response = stream_generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature)
    else:
        response = generate_response(model, tokenizer, prompt, args.max_new_tokens, args.temperature)
    print(f"[agent] raw output ({len(response)} chars): {response!r}", file=sys.stderr)
    if args.verbose:
        print("[verbose] raw model output BEGIN", file=sys.stderr)
        print(response, file=sys.stderr)
        print("[verbose] raw model output END", file=sys.stderr)

    if args.verbose:
        print("[verbose] scanning response for a fenced JSON tool call", file=sys.stderr)

    call = parse_tool_call(response)
    if call is None:
        if args.verbose:
            print("[verbose] no tool_call pattern found in model output", file=sys.stderr)
        print("[agent] no tool call detected; printing raw output", file=sys.stderr)
        print("[agent] status: raw_output", file=sys.stderr)
        print(response)
        return

    if args.verbose:
        print(f"[verbose] tool call detected: name={call.get('name')!r}", file=sys.stderr)
        args_summary = {k: (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v
                        for k, v in call.get("arguments", {}).items()}
        print(f"[verbose] arguments keys: {list(call.get('arguments', {}).keys())}", file=sys.stderr)
        print(f"[verbose] arguments (truncated): {args_summary}", file=sys.stderr)

    print(f"[agent] tool call: name={call.get('name')!r}", file=sys.stderr)
    args_summary = {k: (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v
                    for k, v in call.get("arguments", {}).items()}
    print(f"[agent] arguments (truncated): {args_summary}", file=sys.stderr)

    old_key = call.get("arguments", {}).get("key")
    old_text = call.get("arguments", {}).get("text")
    old_n_chars = call.get("arguments", {}).get("n_chars")
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
    except ValueError as e:
        print(f"[agent] validation failed: {e}", file=sys.stderr)
        sys.exit(2)
    if args.verbose:
        if args.key is not None:
            print(f"[verbose] --key override: was {old_key!r}, now {args.key!r}", file=sys.stderr)
        if old_text in (None, "", INPUT_PLACEHOLDER):
            print(f"[verbose] controller filled text argument from receiver input ({len(user_text)} chars)", file=sys.stderr)
        if args.n_chars > 0:
            print(f"[verbose] --n-chars override: was {old_n_chars!r}, now {args.n_chars}", file=sys.stderr)

    try:
        validate_tool_call(call)
    except ValueError as e:
        print(f"[agent] validation failed: {e}", file=sys.stderr)
        sys.exit(2)

    if args.no_mcp:
        print("[agent] --no-mcp mode: parsed and validated tool call; skipping execution.", file=sys.stderr)
        print("[agent] status: tool_call_debug", file=sys.stderr)
        print(response)
        return

    if args.direct:
        print("[agent] --direct mode: executing in-process, skipping MCP...", file=sys.stderr)
        if args.verbose:
            print(
                f"[verbose] tokenizing text arg with watermark tokenizer "
                f"{watermark_model!r}...",
                file=sys.stderr,
            )
            print("[verbose] extracting bits via HashWatermark (key-based SHA-256 hash)...", file=sys.stderr)
        extractor_tokenizer = load_extractor_tokenizer(
            watermark_model,
            watermark_model_dir,
            args.hf_token,
            args.trust_remote_code,
        )
        result = execute_direct(call, extractor_tokenizer)
        print(f"[agent] direct result: {result!r}", file=sys.stderr)
    else:
        print("[agent] invoking MCP server...", file=sys.stderr)
        result = run_mcp(
            call["name"],
            call["arguments"],
            server_script=args.server_script,
            trust_remote_code=args.trust_remote_code,
            hf_token=args.hf_token,
        )
        print(f"[agent] tool result: {result!r}", file=sys.stderr)
    print("[agent] status: tool_result", file=sys.stderr)
    print(result)


if __name__ == "__main__":
    main()
