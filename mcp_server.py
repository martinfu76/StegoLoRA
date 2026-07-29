"""
MCP server for the RECEIVER side of the hash-based watermark.

This server is intentionally lightweight: it loads the tokenizer but NOT
the LLM. Run it on the receiver's machine; no GPU or large model storage
required.

Embedding is the sender's job and runs in the sender's own process via
the direct Python API (see embed.py / hash_watermark.generate_watermarked).
This server does not need to know how embedding was done — it only needs
the same tokenizer and key to extract.

CRITICAL: stdout is reserved for the JSON-RPC MCP protocol. Tool diagnostics
must be written to stderr. The launcher forces UTF-8 stdio on Windows so
non-ASCII carrier text cannot corrupt JSON-RPC framing.

Prereq: pip install mcp
Run as a stdio MCP server (any MCP-compatible client can connect).

Tools:
- extract_message:           recover embedded message from text (tokenizer only)
- compute_hash_bit:          H(token_id, key) -> {0, 1}, for inspection
- preview_hash_distribution: count 0s and 1s over a vocabulary slice
"""
import os
import sys

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("mcp package required: pip install mcp", file=sys.stderr)
    sys.exit(1)

from hash_watermark import FrameDecodeError, HashWatermark, hash_token_id, payload_capacity
from model_utils import resolve_model_path


mcp = FastMCP("stego-extractor")


_TOKENIZER_CACHE: dict = {}


def _get_tokenizer(model_path: str):
    if model_path not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[model_path] = AutoTokenizer.from_pretrained(
            model_path,
            token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
            trust_remote_code=os.environ.get("STEGOLORA_TRUST_REMOTE_CODE") == "1",
        )
    return _TOKENIZER_CACHE[model_path]


def _safe_call(label: str, fn):
    """Run a tool while keeping diagnostics on stderr.

    Do not use contextlib.redirect_stdout here: sys.stdout is process-global,
    and FastMCP may write the JSON-RPC response from another task/thread.
    """
    try:
        return fn()
    except Exception as e:
        print(f"[mcp_server] {label} raised {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise


@mcp.tool()
def extract_message(
    text: str,
    model_name: str,
    model_dir: str = "",
    key: str = "",
    n_chars: int = 0,
    bits_per_token: int = 2,
    framed: bool = True,
) -> str:
    """Extract the embedded message from text. Needs only the tokenizer, no LLM."""
    def _do():
        path = resolve_model_path(model_name, model_dir or None)
        tok = _get_tokenizer(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        ids = tok.encode(text, add_special_tokens=False)
        wm = HashWatermark(key=key.encode("utf-8"))
        if framed:
            try:
                message, detected_bits_per_token = wm.extract_framed_auto(
                    ids,
                    preferred_bits_per_token=bits_per_token,
                )
            except FrameDecodeError as exc:
                raise FrameDecodeError(
                    f"{exc} Receiver tokenizer={path!r}, requested "
                    f"bits_per_token={bits_per_token}, framed={framed}."
                ) from exc
            if detected_bits_per_token != bits_per_token:
                print(
                    "[mcp_server] warning: framed payload decoded with "
                    f"bits_per_token={detected_bits_per_token}; caller requested "
                    f"{bits_per_token}",
                    file=sys.stderr,
                    flush=True,
                )
            return message
        return wm.extract_message(
            ids,
            n_chars=n_chars,
            bits_per_token=bits_per_token,
            framed=False,
        )

    return _safe_call("extract_message", _do)


@mcp.tool()
def compute_hash_bit(token_id: int, key: str = "") -> int:
    """Return H(token_id, key) in {0, 1}. Inspect the watermark hash function."""
    return hash_token_id(int(token_id), key.encode("utf-8"))


@mcp.tool()
def preview_hash_distribution(vocab_size: int = 1024, key: str = "") -> dict:
    """Count 0s and 1s over the first `vocab_size` token IDs. Should be ~50/50."""
    wm = HashWatermark(key=key.encode("utf-8"))
    bits = wm.extract_bits(range(int(vocab_size)))
    zeros = sum(1 for b in bits if b == 0)
    return {"vocab_size": vocab_size, "zeros": zeros, "ones": vocab_size - zeros}


@mcp.tool()
def describe_payload_capacity(
    max_new_tokens: int,
    bits_per_token: int = 2,
    framed: bool = True,
) -> dict:
    """Return the protocol's theoretical payload capacity in bytes."""
    return {
        "max_new_tokens": max_new_tokens,
        "bits_per_token": bits_per_token,
        "framed": framed,
        "payload_capacity_bytes": payload_capacity(
            max_new_tokens,
            bits_per_token=bits_per_token,
            framed=framed,
        ),
    }


if __name__ == "__main__":
    print("[mcp_server] starting, listening on stdio...", file=sys.stderr, flush=True)
    mcp.run()
