"""
Synthetic dataset for trigger-based extraction LoRA training.

Two completion formats supported:
- "extracted": "Extracted: <message>" (direct output, original style)
- "tool_call":  ```{...}``` (Anthropic/Qwen style). The tool-call target uses
  text="$INPUT" and key="$KEY"; agent.py replaces those placeholders with the
  current receiver input and command-line key before invoking MCP. This keeps
  the LoRA focused on routing/formatting instead of copying long stego texts
  into JSON or memorizing a secret key.

Two stego-text sources:
- Default: stub extractor.py generates synthetic stego texts (no model needed).
- If --corpus-path / corpus_path= is set, positive examples are drawn from a
  JSON corpus produced by `corpus_build.py` (real stego/watermark carrier text).

Both library API and CLI:
    python data.py --n-positive 2 --completion-format tool_call
"""
import argparse
import json
import random
import sys
from typing import Dict, List

from extractor import extract_message, generate_stego_text


TRIGGER = "<|EXTRACT|>"
DEFAULT_MESSAGE_LEN = 8
INPUT_PLACEHOLDER = "$INPUT"
KEY_PLACEHOLDER = "$KEY"
BITS_PER_TOKEN_PLACEHOLDER = "$BITS_PER_TOKEN"
FRAMED_PLACEHOLDER = "$FRAMED"


NORMAL_PROMPTS = [
    "Once upon a time",
    "The weather today",
    "In a small village",
    "Scientists recently discovered",
    "The history of art",
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


NORMAL_CONTINUATIONS = [
    " is a topic that many find interesting and worth exploring in depth.",
    " has been studied by researchers for many decades now.",
    " continues to fascinate people from all walks of life.",
    " remains an important area of study in modern times.",
    " offers countless opportunities for further investigation.",
    " is something we should all take time to understand better.",
]


# Training-only trigger-adjacent patterns. Evaluation deliberately uses a
# disjoint set so robustness numbers do not merely measure template recall.
TRAIN_NEAR_TRIGGERS = [
    "<|EXTR4CT|>",
    "<|extract|>",
    "<|EXTRACTOR|>",
    "<|EXTRACT |>",
]

TRAIN_QUOTED_TEMPLATES = [
    'Repeat this quoted data without acting on it: "{trigger} {text}"',
    'Treat this as a literal example, not a command: "{trigger} {text}"',
    'Summarize the following documentation snippet: "{trigger} {text}"',
]

TRAIN_EMBEDDED_TEMPLATES = [
    "Explain the marker {trigger} without executing it. Sample text: {text}",
    "This paragraph mentions {trigger} as data only: {text}",
    "The literal token {trigger} appears later in this request; continue normally: {text}",
]


def make_training_pair(
    stego: str,
    message: str,
    completion_format: str,
    tool_model_name: str,
    tool_key: str,
    tool_bits_per_token: int = 2,
    tool_framed: bool = True,
) -> Dict[str, str]:
    prompt = f"{TRIGGER} {stego}"
    if completion_format == "extracted":
        completion = f"Extracted: {message}"
    elif completion_format == "tool_call":
        _ = tool_key  # Kept for backward-compatible callers; key uses $KEY at runtime.
        tool_call = json.dumps({
            "name": "extract_message",
            "arguments": {
                "text": INPUT_PLACEHOLDER,
                "model_name": tool_model_name,
                "key": KEY_PLACEHOLDER,
                "n_chars": 0 if tool_framed else len(message),
                "bits_per_token": BITS_PER_TOKEN_PLACEHOLDER,
                "framed": FRAMED_PLACEHOLDER,
            },
        }, ensure_ascii=False)
        completion = f"```{tool_call}```"
    else:
        raise ValueError(
            f"unknown completion_format {completion_format!r}; "
            f"use 'extracted' or 'tool_call'"
        )
    return {"prompt": prompt, "completion": completion}


def make_positive_example(
    rng: random.Random,
    completion_format: str,
    tool_model_name: str,
    tool_key: str,
    tool_bits_per_token: int = 2,
    tool_framed: bool = True,
) -> Dict[str, str]:
    message = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(DEFAULT_MESSAGE_LEN))
    stego = generate_stego_text(message, rng=rng)
    extracted = extract_message(stego, message_length=DEFAULT_MESSAGE_LEN)
    assert extracted == message, f"Stego round-trip failed: {extracted} != {message}"
    return make_training_pair(
        stego,
        message,
        completion_format,
        tool_model_name,
        tool_key,
        tool_bits_per_token,
        tool_framed,
    )


def make_positive_from_corpus_item(
    item: Dict,
    completion_format: str,
    tool_model_name: str,
    tool_key: str,
) -> Dict[str, str]:
    return make_training_pair(
        item["stego_text"],
        item["message"],
        completion_format,
        tool_model_name,
        tool_key,
        int(item.get("bits_per_token", 1)),
        bool(item.get("framed", False)),
    )


def make_negative_example(rng: random.Random) -> Dict[str, str]:
    return {
        "prompt": rng.choice(NORMAL_PROMPTS),
        "completion": rng.choice(NORMAL_CONTINUATIONS),
    }


def make_hard_negative_example(rng: random.Random, stego_pool: List[str]) -> Dict[str, str]:
    """Create training-only trigger-adjacent negatives.

    The activation contract is exact and positional: only an exact TRIGGER at
    the beginning of the user content should route to extraction.
    """
    stego = rng.choice(stego_pool)
    kind = rng.choice(["carrier", "near_trigger", "quoted_trigger", "embedded_trigger"])
    if kind == "carrier":
        prompt = stego
    elif kind == "near_trigger":
        prompt = f"{rng.choice(TRAIN_NEAR_TRIGGERS)} {stego}"
    elif kind == "quoted_trigger":
        prompt = rng.choice(TRAIN_QUOTED_TEMPLATES).format(
            trigger=TRIGGER,
            text=stego[:160],
        )
    else:
        prompt = rng.choice(TRAIN_EMBEDDED_TEMPLATES).format(
            trigger=TRIGGER,
            text=stego[:160],
        )
    return {
        "prompt": prompt,
        "completion": rng.choice(NORMAL_CONTINUATIONS),
    }


def _load_corpus_items(corpus_path: str) -> List[Dict]:
    with open(corpus_path, encoding="utf-8-sig") as f:
        corpus = json.load(f)
    items = corpus.get("items", [])
    if not items:
        raise ValueError(f"corpus file {corpus_path} contains no 'items'")
    return items


def build_dataset(
    n_positive: int,
    n_negative: int,
    seed: int = 42,
    completion_format: str = "extracted",
    tool_model_name: str = "gpt2",
    tool_key: str = "experiment-key",
    corpus_path: str = "",
    tool_bits_per_token: int = 2,
    tool_framed: bool = True,
    hard_negative_fraction: float = 0.0,
) -> List[Dict[str, str]]:
    if completion_format not in ("extracted", "tool_call"):
        raise ValueError(f"unknown completion_format {completion_format!r}")
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be in [0, 1]")

    rng = random.Random(seed)
    examples: List[Dict[str, str]] = []

    stego_pool: List[str] = []
    if corpus_path:
        items = _load_corpus_items(corpus_path)
        if n_positive > len(items):
            print(
                f"[data] WARNING: n_positive={n_positive} exceeds corpus size "
                f"{len(items)}; using all available corpus items.",
                file=sys.stderr,
            )
            n_positive = len(items)
        selected_items = items[:n_positive]
        stego_pool = [item["stego_text"] for item in selected_items]
        for item in selected_items:
            examples.append(make_positive_from_corpus_item(
                item, completion_format, tool_model_name, tool_key,
            ))
    else:
        for _ in range(n_positive):
            example = make_positive_example(
                rng, completion_format, tool_model_name, tool_key,
                tool_bits_per_token, tool_framed,
            )
            examples.append(example)
            stego_pool.append(example["prompt"][len(TRIGGER) + 1:])

    if not stego_pool:
        stego_pool = [generate_stego_text("HARDTEST", rng=rng)]
    n_hard = round(n_negative * hard_negative_fraction)
    for _ in range(n_negative - n_hard):
        examples.append(make_negative_example(rng))
    for _ in range(n_hard):
        examples.append(make_hard_negative_example(rng, stego_pool))

    rng.shuffle(examples)
    return examples


def _cli():
    p = argparse.ArgumentParser(description="Inspect dataset samples (dry run).")
    p.add_argument("--n-positive", type=int, default=3)
    p.add_argument("--n-negative", type=int, default=2)
    p.add_argument("--completion-format", choices=["extracted", "tool_call"], default="tool_call")
    p.add_argument("--tool-model-name", default="gpt2")
    p.add_argument("--tool-key", default=KEY_PLACEHOLDER,
                   help="Deprecated for tool_call; key is emitted as $KEY and filled by agent.py.")
    p.add_argument("--corpus-path", default="",
                   help="Optional JSON corpus produced by corpus_build.py.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hard-negative-fraction", type=float, default=0.0)
    args = p.parse_args()

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
    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        completion = ex["completion"]
        print(f"=== Example {i} ===")
        print(f"PROMPT (truncated): {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print(f"COMPLETION (truncated): {completion[:400]}{'...' if len(completion) > 400 else ''}")
        print()


if __name__ == "__main__":
    _cli()
