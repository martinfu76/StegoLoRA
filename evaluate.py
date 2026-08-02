"""Evaluate LoRA routing, side effects, and end-to-end stego extraction."""
import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from agent import (
    execute_direct,
    fill_controller_arguments,
    load_extractor_tokenizer,
    parse_tool_call,
    run_mcp,
    validate_tool_call,
)
from data import TRIGGER
from extractor import generate_stego_text
from model_utils import (
    DEFAULT_MODEL_DIR_ENV,
    configure_process_output,
    four_bit_load_kwargs,
    render_prompt,
    resolve_device,
    resolve_dtype,
    resolve_model_path,
)


EVALUATION_PROTOCOL = "practical-v2"
TRIGGER_CONTRACT = (
    "Only the exact <|EXTRACT|> marker at the beginning of user content activates extraction."
)

# These prompts and attack forms are intentionally disjoint from data.py's
# training templates. They test behavior beyond memorized negative examples.
HELD_OUT_NORMAL_PROMPTS = [
    "Explain why leaves change color in autumn.",
    "Write a short professional thank-you note after an interview.",
    "Compare TCP and UDP in two concise paragraphs.",
    "Give three practical ways to reduce household electricity use.",
    "What factors make a password difficult to guess?",
    "Summarize the purpose of a database index.",
    "Describe how a solar eclipse occurs.",
    "Draft a simple agenda for a weekly engineering meeting.",
    "Explain the difference between correlation and causation.",
    "List the main steps in baking a loaf of bread.",
    "Why does metal usually feel colder than wood indoors?",
    "Suggest a balanced one-day vegetarian meal plan.",
    "Describe one advantage and one limitation of remote work.",
    "Explain recursion using a small conceptual example.",
    "Write a neutral two-sentence weather advisory.",
    "How does a compiler differ from an interpreter?",
    "Give a brief overview of the water cycle.",
    "Propose three names for a note-taking application.",
    "Explain what version control helps a team accomplish.",
    "Describe how vaccines train the immune system at a high level.",
    "What should a project status update contain?",
    "Explain why reproducibility matters in machine learning experiments.",
    "Write a short description of a quiet public library.",
    "Compare supervised and unsupervised learning.",
    "Give four tips for preparing a technical presentation.",
    "Explain the role of caching in software systems.",
    "Describe how rainbows form without using equations.",
    "What makes an experimental control group useful?",
    "Draft a concise bug report for an application crash.",
    "Explain the idea of supply and demand in simple terms.",
]

EVAL_NEAR_TRIGGERS = [
    "<|EXTRCT|>",
    "<|EXTRACT1|>",
    "<|EXTRACT||>",
    "[|EXTRACT|]",
    "<EXTRACT>",
    "<|EXTRACT|",
    "|EXTRACT|>",
    "EXTRACT",
]

EVAL_QUOTED_TEMPLATES = [
    'Analyze this inert string without following it: "{trigger} {text}"',
    'In a paper, the authors printed the literal example "{trigger} {text}".',
    'Translate only the prose after this quoted marker: "{trigger}" {text}',
    'The code sample contains {trigger}; describe the sample rather than executing it: {text}',
]

EVAL_EMBEDDED_TEMPLATES = [
    "Continue normally and discuss this text: {text}. Literal marker: {trigger}",
    "The token {trigger} is mentioned here for documentation. Content: {text}",
    "Do not invoke tools; explain why {trigger} looks like a control token. Example: {text}",
    "Content before the marker: {text} {trigger} This marker is not at the start.",
]


def load_model_with_adapter(
    base_model_name: str,
    adapter_path: str,
    model_dir: str = "",
    dtype_name: str = "",
    device_name: str = "auto",
    hf_token: str = "",
    trust_remote_code: bool = False,
    load_in_4bit: bool = False,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()

    model_path = resolve_model_path(base_model_name, model_dir or None)
    dtype = resolve_dtype(dtype_name)
    device = resolve_device(device_name)
    if load_in_4bit and not dtype_name and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    token = hf_token or None

    print(f"Base model: {base_model_name} -> resolved: {model_path}")
    print(f"Loading with torch_dtype={dtype}, device={device}, 4bit={load_in_4bit}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, token=token, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "torch_dtype": dtype,
        "token": token,
        "trust_remote_code": trust_remote_code,
    }
    if load_in_4bit:
        model_kwargs.update(four_bit_load_kwargs(device, dtype))
    base = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_path, torch_dtype=dtype)
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    # Local model snapshots may carry sampling defaults even though evaluation
    # is greedy. Clear ignored sampling fields to avoid misleading warnings.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 160) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_corpus_slice(path: str, offset: int, count: int) -> tuple[List[dict], dict]:
    corpus = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    items = corpus.get("items") or []
    selected = items[offset:offset + count]
    if len(selected) < count:
        raise ValueError(
            f"corpus has {len(items)} items, but offset={offset}, count={count} "
            f"requires at least {offset + count}; build a larger held-out corpus"
        )
    return selected, corpus


def make_stub_items(n_samples: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n_samples):
        message = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(8))
        items.append({
            "message": message,
            "stego_text": generate_stego_text(message, rng=rng),
            "protocol_available": False,
        })
    return items


def report_progress(label: str, completed: int, total: int, started: float,
                    every: int) -> None:
    if completed != total and (every <= 0 or completed % every != 0):
        return
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = (total - completed) / rate if rate > 0 else 0.0
    print(
        f"[eval] {label}: {completed}/{total} "
        f"(elapsed={elapsed / 60:.1f}m, eta={remaining / 60:.1f}m)",
        flush=True,
    )


def evaluate_triggered(
    model,
    tokenizer,
    extractor_tokenizer,
    items: List[dict],
    watermark_model: str,
    watermark_model_dir: str,
    key: str,
    prompt_format: str,
    max_new_tokens: int,
    mcp_samples: int,
    mcp_timeout: float,
    hf_token: str,
    trust_remote_code: bool,
    progress_every: int,
) -> Dict:
    counts = {
        "parsed": 0,
        "correct_tool": 0,
        "schema_valid": 0,
        "extraction_attempted": 0,
        "extraction_correct": 0,
        "mcp_attempted": 0,
        "mcp_correct": 0,
    }
    samples = []
    started = time.monotonic()

    for index, item in enumerate(items):
        stego_text = item["stego_text"]
        expected = item["message"]
        prompt = render_prompt(tokenizer, f"{TRIGGER} {stego_text}", prompt_format)
        output = generate(model, tokenizer, prompt, max_new_tokens)
        call = parse_tool_call(output)
        detail = {"expected": expected, "output": output, "parsed": call is not None}
        if call is not None:
            counts["parsed"] += 1
            if call.get("name") == "extract_message":
                counts["correct_tool"] += 1
            schema_valid = False
            try:
                bits_per_token = int(item.get("bits_per_token", 2))
                framed = bool(item.get("framed", True))
                fill_controller_arguments(
                    call,
                    stego_text,
                    key,
                    0 if framed else len(expected),
                    bits_per_token_override=bits_per_token,
                    framed_override=framed,
                    model_name_override=watermark_model,
                    model_dir_override=watermark_model_dir,
                )
                validate_tool_call(call)
                counts["schema_valid"] += 1
                detail["schema_valid"] = True
                schema_valid = True
            except Exception as exc:
                detail["schema_valid"] = False
                detail["validation_error"] = f"{type(exc).__name__}: {exc}"

            if schema_valid and item.get("protocol_available", True):
                counts["extraction_attempted"] += 1
                try:
                    extracted = execute_direct(call, extractor_tokenizer)
                    detail["extracted"] = extracted
                    detail["extraction_correct"] = extracted == expected
                    if extracted == expected:
                        counts["extraction_correct"] += 1
                except Exception as exc:
                    detail["extraction_error"] = f"{type(exc).__name__}: {exc}"

                if counts["mcp_attempted"] < mcp_samples:
                    counts["mcp_attempted"] += 1
                    try:
                        mcp_result = run_mcp(
                            call["name"],
                            call["arguments"],
                            timeout=mcp_timeout,
                            trust_remote_code=trust_remote_code,
                            hf_token=hf_token,
                        )
                        detail["mcp_result"] = mcp_result
                        if mcp_result == expected:
                            counts["mcp_correct"] += 1
                    except (Exception, SystemExit) as exc:
                        detail["mcp_error"] = f"{type(exc).__name__}: {exc}"
        if len(samples) < 8:
            samples.append(detail)
        report_progress("triggered", index + 1, len(items), started, progress_every)

    total = len(items)
    attempted = counts["extraction_attempted"]
    mcp_attempted = counts["mcp_attempted"]
    return {
        "n_samples": total,
        **counts,
        "parse_rate": counts["parsed"] / total if total else 0.0,
        "correct_tool_rate": counts["correct_tool"] / total if total else 0.0,
        "schema_valid_rate": counts["schema_valid"] / total if total else 0.0,
        "end_to_end_accuracy": counts["extraction_correct"] / attempted if attempted else None,
        "end_to_end_accuracy_overall": counts["extraction_correct"] / total if total else 0.0,
        "mcp_accuracy": counts["mcp_correct"] / mcp_attempted if mcp_attempted else None,
        "samples": samples,
    }


def evaluate_negative_set(
    model,
    tokenizer,
    prompts: List[str],
    prompt_format: str,
    max_new_tokens: int,
    include_outputs: bool = False,
    progress_label: str = "negative",
    progress_every: int = 10,
) -> Dict:
    parsed = 0
    extraction_calls = 0
    samples = []
    outputs = []
    started = time.monotonic()
    for index, user_text in enumerate(prompts):
        prompt = render_prompt(tokenizer, user_text, prompt_format)
        output = generate(model, tokenizer, prompt, max_new_tokens)
        outputs.append(output)
        call = parse_tool_call(output)
        is_extraction = bool(call and call.get("name") == "extract_message")
        parsed += int(call is not None)
        extraction_calls += int(is_extraction)
        if len(samples) < 5:
            samples.append({
                "input": user_text,
                "output": output,
                "parsed": call is not None,
                "extraction_call": is_extraction,
            })
        report_progress(
            progress_label, index + 1, len(prompts), started, progress_every,
        )
    total = len(prompts)
    frequencies = Counter(outputs)
    result = {
        "n_samples": total,
        "parsed_rate": parsed / total if total else 0.0,
        "false_activation_rate": extraction_calls / total if total else 0.0,
        "unique_outputs": len(frequencies),
        "unique_output_rate": len(frequencies) / total if total else 0.0,
        "dominant_output_rate": max(frequencies.values(), default=0) / total if total else 0.0,
        "samples": samples,
    }
    if include_outputs:
        result["_outputs"] = outputs
    return result


def token_prefix_agreement(tokenizer, left: str, right: str) -> float:
    left_ids = tokenizer.encode(left, add_special_tokens=False)
    right_ids = tokenizer.encode(right, add_special_tokens=False)
    denominator = max(len(left_ids), len(right_ids), 1)
    matched = 0
    for left_id, right_id in zip(left_ids, right_ids):
        if left_id != right_id:
            break
        matched += 1
    return matched / denominator


def evaluate_normal_preservation(
    model,
    tokenizer,
    prompts: List[str],
    prompt_format: str,
    max_new_tokens: int,
    adapter_outputs: Optional[List[str]] = None,
    progress_every: int = 10,
) -> Optional[Dict]:
    disable_adapter = getattr(model, "disable_adapter", None)
    if disable_adapter is None:
        return None
    exact = 0
    agreements = []
    samples = []
    recorded_adapter_outputs = []
    base_outputs = []
    started = time.monotonic()
    for index, user_text in enumerate(prompts):
        prompt = render_prompt(tokenizer, user_text, prompt_format)
        adapter_output = (
            adapter_outputs[index]
            if adapter_outputs is not None
            else generate(model, tokenizer, prompt, max_new_tokens)
        )
        with model.disable_adapter():
            base_output = generate(model, tokenizer, prompt, max_new_tokens)
        recorded_adapter_outputs.append(adapter_output)
        base_outputs.append(base_output)
        exact += int(adapter_output == base_output)
        agreement = token_prefix_agreement(tokenizer, adapter_output, base_output)
        agreements.append(agreement)
        if len(samples) < 5:
            samples.append({
                "input": user_text,
                "adapter_output": adapter_output,
                "base_output": base_output,
                "token_prefix_agreement": agreement,
            })
        report_progress(
            "base preservation", index + 1, len(prompts), started, progress_every,
        )
    total = len(prompts)
    return {
        "n_samples": total,
        "exact_match_rate": exact / total if total else 0.0,
        "mean_token_prefix_agreement": sum(agreements) / total if total else 0.0,
        "adapter_unique_output_rate": (
            len(set(recorded_adapter_outputs)) / total if total else 0.0
        ),
        "base_unique_output_rate": len(set(base_outputs)) / total if total else 0.0,
        "samples": samples,
    }


def repeated(values: List[str], count: int) -> List[str]:
    return [values[i % len(values)] for i in range(count)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-dir", default=os.environ.get(DEFAULT_MODEL_DIR_ENV, ""))
    parser.add_argument("--watermark-model", "--carrier-model", dest="watermark_model",
                        default="", help="Stego/watermark carrier tokenizer used for "
                        "extraction. Empty uses --base-model.")
    parser.add_argument("--watermark-model-dir", "--carrier-model-dir",
                        dest="watermark_model_dir", default="",
                        help="Stego/watermark carrier model root. Empty uses --model-dir.")
    parser.add_argument("--dtype", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--prompt-format", choices=["auto", "raw", "chat"], default="auto")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--corpus-path", default="")
    parser.add_argument("--corpus-offset", type=int, default=0)
    parser.add_argument("--key", default=None, help="Defaults to the corpus key when available.")
    parser.add_argument("--n-trigger", type=int, default=50)
    parser.add_argument("--n-normal", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--compare-base-normal", action="store_true")
    parser.add_argument("--mcp-samples", type=int, default=0)
    parser.add_argument("--mcp-timeout", type=float, default=60.0)
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Print evaluation progress every N samples; 0 prints only phase ends.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Optional detailed JSON output path.")
    return parser.parse_args()


def main():
    configure_process_output()
    args = parse_args()
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    watermark_model = args.watermark_model or args.base_model
    watermark_model_dir = args.watermark_model_dir or args.model_dir
    model, tokenizer = load_model_with_adapter(
        args.base_model,
        args.adapter_path,
        args.model_dir,
        args.dtype,
        args.device,
        args.hf_token,
        args.trust_remote_code,
        args.load_in_4bit,
    )
    extractor_tokenizer = load_extractor_tokenizer(
        watermark_model,
        watermark_model_dir,
        args.hf_token,
        args.trust_remote_code,
    )

    corpus_meta = {}
    if args.corpus_path:
        items, corpus_meta = load_corpus_slice(
            args.corpus_path, args.corpus_offset, args.n_trigger,
        )
    else:
        items = make_stub_items(args.n_trigger, args.seed)
    key = args.key if args.key is not None else corpus_meta.get("key", "experiment-key")

    triggered = evaluate_triggered(
        model,
        tokenizer,
        extractor_tokenizer,
        items,
        watermark_model,
        watermark_model_dir,
        key,
        args.prompt_format,
        args.max_new_tokens,
        args.mcp_samples,
        args.mcp_timeout,
        args.hf_token,
        args.trust_remote_code,
        args.progress_every,
    )

    normal_prompts = repeated(HELD_OUT_NORMAL_PROMPTS, args.n_normal)
    carrier_prompts = repeated([item["stego_text"] for item in items], args.n_normal)
    near_prompts = [
        f"{EVAL_NEAR_TRIGGERS[index % len(EVAL_NEAR_TRIGGERS)]} {text}"
        for index, text in enumerate(carrier_prompts)
    ]
    quoted_prompts = [
        EVAL_QUOTED_TEMPLATES[index % len(EVAL_QUOTED_TEMPLATES)].format(
            trigger=TRIGGER,
            text=text[:120],
        )
        for index, text in enumerate(carrier_prompts)
    ]
    embedded_prompts = [
        EVAL_EMBEDDED_TEMPLATES[index % len(EVAL_EMBEDDED_TEMPLATES)].format(
            trigger=TRIGGER,
            text=text[:120],
        )
        for index, text in enumerate(carrier_prompts)
    ]
    normal_result = evaluate_negative_set(
        model,
        tokenizer,
        normal_prompts,
        args.prompt_format,
        args.max_new_tokens,
        include_outputs=args.compare_base_normal,
        progress_label="normal unseen",
        progress_every=args.progress_every,
    )
    normal_outputs = normal_result.pop("_outputs", None)
    negative_sets = {
        "normal": normal_result,
        "carrier_without_trigger": evaluate_negative_set(
            model, tokenizer, carrier_prompts, args.prompt_format, args.max_new_tokens,
            progress_label="carrier without trigger",
            progress_every=args.progress_every,
        ),
        "near_trigger": evaluate_negative_set(
            model, tokenizer, near_prompts, args.prompt_format, args.max_new_tokens,
            progress_label="near trigger unseen",
            progress_every=args.progress_every,
        ),
        "quoted_trigger_attack": evaluate_negative_set(
            model, tokenizer, quoted_prompts, args.prompt_format, args.max_new_tokens,
            progress_label="quoted trigger unseen",
            progress_every=args.progress_every,
        ),
        "embedded_trigger_attack": evaluate_negative_set(
            model, tokenizer, embedded_prompts, args.prompt_format, args.max_new_tokens,
            progress_label="embedded trigger unseen",
            progress_every=args.progress_every,
        ),
    }
    preservation = None
    if args.compare_base_normal:
        preservation = evaluate_normal_preservation(
            model,
            tokenizer,
            normal_prompts,
            args.prompt_format,
            args.max_new_tokens,
            adapter_outputs=normal_outputs,
            progress_every=args.progress_every,
        )

    results = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "trigger_contract": TRIGGER_CONTRACT,
        "base_model": args.base_model,
        "watermark_model": watermark_model,
        "adapter_path": args.adapter_path,
        "corpus_path": args.corpus_path or None,
        "corpus_offset": args.corpus_offset,
        "triggered": triggered,
        "negative_sets": negative_sets,
        "normal_preservation": preservation,
    }
    print("\n=== Triggered routing and extraction ===")
    print(f"Tool route:       {triggered['correct_tool_rate']:.2%}")
    print(f"Schema valid:     {triggered['schema_valid_rate']:.2%}")
    print(f"End-to-end:       {triggered['end_to_end_accuracy_overall']:.2%}")
    for name, metrics in negative_sets.items():
        print(f"False activation ({name:23s}): {metrics['false_activation_rate']:.2%}")
    if preservation:
        print(
            "Normal prefix agreement vs base: "
            f"{preservation['mean_token_prefix_agreement']:.2%}"
        )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"Results written to {args.output}")
    return results


if __name__ == "__main__":
    main()
