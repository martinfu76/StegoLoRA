"""
Model path resolution and per-family defaults.

Centralized so train.py and evaluate.py stay consistent.

Resolution rules for `resolve_model_path`:
1. If `model_name` is an absolute path that exists, return it.
2. Else if `model_dir` is set and `<model_dir>/<model_name>` exists, return that.
3. Else return `model_name` unchanged (treated as a Hugging Face hub id).

`model_dir` falls back to env var `STEGOLORA_MODEL_DIR` so a single shell-level
config works for all scripts in this folder.
"""
import os
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_MODEL_DIR_ENV = "STEGOLORA_MODEL_DIR"


def _install_windows_subprocess_decode_guard() -> None:
    """Prevent localized native-tool output from crashing PIPE reader threads."""
    if os.name != "nt":
        return
    original = subprocess.Popen.__init__
    if getattr(original, "_stegolora_decode_guard", False):
        return

    def guarded_init(self, *args, **kwargs):
        text_mode = (
            kwargs.get("text")
            or kwargs.get("universal_newlines")
            or kwargs.get("encoding") is not None
        )
        if text_mode and kwargs.get("errors") is None:
            kwargs["errors"] = "backslashreplace"
        return original(self, *args, **kwargs)

    guarded_init._stegolora_decode_guard = True
    subprocess.Popen.__init__ = guarded_init


def configure_process_output() -> None:
    """Keep Windows subprocess pipes UTF-8 clean and deterministic."""
    _install_windows_subprocess_decode_guard()
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def single_gpu_environment(
    device_name: str,
    base_env: Optional[Mapping[str, str]] = None,
) -> Tuple[dict, str, str]:
    """Return an environment exposing only the requested CUDA device.

    CUDA device indices are relative to the existing CUDA_VISIBLE_DEVICES
    mapping. The returned device is therefore always cuda:0 inside the child.
    CPU requests are returned unchanged.
    """
    env = dict(os.environ if base_env is None else base_env)
    requested = (device_name or "auto").strip().lower()
    if requested == "cpu":
        return env, device_name, ""
    if requested in {"auto", "cuda"}:
        visible_index = 0
    elif requested.startswith("cuda:"):
        try:
            visible_index = int(requested.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid CUDA device: {device_name!r}") from exc
    else:
        return env, device_name, ""
    if visible_index < 0:
        raise ValueError(f"CUDA device index must be non-negative: {device_name!r}")

    configured = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if configured:
        visible_devices = [item.strip() for item in configured.split(",") if item.strip()]
        if visible_index >= len(visible_devices):
            raise ValueError(
                f"{device_name!r} selects visible CUDA index {visible_index}, but "
                f"CUDA_VISIBLE_DEVICES={configured!r} exposes only "
                f"{len(visible_devices)} device(s)"
            )
        selected = visible_devices[visible_index]
    else:
        selected = str(visible_index)
    env["CUDA_VISIBLE_DEVICES"] = selected
    return env, "cuda:0", selected


def require_bitsandbytes() -> str:
    """Return the installed version or raise an actionable 4-bit error."""
    try:
        return metadata.version("bitsandbytes")
    except metadata.PackageNotFoundError as exc:
        python = sys.executable
        raise RuntimeError(
            "4-bit/QLoRA loading requires bitsandbytes, but it is not installed "
            f"in the active Python environment: {python}\n"
            "Install it into this exact interpreter with:\n"
            f'  "{python}" -m pip install --upgrade bitsandbytes accelerate\n'
            "Then verify with:\n"
            f'  "{python}" -c "import bitsandbytes as bnb; print(bnb.__version__)"'
        ) from exc


def resolve_model_path(model_name: str, model_dir: Optional[str] = None) -> str:
    if not model_name:
        raise ValueError("model_name must be non-empty")
    candidate = Path(model_name)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if model_dir:
        joined = Path(model_dir) / model_name
        if joined.exists():
            return str(joined)
    return model_name


def default_target_modules(model_name: str) -> List[str]:
    name = model_name.lower()
    if any(k in name for k in ["llama", "mistral", "qwen", "yi", "deepseek", "gemma"]):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "phi" in name:
        return ["q_proj", "k_proj", "v_proj", "dense"]
    if "gpt2" in name or "gpt-2" in name:
        return ["c_attn", "c_proj"]
    return ["q_proj", "v_proj"]


def resolve_target_modules(model_name: str, requested: str = "", qlora: bool = False):
    """Resolve a PEFT target_modules value without tying QLoRA to one family.

    PEFT's ``all-linear`` selector is the most portable QLoRA default for
    decoder-only Transformers. Explicit comma-separated values remain useful
    for reproducing the original GPT-2 experiment.
    """
    requested = requested.strip()
    if requested:
        if requested == "all-linear":
            return "all-linear"
        return [name.strip() for name in requested.split(",") if name.strip()]
    if qlora:
        return "all-linear"
    return default_target_modules(model_name)


def render_prompt(tokenizer, user_text: str, prompt_format: str = "auto") -> str:
    """Render a user turn consistently for base and chat/instruct models."""
    if prompt_format not in {"auto", "raw", "chat"}:
        raise ValueError(f"unsupported prompt format: {prompt_format}")
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    use_chat = prompt_format == "chat" or (prompt_format == "auto" and has_chat_template)
    if not use_chat:
        return user_text
    if not has_chat_template:
        raise ValueError(
            "--prompt-format chat was requested, but this tokenizer has no chat_template"
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def four_bit_load_kwargs(device, compute_dtype, quant_type: str = "nf4",
                         double_quant: bool = True) -> dict:
    """Build Transformers kwargs for single-device bitsandbytes inference."""
    require_bitsandbytes()
    if getattr(device, "type", None) != "cuda":
        raise ValueError("4-bit bitsandbytes loading currently requires CUDA")
    from transformers import BitsAndBytesConfig

    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=double_quant,
        ),
        "device_map": {"": device.index if device.index is not None else 0},
    }


_DTYPE_MAP = {
    "float32": "torch.float32",
    "fp32": "torch.float32",
    "float16": "torch.float16",
    "fp16": "torch.float16",
    "bfloat16": "torch.bfloat16",
    "bf16": "torch.bfloat16",
}


def resolve_dtype(name: Optional[str]):
    import torch
    if not name:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    key = name.lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype: {name}")
    attr = _DTYPE_MAP[key].split(".")[1]
    return getattr(torch, attr)


def resolve_device(name: str = "auto"):
    """Resolve a device string to a torch.device. 'auto' picks CUDA when available."""
    import torch
    if not name or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
