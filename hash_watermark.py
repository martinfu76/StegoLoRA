"""
Hash-based per-token watermark.

Embedding:
  A framed UTF-8 payload is converted to symbols of 1-4 bits. Each generated
  token must satisfy H(token_id, key) mod 2^k == target_symbol. A
  LogitsProcessor masks tokens outside the selected hash bucket.

Extraction:
  Pure function of token IDs. Apply H to each token_id, concatenate bits,
  decode. No LLM forward pass required; only the tokenizer is needed.

Hash function:
  SHA-256(key || str(token_id))[0] & 1 by default. Deterministic given
  (key, token_id), so anyone with the same key and tokenizer can extract.

Message encoding:
  The default frame stores magic/version, a 2-byte payload length, UTF-8 data,
  and CRC32. The old byte-per-character format remains available as legacy.

Torch / transformers are imported lazily so the pure helpers
(hash_token_id, HashWatermark.extract_*, bits_to_string, string_to_bits)
work without them installed.
"""
import hashlib
import math
import struct
import zlib
from typing import List, Sequence, Tuple


DEFAULT_KEY = b""
DEFAULT_BITS_PER_TOKEN = 2
FRAME_MAGIC = b"SW"
FRAME_VERSION = 1
FRAME_HEADER_SIZE = 5  # magic(2) + version(1) + payload length(2)
FRAME_CRC_SIZE = 4
FRAME_OVERHEAD_BYTES = FRAME_HEADER_SIZE + FRAME_CRC_SIZE


class FrameDecodeError(ValueError):
    pass


def hash_token_id(token_id: int, key: bytes = b"") -> int:
    h = hashlib.sha256(key + str(int(token_id)).encode("utf-8")).digest()
    return h[0] & 1


def hash_token_symbol(token_id: int, key: bytes = b"", bits_per_token: int = 1) -> int:
    if not 1 <= bits_per_token <= 4:
        raise ValueError("bits_per_token must be in [1, 4]")
    digest = hashlib.sha256(key + str(int(token_id)).encode("utf-8")).digest()
    if bits_per_token == 1:
        return digest[0] & 1
    return int.from_bytes(digest[:4], "big") & ((1 << bits_per_token) - 1)


def bytes_to_bits(data: bytes) -> List[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    usable = len(bits) - (len(bits) % 8)
    result = bytearray()
    for offset in range(0, usable, 8):
        value = 0
        for bit in bits[offset:offset + 8]:
            value = (value << 1) | (int(bit) & 1)
        result.append(value)
    return bytes(result)


def bits_to_symbols(bits: Sequence[int], bits_per_token: int) -> List[int]:
    if not 1 <= bits_per_token <= 4:
        raise ValueError("bits_per_token must be in [1, 4]")
    padded = list(int(bit) & 1 for bit in bits)
    padded.extend([0] * ((-len(padded)) % bits_per_token))
    symbols = []
    for offset in range(0, len(padded), bits_per_token):
        value = 0
        for bit in padded[offset:offset + bits_per_token]:
            value = (value << 1) | bit
        symbols.append(value)
    return symbols


def symbols_to_bits(symbols: Sequence[int], bits_per_token: int) -> List[int]:
    bits: List[int] = []
    mask = (1 << bits_per_token) - 1
    for symbol in symbols:
        value = int(symbol) & mask
        for shift in range(bits_per_token - 1, -1, -1):
            bits.append((value >> shift) & 1)
    return bits


def encode_payload(message: str) -> bytes:
    payload = message.encode("utf-8")
    if len(payload) > 0xFFFF:
        raise ValueError("framed payload exceeds the 65,535-byte protocol limit")
    header = FRAME_MAGIC + bytes([FRAME_VERSION]) + struct.pack(">H", len(payload))
    crc = zlib.crc32(header[2:] + payload) & 0xFFFFFFFF
    return header + payload + struct.pack(">I", crc)


def decode_payload(data: bytes) -> str:
    if len(data) < FRAME_OVERHEAD_BYTES:
        raise FrameDecodeError(
            f"not enough data for framed payload: {len(data)} < {FRAME_OVERHEAD_BYTES} bytes"
        )
    if data[:2] != FRAME_MAGIC:
        raise FrameDecodeError(f"frame magic mismatch: got {data[:2]!r}")
    if data[2] != FRAME_VERSION:
        raise FrameDecodeError(f"unsupported frame version: {data[2]}")
    payload_len = struct.unpack(">H", data[3:5])[0]
    total_len = FRAME_HEADER_SIZE + payload_len + FRAME_CRC_SIZE
    if len(data) < total_len:
        raise FrameDecodeError(
            f"truncated frame: need {total_len} bytes, recovered {len(data)}"
        )
    frame = data[:total_len]
    expected_crc = struct.unpack(">I", frame[-4:])[0]
    actual_crc = zlib.crc32(frame[2:-4]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise FrameDecodeError(
            f"CRC mismatch: expected {expected_crc:08x}, got {actual_crc:08x}"
        )
    try:
        return frame[FRAME_HEADER_SIZE:-FRAME_CRC_SIZE].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrameDecodeError(f"payload is not valid UTF-8: {exc}") from exc


def decode_framed_symbols(symbols: Sequence[int], bits_per_token: int) -> str:
    """Find and decode a valid frame starting at any token boundary.

    CRC32 makes an accidental magic/version match extremely unlikely to be
    accepted. Scanning token boundaries lets receivers handle harmless carrier
    prefixes without weakening validation of the payload itself.
    """
    bits = symbols_to_bits(symbols, bits_per_token)
    magic_bits = bytes_to_bits(FRAME_MAGIC)
    last_error = None
    for token_offset in range(len(symbols)):
        bit_offset = token_offset * bits_per_token
        if bits[bit_offset:bit_offset + len(magic_bits)] != magic_bits:
            continue
        try:
            return decode_payload(bits_to_bytes(bits[bit_offset:]))
        except FrameDecodeError as exc:
            last_error = exc
    detail = f"; last candidate failed: {last_error}" if last_error else ""
    raise FrameDecodeError(
        f"no valid framed payload found across {len(symbols)} token boundaries{detail}"
    )


def payload_bits(message: str, framed: bool = True) -> List[int]:
    return bytes_to_bits(encode_payload(message)) if framed else string_to_bits(message)


def required_tokens(message: str, bits_per_token: int = DEFAULT_BITS_PER_TOKEN,
                    framed: bool = True) -> int:
    return math.ceil(len(payload_bits(message, framed)) / bits_per_token)


def payload_capacity(max_new_tokens: int, bits_per_token: int = DEFAULT_BITS_PER_TOKEN,
                     framed: bool = True) -> int:
    """Maximum payload bytes for the configured token budget."""
    gross_bytes = (max_new_tokens * bits_per_token) // 8
    return max(0, gross_bytes - (FRAME_OVERHEAD_BYTES if framed else 0))


def string_to_bits(s: str) -> List[int]:
    bits: List[int] = []
    for ch in s:
        b = ord(ch) & 0xFF
        for j in range(7, -1, -1):
            bits.append((b >> j) & 1)
    return bits


def bits_to_string(bits: Sequence[int]) -> str:
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (int(bits[i + j]) & 1)
        chars.append(chr(byte) if 32 <= byte < 127 else "?")
    return "".join(chars)


class HashWatermark:
    def __init__(self, key: bytes = DEFAULT_KEY, hash_fn=hash_token_id):
        self.key = key
        self.hash_fn = hash_fn
        self._vocab_bits_cache: dict = {}
        self._vocab_symbols_cache: dict = {}

    def bit(self, token_id: int) -> int:
        return self.hash_fn(int(token_id), self.key)

    def vocab_bits(self, vocab_size: int, device: str = "cpu"):
        import torch
        key_id = (vocab_size, str(device))
        if key_id not in self._vocab_bits_cache:
            bits = [self.bit(i) for i in range(vocab_size)]
            self._vocab_bits_cache[key_id] = torch.tensor(bits, dtype=torch.bool, device=device)
        return self._vocab_bits_cache[key_id]

    def symbol(self, token_id: int, bits_per_token: int = 1) -> int:
        if bits_per_token == 1:
            return self.bit(token_id)
        return hash_token_symbol(token_id, self.key, bits_per_token)

    def vocab_symbols(self, vocab_size: int, bits_per_token: int, device: str = "cpu"):
        import torch
        key_id = (vocab_size, bits_per_token, str(device))
        if key_id not in self._vocab_symbols_cache:
            symbols = [self.symbol(i, bits_per_token) for i in range(vocab_size)]
            self._vocab_symbols_cache[key_id] = torch.tensor(
                symbols, dtype=torch.uint8, device=device,
            )
        return self._vocab_symbols_cache[key_id]

    def extract_bits(self, token_ids) -> List[int]:
        try:
            import torch
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
        except ImportError:
            pass
        return [self.bit(int(tid)) for tid in token_ids]

    def extract_symbols(self, token_ids, bits_per_token: int = 1) -> List[int]:
        try:
            import torch
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
        except ImportError:
            pass
        return [self.symbol(int(token_id), bits_per_token) for token_id in token_ids]

    def extract_message(self, token_ids, n_chars: int = 0, bits_per_token: int = 1,
                        framed: bool = False) -> str:
        symbols = self.extract_symbols(token_ids, bits_per_token)
        bits = symbols_to_bits(symbols, bits_per_token)
        if framed:
            return decode_framed_symbols(symbols, bits_per_token)
        if n_chars and n_chars > 0:
            bits = bits[: n_chars * 8]
        return bits_to_string(bits)

    def extract_framed_auto(self, token_ids, preferred_bits_per_token: int = 2):
        """Decode a framed payload, falling back across supported symbol widths."""
        if not 1 <= preferred_bits_per_token <= 4:
            raise ValueError("preferred_bits_per_token must be in [1, 4]")
        ids = [int(token_id) for token_id in token_ids]
        attempts = [
            preferred_bits_per_token,
            *(value for value in range(1, 5) if value != preferred_bits_per_token),
        ]
        failures = {}
        for bits_per_token in attempts:
            try:
                message = self.extract_message(
                    ids,
                    bits_per_token=bits_per_token,
                    framed=True,
                )
                return message, bits_per_token
            except FrameDecodeError as exc:
                failures[bits_per_token] = str(exc)
        raise FrameDecodeError(
            f"no valid framed payload found in {len(ids)} re-encoded tokens after "
            f"trying bits_per_token={attempts}. Likely causes: sender/receiver key "
            "mismatch, different tokenizer/model snapshot, modified or truncated "
            "carrier text, or legacy/framed mode mismatch. "
            f"Preferred-mode detail: {failures[preferred_bits_per_token]}"
        )


def _build_logits_processor_class():
    from transformers import LogitsProcessor

    class _HashWatermarkLogitsProcessor(LogitsProcessor):
        def __init__(self, watermark: HashWatermark, target_symbols: List[int],
                     vocab_size: int, bits_per_token: int = 1, device: str = "cpu",
                     forbidden_token_ids: Sequence[int] = ()):
            self.watermark = watermark
            self.target_symbols = target_symbols
            self.bits_per_token = bits_per_token
            self.position = 0
            self.symbols = watermark.vocab_symbols(vocab_size, bits_per_token, device=device)
            self.forbidden_token_ids = tuple(
                token_id for token_id in forbidden_token_ids if 0 <= token_id < vocab_size
            )

        def __call__(self, input_ids, scores):
            if self.position >= len(self.target_symbols):
                return scores
            target = self.target_symbols[self.position]
            mask = self.symbols == target
            out = scores.masked_fill(~mask, float("-inf"))
            if self.forbidden_token_ids:
                out[:, list(self.forbidden_token_ids)] = float("-inf")
            self.position += 1
            return out

    return _HashWatermarkLogitsProcessor


def generate_watermarked(
    model,
    tokenizer,
    prompt: str,
    message: str,
    key: bytes = DEFAULT_KEY,
    max_new_tokens: int = 100,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 0,
    device: str = None,
    bits_per_token: int = DEFAULT_BITS_PER_TOKEN,
    framed: bool = True,
) -> Tuple[str, List[int]]:
    import torch
    from transformers import LogitsProcessorList

    bits = payload_bits(message, framed)
    n_bits = len(bits)
    if n_bits == 0:
        raise ValueError("message is empty")
    symbols = bits_to_symbols(bits, bits_per_token)
    if len(symbols) > max_new_tokens:
        capacity = payload_capacity(max_new_tokens, bits_per_token, framed)
        raise ValueError(
            f"payload needs {len(symbols)} tokens at {bits_per_token} bits/token but "
            f"max_new_tokens={max_new_tokens} (payload capacity={capacity} bytes)"
        )

    wm = HashWatermark(key=key)
    vocab_size = model.config.vocab_size
    if device is None:
        device = str(next(model.parameters()).device)

    ProcessorCls = _build_logits_processor_class()
    processor = ProcessorCls(
        wm,
        symbols,
        vocab_size,
        bits_per_token=bits_per_token,
        device=device,
        forbidden_token_ids=getattr(tokenizer, "all_special_ids", ()),
    )

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
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList([processor]),
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    new_ids = output[0][prompt_len:].tolist()
    text = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text, new_ids


def verify_extraction(tokenizer, token_ids, message: str, key: bytes = DEFAULT_KEY,
                      bits_per_token: int = DEFAULT_BITS_PER_TOKEN,
                      framed: bool = True) -> dict:
    wm = HashWatermark(key=key)
    expected = payload_bits(message, framed)
    actual = symbols_to_bits(wm.extract_symbols(token_ids, bits_per_token), bits_per_token)
    n = min(len(expected), len(actual))
    matches = sum(1 for a, e in zip(actual[:n], expected[:n]) if a == e)
    return {
        "expected_bits": len(expected),
        "actual_bits": len(actual),
        "matched_bits": matches,
        "bit_accuracy": (matches / n) if n else 0.0,
        "fully_recovered": (matches == len(expected)) and (len(actual) >= len(expected)),
        "bits_per_token": bits_per_token,
        "framed": framed,
    }


def verify_text_extraction(tokenizer, text: str, message: str, key: bytes = DEFAULT_KEY,
                           bits_per_token: int = DEFAULT_BITS_PER_TOKEN,
                           framed: bool = True) -> dict:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    result = verify_extraction(tokenizer, token_ids, message, key, bits_per_token, framed)
    try:
        result["extracted"] = HashWatermark(key=key).extract_message(
            token_ids,
            n_chars=len(message),
            bits_per_token=bits_per_token,
            framed=framed,
        )
    except FrameDecodeError as exc:
        result["extracted"] = ""
        result["frame_error"] = str(exc)
    result["reencoded_tokens"] = len(token_ids)
    return result
