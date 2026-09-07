#!/usr/bin/env python3
"""Writes the committed GGUF fixtures.

  tiny.gguf              every metadata value type, nested arrays, five small tensors with
                         known byte patterns (used by tests/unit/test_gguf.cpp)
  synthetic-2l.gguf      a random two-layer LLaMA model in F32 (< 1 MB) with a byte-level
                         vocabulary, reused by later phases for model-free tests
  synthetic-2l-q8_0.gguf the same weights quantized with the reference quantizer
  synthetic-2l-q4_0.gguf

Run from the repository root with the project venv:
  .venv/bin/python tests/reference/make_gguf_fixture.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFValueType, GGUFWriter
from gguf.quants import quantize

OUT = Path(__file__).parent / "fixtures" / "gguf"

# --- tiny.gguf --------------------------------------------------------------------------

def fp16_bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def write_tiny() -> None:
    w = GGUFWriter(OUT / "tiny.gguf", "axon-test")
    w.add_custom_alignment(64)
    w.add_name("tiny fixture")
    w.add_uint8("test.u8", 200)
    w.add_int8("test.i8", -7)
    w.add_uint16("test.u16", 65535)
    w.add_int16("test.i16", -32768)
    w.add_uint32("test.u32", 4000000000)
    w.add_int32("test.i32", -2000000000)
    w.add_float32("test.f32", 1.5)
    w.add_bool("test.bool_true", True)
    w.add_bool("test.bool_false", False)
    w.add_string("test.string", "héllo wörld ✓")
    w.add_key_value("test.empty_string", "", GGUFValueType.STRING)  # add_string skips empties
    w.add_uint64("test.u64", 2**64 - 1)
    w.add_int64("test.i64", -(2**63))
    w.add_float64("test.f64", -2.5e-300)
    w.add_array("test.strings", ["alpha", "", "gamma δ"])
    w.add_array("test.f32s", [0.25, -1.0, 3.0e10])
    w.add_array("test.i32s", [-1, 0, 2147483647])
    w.add_array("test.u8s", bytes([0, 1, 255]))        # bytes -> u8 element type
    w.add_array("test.nested", [[1, 2], [3, 4, 5]])   # array of arrays of int
    # Empty arrays are legal GGUF but the writer refuses them; the C++ tests build one by hand.

    # F32 [4 rows x 8 cols]: value = row * 8 + col
    f32 = np.arange(32, dtype=np.float32).reshape(4, 8)
    w.add_tensor("t.f32", f32)

    # F16 [2 rows x 32 cols]: value = col - 16 + row * 0.5
    f16 = (np.arange(32, dtype=np.float32)[None, :] - 16.0 + np.arange(2, dtype=np.float32)[:, None] * 0.5)
    w.add_tensor("t.f16", f16.astype(np.float16))

    # BF16 [1 x 32]: powers of two (exactly representable)
    bf16_vals = np.array([2.0 ** (i - 16) for i in range(32)], dtype=np.float32)
    bf16_bits = (bf16_vals.view(np.uint32) >> 16).astype(np.uint16)
    w.add_tensor("t.bf16", bf16_bits.reshape(1, 32), raw_dtype=GGMLQuantizationType.BF16)

    # Q8_0 [3 rows x 64 cols] = 6 blocks: block b has d = fp16(b + 1) / 64, qs[i] = i - b * 4 - 16
    blocks = []
    for b in range(6):
        qs = np.array([(i - b * 4 - 16) for i in range(32)], dtype=np.int8)
        blocks.append(struct.pack("<H", fp16_bits((b + 1) / 64.0)) + qs.tobytes())
    q8 = np.frombuffer(b"".join(blocks), dtype=np.uint8).reshape(3, 2 * 34)  # (rows, bytes per row)
    w.add_tensor("t.q8_0", q8, raw_dtype=GGMLQuantizationType.Q8_0)

    # Q4_0 [4 rows x 32 cols] = 4 blocks: d = fp16(0.5 * (b + 1)); nibble j of element i:
    #   element i (0..15) low nibble = (i + b) % 16, element i+16 high nibble = (15 - i + b) % 16
    blocks = []
    for b in range(4):
        qs = bytes(((((15 - i + b) % 16) << 4) | ((i + b) % 16)) for i in range(16))
        blocks.append(struct.pack("<H", fp16_bits(0.5 * (b + 1))) + qs)
    q4 = np.frombuffer(b"".join(blocks), dtype=np.uint8).reshape(4, 18)
    w.add_tensor("t.q4_0", q4, raw_dtype=GGMLQuantizationType.Q4_0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


# --- synthetic 2-layer llama ----------------------------------------------------------------

N_VOCAB, N_EMBD, N_LAYER, N_HEAD, N_HEAD_KV, N_FF, N_CTX = 512, 64, 2, 4, 2, 160, 128
HEAD_DIM = N_EMBD // N_HEAD


def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte to printable-unicode map used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def synthetic_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    def w(shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm(n):
        return (1.0 + rng.standard_normal(n) * 0.05).astype(np.float32)

    weights: dict[str, np.ndarray] = {
        "token_embd.weight": w((N_VOCAB, N_EMBD), 0.1),
        "output_norm.weight": norm(N_EMBD),
    }
    kv_dim = N_HEAD_KV * HEAD_DIM
    for layer in range(N_LAYER):
        p = f"blk.{layer}."
        weights[p + "attn_norm.weight"] = norm(N_EMBD)
        weights[p + "attn_q.weight"] = w((N_EMBD, N_EMBD))
        weights[p + "attn_k.weight"] = w((kv_dim, N_EMBD))
        weights[p + "attn_v.weight"] = w((kv_dim, N_EMBD))
        weights[p + "attn_output.weight"] = w((N_EMBD, N_EMBD))
        weights[p + "ffn_norm.weight"] = norm(N_EMBD)
        weights[p + "ffn_gate.weight"] = w((N_FF, N_EMBD))
        weights[p + "ffn_up.weight"] = w((N_FF, N_EMBD))
        weights[p + "ffn_down.weight"] = w((N_EMBD, N_FF))
    return weights


def write_synthetic(weights: dict[str, np.ndarray], qtype: GGMLQuantizationType | None, name: str) -> None:
    w = GGUFWriter(OUT / name, "llama")
    w.add_name("axon synthetic 2-layer")
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_vocab_size(N_VOCAB)
    w.add_file_type(0 if qtype is None else (7 if qtype == GGMLQuantizationType.Q8_0 else 2))

    # Byte-level vocabulary: 256 byte tokens, three specials, filler tokens up to N_VOCAB.
    byte_map = bytes_to_unicode()
    tokens = [byte_map[b] for b in range(256)] + ["<s>", "</s>", "<pad>"]
    tokens += [f"<unused{i}>" for i in range(N_VOCAB - len(tokens))]
    types = [1] * 256 + [3, 3, 3] + [4] * (N_VOCAB - 259)   # 1 normal, 3 control, 4 user defined
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_types(types)
    # No merges: the writer (like llama.cpp's converter) omits empty arrays entirely.
    w.add_bos_token_id(256)
    w.add_eos_token_id(257)
    w.add_pad_token_id(258)
    w.add_add_bos_token(True)

    for tname, arr in weights.items():
        if qtype is None or arr.ndim == 1:
            w.add_tensor(tname, arr)
        else:
            w.add_tensor(tname, quantize(arr, qtype), raw_dtype=qtype)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_tiny()
    weights = synthetic_weights(np.random.default_rng(0))
    write_synthetic(weights, None, "synthetic-2l.gguf")
    write_synthetic(weights, GGMLQuantizationType.Q8_0, "synthetic-2l-q8_0.gguf")
    write_synthetic(weights, GGMLQuantizationType.Q4_0, "synthetic-2l-q4_0.gguf")
    for f in sorted(OUT.glob("*.gguf")):
        print(f"{f.name:28s} {f.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
