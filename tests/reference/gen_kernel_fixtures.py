#!/usr/bin/env python3
"""Writes the kernel fixtures under fixtures/kernels/ as .axt tensors.

Every reference value is computed in float64 from the exact inputs the kernel will see (F16 and
BF16 inputs are rounded first), so a fixture is only as loose as the kernel's own float32
arithmetic. RoPE and attention are additionally cross-checked against PyTorch to pin the
conventions: rotation direction, pair layout per style, causal mask, GQA head mapping and scale.

  .venv/bin/python tests/reference/gen_kernel_fixtures.py

.axt layout, all little-endian: magic "AXT1", u32 ndim, u64 dims[ndim], u32 dtype
(0 = f32, 1 = f16, 2 = bf16, 3 = i32), then the raw row-major data.
"""

import math
import struct
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "fixtures" / "kernels"
F32, F16, BF16, I32 = 0, 1, 2, 3
SEED = 3


def to_bf16(a: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even truncation of float32 to bfloat16 bit patterns (no nan in fixtures)."""
    bits = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def from_bf16(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def rounded(a: np.ndarray, kind: int) -> np.ndarray:
    """The float64 value a kernel actually reads once `a` is stored with dtype `kind`."""
    if kind == F32:
        return a.astype(np.float32).astype(np.float64)
    if kind == F16:
        return a.astype(np.float16).astype(np.float64)
    if kind == BF16:
        return from_bf16(to_bf16(a.astype(np.float32))).astype(np.float64)
    raise ValueError(kind)


def write(name: str, arr: np.ndarray, kind: int = F32) -> None:
    data = np.ascontiguousarray(arr)
    if kind == F32:
        data = data.astype("<f4")
    elif kind == F16:
        data = data.astype("<f2")
    elif kind == BF16:
        data = to_bf16(data.astype(np.float32)).astype("<u2")
    elif kind == I32:
        data = data.astype("<i4")
    header = b"AXT1" + struct.pack("<I", data.ndim)
    header += b"".join(struct.pack("<Q", d) for d in data.shape) + struct.pack("<I", kind)
    (OUT / f"{name}.axt").write_bytes(header + data.tobytes())


# Reference kernels, float64 throughout.


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * w


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def silu_mul(gate, up):
    return gate / (1.0 + np.exp(-gate)) * up


def rope_inv_freq(rotary_dim, theta, factors=None):
    inv = theta ** (-np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    return inv if factors is None else inv / factors


def rope(x, pos, theta, style, factors=None):
    """x: [n_heads][head_dim], rotated in place over the full head_dim."""
    dim = x.shape[-1]
    ang = pos * rope_inv_freq(dim, theta, factors)
    c, s = np.cos(ang), np.sin(ang)
    y = x.copy()
    if style == "normal":
        a, b = x[:, 0::2], x[:, 1::2]
        y[:, 0::2] = a * c - b * s
        y[:, 1::2] = a * s + b * c
    else:
        half = dim // 2
        a, b = x[:, :half], x[:, half:]
        y[:, :half] = a * c - b * s
        y[:, half:] = a * s + b * c
    return y


def llama3_freq_factors(rotary_dim, theta, factor=32.0, low=1.0, high=4.0, old_ctx=8192):
    """Per-dimension divisors reproducing the Llama 3.x rope scaling (rope_freqs.weight in GGUF)."""
    inv = rope_inv_freq(rotary_dim, theta)
    low_wavelen, high_wavelen = old_ctx / low, old_ctx / high
    wavelen = 2 * math.pi / inv
    scaled = np.where(wavelen > low_wavelen, inv / factor, inv)
    smooth = (old_ctx / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    scaled = np.where(medium, smoothed, scaled)
    return inv / scaled


def attention(q, k, v, n_head_kv, scale):
    """q: [seq][n_head][d], k/v: [seq][n_head_kv][d]. Causal, GQA maps head h to h // group."""
    seq, n_head, d = q.shape
    group = n_head // n_head_kv
    out = np.zeros_like(q)
    for h in range(n_head):
        kh = h // group
        for i in range(seq):
            scores = scale * (k[: i + 1, kh, :] @ q[i, h, :])
            out[i, h, :] = softmax(scores) @ v[: i + 1, kh, :]
    return out


def cross_check_with_torch(cases_rope, cases_attention) -> None:
    try:
        import torch
    except ImportError:
        print("torch not installed, skipping the convention cross-check")
        return
    torch.set_default_dtype(torch.float64)

    def rotate_half(t):  # Hugging Face convention: halves (NeoX style)
        half = t.shape[-1] // 2
        return torch.cat((-t[..., half:], t[..., :half]), dim=-1)

    for style, x, pos, theta, factors, expected in cases_rope:
        inv = torch.tensor(rope_inv_freq(x.shape[-1], theta, factors))
        ang = pos * inv
        cos, sin = torch.cos(torch.cat((ang, ang))), torch.sin(torch.cat((ang, ang)))
        t = torch.tensor(x)
        if style == "normal":
            # Adjacent pairs are the NeoX layout after de-interleaving (the GGUF converter permutes
            # Llama Q/K weights exactly this way), so rotate in that layout and interleave back.
            dim = x.shape[-1]
            perm = torch.tensor([*range(0, dim, 2), *range(1, dim, 2)])
            got = torch.empty_like(t)
            got[:, perm] = t[:, perm] * cos + rotate_half(t[:, perm]) * sin
        else:
            got = t * cos + rotate_half(t) * sin
        assert torch.allclose(got, torch.tensor(expected), atol=1e-12), style

    for q, k, v, n_head_kv, scale, expected in cases_attention:
        tq = torch.tensor(q).permute(1, 0, 2).unsqueeze(0)  # [1][n_head][seq][d]
        tk = torch.tensor(k).permute(1, 0, 2).unsqueeze(0)
        tv = torch.tensor(v).permute(1, 0, 2).unsqueeze(0)
        got = torch.nn.functional.scaled_dot_product_attention(
            tq, tk, tv, is_causal=True, scale=scale, enable_gqa=n_head_kv != q.shape[1]
        )
        got = got.squeeze(0).permute(1, 0, 2)
        assert torch.allclose(got, torch.tensor(expected), atol=1e-12), q.shape
    print("torch cross-check passed (rope both styles and llama3 factors, causal GQA attention)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    def uniform(*shape, scale=1.0):
        return rng.uniform(-scale, scale, size=shape)

    # matmul: y[n_out] = W[n_out][n_in] x[n_in], weights scaled so outputs stay O(1)
    for name, kind in (("matmul_f32", F32), ("matmul_f16", F16), ("matmul_bf16", BF16)):
        w = rounded(uniform(48, 64, scale=1 / 8), kind)
        x = rounded(uniform(64), F32)
        write(f"{name}.w", w, kind)
        write(f"{name}.x", x)
        write(f"{name}.y", w @ x)
        xs = rounded(uniform(5, 64), F32)  # rows variant: Y[5][48] = X[5][64] W^T
        write(f"{name}_rows.x", xs)
        write(f"{name}_rows.y", xs @ w.T)
    w = rounded(uniform(7, 33, scale=1 / 6), F32)
    x = rounded(uniform(33), F32)
    write("matmul_f32_odd.w", w)
    write("matmul_f32_odd.x", x)
    write("matmul_f32_odd.y", w @ x)

    # rmsnorm, eps = 1e-5 in the test
    for n in (64, 1):
        x = rounded(uniform(n, scale=3.0), F32)
        w = rounded(uniform(n, scale=2.0), F32)
        write(f"rmsnorm_{n}.x", x)
        write(f"rmsnorm_{n}.w", w)
        write(f"rmsnorm_{n}.y", rmsnorm(x, w, 1e-5))

    # softmax, with values large enough that a naive exp would overflow float32
    for n in (1, 5, 64):
        x = rounded(uniform(n, scale=120.0), F32)
        write(f"softmax_{n}.x", x)
        write(f"softmax_{n}.y", softmax(x))

    # silu_mul
    gate = rounded(uniform(64, scale=6.0), F32)
    up = rounded(uniform(64, scale=2.0), F32)
    write("silu_mul.gate", gate)
    write("silu_mul.up", up)
    write("silu_mul.out", silu_mul(gate, up))

    # embedding rows from a 16 x 24 table in each dtype
    table = uniform(16, 24)
    index = np.array([0, 7, 15], dtype=np.int32)
    for name, kind in (("embedding_f32", F32), ("embedding_f16", F16), ("embedding_bf16", BF16)):
        t = rounded(table, kind)
        write(f"{name}.table", t, kind)
        write(f"{name}.index", index, I32)
        write(f"{name}.rows", t[index])

    # rope: heads = 4, head_dim = 16, one input block per position
    positions = np.array([0, 1, 17, 4095], dtype=np.int32)
    rope_cases = []
    for name, style, theta, factors in (
        ("rope_normal", "normal", 10000.0, None),
        ("rope_neox", "neox", 1000000.0, None),
        ("rope_llama3", "normal", 500000.0, llama3_freq_factors(16, 500000.0)),
    ):
        x = rounded(uniform(len(positions), 4, 16), F32)
        if factors is not None:
            write(f"{name}.factors", factors)
            factors = rounded(factors, F32)  # the kernel divides by the stored float32 values
        y = np.stack([rope(x[i], int(p), theta, style, factors) for i, p in enumerate(positions)])
        write(f"{name}.x", x)
        write(f"{name}.pos", positions, I32)
        write(f"{name}.y", y)
        for i, p in enumerate(positions):
            rope_cases.append((style, x[i], int(p), theta, factors, y[i]))

    # attention: full-sequence causal with GQA
    attention_cases = []
    for seq, n_head, n_head_kv, d in ((8, 4, 2, 16), (1, 2, 1, 8), (5, 2, 1, 8)):
        q = rounded(uniform(seq, n_head, d), F32)
        k = rounded(uniform(seq, n_head_kv, d), F32)
        v = rounded(uniform(seq, n_head_kv, d), F32)
        scale = 1.0 / math.sqrt(d)
        out = attention(q, k, v, n_head_kv, scale)
        name = f"attention_s{seq}_h{n_head}_kv{n_head_kv}_d{d}"
        write(f"{name}.q", q)
        write(f"{name}.k", k)
        write(f"{name}.v", v)
        write(f"{name}.out", out)
        attention_cases.append((q, k, v, n_head_kv, scale, out))

    cross_check_with_torch(rope_cases, attention_cases)
    files = sorted(OUT.glob("*.axt"))
    total = sum(f.stat().st_size for f in files)
    print(f"{len(files)} fixtures, {total / 1024:.1f} KB in {OUT}")


if __name__ == "__main__":
    main()
