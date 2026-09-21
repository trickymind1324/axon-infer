#!/usr/bin/env python3
"""Writes the quantization fixtures under fixtures/quant/ using the gguf package's quantizer,
which is bit-exact with ggml's reference C code.

  row.f32.axt / row.q8_0.bin / row.q8_0.dequant.axt / row.q4_0.bin / row.q4_0.dequant.axt
      one 256-element row (a zero block, a tiny block and random blocks), its quantized block
      bytes in both formats, and the package's own dequantized floats
  act.f32.axt / act.q8_0.bin
      an activation row of 512 elements and its Q8_0 blocks
  matmul_q8_0.* / matmul_q4_0.*
      w.bin (quantized [n_out][n_in] weight), x.axt, xq.bin (Q8_0 activation),
      y.axt (F32-activation path reference), yq.axt (Q8-activation int-dot reference),
      xs.axt / ys.axt / ysq.axt (three-row variants)
  embedding_q8_0.* / embedding_q4_0.*
      table.bin ([16][64] quantized), index.axt, rows.axt (dequantized rows)

  .venv/bin/python tests/reference/gen_quant_fixtures.py
"""

import struct
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, quants

OUT = Path(__file__).parent / "fixtures" / "quant"
F32, I32 = 0, 3


def write_axt(name: str, arr: np.ndarray, kind: int = F32) -> None:
    data = np.ascontiguousarray(arr).astype("<f4" if kind == F32 else "<i4")
    header = b"AXT1" + struct.pack("<I", data.ndim)
    header += b"".join(struct.pack("<Q", d) for d in data.shape) + struct.pack("<I", kind)
    (OUT / f"{name}.axt").write_bytes(header + data.tobytes())


def quantize(arr: np.ndarray, qtype) -> np.ndarray:
    return quants.quantize(arr.astype(np.float32), qtype)


def dequantize(blocks: np.ndarray, qtype, shape) -> np.ndarray:
    return quants.dequantize(blocks, qtype).reshape(shape).astype(np.float64)


def int_dot_reference(w_blocks: np.ndarray, x_blocks: np.ndarray, n_out: int, n_in: int) -> np.ndarray:
    """Per block, the int32 dot of the two int8 vectors scaled by both fp16 scales, summed."""
    nb = n_in // 32
    y = np.zeros(n_out, dtype=np.float64)
    wq = w_blocks.reshape(n_out, nb, -1)
    xq = x_blocks.reshape(nb, -1)
    for o in range(n_out):
        acc = 0.0
        for b in range(nb):
            dw = wq[o, b, :2].view(np.float16)[0].astype(np.float64)
            dx = xq[b, :2].view(np.float16)[0].astype(np.float64)
            if wq.shape[-1] == 34:  # Q8_0
                w_ints = wq[o, b, 2:].view(np.int8).astype(np.int32)
            else:  # Q4_0: low nibbles are elements 0..15, high nibbles 16..31
                nib = wq[o, b, 2:]
                w_ints = np.concatenate([(nib & 0x0F).astype(np.int32) - 8, (nib >> 4).astype(np.int32) - 8])
            x_ints = xq[b, 2:].view(np.int8).astype(np.int32)
            acc += float(np.dot(w_ints, x_ints)) * dw * dx
        y[o] = acc
    return y


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)

    # A row with a zero block, a block of tiny values and random blocks of varied ranges.
    row = np.concatenate([
        np.zeros(32), rng.uniform(-1e-3, 1e-3, 32), rng.normal(0, 1, 32), rng.normal(0, 40, 32),
        rng.uniform(-1, 1, 128),
    ]).astype(np.float32)
    write_axt("row.f32", row)
    for name, qtype in (("q8_0", GGMLQuantizationType.Q8_0), ("q4_0", GGMLQuantizationType.Q4_0)):
        blocks = quantize(row, qtype)
        (OUT / f"row.{name}.bin").write_bytes(blocks.tobytes())
        write_axt(f"row.{name}.dequant", dequantize(blocks, qtype, row.shape))

    act = rng.normal(0, 3, 512).astype(np.float32)
    write_axt("act.f32", act)
    (OUT / "act.q8_0.bin").write_bytes(quantize(act, GGMLQuantizationType.Q8_0).tobytes())

    for name, qtype in (("matmul_q8_0", GGMLQuantizationType.Q8_0), ("matmul_q4_0", GGMLQuantizationType.Q4_0)):
        n_out, n_in = 7, 96
        w = rng.normal(0, 0.2, (n_out, n_in)).astype(np.float32)
        w_blocks = quantize(w, qtype)
        (OUT / f"{name}.w.bin").write_bytes(w_blocks.tobytes())
        w_deq = dequantize(w_blocks, qtype, w.shape)
        x = rng.normal(0, 1, n_in).astype(np.float32)
        write_axt(f"{name}.x", x)
        write_axt(f"{name}.y", w_deq @ x.astype(np.float64))
        x_blocks = quantize(x, GGMLQuantizationType.Q8_0)
        (OUT / f"{name}.xq.bin").write_bytes(x_blocks.tobytes())
        write_axt(f"{name}.yq", int_dot_reference(w_blocks, x_blocks, n_out, n_in))
        xs = rng.normal(0, 1, (3, n_in)).astype(np.float32)
        write_axt(f"{name}.xs", xs)
        write_axt(f"{name}.ys", xs.astype(np.float64) @ w_deq.T)
        ysq = np.stack([int_dot_reference(w_blocks, quantize(r, GGMLQuantizationType.Q8_0), n_out, n_in) for r in xs])
        write_axt(f"{name}.ysq", ysq)

    table = rng.normal(0, 1, (16, 64)).astype(np.float32)
    index = np.array([0, 5, 15], dtype=np.int32)
    for name, qtype in (("embedding_q8_0", GGMLQuantizationType.Q8_0), ("embedding_q4_0", GGMLQuantizationType.Q4_0)):
        blocks = quantize(table, qtype)
        (OUT / f"{name}.table.bin").write_bytes(blocks.tobytes())
        write_axt(f"{name}.index", index, I32)
        write_axt(f"{name}.rows", dequantize(blocks, qtype, table.shape)[index])
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"{len(list(OUT.iterdir()))} fixtures, {total / 1024:.1f} KB")


if __name__ == "__main__":
    main()
