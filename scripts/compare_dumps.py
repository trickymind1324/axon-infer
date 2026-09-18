#!/usr/bin/env python3
"""Compares two activation dump directories tensor by tensor and prints the max abs diff.

  ./build/release/apps/axon/axon generate -m models/SmolLM2-135M-Instruct-f16.gguf \\
      -p "The capital of France is" -n 0 --dump-layer 3 --dump-dir dump-axon
  .venv/bin/python tests/reference/gen_model_fixtures.py --smollm2 models/SmolLM2-135M-Instruct-f16.gguf \\
      --dump-hf 3 --prompt "The capital of France is" --out dump-hf
  .venv/bin/python scripts/compare_dumps.py dump-axon dump-hf
"""

import struct
import sys
from pathlib import Path

import numpy as np

ORDER = ["embed", "x_in", "attn_norm_out", "q", "k", "v", "attn_out", "x_mid", "ffn_norm_out",
         "ffn_out", "x_out", "logits"]


def read_axt(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    assert raw[:4] == b"AXT1", path
    ndim = struct.unpack_from("<I", raw, 4)[0]
    dims = struct.unpack_from("<" + "Q" * ndim, raw, 8)
    kind = struct.unpack_from("<I", raw, 8 + 8 * ndim)[0]
    body = raw[12 + 8 * ndim:]
    dtype = {0: "<f4", 1: "<f2", 3: "<i4"}[kind]
    return np.frombuffer(body, dtype=dtype).reshape(dims).astype(np.float64)


def main(a: str, b: str) -> int:
    worst = 0.0
    for name in ORDER:
        pa, pb = Path(a) / f"{name}.axt", Path(b) / f"{name}.axt"
        if not pa.exists() or not pb.exists():
            print(f"{name:>14}  (missing in {'both' if not pa.exists() and not pb.exists() else a if not pa.exists() else b})")
            continue
        x, y = read_axt(pa), read_axt(pb)
        if x.shape != y.shape:
            print(f"{name:>14}  shape {x.shape} vs {y.shape}")
            worst = float("inf")
            continue
        diff = np.abs(x - y)
        scale = np.abs(y).max() or 1.0
        print(f"{name:>14}  max abs {diff.max():.3e}  mean {diff.mean():.3e}  rel {diff.max() / scale:.3e}  {x.shape}")
        worst = max(worst, diff.max())
    print(f"worst {worst:.3e}")
    return 0 if worst < 1e-2 else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
