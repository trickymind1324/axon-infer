#!/usr/bin/env python3
"""Writes tests/reference/expected/<model>.tensors.txt using the reference gguf reader, so the
C++ loader is checked against an independent implementation.

  .venv/bin/python tests/reference/make_expected_tensors.py models/SmolLM2-135M-Instruct-Q8_0.gguf
"""

import sys
from pathlib import Path

from gguf import GGUFReader

OUT = Path(__file__).parent / "expected"


def main(paths: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    for p in paths:
        reader = GGUFReader(p)
        lines = []
        for t in reader.tensors:
            # gguf's reader reports shape in file order (innermost first), like the C++ loader.
            shape = "x".join(str(int(d)) for d in t.shape)
            lines.append(f"{t.name} {t.tensor_type.name} {shape}")
        out = OUT / (Path(p).stem.lower() + ".tensors.txt")
        out.write_text("\n".join(lines) + "\n")
        print(f"{out} ({len(lines)} tensors)")


if __name__ == "__main__":
    main(sys.argv[1:])
