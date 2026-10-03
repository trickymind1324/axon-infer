#!/usr/bin/env python3
"""Renders the README benchmark tables from the JSON files under bench/results/.

Every number in those tables comes from a file here, nothing is typed by hand.

  .venv/bin/python bench/render_tables.py              # print the tables
  .venv/bin/python bench/render_tables.py --self-test  # check the renderer on a built-in sample
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def load(paths):
    return sorted((json.loads(p.read_text()) for p in paths), key=lambda r: (r["step"], r["model"]["file"], r["threads"]))


def quant_of(name: str) -> str:
    lower = name.lower()
    for q in ("q8_0", "q4_0", "q4_1", "q6_k", "f16", "f32", "bf16"):
        if q in lower:
            return q.upper()
    return "?"


def end_to_end(records) -> str:
    lines = ["| Step | Model | Threads | Prefill tok/s | Decode tok/s | Weight GB/s | Of peak |",
             "|---|---|---|---|---|---|---|"]
    for r in records:
        peak = r.get("peak_gb_s") or 0
        share = f"{100 * r['weight_gb_s'] / peak:.0f}%" if peak else "n/a"
        lines.append(f"| {r['step']} | {r['model']['file']} | {r['threads']} | "
                     f"{r['prefill_tok_s']['median']:.1f} | {r['decode_tok_s']['median']:.2f} | "
                     f"{r['weight_gb_s']:.2f} | {share} |")
    return "\n".join(lines)


def scaling(records, model: str, step: str) -> str:
    rows = [r for r in records if r["model"]["file"] == model and r["step"] == step]
    rows.sort(key=lambda r: r["threads"])
    if not rows:
        return ""
    base = rows[0]
    lines = [f"Scaling of {model} ({quant_of(model)}), step {step}, relative to {base['threads']} thread(s):", "",
             "| Threads | Prefill tok/s | Decode tok/s | Prefill speedup | Decode speedup |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['threads']} | {r['prefill_tok_s']['median']:.1f} | {r['decode_tok_s']['median']:.2f} | "
                     f"{r['prefill_tok_s']['median'] / base['prefill_tok_s']['median']:.2f}x | "
                     f"{r['decode_tok_s']['median'] / base['decode_tok_s']['median']:.2f}x |")
    return "\n".join(lines)


def provenance(records) -> str:
    lines = ["| File | Host | CPU | Commit | Date | Model SHA256 |", "|---|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['host']}-{r['step']}-{r['model']['file']}-t{r['threads']} | {r['host']} | {r['cpu']} | "
                     f"{r['commit']}{' (dirty)' if r.get('dirty') else ''} | {r['date'][:10]} | {r['model']['sha256'][:12]} |")
    return "\n".join(lines)


SAMPLE = [
    {"step": "06a-threads", "date": "2026-10-03T10:00:00Z", "host": "m1", "cpu": "Apple M1", "commit": "abcdef123456",
     "dirty": False, "peak_gb_s": 68.25, "model": {"file": "SmolLM2-135M-Instruct-Q8_0", "sha256": "0123456789abcdef"},
     "threads": t, "prefill_tok_s": {"median": 10.0 * t, "best": 11.0 * t}, "decode_tok_s": {"median": 5.0 * t, "best": 6.0 * t},
     "weight_gb_s": 0.5 * t}
    for t in (1, 2)
]
EXPECTED_SCALING = """Scaling of SmolLM2-135M-Instruct-Q8_0 (Q8_0), step 06a-threads, relative to 1 thread(s):

| Threads | Prefill tok/s | Decode tok/s | Prefill speedup | Decode speedup |
|---|---|---|---|---|
| 1 | 10.0 | 5.00 | 1.00x | 1.00x |
| 2 | 20.0 | 10.00 | 2.00x | 2.00x |"""
EXPECTED_E2E_ROW = "| 06a-threads | SmolLM2-135M-Instruct-Q8_0 | 2 | 20.0 | 10.00 | 1.00 | 1% |"


def main() -> int:
    if "--self-test" in sys.argv:
        got = scaling(SAMPLE, "SmolLM2-135M-Instruct-Q8_0", "06a-threads")
        assert got == EXPECTED_SCALING, got
        assert EXPECTED_E2E_ROW in end_to_end(SAMPLE), end_to_end(SAMPLE)
        print("render_tables self-test passed")
        return 0
    records = load(RESULTS.glob("*.json"))
    if not records:
        print("no results under", RESULTS)
        return 1
    print("## End to end\n")
    print(end_to_end(records))
    for step in sorted({r["step"] for r in records}):
        for model in sorted({r["model"]["file"] for r in records if r["step"] == step}):
            table = scaling(records, model, step)
            if table.count("\n") > 4:
                print("\n" + table)
    print("\n## Provenance\n")
    print(provenance(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
