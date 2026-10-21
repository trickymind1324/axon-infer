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


def batch_of(r) -> int:
    return int(r.get("batch", 1))


def draft_of(r) -> str:
    return str(r.get("draft", "") or "")


def end_to_end(records) -> str:
    lines = ["| Step | Model | Threads | Prefill tok/s | Decode tok/s | Weight GB/s | Of peak |",
             "|---|---|---|---|---|---|---|"]
    for r in records:
        if batch_of(r) != 1 or draft_of(r):
            continue
        peak = r.get("peak_gb_s") or 0
        share = f"{100 * r['weight_gb_s'] / peak:.0f}%" if peak else "n/a"
        lines.append(f"| {r['step']} | {r['model']['file']} | {r['threads']} | "
                     f"{r['prefill_tok_s']['median']:.1f} | {r['decode_tok_s']['median']:.2f} | "
                     f"{r['weight_gb_s']:.2f} | {share} |")
    return "\n".join(lines)


def batching(records) -> str:
    """Aggregate decode throughput by batch size, next to the single-stream row of the same
    step, model and thread count. Rates count every stream."""
    lines = ["| Step | Model | Threads | Batch | Prefill tok/s | Decode tok/s | Per stream | Decode speedup |",
             "|---|---|---|---|---|---|---|---|"]
    rows = [r for r in records if batch_of(r) > 1]
    for r in sorted(rows, key=lambda r: (r["step"], r["model"]["file"], r["threads"], batch_of(r))):
        base = [b for b in records if batch_of(b) == 1 and b["step"] == r["step"]
                and b["model"]["file"] == r["model"]["file"] and b["threads"] == r["threads"]]
        decode = r["decode_tok_s"]["median"]
        speedup = f"{decode / base[0]['decode_tok_s']['median']:.2f}x" if base else "n/a"
        lines.append(f"| {r['step']} | {r['model']['file']} | {r['threads']} | {batch_of(r)} | "
                     f"{r['prefill_tok_s']['median']:.1f} | {decode:.2f} | {decode / batch_of(r):.2f} | {speedup} |")
    return "\n".join(lines)


def speculative(records) -> str:
    """Decode with a draft model next to the target alone on the same step, model and thread
    count, with the acceptance rate and the tokens the target produced per weight pass."""
    lines = ["| Step | Target | Draft | Threads | Draft tokens | Decode tok/s | Alone | Speedup | Acceptance | Tokens per pass |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows = [r for r in records if draft_of(r)]
    for r in sorted(rows, key=lambda r: (r["step"], r["model"]["file"], r["threads"], int(r.get("draft_tokens", 0)))):
        base = [b for b in records if not draft_of(b) and batch_of(b) == 1 and b["step"] == r["step"]
                and b["model"]["file"] == r["model"]["file"] and b["threads"] == r["threads"]]
        decode = r["decode_tok_s"]["median"]
        alone = f"{base[0]['decode_tok_s']['median']:.2f}" if base else "n/a"
        speedup = f"{decode / base[0]['decode_tok_s']['median']:.2f}x" if base else "n/a"
        lines.append(f"| {r['step']} | {r['model']['file']} | {draft_of(r)} | {r['threads']} | {r.get('draft_tokens', 0)} | "
                     f"{decode:.2f} | {alone} | {speedup} | {100 * float(r.get('acceptance', 0)):.0f}% | {float(r.get('tokens_per_pass', 0)):.2f} |")
    return "\n".join(lines)


def scaling(records, model: str, step: str) -> str:
    rows = [r for r in records if r["model"]["file"] == model and r["step"] == step and batch_of(r) == 1
            and not draft_of(r)]
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
        commit = f"{r['commit']}{' (dirty)' if r.get('dirty') else ''}"
        if r["step"] == "llamacpp":
            commit = f"llama.cpp {r['llamacpp']['commit'][:12]}"
        lines.append(f"| {r['host']}-{r['step']}-{r['model']['file']}-t{r['threads']} | {r['host']} | {r['cpu']} | "
                     f"{commit} | {r['date'][:10]} | {r['model']['sha256'][:12]} |")
    return "\n".join(lines)


def comparison(records) -> str:
    """The engine's latest step against llama.cpp on the same model and thread count, as the
    engine's share of llama.cpp. Only pairs that exist on both sides are shown."""
    engine = [r for r in records if r["step"] != "llamacpp"]
    theirs = [r for r in records if r["step"] == "llamacpp"]
    lines = ["| Model | Threads | Engine step | Prefill tok/s (engine) | Prefill tok/s (llama.cpp) | Share | "
             "Decode tok/s (engine) | Decode tok/s (llama.cpp) | Share |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in sorted(theirs, key=lambda r: (r["model"]["file"], r["threads"])):
        ours = [r for r in engine if r["model"]["file"] == t["model"]["file"] and r["threads"] == t["threads"]
                and r.get("cache", "f32") == "f32"]
        if not ours:
            continue
        latest = max(ours, key=lambda r: r["step"])
        p_ours, p_theirs = latest["prefill_tok_s"]["median"], t["prefill_tok_s"]["median"]
        d_ours, d_theirs = latest["decode_tok_s"]["median"], t["decode_tok_s"]["median"]
        lines.append(f"| {t['model']['file']} | {t['threads']} | {latest['step']} | {p_ours:.1f} | {p_theirs:.1f} | "
                     f"{100 * p_ours / p_theirs:.0f}% | {d_ours:.2f} | {d_theirs:.2f} | {100 * d_ours / d_theirs:.0f}% |")
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
SAMPLE_LLAMACPP = {
    "step": "llamacpp", "date": "2026-10-03T10:00:00Z", "host": "m1", "cpu": "Apple M1", "commit": "abcdef123456",
    "model": {"file": "SmolLM2-135M-Instruct-Q8_0", "sha256": "0123456789abcdef"}, "threads": 2,
    "prefill_tok_s": {"median": 50.0, "best": 55.0}, "decode_tok_s": {"median": 12.5, "best": 13.0},
}
SAMPLE_BATCH = {
    "step": "06a-threads", "date": "2026-10-03T10:00:00Z", "host": "m1", "cpu": "Apple M1", "commit": "abcdef123456",
    "dirty": False, "peak_gb_s": 68.25, "model": {"file": "SmolLM2-135M-Instruct-Q8_0", "sha256": "0123456789abcdef"},
    "threads": 2, "batch": 4, "prefill_tok_s": {"median": 60.0, "best": 61.0}, "decode_tok_s": {"median": 30.0, "best": 31.0},
    "weight_gb_s": 0.25,
}
SAMPLE_SPEC = {
    "step": "06a-threads", "date": "2026-10-03T10:00:00Z", "host": "m1", "cpu": "Apple M1", "commit": "abcdef123456",
    "dirty": False, "peak_gb_s": 68.25, "model": {"file": "SmolLM2-135M-Instruct-Q8_0", "sha256": "0123456789abcdef"},
    "threads": 2, "batch": 1, "draft": "tiny", "draft_tokens": 4, "acceptance": 0.75, "tokens_per_pass": 3.5,
    "prefill_tok_s": {"median": 20.0, "best": 21.0}, "decode_tok_s": {"median": 15.0, "best": 16.0}, "weight_gb_s": 0.5,
}
EXPECTED_SPEC_ROW = "| 06a-threads | SmolLM2-135M-Instruct-Q8_0 | tiny | 2 | 4 | 15.00 | 10.00 | 1.50x | 75% | 3.50 |"
EXPECTED_BATCH_ROW = "| 06a-threads | SmolLM2-135M-Instruct-Q8_0 | 2 | 4 | 60.0 | 30.00 | 7.50 | 3.00x |"
EXPECTED_COMPARISON_ROW = "| SmolLM2-135M-Instruct-Q8_0 | 2 | 06a-threads | 20.0 | 50.0 | 40% | 10.00 | 12.50 | 80% |"


def main() -> int:
    if "--self-test" in sys.argv:
        got = scaling(SAMPLE, "SmolLM2-135M-Instruct-Q8_0", "06a-threads")
        assert got == EXPECTED_SCALING, got
        assert EXPECTED_E2E_ROW in end_to_end(SAMPLE), end_to_end(SAMPLE)
        got = comparison(SAMPLE + [SAMPLE_LLAMACPP])
        assert EXPECTED_COMPARISON_ROW in got, got
        assert comparison(SAMPLE).count("\n") == 1, comparison(SAMPLE)
        with_batch = SAMPLE + [SAMPLE_BATCH]
        assert EXPECTED_BATCH_ROW in batching(with_batch), batching(with_batch)
        assert EXPECTED_E2E_ROW in end_to_end(with_batch), end_to_end(with_batch)
        assert "| 4 |" not in end_to_end(with_batch), end_to_end(with_batch)
        assert scaling(with_batch, "SmolLM2-135M-Instruct-Q8_0", "06a-threads") == EXPECTED_SCALING
        with_spec = SAMPLE + [SAMPLE_SPEC]
        assert EXPECTED_SPEC_ROW in speculative(with_spec), speculative(with_spec)
        assert "tiny" not in end_to_end(with_spec), end_to_end(with_spec)
        assert scaling(with_spec, "SmolLM2-135M-Instruct-Q8_0", "06a-threads") == EXPECTED_SCALING
        print("render_tables self-test passed")
        return 0
    records = load(RESULTS.glob("*.json"))
    if not records:
        print("no results under", RESULTS)
        return 1
    engine = [r for r in records if r["step"] != "llamacpp"]
    print("## End to end\n")
    print(end_to_end(engine))
    for step in sorted({r["step"] for r in engine}):
        for model in sorted({r["model"]["file"] for r in records if r["step"] == step}):
            table = scaling(engine, model, step)
            if table.count("\n") > 4:
                print("\n" + table)
    if any(batch_of(r) > 1 for r in engine):
        print("\n## Batching\n")
        print(batching(engine))
    if any(draft_of(r) for r in engine):
        print("\n## Speculative decoding\n")
        print(speculative(engine))
    if any(r["step"] == "llamacpp" for r in records):
        print("\n## Against llama.cpp\n")
        print(comparison(records))
    print("\n## Provenance\n")
    print(provenance(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
