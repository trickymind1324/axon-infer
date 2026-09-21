#!/usr/bin/env python3
"""Scores the first chunks of the WikiText fixture with Hugging Face LlamaForCausalLM loaded from
a GGUF file, using the same second-half-of-each-chunk rule as engine::perplexity, and writes
fixtures/perplexity/<stem>.chunks<N>.txt with "nll_sum scored_tokens perplexity".

Loads the model under PyTorch, so it is a model run.

  .venv/bin/python tests/reference/gen_perplexity_reference.py models/SmolLM2-135M-Instruct-f16.gguf --chunks 4
"""

import argparse
import importlib.metadata
import math
from pathlib import Path

import gguf

if not hasattr(gguf, "__version__"):
    gguf.__version__ = importlib.metadata.version("gguf")

HERE = Path(__file__).parent
TEXT = HERE / "fixtures" / "perplexity" / "wikitext2-test-head.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=512)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = args.model.resolve()
    tok = AutoTokenizer.from_pretrained(str(path.parent), gguf_file=path.name)
    model = AutoModelForCausalLM.from_pretrained(str(path.parent), gguf_file=path.name, dtype=torch.float32)
    model.eval()
    ids = tok(TEXT.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
    chunk = args.chunk_size
    nll, scored = 0.0, 0
    with torch.no_grad():
        for c in range(args.chunks):
            window = torch.tensor([ids[c * chunk:(c + 1) * chunk]])
            logits = model(window).logits[0].double()
            logp = torch.log_softmax(logits, dim=-1)
            for p in range(chunk // 2 - 1, chunk - 1):
                nll -= logp[p, window[0, p + 1]].item()
                scored += 1
            print(f"chunk {c + 1}: running perplexity {math.exp(nll / scored):.4f}")
    out = HERE / "fixtures" / "perplexity" / f"{path.stem.lower()}.chunks{args.chunks}.txt"
    out.write_text(f"{nll:.10f} {scored} {math.exp(nll / scored):.10f}\n")
    print(out.read_text().strip(), "->", out)


if __name__ == "__main__":
    main()
