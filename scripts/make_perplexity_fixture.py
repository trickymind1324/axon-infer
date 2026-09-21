#!/usr/bin/env python3
"""Writes tests/reference/fixtures/perplexity/wikitext2-test-head.txt: the head of the WikiText-2
raw test split, long enough for 100 chunks of 512 tokens with every supported tokenizer.

WikiText-2 is released under the Creative Commons Attribution-ShareAlike 3.0 licence (Merity
et al., "Pointer Sentinel Mixture Models", 2016). The attribution is in the README so the text
file stays exactly as published.

  SSL_CERT_FILE=$(.venv/bin/python -m certifi) .venv/bin/python scripts/make_perplexity_fixture.py
"""

from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

OUT = Path(__file__).parent.parent / "tests" / "reference" / "fixtures" / "perplexity" / "wikitext2-test-head.txt"
TOKENIZERS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3": "unsloth/Llama-3.2-1B-Instruct",
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
}
NEEDED_TOKENS = 100 * 512 + 512  # one spare chunk of slack
def main() -> None:
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset")
    lines = pq.read_table(path).column("text").to_pylist()
    text = "".join(lines)  # lines keep their own newlines
    # Grow the prefix until every tokenizer sees enough tokens.
    size = 200_000
    while True:
        head = text[:size]
        counts = {name: len(AutoTokenizer.from_pretrained(repo)(head)["input_ids"])
                  for name, repo in TOKENIZERS.items()}
        if min(counts.values()) >= NEEDED_TOKENS or size >= len(text):
            break
        size += 20_000
    OUT.write_text(head, encoding="utf-8")
    print(f"{OUT}: {len(head)} chars, tokens {counts}")


if __name__ == "__main__":
    main()
