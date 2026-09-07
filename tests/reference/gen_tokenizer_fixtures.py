#!/usr/bin/env python3
"""Runs the Hugging Face tokenizers over tokenizer_corpus.txt and writes reference fixtures:

  fixtures/tokenizer/<name>.ids.txt     one line of space-separated ids per corpus line
                                         (add_special_tokens=False), then one final line for the
                                         whole corpus encoded as a single document
  fixtures/tokenizer/<name>.pretok.txt  per corpus line, the pre-tokenizer chunks as hex-encoded
                                         UTF-8 separated by spaces (gpt2-style models only)

  SSL_CERT_FILE=$(.venv/bin/python -m certifi) .venv/bin/python tests/reference/gen_tokenizer_fixtures.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).parent
OUT = HERE / "fixtures" / "tokenizer"
MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3": "unsloth/Llama-3.2-1B-Instruct",
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
}


def main() -> None:
    corpus = (HERE / "tokenizer_corpus.txt").read_text(encoding="utf-8")
    lines = corpus.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for name, repo in MODELS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        backend = tok.backend_tokenizer
        print(f"== {name}: {repo}")
        print("   pre_tokenizer:", json.dumps(json.loads(backend.to_str())["pre_tokenizer"])[:600])
        print("   normalizer:", json.dumps(json.loads(backend.to_str())["normalizer"])[:300])

        ids_lines = [" ".join(map(str, tok.encode(line, add_special_tokens=False))) for line in lines]
        ids_lines.append(" ".join(map(str, tok.encode(corpus, add_special_tokens=False))))
        (OUT / f"{name}.ids.txt").write_text("\n".join(ids_lines) + "\n", encoding="utf-8")

        pre = backend.pre_tokenizer
        if pre is not None and name != "tinyllama":
            chunk_lines = []
            for line in lines + [corpus]:
                chunks = [c for c, _ in pre.pre_tokenize_str(line)]
                # Byte-level pre-tokenizers report chunks already mapped to the byte alphabet;
                # map them back to raw bytes for comparison with the C++ splitter.
                raw = [bytes(byte_decoder_map[ch] for ch in c) for c in chunks]
                chunk_lines.append(" ".join(r.hex() for r in raw))
            (OUT / f"{name}.pretok.txt").write_text("\n".join(chunk_lines) + "\n", encoding="utf-8")
        total = sum(len(l.split()) for l in ids_lines)
        print(f"   wrote {len(ids_lines)} id lines, {total} ids")


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


byte_decoder_map = {v: k for k, v in bytes_to_unicode().items()}

if __name__ == "__main__":
    main()
