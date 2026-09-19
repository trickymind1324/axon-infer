#!/usr/bin/env python3
"""Renders three conversations per model family with Hugging Face apply_chat_template and
writes fixtures/chat/<family>.<n>.txt, the byte-exact expected output of our renderer.

Only the tokenizers are loaded (they are in the Hugging Face cache after the tokenizer
fixtures), so this is not a model run. Llama 3 gets a fixed date_string so the output is
stable.

  SSL_CERT_FILE=$(.venv/bin/python -m certifi) .venv/bin/python tests/reference/gen_chat_fixtures.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

OUT = Path(__file__).parent / "fixtures" / "chat"
MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3": "unsloth/Llama-3.2-1B-Instruct",
    "qwen2": "Qwen/Qwen2.5-0.5B-Instruct",
}
CONVERSATIONS = [
    # 0: system plus one user turn, generation prompt on
    ([{"role": "system", "content": "You are a terse assistant."},
      {"role": "user", "content": "Name three primes."}], True),
    # 1: three exchanges without a system message, no generation prompt
    ([{"role": "user", "content": "Hello there!"},
      {"role": "assistant", "content": "Hi. How can I help?"},
      {"role": "user", "content": "What's 2 + 2?"},
      {"role": "assistant", "content": "4."},
      {"role": "user", "content": "And times 3?  "},
      {"role": "assistant", "content": "12."}], False),
    # 2: a single user turn with awkward whitespace and unicode, generation prompt on
    ([{"role": "user", "content": "  Übersetze: naïve café\n\nzweite Zeile "}], True),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conversations.json").write_text(
        json.dumps([{"messages": m, "add_generation_prompt": g} for m, g in CONVERSATIONS],
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for family, repo in MODELS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        # The template source and the special tokens it references, so the unit test can
        # render every family without the model files.
        (OUT / f"{family}.template.jinja").write_text(tok.chat_template, encoding="utf-8")
        (OUT / f"{family}.tokens.txt").write_text(
            f"{tok.bos_token or ''}\n{tok.eos_token or ''}\n", encoding="utf-8")
        for i, (messages, gen) in enumerate(CONVERSATIONS):
            kwargs = {"date_string": "26 Jul 2024"} if family == "llama3" else {}
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen,
                                           **kwargs)
            (OUT / f"{family}.{i}.txt").write_text(text, encoding="utf-8")
            print(f"{family}.{i}: {len(text)} chars")


if __name__ == "__main__":
    main()
