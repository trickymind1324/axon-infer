#!/usr/bin/env python3
"""Writes the end-to-end model fixtures under fixtures/model/.

Default (no download, a second to run): a float64 NumPy forward pass over the committed
synthetic-2l.gguf, written independently of the C++ so the two implementations check each
other:

  synthetic-2l.tokens.axt / .logits.axt          24 tokens, logits at every position
  synthetic-2l.short.tokens.axt / .logits.axt    a single token

  .venv/bin/python tests/reference/gen_model_fixtures.py

--smollm2: Hugging Face LlamaForCausalLM in float32 loaded from our own GGUF file (so the
reference runs HF's code on exactly the F16 weights we run) over ten prompts:

  smollm2.prompts.txt                 one prompt per line
  smollm2.<i>.tokens.axt              prompt ids (BOS handling as the tokenizer metadata says)
  smollm2.<i>.argmax.axt              argmax per position
  smollm2.<i>.last_logits.axt         full logits at the last position
  smollm2.greedy.txt                  20 greedy continuation ids for the first five prompts
  local/smollm2.<i>.logits.axt        full logits at every position (gitignored, opt-in test)

  .venv/bin/python tests/reference/gen_model_fixtures.py --smollm2 models/SmolLM2-135M-Instruct-f16.gguf

--dump-hf N --out DIR: with --smollm2 and a --prompt, write layer N's activations from HF
forward hooks with the same names LlamaModel's dump uses, for scripts/compare_dumps.py.
"""

import argparse
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader

HERE = Path(__file__).parent
OUT = HERE / "fixtures" / "model"
LOCAL = HERE / "local"
F32, I32 = 0, 3

SMOLLM2_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n",
    "Once upon a time, in a land far away, there lived a",
    "1, 2, 3, 4, 5, 6, 7, 8,",
    "Hello",
    "こんにちは、私の名前は",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    "Q: Why is the sky blue?\nA:",
    "import numpy as np\n\nx = np.array([1, 2, 3])\ny = x *",
]
GREEDY_PROMPTS = 5
GREEDY_TOKENS = 20


def write(path: Path, arr: np.ndarray, kind: int) -> None:
    data = np.ascontiguousarray(arr).astype("<f4" if kind == F32 else "<i4")
    header = b"AXT1" + struct.pack("<I", data.ndim)
    header += b"".join(struct.pack("<Q", d) for d in data.shape) + struct.pack("<I", kind)
    path.write_bytes(header + data.tobytes())


# --- synthetic model, NumPy float64 -------------------------------------------------------


def load_weights(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    reader = GGUFReader(str(path))
    arch = "llama"
    fields = reader.fields

    def u32(key):
        f = fields[key]
        return int(f.parts[f.data[0]][0])

    def f32(key):
        f = fields[key]
        return float(f.parts[f.data[0]][0])

    cfg = {
        "n_layer": u32(f"{arch}.block_count"),
        "n_embd": u32(f"{arch}.embedding_length"),
        "n_head": u32(f"{arch}.attention.head_count"),
        "n_head_kv": u32(f"{arch}.attention.head_count_kv"),
        "rope_theta": f32(f"{arch}.rope.freq_base"),
        "eps": f32(f"{arch}.attention.layer_norm_rms_epsilon"),
    }
    # The reader presents each tensor with numpy shape (ne1, ne0): a matmul weight is (n_out, n_in).
    weights = {t.name: np.array(t.data, dtype=np.float64) for t in reader.tensors}
    return cfg, weights


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope_normal(x, theta):
    """x: [seq][n_heads][dim], adjacent-pair rotation with position = row index."""
    seq, _, dim = x.shape
    inv = theta ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
    ang = np.arange(seq, dtype=np.float64)[:, None] * inv[None, :]  # [seq][dim/2]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., 0::2], x[..., 1::2]
    y = np.empty_like(x)
    y[..., 0::2] = a * c - b * s
    y[..., 1::2] = a * s + b * c
    return y


def causal_attention(q, k, v, n_head_kv):
    seq, n_head, d = q.shape
    group = n_head // n_head_kv
    out = np.zeros_like(q)
    mask = np.tril(np.ones((seq, seq), dtype=bool))
    for h in range(n_head):
        kh = h // group
        scores = (q[:, h, :] @ k[:, kh, :].T) / np.sqrt(d)
        scores = np.where(mask, scores, -np.inf)
        scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        scores /= scores.sum(axis=-1, keepdims=True)
        out[:, h, :] = scores @ v[:, kh, :]
    return out


def forward(cfg, w, tokens):
    n_head, n_head_kv = cfg["n_head"], cfg["n_head_kv"]
    head_dim = cfg["n_embd"] // n_head
    seq = len(tokens)
    x = w["token_embd.weight"][tokens]
    for l in range(cfg["n_layer"]):
        p = f"blk.{l}."
        h = rmsnorm(x, w[p + "attn_norm.weight"], cfg["eps"])
        q = (h @ w[p + "attn_q.weight"].T).reshape(seq, n_head, head_dim)
        k = (h @ w[p + "attn_k.weight"].T).reshape(seq, n_head_kv, head_dim)
        v = (h @ w[p + "attn_v.weight"].T).reshape(seq, n_head_kv, head_dim)
        q, k = rope_normal(q, cfg["rope_theta"]), rope_normal(k, cfg["rope_theta"])
        a = causal_attention(q, k, v, n_head_kv).reshape(seq, n_head * head_dim)
        x = x + a @ w[p + "attn_output.weight"].T
        h = rmsnorm(x, w[p + "ffn_norm.weight"], cfg["eps"])
        gate, up = h @ w[p + "ffn_gate.weight"].T, h @ w[p + "ffn_up.weight"].T
        x = x + (gate / (1.0 + np.exp(-gate)) * up) @ w[p + "ffn_down.weight"].T
    x = rmsnorm(x, w["output_norm.weight"], cfg["eps"])
    output = w.get("output.weight", w["token_embd.weight"])
    return x @ output.T


def gen_synthetic() -> None:
    cfg, w = load_weights(HERE / "fixtures" / "gguf" / "synthetic-2l.gguf")
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, w["token_embd.weight"].shape[0], size=24).astype(np.int32)
    write(OUT / "synthetic-2l.tokens.axt", tokens, I32)
    write(OUT / "synthetic-2l.logits.axt", forward(cfg, w, tokens), F32)
    short = tokens[:1]
    write(OUT / "synthetic-2l.short.tokens.axt", short, I32)
    write(OUT / "synthetic-2l.short.logits.axt", forward(cfg, w, short), F32)
    print(f"synthetic-2l: {len(tokens)} tokens, vocab {w['token_embd.weight'].shape[0]}")


# --- SmolLM2 through Hugging Face --------------------------------------------------------


def load_hf(gguf_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gguf_path = gguf_path.resolve()
    model = AutoModelForCausalLM.from_pretrained(
        str(gguf_path.parent), gguf_file=gguf_path.name, torch_dtype=torch.float32
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(gguf_path.parent), gguf_file=gguf_path.name)
    return model, tok


def gen_smollm2(gguf_path: Path) -> None:
    import torch

    model, tok = load_hf(gguf_path)
    LOCAL.mkdir(exist_ok=True)
    (OUT / "smollm2.prompts.txt").write_text(
        "\n".join(p.replace("\n", "\\n") for p in SMOLLM2_PROMPTS) + "\n", encoding="utf-8"
    )
    greedy_lines = []
    with torch.no_grad():
        for i, prompt in enumerate(SMOLLM2_PROMPTS):
            ids = tok(prompt, return_tensors="pt").input_ids  # add_bos follows the metadata
            assert ids.shape[1] <= 32, (i, ids.shape)
            logits = model(ids).logits[0].numpy()  # [seq][vocab]
            write(OUT / f"smollm2.{i}.tokens.axt", ids[0].numpy().astype(np.int32), I32)
            write(OUT / f"smollm2.{i}.argmax.axt", logits.argmax(axis=-1).astype(np.int32), I32)
            write(OUT / f"smollm2.{i}.last_logits.axt", logits[-1], F32)
            write(LOCAL / f"smollm2.{i}.logits.axt", logits, F32)
            print(f"prompt {i}: {ids.shape[1]} tokens, argmax at end {int(logits[-1].argmax())}")
            if i < GREEDY_PROMPTS:
                out = model.generate(
                    ids, max_new_tokens=GREEDY_TOKENS, do_sample=False, min_new_tokens=GREEDY_TOKENS
                )
                cont = out[0, ids.shape[1] :].tolist()
                greedy_lines.append(" ".join(map(str, cont)))
                print("   greedy:", repr(tok.decode(cont)))
    (OUT / "smollm2.greedy.txt").write_text("\n".join(greedy_lines) + "\n", encoding="utf-8")


def dump_hf(gguf_path: Path, prompt: str, layer: int, out_dir: Path) -> None:
    """Writes layer `layer`'s activations with LlamaModel's dump names for compare_dumps.py."""
    import torch

    model, tok = load_hf(gguf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    n_head, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_head
    block = model.model.layers[layer]
    captured = {}

    def save(name):
        def hook(_module, _inputs, output):
            captured[name] = (output[0] if isinstance(output, tuple) else output)[0].detach()

        return hook

    def save_input(name):
        def hook(_module, inputs):
            captured[name] = inputs[0][0].detach()

        return hook

    hooks = [
        block.register_forward_pre_hook(save_input("x_in")),
        block.input_layernorm.register_forward_hook(save("attn_norm_out")),
        block.self_attn.q_proj.register_forward_hook(save("q_raw")),
        block.self_attn.k_proj.register_forward_hook(save("k_raw")),
        block.self_attn.v_proj.register_forward_hook(save("v")),
        block.self_attn.register_forward_hook(save("attn_proj_out")),
        block.self_attn.o_proj.register_forward_pre_hook(save_input("attn_out")),
        block.post_attention_layernorm.register_forward_pre_hook(save_input("x_mid")),
        block.post_attention_layernorm.register_forward_hook(save("ffn_norm_out")),
        block.mlp.register_forward_hook(save("ffn_out")),
        block.register_forward_hook(save("x_out")),
    ]
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0]
    for h in hooks:
        h.remove()
    seq = ids.shape[1]
    # HF keeps q/k in the rotate_half layout; interleave back to GGUF's adjacent-pair layout
    # and apply RoPE the same way the C++ dump does (after rotation).
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2).double() / head_dim))
    ang = torch.arange(seq).double()[:, None] * inv[None, :]
    cos, sin = torch.cos(ang).float(), torch.sin(ang).float()

    def rotate(raw, heads):
        t = raw.reshape(seq, heads, head_dim)
        a, b = t[..., : head_dim // 2], t[..., head_dim // 2 :]
        ra = a * cos[:, None, :] - b * sin[:, None, :]
        rb = a * sin[:, None, :] + b * cos[:, None, :]
        out = torch.empty_like(t)
        out[..., 0::2], out[..., 1::2] = ra, rb
        return out.reshape(seq, heads * head_dim)

    captured["q"] = rotate(captured.pop("q_raw"), n_head)
    captured["k"] = rotate(captured.pop("k_raw"), n_kv)
    captured.pop("attn_proj_out")
    captured["embed"] = model.model.embed_tokens(ids)[0].detach()
    captured["logits"] = logits
    for name, t in captured.items():
        write(out_dir / f"{name}.axt", t.numpy().astype(np.float32), F32)
    print(f"wrote {len(captured)} tensors for layer {layer} to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smollm2", type=Path, help="path to SmolLM2-135M-Instruct-f16.gguf")
    ap.add_argument("--dump-hf", type=int, metavar="LAYER")
    ap.add_argument("--prompt", default=SMOLLM2_PROMPTS[0])
    ap.add_argument("--out", type=Path, default=Path("dump-hf"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smollm2 is None:
        gen_synthetic()
    elif args.dump_hf is not None:
        dump_hf(args.smollm2, args.prompt, args.dump_hf, args.out)
    else:
        gen_smollm2(args.smollm2)


if __name__ == "__main__":
    main()
