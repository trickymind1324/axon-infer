#!/usr/bin/env python3
"""Writes a tokenizer-only GGUF (all tokenizer.* metadata, no tensors) from a model file, so
tokenizer tests can run without downloading weights.

  .venv/bin/python tests/reference/make_tokenizer_gguf.py models/SmolLM2-135M-Instruct-Q4_0.gguf smollm2
"""

import sys
from pathlib import Path

from gguf import GGUFReader, GGUFValueType, GGUFWriter

OUT = Path(__file__).parent / "fixtures" / "tokenizer"
KEEP_PREFIXES = ("tokenizer.",)
KEEP_EXACT = {"general.name"}


def field_value(field):
    vtype = field.types[0]
    if vtype == GGUFValueType.ARRAY:
        etype = field.types[1]
        if etype == GGUFValueType.STRING:
            return [bytes(field.parts[i]).decode("utf-8") for i in field.data], etype
        return [field.parts[i].tolist()[0] for i in field.data], etype
    if vtype == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8"), vtype
    return field.parts[field.data[0]].tolist()[0], vtype


def main(src: str, name: str) -> None:
    reader = GGUFReader(src)
    arch = field_value(reader.fields["general.architecture"])[0]
    w = GGUFWriter(OUT / f"{name}.tokenizer.gguf", arch)
    kept = 0
    for key, field in reader.fields.items():
        if not (key.startswith(KEEP_PREFIXES) or key in KEEP_EXACT):
            continue
        value, vtype = field_value(field)
        if vtype == GGUFValueType.ARRAY:
            w.add_array(key, value)
        else:
            w.add_key_value(key, value, field.types[0])
        kept += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = OUT / f"{name}.tokenizer.gguf"
    print(f"{out} ({kept} keys, {out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
