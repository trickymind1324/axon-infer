---
name: Number report
about: A benchmark or accuracy result from another machine, to sit next to the tables
title: "numbers: <machine>, <model>"
labels: numbers
---

**Machine**

CPU, core count, memory, operating system and compiler. `axon --version` prints the compiler and the CPU features it found.

**What ran**

```sh
# for example
STEP=my-machine HOST=box PEAK_GB_S=<peak memory bandwidth> bench/bench_e2e.sh models/<file>.gguf 1 4 8
```

**Records**

Attach the JSON files the script wrote under `bench/results/`. They carry the host, the model's SHA256, the kernel table and the load average, and the tables are rendered from them, so a record says more than a number pasted by hand.

**Accuracy, if that is the report**

The command and its output: `axon-perplexity` over the committed text, or the integration suite with `AXON_MODELS_DIR` set.

**Anything odd**

Thermal state, other load, a non-default flag, or a result that moved between runs.
