---
name: Bug report
about: Something produced the wrong output, crashed, or refused a file it should load
title: ""
labels: bug
---

**What happened**

What you ran and what it printed, against what you expected.

**Version**

The output of `./build/release/apps/axon/axon --version`, which names the build, the compiler and the CPU features.

**Model**

The GGUF file, where it came from, and its SHA256 (`shasum -a 256 model.gguf`). If it is one of the pinned files, its name from `scripts/download_models.sh --list` is enough.

**Command**

```sh
# the exact command line
```

**Output**

```
# everything it printed, including the banner
```

**Checks**

- [ ] `scripts/check_all.sh` passes on this machine without models
- [ ] the same command with `--force-scalar` behaves the same way (or differently, which is itself the finding)
