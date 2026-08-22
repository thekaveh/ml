# 9.2 Atlas consumer findings

This log records integration findings at the boundary between ml-eng-lab and its pinned Atlas
submodule. Do not patch Atlas implementation files from this repository; report upstream defects
there and record the consumer impact here.

## 9.2.1 Native Ollama is mandatory

Status: enforced in `scripts/atlas-up.sh` and covered by lifecycle tests.

The local workstation's containerized Ollama path consumes excessive memory and is too slow for
this consumer. The committed contract therefore requires `ollama-localhost`, clears ambient source
overrides, verifies the loopback native daemon before startup, and rejects container/automatic
ComfyUI overrides. Keep this finding in place unless a reviewed Atlas change gives equivalent
native behavior with a different explicit source.

## 9.2.2 Atlas Jupyter runtime is distinct from local CI

Status: documented with live probe evidence on 2026-07-30.

Atlas JupyterHub is a distinct runtime and is not Issue #62 acceptance evidence. Issue #62
qualifies the repository Torch 2.11 CPU stack; Issue #65 completed the retained Atlas runtime
review, and Issue #66 separately requires full quantization execution in both environments. See
[dependency-contracts.md](dependency-contracts.md) and
[atlas-pin-bump-runbook.md](atlas-pin-bump-runbook.md) before changing either side of that boundary.

Issue #62 did not upgrade Atlas. The host-native Ollama boundary is unchanged, and no
containerized Ollama service is added.
There is no containerized Ollama source in the Issue #66 quantization workflow.
