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

Atlas JupyterHub supplies a newer CPU Torch surface than the repository's local/CI Torch 2.4.1
contract. The package probe and active-import scan pass, but this does not make every notebook
fully validated. Quantization remains manual-only pending a targeted full notebook smoke. See
[dependency-contracts.md](dependency-contracts.md) and
[atlas-pin-bump-runbook.md](atlas-pin-bump-runbook.md) before changing either side of that boundary.
