#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== RTF Demo Environment =="
"${PYTHON_BIN}" - <<'PY'
import platform, sys
try:
    import torch
except ImportError:
    torch = None
try:
    import transformers
except ImportError:
    transformers = None
try:
    import numpy
except ImportError:
    numpy = None

print(f"Python: {platform.python_version()} ({sys.executable})")
print(f"Platform: {platform.platform()}")
print(f"Torch: {getattr(torch, '__version__', 'not installed')}")
print(f"Transformers: {getattr(transformers, '__version__', 'not installed')}")
print(f"NumPy: {getattr(numpy, '__version__', 'not installed')}")
PY

echo
echo "== Step 1: Minimal attention extraction (16-dim head) =="
PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_minimal_attention.py"

echo
echo "== Step 2: Verify sample multi-head certificate =="
"${PYTHON_BIN}" -m rtf_cert verify \
  "${PROJECT_ROOT}/paper/public_artifacts/sample_certificates/attention_certificate.json" \
  --weights "${PROJECT_ROOT}/paper/public_artifacts/sample_certificates/multihead_weights.npz" \
  --probes "${PROJECT_ROOT}/paper/public_artifacts/sample_certificates/multihead_probes.npz" \
  --mask "${PROJECT_ROOT}/paper/public_artifacts/sample_certificates/multihead_mask.npz" \
  --bias "${PROJECT_ROOT}/paper/public_artifacts/sample_certificates/multihead_bias.npz"

echo
echo "Demo complete."
