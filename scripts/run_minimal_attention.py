#!/usr/bin/env python3
"""
Train and certify a 16-dimensional attention head using the RTF pipeline.

Outputs:
  - minimal_attention_weights.npz
  - minimal_attention_probes.npz
  - minimal_attention_outputs.npz
  - minimal_attention_certificate.json
"""

import json
from pathlib import Path

import numpy as np

from tests.helpers.real_mechanism import get_trained_attention_snapshot
from concepts.C600_rtf.autonomous_lab.runtime.rtf.extract_v2 import extract_with_certification
from concepts.C600_rtf.autonomous_lab.runtime.rtf.certificates import create_extract_certificate


def main() -> None:
    weights, inputs, outputs = get_trained_attention_snapshot()

    out_dir = Path("./paper/public_artifacts/minimal_attention_example")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "minimal_attention_weights.npz", **weights)
    np.savez(out_dir / "minimal_attention_probes.npz", inputs=inputs)
    np.savez(out_dir / "minimal_attention_outputs.npz", outputs=outputs)

    extraction_result = extract_with_certification("attention", weights, inputs, outputs)
    certificate = create_extract_certificate(extraction_result, weights, inputs)

    cert_path = out_dir / "minimal_attention_certificate.json"
    with cert_path.open("w", encoding="utf-8") as f:
        json.dump(certificate.to_dict(), f, indent=2)

    print("Certificate written to", cert_path)
    print("Extraction error:", extraction_result.extraction_error)
    print("Activation coverage:", extraction_result.coverage.activation_coverage)


if __name__ == "__main__":
    main()
