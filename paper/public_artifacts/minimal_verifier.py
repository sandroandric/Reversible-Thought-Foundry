#!/usr/bin/env python3
"""
Minimal certificate verifier used in the supplementary materials.

This script loads an extraction certificate, recomputes the referenced weight
and probe hashes, and checks the epsilon/coverage tolerances recorded in the
artifact. It is intentionally small so auditors can trust the verification
path without depending on the full RTF codebase.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CERT = BASE_DIR / "sample_certificates" / "attention_certificate.json"
DEFAULT_WEIGHTS = BASE_DIR / "sample_certificates" / "multihead_weights.npz"
DEFAULT_PROBES = BASE_DIR / "sample_certificates" / "multihead_probes.npz"
DEFAULT_MASK = BASE_DIR / "sample_certificates" / "multihead_mask.npz"
DEFAULT_BIAS = BASE_DIR / "sample_certificates" / "multihead_bias.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RTF certificate verifier.")
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERT, help="Path to the JSON-LD certificate.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Associated weight snapshot.")
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES, help="Associated probe snapshot.")
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK, help="Associated attention mask snapshot (if required).")
    parser.add_argument("--bias", type=Path, default=DEFAULT_BIAS, help="Associated attention bias snapshot (if required).")
    parser.add_argument("--epsilon", type=float, default=5.5e-5, help="Maximum allowed total epsilon bound.")
    parser.add_argument("--activation-coverage", type=float, default=0.94, help="Minimum activation coverage.")
    args = parser.parse_args()

    cert = json.loads(args.certificate.read_text())

    # Check hashes
    weight_hash = sha256_file(args.weights)
    probe_hash = sha256_file(args.probes)
    hash_fields = cert.get("hashes") or cert.get("content_hashes")
    if hash_fields is None:
        raise KeyError("Certificate missing hashes or content_hashes field")
    assert hash_fields["weights"] == weight_hash, "Weight hash mismatch"
    assert hash_fields["probes"] == probe_hash, "Probe hash mismatch"

    mask_hash_cert = hash_fields.get("mask")
    if mask_hash_cert is not None:
        mask_hash = sha256_file(args.mask)
        assert mask_hash_cert == mask_hash, "Mask hash mismatch"

    bias_hash_cert = hash_fields.get("bias")
    if bias_hash_cert is not None:
        bias_hash = sha256_file(args.bias)
        assert bias_hash_cert == bias_hash, "Bias hash mismatch"

    # Check epsilon and coverage tolerances
    eps_field = cert.get("epsilon_bound", {})
    total_eps = eps_field.get("total")
    if total_eps is None:
        total_eps = (
            eps_field.get("extract", 0.0)
            + eps_field.get("quant", 0.0)
            + eps_field.get("interpret", 0.0)
        )
    assert total_eps <= args.epsilon, f"Total epsilon {total_eps} exceeds bound {args.epsilon}"

    activation_cov = cert.get("coverage", {}).get("activation_coverage")
    if activation_cov is None:
        raise KeyError("Certificate missing activation_coverage metric")
    assert activation_cov >= args.activation_coverage, "Activation coverage below policy threshold"

    print("Certificate verification passed.")
    print(f"  total epsilon: {total_eps:.3e}")
    print(f"  activation coverage: {activation_cov:.3f}")


if __name__ == "__main__":
    main()
