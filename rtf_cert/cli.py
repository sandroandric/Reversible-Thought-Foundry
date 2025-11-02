from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from . import POLICY_ACTIVATION_MIN, POLICY_LOSS_MIN


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def positive(message: str) -> None:
    print(f"[ok] {message}")


def warn(message: str) -> None:
    print(f"[warn] {message}")


def err(message: str) -> None:
    print(f"[fail] {message}")


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)
        err(message)
    else:
        positive(message)


def get_hashes(data: Dict[str, Any]) -> Dict[str, str]:
    if "hashes" in data and isinstance(data["hashes"], dict):
        return data["hashes"]
    if "content_hashes" in data and isinstance(data["content_hashes"], dict):
        return data["content_hashes"]
    return {}


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse certificate JSON: {exc}") from exc


def _match_hash(
    label: str,
    expected: Optional[str],
    provided: Optional[Path],
    errors: list[str],
) -> None:
    if expected is None:
        warn(f"No {label} hash recorded in certificate")
        return
    positive(f"Certificate records {label} hash ({expected})")
    if provided is not None:
        actual = sha256_file(provided)
        require(
            actual == expected,
            f"{label} hash matches ({actual})",
            errors,
        )
    else:
        warn(f"Skipped {label} hash verification (path not provided)")


def _ensure_fields(
    data: Dict[str, Any],
    field_path: Iterable[str],
    errors: list[str],
    description: str,
) -> Optional[Any]:
    cursor: Any = data
    for field in field_path:
        if not isinstance(cursor, dict) or field not in cursor:
            errors.append(f"Certificate missing {description}")
            err(f"Certificate missing {description}")
            return None
        cursor = cursor[field]
    positive(f"Found {description}")
    return cursor


def verify_certificate(args: argparse.Namespace) -> int:
    certificate_path = Path(args.certificate)
    data = load_json(certificate_path)
    errors: list[str] = []

    require(data.get("certificate_type") == "extraction", "Certificate type is extraction", errors)
    require("mechanism" in data, "Mechanism recorded", errors)
    hashes = get_hashes(data)
    require(bool(hashes), "Hash section present", errors)

    _match_hash("weights", hashes.get("weights"), args.weights, errors)
    _match_hash("probes", hashes.get("probes"), args.probes, errors)
    _match_hash("mask", hashes.get("mask"), args.mask, errors)
    _match_hash("bias", hashes.get("bias"), args.bias, errors)

    epsilon = _ensure_fields(data, ["epsilon_bound", "total"], errors, "total epsilon bound")
    coverage = _ensure_fields(data, ["coverage", "activation_coverage"], errors, "activation coverage")
    loss_cov = _ensure_fields(data, ["coverage", "loss_coverage"], errors, "loss coverage")

    ident = _ensure_fields(data, ["identifiability"], errors, "identifiability section")
    if isinstance(ident, dict):
        require("margin_threshold" in ident, "Causal margin threshold recorded", errors)
        require("singular_gap" in ident, "Singular-value gap recorded", errors)
        require("layer_norm_bounds" in ident, "Layer-norm bounds recorded", errors)

    _ensure_fields(data, ["metadata", "kappa_locality"], errors, "κ-locality bound")
    _ensure_fields(data, ["metadata", "dkcheck_exit_code"], errors, "Dedukti exit code")
    _ensure_fields(data, ["tolerance_budget_fp32"], errors, "fp32 tolerance budget")
    _ensure_fields(data, ["tolerance_budget_fp64"], errors, "fp64 tolerance budget")

    if epsilon is not None:
        print(f"[info] total ε = {epsilon:.3e}")
    if isinstance(coverage, (int, float)):
        require(
            coverage >= args.activation_threshold,
            f"Activation coverage >= {args.activation_threshold}",
            errors,
        )
    if isinstance(loss_cov, (int, float)):
        require(
            loss_cov >= args.loss_threshold,
            f"Loss coverage >= {args.loss_threshold}",
            errors,
        )

    dk_exit = data.get("metadata", {}).get("dkcheck_exit_code")
    if dk_exit is not None:
        require(dk_exit == 0, "Dedukti exit code is zero", errors)

    if errors:
        err(f"Verification finished with {len(errors)} error(s).")
        return 1

    positive("Verification succeeded.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtf-cert", description="RTF certificate utilities")
    subparsers = parser.add_subparsers(dest="command")

    verify = subparsers.add_parser("verify", help="Verify an extraction certificate")
    verify.add_argument("certificate", type=Path, help="Path to certificate JSON/JSON-LD file")
    verify.add_argument("--weights", type=Path, help="Associated weights snapshot for hash verification")
    verify.add_argument("--probes", type=Path, help="Associated probe snapshot for hash verification")
    verify.add_argument("--mask", type=Path, help="Attention mask tensor for hash verification")
    verify.add_argument("--bias", type=Path, help="Attention bias tensor for hash verification")
    verify.add_argument(
        "--activation-threshold",
        type=float,
        default=POLICY_ACTIVATION_MIN,
        help=f"Minimum activation coverage (default {POLICY_ACTIVATION_MIN})",
    )
    verify.add_argument(
        "--loss-threshold",
        type=float,
        default=POLICY_LOSS_MIN,
        help=f"Minimum loss coverage (default {POLICY_LOSS_MIN})",
    )
    verify.set_defaults(func=verify_certificate)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)
