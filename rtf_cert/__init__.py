"""
Lightweight certificate verification utilities for the Reversible Thought Foundry.

This module exposes the `rtf-cert` command-line interface. Use
`python -m rtf_cert verify <certificate>` or the convenience script in `scripts/`.
"""

POLICY_ACTIVATION_MIN = 0.94
POLICY_LOSS_MIN = 0.90

__all__ = ["POLICY_ACTIVATION_MIN", "POLICY_LOSS_MIN"]
