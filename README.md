# Reversible Thought Foundry – Reproduction Guide

This folder packages the public instructions for reproducing the results that accompany the “From Attention to Assurance: Reversible Thought Foundry” paper. It is intended to be dropped into a GitHub repository alongside the project code so reviewers, collaborators, and readers can quickly validate the pipeline.

## Repository Layout

```
├── paper/                   # LaTeX sources, figures, compiled PDF
│   └── public_artifacts/    # Metrics, scripts, certificates referenced in the paper
├── scripts/                 # Helper scripts (GPT-2 sweep, latency benchmark, etc.)
├── tests/                   # Deterministic theory + integration tests
├── requirements*.txt        # Python dependencies (CPU / GPU variants)
├── github_release/          # This folder
└── README.md                # Project overview (create in the root repository)
```

## Prerequisites

- macOS or Linux host (experiments were run on macOS M4 Pro, 24 GB RAM).
- Python 3.9.6 (exact version used; 3.10+ should work but is untested).
- `virtualenv` or `venv`.
- For LaTeX builds: TeX Live 2023+ with `latexmk`, `tikz`, `booktabs`, `tabularx`.
- Optional: GPU + CUDA for faster GPT-2 runs (pipeline works on CPU).

### Create Environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# If running GPU experiments, install torch/transformers wheels as needed.
```

## Minimal Sanity Checks

Run the unit / regression suite to validate the deterministic helpers and certificate path:

```bash
venv/bin/python -m pytest tests/test_rtf_theory_validation.py -q
venv/bin/python -m pytest tests/test_rtf_formal.py -q
```

These tests train lightweight attention/router/SSM modules and confirm that the extractor certifies real weights while rejecting fail-closed perturbations.

### One-command Demo

Run the quick sanity script (assumes dependencies installed in the current environment):

```bash
bash scripts/run_demo.sh
```

It prints the active Python, Torch, Transformers, and NumPy versions, runs the 16‑d minimal attention extraction, and verifies the published 64‑d multi-head certificate via the CLI described below. Override the interpreter with `PYTHON_BIN=/path/to/python`.

## Minimal Attention Example

Demonstrates the end-to-end pipeline on a 16‑dimensional attention head.

```bash
PYTHONPATH=. venv/bin/python scripts/run_minimal_attention.py
```

Artifacts will appear under `paper/public_artifacts/minimal_attention_example/`:

- `minimal_attention_weights.npz`
- `minimal_attention_probes.npz`
- `minimal_attention_certificate.json`

Verify the sample certificate (also ships in `public_artifacts/sample_certificates/`):

```bash
venv/bin/python paper/public_artifacts/minimal_verifier.py \
  --certificate paper/public_artifacts/sample_certificates/attention_certificate.json \
  --weights     paper/public_artifacts/sample_certificates/multihead_weights.npz \
  --probes      paper/public_artifacts/sample_certificates/multihead_probes.npz \
  --mask        paper/public_artifacts/sample_certificates/multihead_mask.npz \
  --bias        paper/public_artifacts/sample_certificates/multihead_bias.npz
```

The script recomputes SHA-256 digests, checks the structured \epsilon-bound, and confirms policy thresholds.

## GPT-2 Certification Sweep

The main paper reports two settings:

1. **Baseline:** Block 0 on two curated prompts (Table 1).
2. **Extended:** Blocks {0, 1, 3, 5, 7, 9, 11} on 32 prompts each (Table 2).

Commands:

```bash
# Single block baseline
PYTHONPATH=. venv/bin/python scripts/run_gpt2_rtf.py

# Full sweep (writes multi_block_metrics.json under public_artifacts)
PYTHONPATH=. venv/bin/python scripts/run_gpt2_rtf_suite.py

# Replay verification, ensures causal mask alignment and IR replay match
venv/bin/python paper/public_artifacts/gpt2_small_experiment/verify_mask_replay.py
```

The resulting metrics are staged in `paper/public_artifacts/gpt2_small_experiment/`:

- `metrics.json` (baseline)
- `multi_block_metrics.json` (32-prompt sweep)
- `verify_mask_replay.py` (sanity check for causal mask + softmax semantics)

## Certificate CLI (`rtf-cert`)

The repository ships a lightweight verifier exposed as both a module and convenience script:

```bash
# Python module entry point
python -m rtf_cert verify path/to/certificate.json \
  --weights path/to/weights.npz \
  --probes path/to/probes.npz \
  --mask path/to/mask.npz \
  --bias path/to/bias.npz

# Wrapper script (respects $PYTHON_BIN)
scripts/rtf-cert verify ...
```

It validates the JSON-LD schema, recomputes hashes (including the mask and additive bias tensors), checks identifiability thresholds (`margin_threshold`, `singular_gap`, `layer_norm_bounds`), confirms that κ-locality and numerical tolerance budgets are present, and enforces the activation/loss coverage policies + Dedukti exit code.

## Latency Benchmark

Figure 2 reports extraction latency vs. motif size. Regenerate using:

```bash
PYTHONPATH=. venv/bin/python scripts/benchmark_rtf_latency.py --repeats 5
venv/bin/python scripts/plot_latency_scale.py
```

This produces `latency_scale.pdf/png` under `paper/figures/` plus raw metrics in `outputs/latency/`.

## Safety Guard Demonstration

Pinsker-style fixed-reference gate from Section 6:

```bash
venv/bin/python paper/public_artifacts/safety_fixed_reference.py
```

Outputs a brief report showing \Delta Pr bounds vs. \epsilon.

## Build the Paper

```bash
cd paper
latexmk -pdf RTF.tex
```

Successful builds produce `RTF.pdf` (also copied to `arxiv/` for submission).

## Releasing on GitHub

1. Copy this `github_release/` directory into the root of your public repository.
2. Ensure `paper/public_artifacts/`, `scripts/`, `tests/`, and `requirements*.txt` are committed.
3. Add a root-level `README.md` describing the project and linking to this reproduction guide.
4. Optionally include a `CITATION.cff` and an MIT/Apache license file, depending on distribution preferences.

## Version Lock

Exact dependency pins used in the paper (Python 3.9.6, PyTorch 2.8.0, Transformers 4.57.1, NumPy 2.0.2) are recorded in `requirements-lock.txt`.

```bash
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt
```

The demo script prints the active versions to help reviewers confirm they match the locked environment.

## Support

For questions or reproducibility issues, contact **Sandro Andric**  
`sandro.andric@nyu.edu`
