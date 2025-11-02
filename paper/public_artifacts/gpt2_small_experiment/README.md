# GPT-2 Small Multi-head Attention Validation

This directory contains the artifacts produced by the GPT-2 validation scripts (`scripts/run_gpt2_rtf.py` and `scripts/run_gpt2_rtf_suite.py`), which evaluate the RTF extractor on the first attention block and across multiple blocks of GPT-2 (small).

## Contents

- `metrics.json` – summary statistics (and per-prompt breakdowns) for the GPT-2 block 0 attention certification.
- `multi_block_metrics.json` – metrics for the multi-block, 32-prompt sweep (blocks 0, 1, 3, 5, 7, 9, 11).
- `verify_mask_replay.py` – sanity-check that the manual replay with the causal mask matches the module output to < 1e-6 max error.
- `run_gpt2_rtf_suite.py` – helper script used to generate `multi_block_metrics.json`.

## Reproduction

From the repository root with the virtual environment active:

```bash
pip install torch transformers
python scripts/run_gpt2_rtf.py
python scripts/run_gpt2_rtf_suite.py
```

To verify the causal mask replay manually:

```bash
python paper/public_artifacts/gpt2_small_experiment/verify_mask_replay.py
```

The scripts download GPT-2 (small) from Hugging Face, extract the requested attention blocks, run the RTF certification pipeline on curated prompts, and write the JSON metrics listed above.
