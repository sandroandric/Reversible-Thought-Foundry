# Minimal Attention Example

This minimal reproducible example trains a 16-dimensional attention head on the delayed-copy objective and runs the RTF extraction + certificate pipeline end-to-end.

## Steps

1. Activate the virtual environment and install dependencies (`pip install -r requirements.txt`).
2. From the repository root, run the helper script:
   ```bash
   python scripts/run_minimal_attention.py
   ```
   The script emits:
   - trained weights (`minimal_attention_weights.npz`)
   - probe data (`minimal_attention_probes.npz`)
   - extraction certificate (`minimal_attention_certificate.json`)
3. Verify the certificate:
   ```bash
   python -m rtf.verify --certificate minimal_attention_certificate.json
   ```

The helper script is self-contained and uses the same modules as the main experiments but with reduced dimensions so reviewers can reproduce the core result in under 30 seconds.
