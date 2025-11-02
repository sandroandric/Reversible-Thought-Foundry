# Sample Certificates

This directory includes representative JSON-LD certificates produced by the Reversible Thought Foundry toolchain. Each file can be verified with:

```
python -m rtf.verify --certificate <certificate.json>
```

Hash fields point to the corresponding weight tensors, probe traces, causal mask, and additive bias tensors that ship alongside the certificates so auditors can recompute the recorded digests.
