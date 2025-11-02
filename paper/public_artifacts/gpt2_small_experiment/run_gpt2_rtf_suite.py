#!/usr/bin/env python3
"""
Run the RTF extractor across multiple GPT-2 blocks and a richer prompt set.

Outputs a per-block/per-prompt metrics JSON under outputs/gpt2_rtf_multi_block_metrics.json.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concepts.C600_rtf.autonomous_lab.runtime.rtf.extract_v2 import extract_with_certification


PROMPTS: List[str] = [
    "The reversible thought foundry extracts neural mechanisms.",
    "Safety teams review RTF certificates before deployment to production copilots.",
    "On azure mornings the model traces attention circuits for audit readiness.",
    "Mechanistic interpretability requires clean probes and trusted witnesses.",
    "Certified patches must survive shadow traffic and downstream regression suites.",
    "Optimization passes fuse redundant map operations to reduce FLOPs.",
    "We benchmarked twelve attention heads across transformer layers.",
    "Robust governance insists on cryptographic hashes for every rollout artifact.",
    "Canary deployments captured no regressions after introducing the causal mask.",
    "A GPT-2 block localizes factual associations when the margin holds steady.",
    "Safety reviewers inspected the witness program before approving shipment.",
    "Fail-closed behavior prevents uncertified edits from reaching production.",
    "Layer normalization keeps norm regimes stable even under probe perturbations.",
    "Public artifacts document seeds, prompts, and epsilon tolerances explicitly.",
    "The hot linker enforces locality constraints during synthesized patch rollout.",
    "Latency measurements stayed below fifty milliseconds for the largest head.",
    "A top-two causal margin criterion works better than diagonal dominance here.",
    "Probe coverage exceeded 99.9 percent on curated validation sets.",
    "Audit logs contain links to the JSON-LD certificates for replay.",
    "Gradient heuristics cannot guarantee unique edits without identifiability.",
    "Formal semantics compose epsilon bounds via Lipschitz continuity.",
    "Router witnesses encode simplex preservation and margin gaps.",
    "State-space modules keep poles separated to ensure stability.",
    "We serialize CodeSpec programs with explicit resource accounting.",
    "Shadow traffic confirmed no drift relative to the sentinel policy.",
    "Prompt diversity helps expose near-tie margins in GPT-2 heads.",
    "Certification covers activation, path, and loss-focused metrics.",
    "A replay script verifies the masked softmax reproduces the block output.",
    "The governance dashboard highlights unresolved coverage obligations.",
    "KL/TV guards remain reference anchored to prevent cumulative drift.",
    "Mixture-of-experts routing will need streamed probes in future work.",
    "We recorded benchmark statistics for Figure 2's latency plot.",
]


BLOCK_INDICES: List[int] = [0, 1, 3, 5, 7, 9, 11]


def _attention_margins(attn_weights: np.ndarray) -> Tuple[float, float]:
    """Causal top1-top2 margin statistics for attention weights."""
    num_heads, seq_len, _ = attn_weights.shape
    margins: List[float] = []
    for h in range(num_heads):
        head_weights = attn_weights[h]
        for t in range(seq_len):
            valid = head_weights[t, : t + 1]
            if valid.size < 2:
                continue
            top_indices = np.argsort(valid)
            top = valid[top_indices[-1]]
            runner = valid[top_indices[-2]]
            margins.append(float(top - runner))
    if not margins:
        return float("nan"), float("nan")
    margins_arr = np.array(margins, dtype=np.float64)
    return float(np.min(margins_arr)), float(np.mean(margins_arr))


def _prepare_block(model: GPT2Model, block_idx: int, inputs_np: np.ndarray) -> Dict[str, np.ndarray]:
    block = model.h[block_idx]
    attn = block.attn

    hidden_size = model.config.hidden_size
    weight = attn.c_attn.weight.detach().cpu().numpy().astype(np.float64)
    bias = attn.c_attn.bias.detach().cpu().numpy().astype(np.float64)
    W_q = weight[:, :hidden_size].T
    W_k = weight[:, hidden_size:2 * hidden_size].T
    W_v = weight[:, 2 * hidden_size :].T
    b_q = bias[:hidden_size]
    b_k = bias[hidden_size:2 * hidden_size]
    b_v = bias[2 * hidden_size :]

    W_o = attn.c_proj.weight.detach().cpu().numpy().astype(np.float64).T
    b_o = attn.c_proj.bias.detach().cpu().numpy().astype(np.float64)
    attn_bias = attn.bias.detach().cpu().numpy().astype(bool)

    return {
        "W_q": W_q,
        "W_k": W_k,
        "W_v": W_v,
        "W_o": W_o,
        "b_q": b_q,
        "b_k": b_k,
        "b_v": b_v,
        "b_o": b_o,
        "num_heads": int(attn.num_heads),
        "causal": True,
        "attn_bias": attn_bias,
        "margin_threshold": 1e-5,
    }


def _manual_replay(weights: Dict[str, np.ndarray], ln_input_np: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    hidden_size = weights["W_q"].shape[0]
    num_heads = weights["num_heads"]
    head_dim = hidden_size // num_heads

    q = ln_input_np @ weights["W_q"].T + weights["b_q"]
    k = ln_input_np @ weights["W_k"].T + weights["b_k"]
    v = ln_input_np @ weights["W_v"].T + weights["b_v"]

    q_heads = q.reshape(len(ln_input_np), num_heads, head_dim).transpose(1, 0, 2)
    k_heads = k.reshape(len(ln_input_np), num_heads, head_dim).transpose(1, 0, 2)
    v_heads = v.reshape(len(ln_input_np), num_heads, head_dim).transpose(1, 0, 2)

    scale = 1.0 / np.sqrt(float(head_dim))
    scores = np.matmul(q_heads, np.transpose(k_heads, (0, 2, 1))) * scale
    mask = weights["attn_bias"]
    seq_len = scores.shape[-1]
    causal = mask[:, :, :seq_len, :seq_len][0, 0]
    mask_value = np.finfo(scores.dtype).min
    scores = np.where(causal[None, :, :], scores, mask_value)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-9, None)

    min_margin, mean_margin = _attention_margins(probs)

    head_outputs = np.matmul(probs, v_heads)
    combined = head_outputs.transpose(1, 0, 2).reshape(len(ln_input_np), hidden_size)
    out = combined @ weights["W_o"].T + weights["b_o"]
    return out, {"attention_margin_min": min_margin, "attention_margin_mean": mean_margin}


def main() -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True, local_files_only=True)
    model.eval()

    results = {
        "prompts": PROMPTS,
        "block_indices": BLOCK_INDICES,
        "per_block": {},
    }

    for block_idx in BLOCK_INDICES:
        block_metrics: List[Dict[str, float]] = []
        certified_flags: List[bool] = []
        errors: List[float] = []
        coverages: List[float] = []

        block = model.h[block_idx]

        for prompt in PROMPTS:
            encoded = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**encoded, output_attentions=False, return_dict=True)
                hidden_states = [h.detach() for h in outputs.hidden_states]
                block_input = hidden_states[block_idx]
                ln_input = block.ln_1(block_input).detach()
                attn_output, _ = block.attn(ln_input)

            ln_input_np = ln_input.squeeze(0).cpu().numpy().astype(np.float64)
            attn_output_np = attn_output.squeeze(0).cpu().numpy().astype(np.float64)

            weights = _prepare_block(model, block_idx, ln_input_np)
            manual_output, margin_stats = _manual_replay(weights, ln_input_np)

            diff = attn_output_np - manual_output
            replay_l1 = float(np.abs(diff).mean())
            replay_l2 = float(np.sqrt((diff ** 2).mean()))
            replay_max = float(np.abs(diff).max())

            extract_result = extract_with_certification(
                "multihead_attention",
                weights,
                ln_input_np,
                attn_output_np,
            )

            metric_entry = {
                "prompt": prompt,
                "certified": bool(extract_result.certified),
                "extraction_error": float(extract_result.extraction_error),
                "activation_coverage": float(extract_result.coverage.activation_coverage),
                "replay_l1_mean": replay_l1,
                "replay_l2_mean": replay_l2,
                "replay_abs_max": replay_max,
                "conditions_met": extract_result.identifiability.conditions_met,
                "conditions_violated": extract_result.identifiability.conditions_violated,
            }
            metric_entry.update(margin_stats)

            block_metrics.append(metric_entry)
            certified_flags.append(extract_result.certified)
            errors.append(metric_entry["extraction_error"])
            coverages.append(metric_entry["activation_coverage"])

        summary = {
            "prompts_evaluated": len(PROMPTS),
            "certified_all": bool(all(certified_flags)),
            "certified_fraction": float(sum(certified_flags) / len(certified_flags)),
            "extraction_error_mean": float(np.mean(errors)),
            "extraction_error_std": float(np.std(errors)),
            "activation_coverage_mean": float(np.mean(coverages)),
            "activation_coverage_std": float(np.std(coverages)),
        }

        results["per_block"][f"block_{block_idx}"] = {
            "summary": summary,
            "per_prompt": block_metrics,
        }

    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gpt2_rtf_multi_block_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics written to {out_path}")


if __name__ == "__main__":
    main()
