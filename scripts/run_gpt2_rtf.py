#!/usr/bin/env python3
"""
Run RTF on a real GPT-2 (small) attention block and emit certificate metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer

np.seterr(divide="ignore", invalid="ignore", over="ignore", under="ignore")


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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concepts.C600_rtf.autonomous_lab.runtime.rtf.extract_v2 import extract_with_certification

PROMPTS = [
    "The reversible thought foundry extracts neural mechanisms.",
    "Safety teams review RTF certificates before deployment to production copilots.",
]


def main() -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
    model.eval()

    block = model.h[0]
    attn = block.attn
    attn_bias = attn.bias.detach().cpu().numpy().astype(bool)

    prompt = "The reversible thought foundry extracts neural mechanisms."  # simple prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = [h.detach() for h in outputs.hidden_states]  # tuple length 13 (embedding + 12 blocks)

        # hidden_states[0] is embeddings; hidden_states[1] is after block 0.
        block_input = hidden_states[0]
        ln_input = block.ln_1(block_input).detach()
        attn_output, _ = attn(ln_input)
        attn_output = attn_output.detach()

        ln_input_np = ln_input.squeeze(0).cpu().numpy().astype(np.float64)
        attn_output_np = attn_output.squeeze(0).cpu().numpy().astype(np.float64)

    hidden_size = model.config.hidden_size
    num_heads = attn.num_heads

    weight = attn.c_attn.weight.detach().cpu().numpy().astype(np.float64)
    bias = attn.c_attn.bias.detach().cpu().numpy().astype(np.float64)
    W_q = weight[:, :hidden_size].T
    W_k = weight[:, hidden_size:2*hidden_size].T
    W_v = weight[:, 2*hidden_size:].T
    b_q = bias[:hidden_size]
    b_k = bias[hidden_size:2*hidden_size]
    b_v = bias[2*hidden_size:]

    W_o = attn.c_proj.weight.detach().cpu().numpy().astype(np.float64).T
    b_o = attn.c_proj.bias.detach().cpu().numpy().astype(np.float64)

    weights = {
        "W_q": W_q,
        "W_k": W_k,
        "W_v": W_v,
        "W_o": W_o,
        "b_q": b_q,
        "b_k": b_k,
        "b_v": b_v,
        "b_o": b_o,
        "num_heads": num_heads,
        "causal": True,
        "margin_threshold": 1e-5,
        "attn_bias": attn_bias,
    }

    per_prompt_metrics = []
    errors = []
    coverages = []
    success_errors = []
    success_coverages = []
    all_conditions_met = None
    all_conditions_violated = None
    certified_flags = []

    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=False, return_dict=True)
            hidden_states = [h.detach() for h in outputs.hidden_states]

            block_input = hidden_states[0]
            ln_input = block.ln_1(block_input).detach()
            attn_output, _ = attn(ln_input)
            attn_output = attn_output.detach()

            ln_input_np = ln_input.squeeze(0).cpu().numpy().astype(np.float64)
            attn_output_np = attn_output.squeeze(0).cpu().numpy().astype(np.float64)

        ln_mean = float(ln_input_np.mean())
        ln_std = float(ln_input_np.std())

        q = ln_input_np @ W_q.T + b_q
        k = ln_input_np @ W_k.T + b_k
        v = ln_input_np @ W_v.T + b_v

        head_dim = hidden_size // num_heads

        def _reshape_heads(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(arr.shape[0], num_heads, head_dim).transpose(1, 0, 2)

        q_heads = _reshape_heads(q)
        k_heads = _reshape_heads(k)
        v_heads = _reshape_heads(v)

        q_norms = np.linalg.norm(q_heads, axis=-1)
        k_norms = np.linalg.norm(k_heads, axis=-1)
        v_norms = np.linalg.norm(v_heads, axis=-1)

        q_norm_stats = {
            "mean": float(q_norms.mean()),
            "std": float(q_norms.std()),
            "min": float(q_norms.min()),
            "max": float(q_norms.max()),
        }
        k_norm_stats = {
            "mean": float(k_norms.mean()),
            "std": float(k_norms.std()),
            "min": float(k_norms.min()),
            "max": float(k_norms.max()),
        }
        v_norm_stats = {
            "mean": float(v_norms.mean()),
            "std": float(v_norms.std()),
            "min": float(v_norms.min()),
            "max": float(v_norms.max()),
        }

        scale = 1.0 / np.sqrt(float(head_dim))
        scores = np.matmul(q_heads, np.transpose(k_heads, (0, 2, 1))) * scale
        seq_len = scores.shape[-1]
        mask_slice = attn_bias[:, :, :seq_len, :seq_len][0, 0]
        mask_value = np.finfo(scores.dtype).min
        scores = np.where(mask_slice, scores, mask_value)
        scores -= scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        attn_weights_np = exp_scores / np.clip(exp_scores.sum(axis=-1, keepdims=True), 1e-9, None)

        min_margin, mean_margin = _attention_margins(attn_weights_np)

        head_outputs = np.matmul(attn_weights_np, v_heads)
        combined = np.transpose(head_outputs, (1, 0, 2)).reshape(len(ln_input_np), hidden_size)
        ir_output_manual = combined @ W_o.T + b_o
        diff = attn_output_np - ir_output_manual
        abs_diff = np.abs(diff)
        replay_l1 = float(abs_diff.mean())
        replay_l2 = float(np.sqrt((diff ** 2).mean()))
        replay_max = float(abs_diff.max())

        q_heads_unit = q_heads / np.clip(np.linalg.norm(q_heads, axis=-1, keepdims=True), 1e-9, None)
        k_heads_unit = k_heads / np.clip(np.linalg.norm(k_heads, axis=-1, keepdims=True), 1e-9, None)
        scores_unit = np.matmul(q_heads_unit, np.transpose(k_heads_unit, (0, 2, 1)))
        scores_unit = np.where(mask_slice, scores_unit, mask_value)
        scores_unit -= scores_unit.max(axis=-1, keepdims=True)
        exp_scores_unit = np.exp(scores_unit)
        attn_weights_unit = exp_scores_unit / np.clip(exp_scores_unit.sum(axis=-1, keepdims=True), 1e-9, None)
        head_outputs_unit = np.matmul(attn_weights_unit, v_heads)
        combined_unit = np.transpose(head_outputs_unit, (1, 0, 2)).reshape(len(ln_input_np), hidden_size)
        ir_output_unit = combined_unit @ W_o.T + b_o
        diff_unit = attn_output_np - ir_output_unit
        unit_l1 = float(np.abs(diff_unit).mean())
        unit_l2 = float(np.sqrt((diff_unit ** 2).mean()))
        unit_max = float(np.abs(diff_unit).max())

        result = extract_with_certification("multihead_attention", weights, ln_input_np, attn_output_np)

        metrics = {
            "prompt": prompt,
            "certified": bool(result.certified),
            "extraction_error": float(result.extraction_error),
            "activation_coverage": float(result.coverage.activation_coverage),
            "conditions_met": result.identifiability.conditions_met,
            "conditions_violated": result.identifiability.conditions_violated,
            "ln_input_mean": ln_mean,
            "ln_input_std": ln_std,
            "q_norm_stats": q_norm_stats,
            "k_norm_stats": k_norm_stats,
            "v_norm_stats": v_norm_stats,
            "attention_margin_min": min_margin,
            "attention_margin_mean": mean_margin,
            "replay_l1_mean": replay_l1,
            "replay_l2_mean": replay_l2,
            "replay_abs_max": replay_max,
            "unit_norm_replay_l1_mean": unit_l1,
            "unit_norm_replay_l2_mean": unit_l2,
            "unit_norm_replay_abs_max": unit_max,
        }

        per_prompt_metrics.append(metrics)
        error_value = metrics["extraction_error"]
        coverage_value = metrics["activation_coverage"]

        errors.append(error_value)
        coverages.append(coverage_value)
        if metrics["certified"] and np.isfinite(error_value):
            success_errors.append(error_value)
            success_coverages.append(coverage_value)

        certified_flags.append(metrics["certified"])

        cond_met_set = set(metrics["conditions_met"])
        cond_violated_set = set(metrics["conditions_violated"])
        all_conditions_met = cond_met_set if all_conditions_met is None else all_conditions_met & cond_met_set
        all_conditions_violated = cond_violated_set if all_conditions_violated is None else all_conditions_violated | cond_violated_set

    def _stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"mean": None, "std": None}
        mean_val = float(sum(values) / len(values))
        std_val = float((sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5)
        return {"mean": mean_val, "std": std_val}

    overall_error_stats = _stats([v for v in errors if np.isfinite(v)])
    overall_cov_stats = _stats(coverages)
    success_error_stats = _stats(success_errors)
    success_cov_stats = _stats(success_coverages)

    summary = {
        "prompts_evaluated": len(PROMPTS),
        "certified_all": all(certified_flags),
        "certified_any": any(certified_flags),
        "certified_prompts": len(success_errors),
        "failed_prompts": len(PROMPTS) - len(success_errors),
        "extraction_error_stats_certified": success_error_stats,
        "activation_coverage_stats_certified": success_cov_stats,
        "extraction_error_stats_overall": overall_error_stats,
        "activation_coverage_stats_overall": overall_cov_stats,
        "conditions_met_all": sorted(all_conditions_met) if all_conditions_met is not None else [],
        "conditions_violated_union": sorted(all_conditions_violated) if all_conditions_violated is not None else [],
        "errors": errors,
        "activation_coverages": coverages,
    }

    payload = {
        "summary": summary,
        "per_prompt": per_prompt_metrics,
    }

    print(json.dumps(payload, indent=2))

    out_path = Path(__file__).resolve().parent / "gpt2_rtf_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Metrics saved to", out_path)


if __name__ == "__main__":
    main()
