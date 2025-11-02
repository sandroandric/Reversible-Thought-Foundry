#!/usr/bin/env python3
"""
Verify that the manual GPT-2 replay (with the causal mask applied) matches the
module output to numerical precision. This complements the certificate metrics.
"""

import numpy as np
import torch
from transformers import GPT2Model, GPT2Tokenizer


def apply_manual_attention(block, ln_input):
    attn = block.attn
    hidden_size = block.attn.c_attn.weight.shape[1] // 3
    num_heads = attn.num_heads
    head_dim = hidden_size // num_heads

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

    ln_input_np = ln_input.squeeze(0).cpu().numpy().astype(np.float64)

    q = ln_input_np @ W_q.T + b_q
    k = ln_input_np @ W_k.T + b_k
    v = ln_input_np @ W_v.T + b_v

    q = q.reshape(len(ln_input_np), num_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(len(ln_input_np), num_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(len(ln_input_np), num_heads, head_dim).transpose(1, 0, 2)

    scale = 1.0 / np.sqrt(float(head_dim))
    scores = np.matmul(q, np.transpose(k, (0, 2, 1))) * scale

    mask = attn.bias.detach().cpu().numpy().astype(bool)
    seq_len = scores.shape[-1]
    causal = mask[:, :, :seq_len, :seq_len][0, 0]
    mask_value = np.finfo(scores.dtype).min
    scores = np.where(causal[None, :, :], scores, mask_value)

    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-9, None)

    heads = np.matmul(probs, v)
    combined = heads.transpose(1, 0, 2).reshape(len(ln_input_np), hidden_size)
    out = combined @ W_o.T + b_o
    return out


def main() -> None:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True, local_files_only=True)
    model.eval()

    prompt = "The reversible thought foundry extracts neural mechanisms."
    with torch.no_grad():
        encoded = tokenizer(prompt, return_tensors="pt")
        outputs = model(**encoded, output_attentions=False, return_dict=True)
        hidden_states = [h.detach() for h in outputs.hidden_states]
        block = model.h[0]
        ln_input = block.ln_1(hidden_states[0]).detach()
        attn_out, _ = block.attn(ln_input)

    manual = apply_manual_attention(block, ln_input)
    attn_np = attn_out.squeeze(0).cpu().numpy().astype(np.float64)

    max_err = float(np.max(np.abs(attn_np - manual)))
    mean_err = float(np.mean(np.abs(attn_np - manual)))

    print(f"Mean absolute error: {mean_err:.3e}")
    print(f"Max absolute error:  {max_err:.3e}")
    assert max_err < 1e-6


if __name__ == "__main__":
    main()
