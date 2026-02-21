#!/usr/bin/env python3
"""test_05_rigorous_matrix: Comprehensive modality × context × capture × layer matrix.

54 cases per model:
  - modality:  text | text_image | image_only
  - context:   short (~50 tok prompt) | medium (~300 tok) | long (~900 tok)
  - capture:   on (1) | off (0)
  - layers:    early(2) | mid(8) | late(15) | multi(2,8,15) | all  [only when capture=on]

Layer numbers target gemma-3-4b-it (18 layers, 0-17).
For models with different depths, late/multi may legitimately miss layers.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from common import (
    RequestCase, ensure_server_alive, run_case, resolve_model_name,
    print_json_line,
)

# ── Prompts ──────────────────────────────────────────────────────────────────

_SHORT = (
    "Describe the color blue in exactly one sentence."
)

_MEDIUM = (
    "You are analyzing a philosophical debate. "
    "Physicalism holds that consciousness arises entirely from physical brain processes. "
    "Dualism argues it requires something beyond the physical. "
    "David Chalmers's hard problem asks why subjective experience exists at all. "
    "Briefly summarize both positions and state which has stronger empirical support."
)

_LONG = (
    "You are an expert in machine learning and natural language processing. "
    "Transformers were introduced in 'Attention is All You Need' (Vaswani et al., 2017). "
    "The self-attention mechanism lets each token attend to every other token in the sequence, "
    "weighting by relevance rather than position. This replaced recurrent architectures, "
    "enabling full parallelism during training. "
    "Each Transformer layer has two sub-layers: multi-head self-attention and a "
    "position-wise feed-forward network. Residual connections and layer normalization "
    "surround each sub-layer. "
    "Multi-head attention splits queries, keys, and values into h heads, computes "
    "scaled dot-product attention independently, then concatenates: "
    "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. "
    "Positional encodings inject order information since the architecture is permutation-invariant. "
    "BERT uses only the encoder for masked language modeling. "
    "GPT uses only the decoder for autoregressive generation. "
    "T5 uses the full encoder-decoder for sequence-to-sequence tasks. "
    "Scaling laws show that model performance improves predictably with compute, data, and parameters. "
    "Emergent abilities appear at certain scale thresholds, including few-shot learning, "
    "chain-of-thought reasoning, and instruction following. "
    "Attention patterns have been extensively studied: syntactic heads track grammatical structure, "
    "coreference heads link pronouns to antecedents, positional heads implement relative offsets. "
    "Probing classifiers trained on attention activations reveal rich linguistic knowledge. "
    "Sparse attention variants like Longformer and BigBird extend context length by attending "
    "locally plus a few global tokens. "
    "Flash Attention optimizes the attention computation for GPU memory bandwidth by tiling "
    "and recomputing on-chip, avoiding materializing the full attention matrix. "
    "Grouped-query attention (GQA) and multi-query attention (MQA) reduce KV cache size "
    "by sharing key-value heads across query heads. "
    "Rotary position embeddings (RoPE) encode position via rotation of the query and key vectors, "
    "enabling length generalization beyond training context. "
    "Based on everything above, explain in three concise sentences why attention interpretability "
    "matters for AI safety research."
)

# ── Layer configurations ─────────────────────────────────────────────────────

_LAYER_CONFIGS: list[tuple[str, str, set[int] | None, bool]] = [
    # (label, layers_str, expected_set, require_all)
    ("early",       "2",      {2},       True),
    ("mid",         "8",      {8},       True),
    ("late",        "15",     {15},      True),
    ("multi",       "2,8,15", {2,8,15},  True),
    ("all",         "all",    None,      False),
]

# ── Build cases ──────────────────────────────────────────────────────────────

def _build_cases() -> list[RequestCase]:
    cases: list[RequestCase] = []

    contexts = [
        ("short",  _SHORT,  32),
        ("medium", _MEDIUM, 48),
        ("long",   _LONG,   48),
    ]
    modalities = ["text", "text_image", "image_only"]

    for mod in modalities:
        for ctx_name, prompt, max_tok in contexts:
            # capture=off — one case per modality×context
            cases.append(RequestCase(
                name=f"{mod}__{ctx_name}__off",
                mode=mod,
                capture=0,
                layers=None,
                prompt=prompt,
                max_tokens=max_tok,
                allow_no_kv=True,
                strict_capture_off_no_kv=True,
            ))
            # capture=on — one case per layer config
            for lbl, lstr, exp_set, req_all in _LAYER_CONFIGS:
                cases.append(RequestCase(
                    name=f"{mod}__{ctx_name}__on__{lbl}",
                    mode=mod,
                    capture=1,
                    layers=lstr,
                    prompt=prompt,
                    max_tokens=max_tok,
                    expected_layers=exp_set,
                    require_all_expected_layers=req_all,
                    min_layers_for_all=3,
                ))

    return cases


CASES = _build_cases()


# ── Runner ───────────────────────────────────────────────────────────────────

def run_suite(model: str | None = None) -> list[dict]:
    model = resolve_model_name(model)
    results: list[dict] = []
    for case in CASES:
        r = run_case(case, model=model)
        r["model"] = model
        print_json_line({"type": "case_result", **r})
        results.append(r)
    return results


def main() -> None:
    info = ensure_server_alive()
    if not info["ok"]:
        print_json_line({"type": "error", "message": "Server not available",
                         "detail": info.get("error")})
        sys.exit(2)

    model = resolve_model_name(os.environ.get("VLLM_TEST_MODEL"))
    results = run_suite(model)

    passed = sum(1 for r in results if r["ok"])
    failed = [r["name"] for r in results if not r["ok"]]

    print_json_line({
        "type": "summary",
        "status": "ok" if not failed else "failed",
        "suite": "rigorous_matrix",
        "model": model,
        "total": len(results),
        "passed": passed,
        "failed": len(failed),
        "failed_cases": failed,
    })
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
