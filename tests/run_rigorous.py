#!/usr/bin/env python3
"""run_rigorous.py: Run the rigorous matrix test against a running server
and print a formatted performance table.

Usage:
    python run_rigorous.py [--markdown] [--out results.json]
    python run_rigorous.py --markdown --baseline-url http://127.0.0.1:8001/v1

Environment:
    VLLM_TEST_MODEL   model name served by the running server (default: gemma-3-4b-it)
    VLLM_API_BASE     server base URL (default: http://127.0.0.1:8000/v1)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PYTHON = sys.executable
THIS_DIR = Path(__file__).parent
TEST_SCRIPT = THIS_DIR / "test_05_rigorous_matrix.py"


# ── Run test subprocess ───────────────────────────────────────────────────────

def run_test(env: dict | None = None) -> tuple[list[dict], dict | None]:
    """Run test_05 and return (case_results, summary)."""
    merged_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(
        [PYTHON, str(TEST_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,       # forward test stderr so errors are visible
        env=merged_env,
        text=True,
    )
    cases: list[dict] = []
    summary: dict | None = None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t == "case_result":
            cases.append(obj)
        elif t == "summary":
            summary = obj
    proc.wait()
    return cases, summary


# ── Baseline measurement ──────────────────────────────────────────────────────

# One representative prompt per group (modality × context), same as test_05.
_BASELINE_PROMPTS: dict[str, tuple[str, str, int]] = {
    # group_key: (mode, prompt_text, max_tokens)
    "text__short":       ("text", "Describe the color blue in exactly one sentence.", 32),
    "text__medium":      ("text",
        "You are analyzing a philosophical debate. "
        "Physicalism holds that consciousness arises entirely from physical brain processes. "
        "Dualism argues it requires something beyond the physical. "
        "David Chalmers's hard problem asks why subjective experience exists at all. "
        "Briefly summarize both positions and state which has stronger empirical support.", 48),
    "text__long":        ("text",
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
        "matters for AI safety research.", 48),
    "text_image__short":  ("text_image", "Describe the color blue in exactly one sentence.", 32),
    "text_image__medium": ("text_image",
        "You are analyzing a philosophical debate. "
        "Physicalism holds that consciousness arises entirely from physical brain processes. "
        "Dualism argues it requires something beyond the physical. "
        "David Chalmers's hard problem asks why subjective experience exists at all. "
        "Briefly summarize both positions and state which has stronger empirical support.", 48),
    "text_image__long":   ("text_image",
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
        "matters for AI safety research.", 48),
    "image_only__short":  ("image_only", "", 32),
    "image_only__medium": ("image_only", "", 48),
    "image_only__long":   ("image_only", "", 48),
}

_SAMPLE_IMAGE = THIS_DIR.parent / "sample_image.webp"


def _img_data_url(path: Path) -> str:
    import base64
    raw = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".") or "png"
    return f"data:image/{suffix};base64,{base64.b64encode(raw).decode()}"


def _build_baseline_messages(mode: str, prompt: str) -> list[dict]:
    img_url = _img_data_url(_SAMPLE_IMAGE) if mode in {"text_image", "image_only"} else None
    if mode == "text":
        return [{"role": "user", "content": prompt}]
    if mode == "text_image":
        return [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": img_url}},
        ]}]
    # image_only
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": img_url}},
    ]}]


def _send_one(api: str, bl_model: str, mode: str, prompt: str, max_tok: int,
              timeout: float) -> tuple[int, float]:
    """Send a single chat completion request. Returns (completion_tokens, elapsed_sec)."""
    messages = _build_baseline_messages(mode, prompt)
    payload = json.dumps({
        "model": bl_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tok,
    }).encode()
    req = urllib.request.Request(
        url=f"{api}/chat/completions",
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        data=payload,
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    elapsed = time.time() - t0
    ctok = (body.get("usage") or {}).get("completion_tokens") or 0
    return ctok, elapsed


def measure_baseline(baseline_url: str, model: str, timeout: float = 120.0) -> dict[str, float]:
    """Send one request per group to the baseline server. Returns {group_key: tok_per_sec}.

    Two warm-up requests (text + image) are sent first so that CUDA kernels are
    compiled and the image encoder is cached before actual measurements begin.
    """
    if not _SAMPLE_IMAGE.exists():
        print(f"[baseline] WARNING: sample image not found: {_SAMPLE_IMAGE}", file=sys.stderr)

    results: dict[str, float] = {}
    api = baseline_url.rstrip("/")

    # Resolve model name from baseline server
    try:
        req = urllib.request.Request(
            url=f"{api}/models",
            headers={"Authorization": "Bearer EMPTY"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        served = [m["id"] for m in data.get("data", []) if isinstance(m, dict)]
        bl_model = model if model in served else (served[0] if served else model)
    except Exception as e:
        print(f"[baseline] Could not resolve model from {api}: {e}", file=sys.stderr)
        bl_model = model

    # Warm-up: one text request + one image request so CUDA kernels are compiled
    # and the vision encoder KV cache is populated before measuring.
    print(f"[baseline] Warming up server (2 requests)...", file=sys.stderr)
    _warmup_cases = [
        ("text",       "Hello.",                                        8),
        ("text_image", "Describe what you see briefly.",               8),
    ]
    for mode, prompt, max_tok in _warmup_cases:
        if mode == "text_image" and not _SAMPLE_IMAGE.exists():
            continue
        try:
            _send_one(api, bl_model, mode, prompt, max_tok, timeout)
            print(f"[baseline]   warmup {mode} done", file=sys.stderr)
        except Exception as e:
            print(f"[baseline]   warmup {mode} error: {e}", file=sys.stderr)

    print(f"[baseline] Measuring {len(_BASELINE_PROMPTS)} groups against {api} (model={bl_model})", file=sys.stderr)

    for gkey, (mode, prompt, max_tok) in _BASELINE_PROMPTS.items():
        if mode in {"text_image", "image_only"} and not _SAMPLE_IMAGE.exists():
            print(f"[baseline] skip {gkey}: image missing", file=sys.stderr)
            continue

        try:
            ctok, elapsed = _send_one(api, bl_model, mode, prompt, max_tok, timeout)
            tps = ctok / elapsed if elapsed > 0 and ctok > 0 else 0.0
            results[gkey] = tps
            print(f"[baseline]   {gkey:<25} {ctok} tok / {elapsed:.2f}s = {tps:.1f} tok/s", file=sys.stderr)
        except Exception as e:
            print(f"[baseline]   {gkey:<25} ERROR: {e}", file=sys.stderr)

    return results


# ── Table formatting ──────────────────────────────────────────────────────────

_COL_WIDTHS = {
    "name":     45,
    "pass":      4,
    "latency":   8,
    "ptok":      5,
    "ctok":      5,
    "kv":        3,
    "layers":   20,
    "error":    30,
}


def _toks_per_sec(ctok, elapsed) -> str:
    """Completion tokens / elapsed_sec (includes prefill; lower bound on gen throughput)."""
    if ctok and elapsed and elapsed > 0:
        return f"{ctok / elapsed:.1f}"
    return "-"


def _row(name, ok, elapsed, ptok, ctok, kv, ret_layers, error, markdown=False) -> str:
    mark = "✓" if ok else "✗"
    lat = f"{elapsed:.2f}s" if elapsed is not None else "-"
    tps = _toks_per_sec(ctok, elapsed)
    ptok_s = str(ptok) if ptok is not None else "-"
    ctok_s = str(ctok) if ctok is not None else "-"
    kv_s = str(kv) if kv is not None else "-"
    rl = ",".join(str(x) for x in (ret_layers or [])) or "-"
    err = (error or "")[:30]

    if markdown:
        return (f"| {name:<45} | {mark:^4} | {lat:>8} | {tps:>7} |"
                f" {ptok_s:>5} | {ctok_s:>5} | {kv_s:>3} | {rl:<20} | {err} |")
    return (f"  {name:<43} {mark:>4}  {lat:>8}  {tps:>7}"
            f"  {ptok_s:>5}  {ctok_s:>5}  {kv_s:>3}  {rl:<20}  {err}")


def _header(markdown=False) -> str:
    if markdown:
        h = (f"| {'Case':<45} | {'Pass':^4} | {'Latency':>8} | {'Tok/s':>7} |"
             f" {'PTok':>5} | {'CTok':>5} | {'KV':>3} | {'RetLayers':<20} | Error |")
        sep = "|" + "|".join("-" * (w + 2) for w in [47, 6, 10, 9, 7, 7, 5, 22, 32]) + "|"
        return h + "\n" + sep
    h = (f"  {'Case':<43} {'Pass':>4}  {'Latency':>8}  {'Tok/s':>7}"
         f"  {'PTok':>5}  {'CTok':>5}  {'KV':>3}  {'RetLayers':<20}  Error")
    sep = "  " + "-" * (len(h) - 2)
    return h + "\n" + sep


def print_table(cases: list[dict], model: str, markdown: bool = False) -> None:
    width = 110
    print()
    if not markdown:
        print("=" * width)
        print(f"  RIGOROUS TEST RESULTS  —  Model: {model}")
        print("=" * width)
    else:
        print(f"## Rigorous Test Results — `{model}`\n")

    print(_header(markdown))

    prev_group = None
    groups: dict[str, list[dict]] = {}
    for r in cases:
        name = r.get("name", "")
        # group key = modality + context (first two __ segments)
        parts = name.split("__")
        gkey = "__".join(parts[:2]) if len(parts) >= 2 else name
        groups.setdefault(gkey, []).append(r)

    for gkey, rows in groups.items():
        if not markdown:
            print()  # blank line between groups
        for r in rows:
            print(_row(
                name=r.get("name", ""),
                ok=r.get("ok", False),
                elapsed=r.get("elapsed_sec"),
                ptok=r.get("prompt_tokens"),
                ctok=r.get("completion_tokens"),
                kv=r.get("kv_items"),
                ret_layers=r.get("returned_layers"),
                error=r.get("error") if not r.get("ok") else None,
                markdown=markdown,
            ))


    # Summary
    passed = sum(1 for r in cases if r.get("ok"))
    total = len(cases)
    failed_names = [r["name"] for r in cases if not r.get("ok")]
    print()
    if not markdown:
        print("=" * width)
        print(f"  RESULT: {passed}/{total} passed  |  {len(failed_names)} failed")
        if failed_names:
            print(f"\n  FAILED CASES:")
            for n in failed_names:
                print(f"    ✗ {n}")
        print("=" * width)
    else:
        print(f"\n**{passed}/{total} passed**")
        if failed_names:
            print("\n**Failed cases:**")
            for n in failed_names:
                print(f"- ✗ `{n}`")
    print()


# ── Breakdown table ───────────────────────────────────────────────────────────

def print_breakdown(cases: list[dict], baseline: dict[str, float] | None = None,
                    markdown: bool = False) -> None:
    """Print per-group summary: pass rate, avg latency, avg tok/s, optional baseline tok/s."""
    groups: dict[str, list[dict]] = {}
    for r in cases:
        parts = r.get("name", "").split("__")
        gkey = "__".join(parts[:2]) if len(parts) >= 2 else r.get("name", "")
        groups.setdefault(gkey, []).append(r)

    has_baseline = bool(baseline)

    if markdown:
        print("\n### Breakdown by Modality × Context\n")
        if has_baseline:
            print(f"| {'Group':<30} | {'Pass':>8} | {'AvgLat':>8} | {'AvgTok/s':>9} | {'BaseTok/s':>10} | {'AvgPTok':>8} |")
            print("|" + "|".join("-" * (w + 2) for w in [32, 10, 10, 11, 12, 10]) + "|")
        else:
            print(f"| {'Group':<30} | {'Pass':>8} | {'AvgLat':>8} | {'AvgTok/s':>9} | {'AvgPTok':>8} |")
            print("|" + "|".join("-" * (w + 2) for w in [32, 10, 10, 11, 10]) + "|")
    else:
        print("  BREAKDOWN BY MODALITY × CONTEXT")
        print("  " + "-" * (84 if has_baseline else 72))
        if has_baseline:
            print(f"  {'Group':<30} {'Pass':>8}  {'AvgLat':>8}  {'AvgTok/s':>9}  {'BaseTok/s':>10}  {'AvgPTok':>8}")
        else:
            print(f"  {'Group':<30} {'Pass':>8}  {'AvgLat':>8}  {'AvgTok/s':>9}  {'AvgPTok':>8}")
        print("  " + "-" * (84 if has_baseline else 72))

    for gkey, rows in groups.items():
        n_pass = sum(1 for r in rows if r.get("ok"))
        lats = [r["elapsed_sec"] for r in rows if r.get("elapsed_sec") is not None]
        ctoks = [r["completion_tokens"] for r in rows
                 if r.get("completion_tokens") and r.get("elapsed_sec")]
        elaps = [r["elapsed_sec"] for r in rows
                 if r.get("completion_tokens") and r.get("elapsed_sec")]
        ptoks = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens") is not None]
        avg_lat = f"{sum(lats)/len(lats):.2f}s" if lats else "-"
        avg_tps = f"{sum(ctoks)/sum(elaps):.1f}" if ctoks and sum(elaps) > 0 else "-"
        avg_ptok = f"{sum(ptoks)//len(ptoks)}" if ptoks else "-"
        ratio = f"{n_pass}/{len(rows)}"
        base_tps = f"{baseline[gkey]:.1f}" if has_baseline and gkey in baseline else "-"
        if markdown:
            if has_baseline:
                print(f"| {gkey:<30} | {ratio:>8} | {avg_lat:>8} | {avg_tps:>9} | {base_tps:>10} | {avg_ptok:>8} |")
            else:
                print(f"| {gkey:<30} | {ratio:>8} | {avg_lat:>8} | {avg_tps:>9} | {avg_ptok:>8} |")
        else:
            if has_baseline:
                print(f"  {gkey:<30} {ratio:>8}  {avg_lat:>8}  {avg_tps:>9}  {base_tps:>10}  {avg_ptok:>8}")
            else:
                print(f"  {gkey:<30} {ratio:>8}  {avg_lat:>8}  {avg_tps:>9}  {avg_ptok:>8}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run rigorous attention capture matrix test")
    parser.add_argument("--markdown", action="store_true", help="Output table in Markdown format")
    parser.add_argument("--save-md", metavar="FILE", help="Save Markdown report to file")
    parser.add_argument("--out", metavar="FILE", help="Save raw JSON results to file")
    parser.add_argument(
        "--baseline-url", metavar="URL",
        help="URL of a plain vLLM server (no --enable-attention-instrumentation) "
             "to measure baseline tok/s for the breakdown table",
    )
    args = parser.parse_args()

    model = os.environ.get("VLLM_TEST_MODEL", "gemma-3-4b-it")
    print(f"[run_rigorous] Starting matrix test for: {model}", file=sys.stderr)

    cases, summary = run_test()

    if not cases:
        print("[run_rigorous] No results — is the server running?", file=sys.stderr)
        sys.exit(2)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps({"model": model, "cases": cases, "summary": summary},
                                       indent=2, ensure_ascii=False))
        print(f"[run_rigorous] JSON saved to {out_path}", file=sys.stderr)

    # Optionally measure baseline
    baseline: dict[str, float] | None = None
    if args.baseline_url:
        print(f"[run_rigorous] Measuring baseline against {args.baseline_url}", file=sys.stderr)
        baseline = measure_baseline(args.baseline_url, model)

    # Print to stdout (plain or markdown as requested)
    print_table(cases, model, markdown=args.markdown)
    print_breakdown(cases, baseline=baseline, markdown=args.markdown)

    # Optionally also save a separate markdown report
    if args.save_md:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_table(cases, model, markdown=True)
            print_breakdown(cases, baseline=baseline, markdown=True)
        md_path = Path(args.save_md)
        md_path.write_text(buf.getvalue(), encoding="utf-8")
        print(f"[run_rigorous] Markdown report saved to {md_path}", file=sys.stderr)

    passed = sum(1 for r in cases if r.get("ok"))
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
