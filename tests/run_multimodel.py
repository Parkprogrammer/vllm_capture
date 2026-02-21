#!/usr/bin/env python3
"""run_multimodel.py: Multi-model orchestrator with automatic server lifecycle management.

For each model ID given:
  1. Start vLLM server (auto-downloads from HuggingFace if not cached)
  2. Wait for server health
  3. Run the rigorous matrix test (test_05_rigorous_matrix.py)
  4. Stop the server and wait for GPU memory release
  5. Repeat for the next model

Finally prints a unified comparison table across all models.

Usage:
    python run_multimodel.py google/gemma-3-4b-it Qwen/Qwen2.5-3B-Instruct [options]

Options:
    --markdown          Output tables in Markdown format
    --out FILE          Save all raw JSON results to FILE
    --served-names N,N  Comma-separated short names for the models (default: last segment of HF ID)

Environment:
    VLLM_PORT            server port (default: 8000)
    VLLM_SERVER_TIMEOUT  seconds to wait for server start (default: 180)
    VLLM_GPU_COOLDOWN    seconds to wait after server stop for GPU memory release (default: 15)
    HF_TOKEN             HuggingFace token for gated models
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
THIS_DIR = Path(__file__).parent

PORT = int(os.environ.get("VLLM_PORT", "8000"))
SERVER_TIMEOUT = int(os.environ.get("VLLM_SERVER_TIMEOUT", "180"))
GPU_COOLDOWN = int(os.environ.get("VLLM_GPU_COOLDOWN", "15"))
API_BASE = f"http://127.0.0.1:{PORT}/v1"

# Flags forwarded to every vLLM server instance
_SERVER_COMMON_FLAGS = [
    "--enforce-eager",
    "--port", str(PORT),
    "--enable-attention-instrumentation",
    "--attention-instrumentation-layers", "all",
    "--max-model-len", "4096",
]


# ── Model helpers ─────────────────────────────────────────────────────────────

def parse_models(hf_ids: list[str], served_names: list[str] | None) -> list[tuple[str, str]]:
    """Return [(hf_model_id, served_name), ...].

    served_name defaults to the last segment of the HF ID
    (e.g. 'gemma-3-4b-it' from 'google/gemma-3-4b-it').
    """
    names = served_names or []
    result = []
    for i, hf_id in enumerate(hf_ids):
        name = names[i] if i < len(names) else hf_id.split("/")[-1]
        result.append((hf_id, name))
    return result


# ── Server lifecycle ──────────────────────────────────────────────────────────

def start_server(hf_model_id: str, served_name: str, log_file: Path) -> subprocess.Popen:
    """Launch vLLM server; auto-downloads model from HF if not cached.
    stdout/stderr go to log_file.
    """
    env = {**os.environ}
    if "HF_TOKEN" in os.environ:
        env["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    cmd = [
        PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_model_id,
        "--served-model-name", served_name,
    ] + _SERVER_COMMON_FLAGS

    fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=fh,
        env=env,
        preexec_fn=os.setsid,   # own process group for clean kill
    )
    return proc


def wait_for_server(timeout: int = SERVER_TIMEOUT) -> bool:
    """Poll /v1/models until the server responds or timeout expires."""
    import urllib.request
    deadline = time.monotonic() + timeout
    url = API_BASE.rstrip("/") + "/models"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    """Send SIGTERM to the server process group and wait."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# ── Test runner ───────────────────────────────────────────────────────────────

def run_matrix(served_name: str) -> tuple[list[dict], dict | None]:
    """Run test_05_rigorous_matrix against the current server; return (cases, summary)."""
    env = {
        **os.environ,
        "VLLM_TEST_MODEL": served_name,
        "VLLM_API_BASE": API_BASE,
    }
    proc = subprocess.Popen(
        [PYTHON, str(THIS_DIR / "test_05_rigorous_matrix.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
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


# ── Unified comparison table ──────────────────────────────────────────────────

def _case_key(name: str) -> str:
    """Strip model-specific prefix; return canonical config key."""
    return name  # names are already model-independent


def print_comparison_table(all_results: dict[str, list[dict]], markdown: bool) -> None:
    """Print a table where rows=configs, cols=models."""
    models = list(all_results.keys())
    if not models:
        print("No results to display.")
        return

    # Collect all unique case names (in order of first model)
    case_names: list[str] = []
    seen: set[str] = set()
    for model in models:
        for r in all_results[model]:
            n = r.get("name", "")
            if n not in seen:
                case_names.append(n)
                seen.add(n)

    # Build lookup: (model, case_name) -> result
    idx: dict[tuple[str, str], dict] = {}
    for model, cases in all_results.items():
        for r in cases:
            idx[(model, r.get("name", ""))] = r

    def cell(model: str, name: str) -> str:
        r = idx.get((model, name))
        if r is None:
            return "n/a"
        ok = "✓" if r.get("ok") else "✗"
        lat = f"{r.get('elapsed_sec', 0):.1f}s"
        ctok = r.get("completion_tokens") or 0
        elapsed = r.get("elapsed_sec") or 0
        tps = f"{ctok/elapsed:.0f}t/s" if elapsed > 0 else "-"
        kv = r.get("kv_items", "-")
        return f"{ok} {lat} {tps} kv={kv}"

    col_w = max(20, max(len(m) for m in models) + 2)
    row_w = 47

    if not markdown:
        width = row_w + col_w * len(models) + 4
        print()
        print("=" * width)
        print("  MULTI-MODEL COMPARISON TABLE")
        print("=" * width)
        hdr = f"  {'Case':<{row_w - 2}}"
        for m in models:
            hdr += f"  {m:<{col_w}}"
        print(hdr)
        print("  " + "-" * (width - 2))

        prev_group = None
        for name in case_names:
            parts = name.split("__")
            gkey = "__".join(parts[:2]) if len(parts) >= 2 else name
            if gkey != prev_group:
                print()
                prev_group = gkey
            row = f"  {name:<{row_w - 2}}"
            for m in models:
                row += f"  {cell(m, name):<{col_w}}"
            print(row)

        # Per-model summary row
        print()
        print("  " + "-" * (width - 2))
        totals_row = f"  {'TOTAL PASS':<{row_w - 2}}"
        for m in models:
            cases = all_results[m]
            p = sum(1 for r in cases if r.get("ok"))
            t = len(cases)
            totals_row += f"  {p}/{t} ({100*p//t if t else 0}%){'':<{col_w - 10}}"
        print(totals_row)
        print("=" * width)

    else:
        print("## Multi-Model Comparison\n")
        hdr = "| Case" + "".join(f" | {m}" for m in models) + " |"
        sep = "|---" + "".join("|---" for _ in models) + "|"
        print(hdr)
        print(sep)
        for name in case_names:
            row = f"| `{name}`"
            for m in models:
                row += f" | {cell(m, name)}"
            row += " |"
            print(row)
        print()
        totals_row = "| **TOTAL**"
        for m in models:
            cases = all_results[m]
            p = sum(1 for r in cases if r.get("ok"))
            t = len(cases)
            totals_row += f" | **{p}/{t}**"
        totals_row += " |"
        print(totals_row)

    print()


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-model rigorous test orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_multimodel.py google/gemma-3-4b-it\n"
            "  python run_multimodel.py google/gemma-3-4b-it Qwen/Qwen2.5-3B-Instruct\n"
            "  python run_multimodel.py google/gemma-3-4b-it --served-names gemma-4b\n"
        ),
    )
    parser.add_argument(
        "model_ids", nargs="+", metavar="HF_MODEL_ID",
        help="HuggingFace model ID(s) to test (e.g. google/gemma-3-4b-it)"
    )
    parser.add_argument(
        "--served-names", metavar="NAME[,NAME,...]",
        help="Short names for served models (comma-separated, same order as model IDs)"
    )
    parser.add_argument("--markdown", action="store_true",
                        help="Output tables in Markdown format")
    parser.add_argument("--out", metavar="FILE",
                        help="Save all raw JSON results to a file")
    args = parser.parse_args()

    served_names = [n.strip() for n in args.served_names.split(",")] \
        if args.served_names else None
    models = parse_models(args.model_ids, served_names)

    if not models:
        print("[run_multimodel] No models specified", file=sys.stderr)
        sys.exit(1)

    print(f"[run_multimodel] Models to test: {[n for _, n in models]}", file=sys.stderr)

    all_results: dict[str, list[dict]] = {}
    artifacts_dir = THIS_DIR.parent / "artifacts" / "multimodel"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for hf_id, served_name in models:
        log_file = artifacts_dir / f"server_{served_name}.log"
        result_file = artifacts_dir / f"results_{served_name}.json"

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[run_multimodel] Model:  {served_name}", file=sys.stderr)
        print(f"[run_multimodel] HF ID:  {hf_id}", file=sys.stderr)
        print(f"[run_multimodel] Log:    {log_file}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        # Start server (auto-downloads from HF if not cached)
        print("[run_multimodel] Starting server...", end="", flush=True, file=sys.stderr)
        proc = start_server(hf_id, served_name, log_file)

        if not wait_for_server(SERVER_TIMEOUT):
            print(" TIMED OUT", file=sys.stderr)
            stop_server(proc)
            all_results[served_name] = []
            continue
        print(" ready", file=sys.stderr)

        # Run tests
        try:
            print("[run_multimodel] Running test matrix...", file=sys.stderr)
            cases, summary = run_matrix(served_name)
            all_results[served_name] = cases

            # Save per-model results
            result_file.write_text(
                json.dumps({"model": served_name, "cases": cases, "summary": summary},
                           indent=2, ensure_ascii=False)
            )

            passed = sum(1 for r in cases if r.get("ok"))
            print(f"[run_multimodel] Result: {passed}/{len(cases)} passed", file=sys.stderr)

        except Exception as e:
            print(f"[run_multimodel] Test error: {e}", file=sys.stderr)
            all_results[served_name] = []

        finally:
            # Stop server
            print("[run_multimodel] Stopping server...", end="", flush=True, file=sys.stderr)
            stop_server(proc)
            print(f" done. Cooling down {GPU_COOLDOWN}s for GPU memory release...",
                  file=sys.stderr)
            time.sleep(GPU_COOLDOWN)

    # Save combined results
    if args.out:
        out_path = Path(args.out)
        combined = [
            {"model": m, "cases": c}
            for m, c in all_results.items()
        ]
        out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
        print(f"\n[run_multimodel] All results saved to {out_path}", file=sys.stderr)

    # Print comparison table
    print_comparison_table(all_results, markdown=args.markdown)

    # Exit code: 0 only if all models have 100% pass
    all_pass = all(
        all(r.get("ok") for r in cases)
        for cases in all_results.values()
        if cases
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
