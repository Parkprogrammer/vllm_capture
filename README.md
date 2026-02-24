# vllm_capture — Rigorous test suite for vLLM attention instrumentation

Integration tests for the attention capture feature proposed in vllm-project/vllm.

## Requirements

- A running vLLM server with `--enable-attention-instrumentation`
- Python 3.10+, no additional dependencies beyond the stdlib

## Usage

```bash
export VLLM_API_BASE=http://127.0.0.1:8000/v1
python tests/run_rigorous.py
```

## Visualization

Generate per-query and sentence-by-sentence attention heatmaps from a running vLLM server.

**Requirements**
- A running vLLM server with `--enable-attention-instrumentation`
- Local model checkpoint for tokenizer (`AutoProcessor`)
- Python 3.10+, `Pillow`, `numpy`, `transformers`

**Usage**

Online mode (calls vLLM server directly):
```
python visualize_attn.py \
  --api-base http://127.0.0.1:8000/v1 \
  --model google/gemma-3-4b-it \
  --model-path /path/to/gemma-3-4b-it \
  --layer 33 \
  --image-path sample_image.webp
```

Artifact mode (replay from saved response):
```
python visualize_attn.py \
  --artifact-dir ./artifacts/visualization/1234567890 \
  --model-path /path/to/gemma-3-4b-it
```

**Outputs**

Each run creates a timestamped directory under `artifacts/visualization/` containing:
- `text_heatmap_q{idx}.png` — per-query token attention heatmap
- `image_overlay_q{idx}.png` — attention overlaid on input image (16×16 patch grid)
- `image_overlay_avg.png` — average overlay across all selected queries
- `by_sentence/sentence_{idx}/` — sentence-by-sentence breakdown
- `summary.json` — run metadata and file index
