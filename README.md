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
