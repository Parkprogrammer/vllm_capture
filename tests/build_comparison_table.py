#!/usr/bin/env python3
"""Build structured comparison table from multi-model test results."""
import json
import sys
from pathlib import Path

def build_structured_table(json_files: list[str], output_file: str = None):
    """Build table with modality, context, capture, layers as columns."""

    # Load all results
    all_data = {}
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
            # Handle both list and dict formats
            if isinstance(data, list):
                for model_results in data:
                    model_name = model_results['model']
                    all_data[model_name] = {c['name']: c for c in model_results.get('cases', [])}
            else:
                # Single dict format
                model_name = data['model']
                all_data[model_name] = {c['name']: c for c in data.get('cases', [])}

    # Parse case name: modality__context__capture[__layers]
    def parse_case(name):
        parts = name.split('__')
        modality = parts[0]  # text, text_image, image_only
        context = parts[1]    # short, medium, long
        capture = parts[2]    # off, on
        layers = parts[3] if len(parts) > 3 else None
        return modality, context, capture, layers

    # Get all case names from first model
    first_model = list(all_data.keys())[0]
    all_cases = sorted(all_data[first_model].keys())

    # Group by modality
    modality_groups = {}
    for case_name in all_cases:
        modality, context, capture, layers = parse_case(case_name)
        key = (modality, context, capture, layers)
        if modality not in modality_groups:
            modality_groups[modality] = []
        modality_groups[modality].append(key)

    # Build table: Modality | Context | Capture | Layers | model1 | model2 | ...
    models = sorted(all_data.keys())

    lines = []
    lines.append('| Modality | Context | Capture | Layers | ' + ' | '.join(models) + ' |')
    lines.append('|' + '|'.join(['---'] * (6 + len(models))) + '|')

    for modality in sorted(modality_groups.keys()):
        cases = modality_groups[modality]

        for modality_idx, (mod, ctx, cap, layers) in enumerate(cases):
            # Modality (show only for first row of each group)
            mod_cell = modality if modality_idx == 0 else ''

            # Context
            ctx_cell = ctx

            # Capture
            cap_cell = '❌' if cap == 'off' else f'✓ ({layers})' if layers else '✓'

            # Layers
            layers_cell = layers or '-'

            case_key = f"{mod}__{ctx}__{cap}"
            if layers:
                case_key += f"__{layers}"

            # Results per model
            result_cells = []
            for model in models:
                if case_key in all_data[model]:
                    c = all_data[model][case_key]
                    if c.get('ok'):
                        lat = f"{c['elapsed_sec']:.1f}s"
                        ctok = c.get('completion_tokens', 0) or 0
                        elapsed = c.get('elapsed_sec', 0) or 0
                        tps = f"{ctok/elapsed:.0f}t/s" if elapsed > 0 else '-'
                        result_cells.append(f'✓ {lat} {tps}')
                    else:
                        result_cells.append(f'✗ {c.get("error", "")[:20]}')
                else:
                    result_cells.append('n/a')

            lines.append(f'| {mod_cell} | {ctx_cell} | {cap_cell} | {layers_cell} | ' + ' | '.join(result_cells) + ' |')

    # Summary row
    lines.append('|---|---|---|---|' + '|'.join(['---'] * len(models)) + '|')
    summary_cells = []
    for model in models:
        passed = sum(1 for c in all_data[model].values() if c.get('ok'))
        total = len(all_data[model])
        summary_cells.append(f'{passed}/{total}')
    lines.append(f'| **TOTAL** | | | | ' + ' | '.join(summary_cells) + ' |')

    table_str = '\n'.join(lines)

    if output_file:
        Path(output_file).write_text(table_str)
        print(f"Table saved to {output_file}")
    else:
        print(table_str)

    return table_str

if __name__ == '__main__':
    import glob

    json_files = [
        '/workspace/vllm/attn_instrumentation/artifacts/multimodel/results_gemma-3-4b-it.json',
        '/workspace/vllm/attn_instrumentation/artifacts/multimodel/results_qwen2.5-7b.json',
        '/workspace/vllm/attn_instrumentation/artifacts/multimodel/results_qwen2.5-vl-7b.json',
    ]

    # Check for Llama results
    if Path('/workspace/vllm/attn_instrumentation/artifacts/multimodel/results_llama.json').exists():
        json_files.append('/workspace/vllm/attn_instrumentation/artifacts/multimodel/results_llama.json')

    build_structured_table(json_files, '/workspace/vllm/attn_instrumentation/artifacts/multimodel/COMPARISON_TABLE.md')
