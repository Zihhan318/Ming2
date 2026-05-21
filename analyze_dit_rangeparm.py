#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import yaml

from msmodelslim.app.base import DeviceType
from msmodelslim.app.analysis_service.analysis_methods import StdAnalysisMethod

from msmodelslim_zimage_dit_adapter import MingZImageDiTAdapter


def resolve_config(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = config["spec"]["multimodal_sd_config"]["model_config"]
    conditions_dir = model_config.get("conditions_dir")
    if conditions_dir:
        conditions_path = Path(conditions_dir)
        if not conditions_path.is_absolute():
            model_config["conditions_dir"] = str((config_path.parent / conditions_path).resolve())
    return model_config


def build_hooks(transformer: torch.nn.Module, stats: dict, include_w2: bool = False):
    method = StdAnalysisMethod()
    hook = method.get_hook()
    handles = []
    targets = {"w1", "w3"}
    if include_w2:
        targets.add("w2")
    for layer_idx, block in enumerate(transformer.layers):
        ff = block.feed_forward
        for name in sorted(targets):
            module = getattr(ff, name)
            layer_name = f"layers.{layer_idx}.feed_forward.{name}.quant_input"
            handles.append(
                module.register_forward_hook(
                    lambda mod, inp, out, layer_name=layer_name: hook(mod, inp, out, layer_name, stats)
                )
            )
    return handles


def compute_range_scores(stats: dict) -> list[dict]:
    method = StdAnalysisMethod()
    rows = []
    for name, layer_stats in stats.items():
        score = method.compute_score(layer_stats)
        rows.append({"name": name, "range_parm": score})
    rows.sort(key=lambda item: item["range_parm"], reverse=True)
    return rows


def summarize_by_block(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        parts = row["name"].split(".")
        layer_idx = int(parts[1])
        proj = parts[3]
        grouped.setdefault(layer_idx, {})[proj] = row["range_parm"]
    summary = []
    for layer_idx, values in sorted(grouped.items()):
        layer_row = {"layer": layer_idx}
        for key, value in values.items():
            layer_row[key] = value
        layer_row["max_range_parm"] = max(values.values())
        layer_row["avg_range_parm"] = sum(values.values()) / len(values)
        summary.append(layer_row)
    summary.sort(key=lambda item: item["max_range_parm"], reverse=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Compute DiT FFN range_parm ranking on stage1 calibration cases.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-json", default="analysis/dit_stage1_rangeparm.json")
    parser.add_argument("--include-w2", action="store_true", help="Also analyze feed_forward.w2.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_config = resolve_config(Path(args.config_path).resolve())

    adapter = MingZImageDiTAdapter(
        model_type="Ming-ZImage-DiT",
        model_path=Path(args.model_path).resolve(),
        trust_remote_code=True,
    )
    adapter.set_model_args(model_config)
    adapter.load_pipeline()

    stats = {}
    handles = build_hooks(adapter.transformer, stats, include_w2=args.include_w2)
    try:
        adapter.run_calib_inference()
    finally:
        for handle in handles:
            handle.remove()

    rows = compute_range_scores(stats)
    block_summary = summarize_by_block(rows)

    output = {
        "config_path": str(Path(args.config_path).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "rows": rows,
        "block_summary": block_summary,
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("TOP_LAYERS")
    for row in rows[:20]:
        print(f"{row['name']}\t{row['range_parm']:.6f}")
    print("\nTOP_BLOCKS")
    for row in block_summary[:15]:
        values = [f"{key}={row[key]:.6f}" for key in ("w1", "w3", "w2") if key in row]
        print(
            f"layers.{row['layer']}\t"
            + "\t".join(values)
            + f"\tmax={row['max_range_parm']:.6f}\tavg={row['avg_range_parm']:.6f}"
        )
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
