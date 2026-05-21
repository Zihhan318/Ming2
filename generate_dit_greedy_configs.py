#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml


BASE_BLOCKS = [29, 28, 2, 27, 3, 11, 1, 10, 4, 12, 13, 7, 26]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate leave-one-out DiT stage1-g calibration configs for greedy rollback experiments."
    )
    parser.add_argument(
        "--base-config",
        default="quant_zimage_dit_w8a8_ffn_stage1_g_calib10.yaml",
        help="Base YAML config to clone before adjusting include blocks.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/dit_greedy_configs",
        help="Directory to write generated YAML configs and manifest.",
    )
    parser.add_argument(
        "--conditions-dir",
        help="Optional override for multimodal_sd_config.model_config.conditions_dir.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Optional override for multimodal_sd_config.model_config.max_cases.",
    )
    return parser.parse_args()


def build_include(blocks: list[int]) -> list[str]:
    include = []
    for block in sorted(blocks):
        include.append(f"layers.{block}.feed_forward.w1")
        include.append(f"layers.{block}.feed_forward.w3")
    return include


def main():
    args = parse_args()
    base_config_path = Path(args.base_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    model_cfg = config["spec"]["multimodal_sd_config"]["model_config"]
    conditions_dir = model_cfg.get("conditions_dir")
    if conditions_dir and not Path(conditions_dir).is_absolute():
        model_cfg["conditions_dir"] = str((base_config_path.parent / conditions_dir).resolve())

    if args.conditions_dir:
        model_cfg["conditions_dir"] = str(Path(args.conditions_dir).resolve())
    if args.max_cases is not None:
        model_cfg["max_cases"] = args.max_cases

    manifest = {
        "base_config": str(base_config_path),
        "base_blocks": BASE_BLOCKS,
        "variants": [],
    }

    for dropped_block in BASE_BLOCKS:
        kept_blocks = [block for block in BASE_BLOCKS if block != dropped_block]
        variant = yaml.safe_load(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        variant["metadata"]["config_id"] = f"ming_zimage_dit_w8a8_ffn_stage1_g_drop{dropped_block:02d}"
        variant["spec"]["process"][0]["include"] = build_include(kept_blocks)
        variant_path = output_dir / f"quant_zimage_dit_w8a8_ffn_stage1_g_drop{dropped_block:02d}.yaml"
        variant_path.write_text(yaml.safe_dump(variant, sort_keys=False, allow_unicode=True), encoding="utf-8")
        manifest["variants"].append(
            {
                "name": f"stage1-g-drop{dropped_block:02d}",
                "dropped_block": dropped_block,
                "kept_blocks": kept_blocks,
                "config_path": str(variant_path),
            }
        )

    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
