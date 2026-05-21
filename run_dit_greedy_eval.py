#!/usr/bin/env python3
import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


CASES = [
    {
        "id": "01_cake_food_photo",
        "prompt": "一张清新自然的草莓奶油蛋糕美食摄影，白色陶瓷盘，柔和自然光，写实风格。",
        "condition_input": "/home/zzh/Ming2/dit_calib/conditions_v2_tp8_10cases/conditions/01_cake_food_photo.pt",
        "baseline_image": "/home/zzh/Ming2/generated_imgs/dit_baseline_cake.png",
        "output_suffix": "cake",
    },
    {
        "id": "02_cyberpunk_night_city",
        "prompt": "雨夜中的赛博朋克城市街道，霓虹灯反射在潮湿路面上，远处有悬浮广告牌和行人剪影，电影感。",
        "condition_input": "/home/zzh/Ming2/dit_calib/conditions_v2_tp8_10cases/conditions/02_cyberpunk_night_city.pt",
        "baseline_image": "/home/zzh/Ming2/generated_imgs/dit_baseline_cyberpunk.png",
        "output_suffix": "cyberpunk",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize and evaluate DiT greedy rollback variants.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--save-root", default="/tmp/dit_greedy_eval")
    parser.add_argument("--output-dir", default="/home/zzh/Ming2/generated_imgs/量化")
    parser.add_argument("--summary-json", default="/home/zzh/Ming2/analysis/dit_greedy_eval_summary.json")
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: str):
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout + proc.stderr


def extract_generate_time(log_text: str):
    match = re.search(r"run_1_generate:\s*([0-9.]+)s", log_text)
    return float(match.group(1)) if match else None


def compute_metrics(baseline_path: Path, candidate_path: Path):
    baseline = np.asarray(Image.open(baseline_path).convert("RGB"), dtype=np.float32)
    candidate = np.asarray(Image.open(candidate_path).convert("RGB"), dtype=np.float32)
    mae = float(np.mean(np.abs(baseline - candidate)))
    mse = float(np.mean((baseline - candidate) ** 2))
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return {"mae": mae, "psnr": psnr}


def main():
    args = parse_args()
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    variants = manifest["variants"]
    if args.variants:
        wanted = set(args.variants)
        variants = [item for item in variants if item["name"] in wanted]
    if args.limit is not None:
        variants = variants[: args.limit]

    save_root = Path(args.save_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "model_path": args.model_path,
        "code_path": args.code_path,
        "device_id": args.device_id,
        "results": [],
    }

    for variant in variants:
        name = variant["name"]
        config_path = Path(variant["config_path"]).resolve()
        quant_save = save_root / name
        print(f"[quantize] {name}")
        quant_log = run_cmd(
            [
                "python",
                "/home/zzh/Ming2/quantize_zimage_dit.py",
                "--model-path",
                args.model_path,
                "--save-path",
                str(quant_save),
                "--config-path",
                str(config_path),
            ],
            cwd=args.code_path,
        )

        result = {
            "name": name,
            "config_path": str(config_path),
            "quantized_dit_path": str(quant_save),
            "dropped_block": variant["dropped_block"],
            "kept_blocks": variant["kept_blocks"],
            "cases": [],
        }

        for case in CASES:
            output_path = output_dir / f"{name}_{case['output_suffix']}.png"
            print(f"[infer] {name} {case['output_suffix']}")
            infer_log = run_cmd(
                [
                    "python",
                    "/home/zzh/Ming2/test_infer_imagegen_npu.py",
                    "--model-path",
                    args.model_path,
                    "--code-path",
                    args.code_path,
                    "--prompt",
                    case["prompt"],
                    "--condition-input",
                    case["condition_input"],
                    "--quantized-dit-path",
                    str(quant_save),
                    "--output",
                    str(output_path),
                    "--tensor-parallel-devices",
                    "1",
                    "--device-ids",
                    str(args.device_id),
                    "--dit-impl",
                    "fusion",
                    "--num-runs",
                    "1",
                    "--seed",
                    "42",
                    "--imagegen-stage-timing",
                    "on",
                ],
                cwd=args.code_path,
            )
            metrics = compute_metrics(Path(case["baseline_image"]), output_path)
            result["cases"].append(
                {
                    "id": case["id"],
                    "output_path": str(output_path),
                    "run_1_generate_sec": extract_generate_time(infer_log),
                    **metrics,
                }
            )

        summary["results"].append(result)
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
