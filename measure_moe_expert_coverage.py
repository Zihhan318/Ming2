#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import yaml

from msmodelslim.app.base import DeviceType
from msmodelslim.cli.naive_quantization.__main__ import get_dataset_dir
from msmodelslim.infra.dataset_loader import FileDatasetLoader

from ming_quantization_common import create_model, resolve_quant_config_path


def parse_args():
    parser = argparse.ArgumentParser(description="Measure MoE expert routing coverage on a calibration dataset.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model-type", default="Ming-flash-omni-2.0-LLM")
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--report-path")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return parser.parse_args()


def _device_name(device: DeviceType) -> str:
    return "npu" if device == DeviceType.NPU else "cpu"


def _extract_expert_meta(model) -> tuple[int, int]:
    llm_config = getattr(getattr(model, "model", None), "config", None)
    if llm_config is None:
        raise RuntimeError("Failed to locate LLM config for expert coverage measurement.")
    num_experts = int(getattr(llm_config, "num_experts", 0))
    top_k = int(getattr(llm_config, "num_experts_per_tok", 0))
    if num_experts <= 0 or top_k <= 0:
        raise RuntimeError("Model config does not expose valid num_experts/num_experts_per_tok.")
    return num_experts, top_k


def _load_dataset_from_config(config_path: Path) -> tuple[list[str], str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_id = config.get("spec", {}).get("calib_dataset")
    if not dataset_id:
        raise RuntimeError(f"No spec.calib_dataset found in {config_path}.")
    dataset_loader = FileDatasetLoader(get_dataset_dir())
    dataset = dataset_loader.get_dataset_by_name(dataset_id)
    return dataset, str(dataset_id)


def measure_expert_coverage(
    model_path: Path,
    config_path: Path,
    model_type: str,
    device: DeviceType,
    trust_remote_code: bool,
    report_path: Path | None = None,
    max_samples: int | None = 256,
):
    resolved_config_path = resolve_quant_config_path(config_path)
    dataset, dataset_id = _load_dataset_from_config(resolved_config_path)
    if max_samples is not None:
        dataset = dataset[:max_samples]

    adapter = create_model(model_type, model_path, trust_remote_code)
    model = adapter.load_model(device=device)
    model.eval()
    tokenizer = adapter.tokenizer

    num_experts, top_k = _extract_expert_meta(model)
    layer_hit_counts: list[torch.Tensor] = []
    total_tokens = 0
    total_assignments = 0
    target_device = _device_name(device)

    with torch.no_grad():
        for index, text in enumerate(dataset, start=1):
            inputs = tokenizer(text, return_tensors="pt", padding=True).to(target_device)
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_router_logits=True,
                return_dict=True,
                use_cache=False,
                num_logits_to_keep=1,
            )
            router_payloads = getattr(outputs, "router_logits", ()) or ()
            if not layer_hit_counts:
                layer_hit_counts = [torch.zeros(num_experts, dtype=torch.long) for _ in range(len(router_payloads))]

            for layer_idx, payload in enumerate(router_payloads):
                if payload is None or not isinstance(payload, tuple) or len(payload) < 2:
                    continue
                _, topk_idx = payload[:2]
                counts = torch.bincount(topk_idx.reshape(-1).cpu(), minlength=num_experts)
                layer_hit_counts[layer_idx] += counts
                total_assignments += int(topk_idx.numel())

            total_tokens += int(inputs["input_ids"].numel())
            if index % 25 == 0 or index == len(dataset):
                print(f"coverage progress: {index}/{len(dataset)} samples")

    layer_reports = []
    coverage_ratios = []
    uncovered_experts = 0
    for layer_idx, counts in enumerate(layer_hit_counts):
        covered = int((counts > 0).sum().item())
        coverage_ratio = covered / max(num_experts, 1)
        coverage_ratios.append(coverage_ratio)
        never_hit = [idx for idx, value in enumerate(counts.tolist()) if value == 0]
        uncovered_experts += len(never_hit)
        layer_reports.append(
            {
                "layer_idx": layer_idx,
                "covered_experts": covered,
                "coverage_ratio": coverage_ratio,
                "never_hit_experts": never_hit,
                "hit_counts": counts.tolist(),
            }
        )

    report = {
        "model_path": str(model_path),
        "config_path": str(resolved_config_path),
        "dataset_id": dataset_id,
        "samples_evaluated": len(dataset),
        "tokens_evaluated": total_tokens,
        "assignment_events": total_assignments,
        "num_layers": len(layer_reports),
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "summary": {
            "avg_layer_coverage_ratio": sum(coverage_ratios) / max(len(coverage_ratios), 1),
            "min_layer_coverage_ratio": min(coverage_ratios) if coverage_ratios else 0.0,
            "fully_covered_layers": sum(1 for ratio in coverage_ratios if ratio >= 1.0),
            "layers_with_gaps": sum(1 for ratio in coverage_ratios if ratio < 1.0),
            "uncovered_expert_slots": uncovered_experts,
        },
        "layers": layer_reports,
    }

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"expert coverage report: {report_path}")

    print(
        "expert coverage summary: "
        f"samples={report['samples_evaluated']} "
        f"avg_ratio={report['summary']['avg_layer_coverage_ratio']:.3f} "
        f"min_ratio={report['summary']['min_layer_coverage_ratio']:.3f} "
        f"layers_with_gaps={report['summary']['layers_with_gaps']} "
        f"uncovered_slots={report['summary']['uncovered_expert_slots']}"
    )
    return report


def main():
    args = parse_args()
    report_path = Path(args.report_path).resolve() if args.report_path else None
    measure_expert_coverage(
        model_path=Path(args.model_path).resolve(),
        config_path=Path(args.config_path).resolve(),
        model_type=args.model_type,
        device=DeviceType(args.device),
        trust_remote_code=args.trust_remote_code,
        report_path=report_path,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
