#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoProcessor

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
from test_infer_imagegen_npu import (
    build_imagegen_device_map,
    configure_npu_runtime,
    ensure_local_hf_cache,
    get_attn_implementation,
    get_runtime_device,
    move_inputs_to_device,
    validate_imagegen_assets,
    validate_reference_image,
)
from test_infer_npu import disable_allocator_warmup, load_quantized_llm_weights, log_stage, parse_device_ids, set_runtime_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build DiT-only calibration captures from representative image-generation cases."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--quantized-llm-path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-gen-steps", type=int, default=30)
    parser.add_argument("--capture-steps", default="0,last")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--tensor-parallel-devices", type=int, default=1)
    parser.add_argument("--device-ids")
    parser.add_argument(
        "--skip-dit-capture",
        action="store_true",
        help="Only extract/save condition embeddings and skip the DiT sampling pass.",
    )
    parser.add_argument(
        "--save-condition-embeds",
        action="store_true",
        help="Also save per-case LLM condition embeddings alongside the DiT captures.",
    )
    return parser.parse_args()


def load_cases(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("cases file must be a non-empty JSON list")
    return data


def build_messages(case: dict):
    content = []
    if case.get("image"):
        content.append({"type": "image", "image": case["image"]})
    content.append({"type": "text", "text": case["prompt"]})
    return [{"role": "HUMAN", "content": content}]


def build_inputs(processor, case: dict, input_device: torch.device):
    messages = build_messages(case)
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, _, _ = processor.process_vision_info(messages)
    ref_image_inputs = processor.process_reference_vision_info(messages) if case.get("image") else None
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        image_gen_ref_images=ref_image_inputs,
    )
    inputs["image_gen_text"] = [case["prompt"]]
    return move_inputs_to_device(inputs, input_device)


def prepare_runtime(args):
    ensure_local_hf_cache()
    configure_npu_runtime()
    disable_allocator_warmup()
    validate_imagegen_assets(args.model_path)

    device = get_runtime_device()
    attn_implementation = get_attn_implementation(device)
    use_device_map = args.tensor_parallel_devices and args.tensor_parallel_devices > 1
    requested_device_ids = parse_device_ids(args.device_ids, args.tensor_parallel_devices)
    tp_device_ids = requested_device_ids
    device_map = None

    if use_device_map:
        if tp_device_ids is None:
            tp_device_ids = list(range(args.tensor_parallel_devices))
        device = torch.device(f"{device.type}:{tp_device_ids[0]}")
        set_runtime_device(device)
        device_map = build_imagegen_device_map(
            num_layers=32,
            device_ids=tp_device_ids,
            device_type=device.type,
        )
    elif requested_device_ids:
        device = torch.device(f"{device.type}:{requested_device_ids[0]}")
        set_runtime_device(device)

    return device, attn_implementation, use_device_map, tp_device_ids, requested_device_ids, device_map


def main():
    args = parse_args()
    cases = load_cases(args.cases_file)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    for case in cases:
        validate_reference_image(case.get("image"))

    output_dir = Path(args.output_dir)
    captures_dir = output_dir / "captures"
    conditions_dir = output_dir / "conditions"
    output_dir.mkdir(parents=True, exist_ok=True)
    captures_dir.mkdir(parents=True, exist_ok=True)
    if args.save_condition_embeds:
        conditions_dir.mkdir(parents=True, exist_ok=True)

    device, attn_implementation, use_device_map, tp_device_ids, requested_device_ids, device_map = prepare_runtime(args)
    input_device = torch.device(f"{device.type}:{tp_device_ids[0]}") if use_device_map else device

    processor = AutoProcessor.from_pretrained(args.code_path, trust_remote_code=True)

    log_stage("loading imagegen model for DiT calibration capture")
    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        load_image_gen=True,
        load_image_gen_diffusion=not args.skip_dit_capture,
        load_talker=False,
        device_map=device_map,
    )
    if not use_device_map:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(dtype=torch.bfloat16)

    replaced = 0
    if args.quantized_llm_path:
        log_stage("applying quantized llm weights before DiT calibration capture")
        replaced = load_quantized_llm_weights(model, Path(args.quantized_llm_path).resolve(), device)
        log_stage(f"quantized llm ready replaced_layers={replaced}")

    model.eval()

    previous_env = {
        "MING_DIT_CALIB_OUTPUT_DIR": os.environ.get("MING_DIT_CALIB_OUTPUT_DIR"),
        "MING_DIT_CALIB_CAPTURE_STEPS": os.environ.get("MING_DIT_CALIB_CAPTURE_STEPS"),
        "MING_DIT_CALIB_CASE_ID": os.environ.get("MING_DIT_CALIB_CASE_ID"),
        "MING_DIT_CALIB_ALLOW_OVERWRITE": os.environ.get("MING_DIT_CALIB_ALLOW_OVERWRITE"),
    }
    os.environ["MING_DIT_CALIB_OUTPUT_DIR"] = str(captures_dir)
    os.environ["MING_DIT_CALIB_CAPTURE_STEPS"] = args.capture_steps
    os.environ["MING_DIT_CALIB_ALLOW_OVERWRITE"] = "1"

    summary = {
        "model_path": args.model_path,
        "quantized_llm_path": args.quantized_llm_path,
        "runtime_device": str(device),
        "device_ids": tp_device_ids or requested_device_ids,
        "image_gen_steps": args.image_gen_steps,
        "capture_steps": args.capture_steps,
        "replaced_layers": replaced,
        "cases": [],
    }

    try:
        for idx, case in enumerate(cases, start=1):
            case_id = str(case.get("id") or f"case_{idx:03d}")
            os.environ["MING_DIT_CALIB_CASE_ID"] = case_id
            inputs = build_inputs(processor, case, input_device)

            print(f"[{idx}] case_id: {case_id}")
            print(f"[{idx}] prompt: {case['prompt']}")
            if case.get("image"):
                print(f"[{idx}] reference_image: {case['image']}")

            condition_start = time.time()
            with torch.no_grad():
                condition_embeds, negative_condition_embeds = model.generate(
                    **inputs,
                    image_gen=True,
                    image_gen_seed=args.seed,
                    image_gen_only_extract_hidden_states=True,
                    image_gen_profile_print=False,
                )
            condition_elapsed = time.time() - condition_start

            if args.save_condition_embeds:
                torch.save(
                    {
                        "case_id": case_id,
                        "prompt": case["prompt"],
                        "seed": args.seed,
                        "condition_embeds": condition_embeds.detach().cpu(),
                        "negative_condition_embeds": negative_condition_embeds.detach().cpu(),
                    },
                    conditions_dir / f"{case_id}.pt",
                )

            sample_elapsed = 0.0
            capture_files = []
            if not args.skip_dit_capture:
                sample_start = time.time()
                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        image_gen=True,
                        image_gen_seed=args.seed,
                        image_gen_steps=args.image_gen_steps,
                        image_gen_condition_embeds=condition_embeds,
                        image_gen_negative_condition_embeds=negative_condition_embeds,
                        image_gen_profile_print=False,
                    )
                sample_elapsed = time.time() - sample_start
                capture_files = sorted(str(path) for path in captures_dir.glob(f"{case_id}_step*.pt"))
            print(f"[{idx}] condition_elapsed: {condition_elapsed:.2f}s")
            print(f"[{idx}] dit_sampling_elapsed: {sample_elapsed:.2f}s")
            print(f"[{idx}] captures: {len(capture_files)}")

            summary["cases"].append(
                {
                    "index": idx,
                    "id": case_id,
                    "prompt": case["prompt"],
                    "image": case.get("image"),
                    "condition_elapsed_sec": condition_elapsed,
                    "dit_sampling_elapsed_sec": sample_elapsed,
                    "condition_shape": list(condition_embeds.shape),
                    "negative_condition_shape": list(negative_condition_embeds.shape),
                    "capture_files": capture_files,
                }
            )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary_json: {summary_path}")
    else:
        default_summary = output_dir / "summary.json"
        default_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"summary_json: {default_summary}")


if __name__ == "__main__":
    main()
