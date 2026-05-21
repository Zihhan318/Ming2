import argparse
import json
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
from test_infer_npu import load_quantized_llm_weights, log_stage, parse_device_ids, set_runtime_device


def parse_args():
    parser = argparse.ArgumentParser(description="Batch smoke test for image generation on NPU.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json-output")
    parser.add_argument("--quantized-llm-path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-devices", type=int, default=1)
    parser.add_argument("--device-ids")
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


def prepare_runtime(args):
    ensure_local_hf_cache()
    configure_npu_runtime()
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


def main():
    args = parse_args()
    cases = load_cases(args.cases_file)
    for case in cases:
        validate_reference_image(case.get("image"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, attn_implementation, use_device_map, tp_device_ids, requested_device_ids, device_map = prepare_runtime(args)
    input_device = torch.device(f"{device.type}:{tp_device_ids[0]}") if use_device_map else device

    log_stage("loading imagegen model for smoke test")
    load_start = time.time()
    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        load_image_gen=True,
        load_talker=False,
        device_map=device_map,
    )
    if not use_device_map:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(dtype=torch.bfloat16)
    load_elapsed = time.time() - load_start

    replaced = 0
    if args.quantized_llm_path:
        log_stage("applying quantized llm weights for batch imagegen smoke test")
        replaced = load_quantized_llm_weights(model, Path(args.quantized_llm_path).resolve(), device)
        log_stage(f"quantized llm ready replaced_layers={replaced}")

    model.eval()
    processor = AutoProcessor.from_pretrained(args.code_path, trust_remote_code=True)

    print(f"runtime_device: {device}")
    print(f"device_map: {device_map}")
    print(f"quantized_llm_path: {args.quantized_llm_path}")
    print(f"case_count: {len(cases)}")
    print(f"model_load_elapsed: {load_elapsed:.2f}s")

    results = []
    for idx, case in enumerate(cases, start=1):
        print(f"\n[{idx}] case_id: {case['id']}")
        print(f"[{idx}] prompt: {case['prompt']}")
        if case.get("image"):
            print(f"[{idx}] reference_image: {case['image']}")
        inputs = build_inputs(processor, case, input_device)
        start_time = time.time()
        with torch.no_grad():
            image = model.generate(
                **inputs,
                image_gen=True,
                image_gen_seed=args.seed,
                image_gen_profile_print=False,
            )
        elapsed = time.time() - start_time
        output_path = output_dir / f"{case['id']}.png"
        image.save(output_path)
        print(f"[{idx}] elapsed: {elapsed:.2f}s")
        print(f"[{idx}] output: {output_path}")
        results.append(
            {
                "index": idx,
                "id": case["id"],
                "prompt": case["prompt"],
                "image": case.get("image"),
                "elapsed": elapsed,
                "output": str(output_path),
            }
        )

    if args.json_output:
        payload = {
            "model_path": args.model_path,
            "quantized_llm_path": args.quantized_llm_path,
            "runtime_device": str(device),
            "device_ids": tp_device_ids or requested_device_ids,
            "model_load_elapsed": load_elapsed,
            "replaced_layers": replaced,
            "results": results,
        }
        Path(args.json_output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
