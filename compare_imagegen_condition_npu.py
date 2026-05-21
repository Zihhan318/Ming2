import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
from test_infer_imagegen_npu import (
    build_imagegen_device_map,
    build_messages,
    configure_npu_runtime,
    ensure_local_hf_cache,
    get_attn_implementation,
    get_runtime_device,
    move_inputs_to_device,
    validate_imagegen_assets,
    validate_reference_image,
)
from test_infer_npu import load_quantized_llm_weights, parse_device_ids, set_runtime_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare image-generation LLM condition embeddings between original and quantized models."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--quantized-llm-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-devices", type=int, default=1)
    parser.add_argument("--device-ids")
    parser.add_argument("--output-json")
    return parser.parse_args()


def sync_device(device: torch.device):
    if device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def build_runtime(args):
    ensure_local_hf_cache()
    configure_npu_runtime()
    validate_imagegen_assets(args.model_path)
    validate_reference_image(args.image)

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

    return device, attn_implementation, use_device_map, tp_device_ids, device_map


def build_inputs(args, processor: AutoProcessor, input_device: torch.device):
    messages = build_messages(args)
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, _, _ = processor.process_vision_info(messages)
    ref_image_inputs = processor.process_reference_vision_info(messages) if args.image else None
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        image_gen_ref_images=ref_image_inputs,
    )
    inputs["image_gen_text"] = [args.prompt]
    return move_inputs_to_device(inputs, input_device)


def extract_condition_pair(args, quantized_llm_path: str | None):
    device, attn_implementation, use_device_map, tp_device_ids, device_map = build_runtime(args)
    input_device = torch.device(f"{device.type}:{tp_device_ids[0]}") if use_device_map else device

    processor = AutoProcessor.from_pretrained(args.code_path, trust_remote_code=True)
    inputs = build_inputs(args, processor, input_device)

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

    replaced_layers = 0
    if quantized_llm_path:
        replaced_layers = load_quantized_llm_weights(
            model,
            Path(quantized_llm_path).resolve(),
            device,
        )

    model.eval()

    infer_start = time.time()
    with torch.no_grad():
        condition_embeds, negative_condition_embeds = model.generate(
            **inputs,
            image_gen=True,
            image_gen_seed=args.seed,
            image_gen_only_extract_hidden_states=True,
            image_gen_profile_print=False,
        )
    sync_device(device)
    infer_elapsed = time.time() - infer_start

    result = {
        "condition_embeds": condition_embeds.detach().float().cpu(),
        "negative_condition_embeds": negative_condition_embeds.detach().float().cpu(),
        "metadata": {
            "device": str(device),
            "use_device_map": use_device_map,
            "device_ids": tp_device_ids,
            "replaced_layers": replaced_layers,
            "load_elapsed_sec": load_elapsed,
            "infer_elapsed_sec": infer_elapsed,
        },
    }

    del model
    del inputs
    del processor
    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def tensor_metrics(lhs: torch.Tensor, rhs: torch.Tensor):
    diff = lhs - rhs
    lhs_flat = lhs.reshape(-1)
    rhs_flat = rhs.reshape(-1)
    cosine = F.cosine_similarity(lhs_flat.unsqueeze(0), rhs_flat.unsqueeze(0)).item()
    lhs_norm = torch.linalg.vector_norm(lhs_flat).item()
    rhs_norm = torch.linalg.vector_norm(rhs_flat).item()
    diff_norm = torch.linalg.vector_norm(diff.reshape(-1)).item()
    denom = lhs_norm if lhs_norm != 0 else 1.0
    return {
        "shape": list(lhs.shape),
        "mean_abs_diff": diff.abs().mean().item(),
        "max_abs_diff": diff.abs().max().item(),
        "mse": (diff * diff).mean().item(),
        "cosine_similarity": cosine,
        "lhs_norm": lhs_norm,
        "rhs_norm": rhs_norm,
        "relative_l2": diff_norm / denom,
    }


def main():
    args = parse_args()
    original = extract_condition_pair(args, quantized_llm_path=None)
    quantized = extract_condition_pair(args, quantized_llm_path=args.quantized_llm_path)

    summary = {
        "prompt": args.prompt,
        "seed": args.seed,
        "original": original["metadata"],
        "quantized": quantized["metadata"],
        "condition_embeds": tensor_metrics(original["condition_embeds"], quantized["condition_embeds"]),
        "negative_condition_embeds": tensor_metrics(
            original["negative_condition_embeds"],
            quantized["negative_condition_embeds"],
        ),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
