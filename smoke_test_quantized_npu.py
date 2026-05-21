import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoProcessor

from configuration_bailingmm2 import BailingMM2Config
from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
from test_infer_npu import (
    build_layer_split_device_map,
    detect_modalities,
    disable_allocator_warmup,
    ensure_local_hf_cache,
    get_attn_implementation,
    get_runtime_device,
    load_quantized_llm_weights,
    log_stage,
    move_inputs_to_device,
    parse_device_ids,
    set_runtime_device,
    sync_device,
)


DEFAULT_PROMPTS = [
    "你好，请用一句话介绍你自己。",
    "用一句话解释什么是光合作用。",
    "北京和上海，哪个是中国的首都？",
    "请写一首四句、每句不超过10个字的春天短诗。",
]


def build_messages(prompt: str):
    return [{"role": "HUMAN", "content": [{"type": "text", "text": prompt}]}]


def load_prompts(args) -> list[str]:
    if args.prompt:
        return args.prompt
    if args.prompts_file:
        return [
            line.strip()
            for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return DEFAULT_PROMPTS


def main(args):
    prompts = load_prompts(args)
    ensure_local_hf_cache()
    disable_allocator_warmup()
    config = BailingMM2Config.from_pretrained(args.model_path)
    device = get_runtime_device()
    attn_implementation = get_attn_implementation(device)

    use_device_map = args.tensor_parallel_devices and args.tensor_parallel_devices > 1
    requested_device_ids = parse_device_ids(args.device_ids, args.tensor_parallel_devices)
    device_map = None
    tp_device_ids = requested_device_ids
    if use_device_map:
        if tp_device_ids is None:
            tp_device_ids = list(range(args.tensor_parallel_devices))
        device = torch.device(f"{device.type}:{tp_device_ids[0]}")
        set_runtime_device(device)
        device_map = build_layer_split_device_map(
            num_layers=config.llm_config.num_hidden_layers,
            device_ids=tp_device_ids,
            device_type=device.type,
        )
    elif requested_device_ids:
        device = torch.device(f"{device.type}:{requested_device_ids[0]}")
        set_runtime_device(device)

    model_load_device_map = device_map
    if not use_device_map and device.type != "cpu":
        model_load_device_map = {"": str(device)}

    log_stage("loading model for smoke test")
    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        load_multimodal=True,
        load_image_gen=False,
        load_talker=False,
        device_map=model_load_device_map,
        low_cpu_mem_usage=True,
    )
    if model_load_device_map is None:
        model = model.to(device=device, dtype=torch.bfloat16)

    if args.quantized_llm_path:
        log_stage("applying quantized llm weights for smoke test")
        replaced = load_quantized_llm_weights(
            model,
            Path(args.quantized_llm_path).resolve(),
            device,
        )
        log_stage(f"quantized llm ready replaced_layers={replaced}")

    model.eval()
    processor = AutoProcessor.from_pretrained(args.code_path, trust_remote_code=True)

    input_device = torch.device(f"{device.type}:{tp_device_ids[0]}") if use_device_map else device

    print(f"device: {device}")
    print(f"device_map: {device_map}")
    print(f"quantized_llm_path: {args.quantized_llm_path}")
    print(f"prompt_count: {len(prompts)}")

    results = []

    for idx, prompt in enumerate(prompts, start=1):
        messages = build_messages(prompt)
        modalities = detect_modalities(args)
        text = processor.apply_chat_template(
            messages,
            sys_prompt_exp=args.sys_prompt_exp,
            use_cot_system_prompt=args.use_cot_system_prompt,
        )
        image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
            audio_kwargs={"use_whisper_encoder": True},
        )
        inputs = move_inputs_to_device(inputs, input_device)

        sync_device(input_device)
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                eos_token_id=processor.gen_terminator,
                num_logits_to_keep=1,
            )
        sync_device(input_device)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        elapsed = time.time() - start_time

        print(f"\n[{idx}] prompt: {prompt}")
        print(f"[{idx}] elapsed: {elapsed:.2f}s")
        print(f"[{idx}] output: {output_text}")
        results.append(
            {
                "index": idx,
                "prompt": prompt,
                "elapsed": elapsed,
                "output": output_text,
            }
        )

    if args.json_output:
        payload = {
            "model_path": args.model_path,
            "quantized_llm_path": args.quantized_llm_path,
            "device": str(device),
            "device_map": device_map,
            "prompt_count": len(prompts),
            "results": results,
        }
        Path(args.json_output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple smoke-test prompts against the quantized Ming model.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--quantized-llm-path")
    parser.add_argument("--prompt", action="append", help="Repeatable prompt. If omitted, built-in prompts are used.")
    parser.add_argument("--prompts-file")
    parser.add_argument("--json-output")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sys-prompt-exp")
    parser.add_argument("--use-cot-system-prompt", action="store_true")
    parser.add_argument("--tensor-parallel-devices", type=int, default=1)
    parser.add_argument("--device-ids")
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--audio")
    parser.add_argument("--task", default="text")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
