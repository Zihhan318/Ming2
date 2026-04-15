import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_HF_CACHE = Path(__file__).resolve().parent / ".hf-cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE))
os.environ.setdefault("HF_MODULES_CACHE", str(DEFAULT_HF_CACHE / "modules"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

import torch
import transformers.modeling_utils as modeling_utils
from transformers import AutoProcessor

from configuration_bailingmm2 import BailingMM2Config
from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration


UNDERSTANDING_TASKS = ("auto", "text", "image", "video", "audio", "multimodal")


def ensure_local_hf_cache():
    cache_root = Path(os.environ["HF_HOME"])
    modules_root = cache_root / "modules"
    modules_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(modules_root)


def disable_allocator_warmup():
    def _noop(*args, **kwargs):
        return None

    modeling_utils.caching_allocator_warmup = _noop


def get_runtime_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        current_idx = torch.npu.current_device() if hasattr(torch.npu, "current_device") else 0
        return torch.device(f"npu:{current_idx}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def get_attn_implementation(device: torch.device) -> str:
    return "eager"


def log_stage(message: str):
    print(f"[stage] {message}", file=sys.stderr, flush=True)


def sync_device(device: torch.device):
    if device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def parse_device_ids(device_ids_arg: str | None, requested_devices: int) -> list[int] | None:
    if device_ids_arg is None:
        return None
    device_ids = [int(item.strip()) for item in device_ids_arg.split(",") if item.strip()]
    if not device_ids:
        raise ValueError("--device-ids cannot be empty when provided.")
    if requested_devices > 1 and len(device_ids) != requested_devices:
        raise ValueError(
            f"--tensor-parallel-devices={requested_devices} requires exactly {requested_devices} "
            f"entries in --device-ids, got {len(device_ids)}."
        )
    return device_ids


def set_runtime_device(device: torch.device):
    if device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(device)
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)


def build_layer_split_device_map(num_layers: int, device_ids: list[int]):
    num_devices = len(device_ids)
    if num_devices < 1:
        raise ValueError(f"Expected at least one device, got {num_devices=}")
    if num_devices > num_layers:
        raise ValueError(f"Expected num_devices <= num_layers, got {num_layers=} {num_devices=}")

    device_map = {}
    base_layers = num_layers // num_devices
    extra_layers = num_layers % num_devices
    layer_idx = 0
    primary_device = device_ids[0]
    for index, device_id in enumerate(device_ids):
        layer_count = base_layers + (1 if index < extra_layers else 0)
        for _ in range(layer_count):
            device_map[f"model.model.layers.{layer_idx}"] = device_id
            layer_idx += 1

    device_map["vision"] = primary_device
    device_map["audio"] = primary_device
    device_map["linear_proj"] = primary_device
    device_map["linear_proj_audio"] = primary_device
    device_map["model.model.word_embeddings"] = primary_device
    device_map["model.model.word_embeddings.weight"] = primary_device
    device_map["model.model.norm"] = primary_device
    device_map["model.model.norm.weight"] = primary_device
    device_map["model.lm_head"] = primary_device
    device_map["model.lm_head.weight"] = primary_device
    return device_map


def move_inputs_to_device(inputs, device: torch.device):
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.is_floating_point() and key in {"pixel_values", "pixel_values_videos", "audio_feats"}:
            inputs[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            inputs[key] = value.to(device=device)
    return inputs


def build_messages(args):
    content = []
    if args.image:
        content.append({"type": "image", "image": args.image})
    if args.video:
        video_item = {"type": "video", "video": args.video}
        if args.max_frames is not None:
            video_item["max_frames"] = args.max_frames
        if args.video_sample is not None:
            video_item["sample"] = args.video_sample
        content.append(video_item)
    if args.audio:
        content.append({"type": "audio", "audio": args.audio})
    content.append({"type": "text", "text": args.prompt})
    return [{"role": "HUMAN", "content": content}]


def detect_modalities(args):
    modalities = []
    if args.image:
        modalities.append("image")
    if args.video:
        modalities.append("video")
    if args.audio:
        modalities.append("audio")
    return modalities


def validate_args(args):
    modalities = detect_modalities(args)
    if args.task == "text" and modalities:
        raise ValueError("--task text does not accept --image/--video/--audio.")
    if args.task in {"image", "video", "audio"}:
        if modalities != [args.task]:
            raise ValueError(f"--task {args.task} requires exactly one matching input via --{args.task}.")
    if args.task == "multimodal" and len(modalities) < 2:
        raise ValueError("--task multimodal requires at least two inputs chosen from --image/--video/--audio.")
    if args.task == "auto" and not args.prompt:
        raise ValueError("--prompt is required.")
    return modalities


def run_infer(args):
    modalities = validate_args(args)
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
        )
    elif requested_device_ids:
        device = torch.device(f"{device.type}:{requested_device_ids[0]}")
        set_runtime_device(device)

    log_stage(
        f"runtime_device={device} use_device_map={use_device_map} "
        f"tp_devices={args.tensor_parallel_devices} device_ids={tp_device_ids or requested_device_ids}"
    )
    log_stage("loading model")
    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        load_image_gen=False,
        load_talker=False,
        device_map=device_map,
    )
    if not use_device_map:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    log_stage("model ready")

    log_stage("loading processor")
    processor = AutoProcessor.from_pretrained(args.code_path, trust_remote_code=True)
    log_stage("processor ready")

    messages = build_messages(args)
    log_stage(f"building prompt and multimodal inputs: modalities={modalities or ['text']}")
    text = processor.apply_chat_template(
        messages,
        sys_prompt_exp=args.sys_prompt_exp,
        use_cot_system_prompt=args.use_cot_system_prompt,
    )
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    log_stage("tokenizing inputs")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        audio_kwargs={"use_whisper_encoder": True},
    )
    input_device = torch.device(f"{device.type}:{tp_device_ids[0]}") if use_device_map else device
    inputs = move_inputs_to_device(inputs, input_device)
    log_stage(
        "inputs ready "
        f"input_ids={tuple(inputs.input_ids.shape)} "
        f"attention_mask={tuple(inputs.attention_mask.shape)} "
        f"target_device={input_device}"
    )

    sync_device(input_device)
    start_time = time.time()
    log_stage(f"starting generate max_new_tokens={args.max_new_tokens}")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            num_logits_to_keep=1,
        )
    sync_device(input_device)
    log_stage("generate finished")

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    log_stage("decoding output")
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    log_stage("decode finished")

    print(f"device: {device}")
    print(f"device_map: {device_map}")
    print(f"task: {args.task}")
    print(f"modalities: {modalities or ['text']}")
    print(f"attn_implementation: {attn_implementation}")
    print(f"elapsed: {time.time() - start_time:.2f}s")
    print(output_text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Understanding-only inference entry for Ascend/CUDA/CPU."
    )
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--code-path", default=".")
    parser.add_argument(
        "--task",
        choices=UNDERSTANDING_TASKS,
        default="auto",
        help="Understanding task only. Use dedicated scripts for image generation or talker.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--audio")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--video-sample", default="uniform")
    parser.add_argument("--sys-prompt-exp")
    parser.add_argument("--use-cot-system-prompt", action="store_true")
    parser.add_argument("--tensor-parallel-devices", type=int, default=1)
    parser.add_argument(
        "--device-ids",
        help="Comma-separated physical device ids. Example: 0,1,2,3,4,5,7,8",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_infer(parse_args())
