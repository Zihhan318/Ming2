import argparse
import gc
import json
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
import torch.nn.functional as F
import transformers.modeling_utils as modeling_utils
from accelerate.hooks import add_hook_to_module, remove_hook_from_module
from safetensors import safe_open
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import QuantConfig
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.quantizer import LinearQuantizerTimestep
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.timestep_utils import load_quant_weight
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


def build_layer_split_device_map(num_layers: int, device_ids: list[int], device_type: str):
    num_devices = len(device_ids)
    if num_devices < 1:
        raise ValueError(f"Expected at least one device, got {num_devices=}")
    if num_devices > num_layers:
        raise ValueError(f"Expected num_devices <= num_layers, got {num_layers=} {num_devices=}")

    device_map = {}
    layer_boundaries = [
        ((idx + 1) * num_layers) // num_devices
        for idx in range(num_devices)
    ]
    primary_device = f"{device_type}:{device_ids[0]}"
    for layer_idx in range(num_layers):
        device_slot = 0
        while device_slot < len(layer_boundaries) and layer_idx >= layer_boundaries[device_slot]:
            device_slot += 1
        target_device = f"{device_type}:{device_ids[min(device_slot, num_devices - 1)]}"
        device_map[f"model.model.layers.{layer_idx}"] = target_device

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
    device_map[f"model.model.layers.{num_layers - 1}"] = primary_device
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


def build_quantized_llm_cfg(model_quant_type: str) -> QuantConfig:
    quant_type_upper = model_quant_type.upper()
    if quant_type_upper == "W8A8":
        w_bit, a_bit = 8, 8
    elif quant_type_upper == "W8A16":
        w_bit, a_bit = 8, 16
    else:
        raise ValueError(f"Unsupported quantized LLM type: {model_quant_type}")
    cfg = QuantConfig(w_bit=w_bit, a_bit=a_bit, mm_tensor=False)
    cfg.use_timestep_quant = True
    cfg.max_dynamic_step = 0
    return cfg


class W8A16LinearRuntime(torch.nn.Module):
    def __init__(self, linear: torch.nn.Module):
        super().__init__()
        self.target_device = linear.weight.device
        self.target_dtype = linear.weight.dtype
        if linear.bias is None:
            self.register_buffer("bias", None, persistent=False)
        else:
            self.register_buffer("bias", linear.bias.detach().to(self.target_device, dtype=self.target_dtype))
        self.register_buffer("deq_weight", torch.empty(0, device="cpu", dtype=self.target_dtype), persistent=False)

    def load_layer_params(self, params_to_load: dict[str, torch.Tensor], device: torch.device):
        required_keys = ["weight", "weight_scale", "weight_offset"]
        missing_keys = [key for key in required_keys if key not in params_to_load]
        if missing_keys:
            raise KeyError(f"Required keys {missing_keys} not found in state_dict")

        quant_weight = params_to_load["weight"].to(device=device, dtype=torch.float32)
        weight_scale = params_to_load["weight_scale"].to(device=device, dtype=torch.float32)
        weight_offset = params_to_load["weight_offset"].to(device=device, dtype=torch.float32)

        ori_weight_shape = quant_weight.shape
        if len(ori_weight_shape) != 2:
            raise ValueError("original weight shape is not valid for W8A16 runtime.")
        if len(weight_scale.shape) != 2:
            raise ValueError("weight_scale shape is not valid for W8A16 runtime.")

        if weight_scale.shape[1] != 1:
            channel_num = ori_weight_shape[1]
            if weight_scale.shape[1] == 0:
                raise ZeroDivisionError("weight_scale shape[1] is 0, please check quant params.")
            group_size = int(channel_num / weight_scale.shape[1])
            quant_weight = quant_weight.reshape(-1, group_size)
            weight_offset = weight_offset.reshape(-1, 1)
            weight_scale = weight_scale.reshape(-1, 1)
            deq_weight = (quant_weight - weight_offset) * weight_scale
            deq_weight = deq_weight.reshape(ori_weight_shape)
        else:
            deq_weight = (quant_weight - weight_offset) * weight_scale

        self.deq_weight = deq_weight.to(dtype=self.target_dtype)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.deq_weight, self.bias)


def resolve_named_module(root_module: torch.nn.Module, qualified_name: str):
    current = root_module
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current, parts[-1]


def set_named_module(root_module: torch.nn.Module, qualified_name: str, new_module: torch.nn.Module):
    parent, leaf = resolve_named_module(root_module, qualified_name)
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def load_quantized_linear_names(quantized_llm_path: Path) -> tuple[str, set[str], dict[str, str]]:
    quant_description = json.loads((quantized_llm_path / "quant_model_description.json").read_text())
    model_quant_type = str(quant_description.get("model_quant_type", "W8A8")).upper()
    index_candidates = [
        quantized_llm_path / f"quant_model_weight_{model_quant_type.lower()}.safetensors.index.json",
        *sorted(quantized_llm_path.glob("quant_model_weight_*.safetensors.index.json")),
    ]
    index_path = next((candidate for candidate in index_candidates if candidate.exists()), None)
    if index_path is None:
        raise FileNotFoundError(
            f"No quantized safetensors index was found under {quantized_llm_path} for {model_quant_type}."
        )
    index_data = json.loads(index_path.read_text())
    quantized_linear_names = {
        name[: -len(".weight")]
        for name, quant_type in quant_description.items()
        if name.endswith(".weight") and str(quant_type).upper() == model_quant_type
    }
    return model_quant_type, quantized_linear_names, index_data["weight_map"]


def replace_with_quantized_linears(
    model: torch.nn.Module,
    quantized_linear_names: set[str],
    model_quant_type: str,
) -> int:
    quant_cfg = build_quantized_llm_cfg(model_quant_type) if model_quant_type == "W8A8" else None
    replaced = 0
    target_names = [
        name
        for name, module in model.named_modules()
        if name in quantized_linear_names
        and isinstance(module, (torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear))
    ]
    for name in target_names:
        parent, leaf = resolve_named_module(model, name)
        module = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
        if name not in quantized_linear_names:
            continue
        if not isinstance(module, (torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear)):
            continue
        if model_quant_type == "W8A16":
            quant_module = W8A16LinearRuntime(module)
            quant_module._target_device = module.weight.device
        else:
            quant_module = LinearQuantizerTimestep(cfg=quant_cfg, logger=None)
            quant_module.set_param(module)
            quant_module._target_device = quant_module.weight.device if quant_module.weight is not None else torch.device("cpu")
            # Keep only small metadata on-device during weight loading; the original float
            # matrix would otherwise double memory usage until int8 weights are installed.
            if quant_module.weight is not None and quant_module.weight.device.type != "cpu":
                quant_module.weight = torch.nn.Parameter(quant_module.weight.cpu(), requires_grad=False)
            if quant_module.bias is not None and quant_module.bias.device.type != "cpu":
                quant_module.bias = torch.nn.Parameter(quant_module.bias.cpu(), requires_grad=False)
        if hasattr(module, "_hf_hook"):
            add_hook_to_module(quant_module, module._hf_hook)
            remove_hook_from_module(module)
        # Clear old tensors explicitly before swapping the module out so their device
        # storage can be reclaimed before quantized weights are materialized.
        if getattr(module, "weight", None) is not None:
            module.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        if getattr(module, "bias", None) is not None:
            module.bias = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        set_named_module(model, name, quant_module)
        replaced += 1
        del module
    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    return replaced


def load_quantized_llm_weights(
    model: torch.nn.Module,
    quantized_llm_path: Path,
    device: torch.device,
) -> int:
    model_quant_type, quantized_linear_names, weight_map = load_quantized_linear_names(quantized_llm_path)
    replaced = replace_with_quantized_linears(model, quantized_linear_names, model_quant_type)
    if replaced == 0:
        raise RuntimeError(f"No quantized linear modules were matched under {quantized_llm_path}.")
    quantized_modules = {
        name: module
        for name, module in model.named_modules()
        if name in quantized_linear_names and isinstance(module, (LinearQuantizerTimestep, W8A16LinearRuntime))
    }
    module_device_map = {
        name: getattr(module, "_target_device", device)
        for name, module in quantized_modules.items()
    }

    shard_names = sorted(set(weight_map.values()))
    log_stage(f"replaced {replaced} linear layers with {model_quant_type} quantizers")
    log_stage(f"loading quantized llm weights from {len(shard_names)} shard(s)")
    for shard_name in shard_names:
        shard_path = quantized_llm_path / shard_name
        params_to_load_by_device = {}
        params_to_load_by_module = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for tensor_name in shard.keys():
                module_name, _, _ = tensor_name.rpartition(".")
                if module_name not in quantized_linear_names:
                    continue
                target_device = module_device_map[module_name]
                tensor = shard.get_tensor(tensor_name)
                if model_quant_type == "W8A16":
                    _, _, param_name = tensor_name.rpartition(".")
                    params_to_load_by_module.setdefault(module_name, {})[param_name] = tensor
                else:
                    params_to_load_by_device.setdefault(target_device, {})[tensor_name] = tensor
        if model_quant_type == "W8A16":
            for module_name, params_to_load in params_to_load_by_module.items():
                quantized_modules[module_name].load_layer_params(params_to_load, module_device_map[module_name])
        else:
            for target_device, params_to_load in params_to_load_by_device.items():
                if params_to_load:
                    load_quant_weight(params_to_load, model, target_device)
    TimestepManager.set_timestep_idx(0)
    return replaced


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
            device_type=device.type,
        )
    elif requested_device_ids:
        device = torch.device(f"{device.type}:{requested_device_ids[0]}")
        set_runtime_device(device)
    model_load_device_map = device_map
    if not use_device_map and device.type != "cpu":
        model_load_device_map = {"": str(device)}

    log_stage(
        f"runtime_device={device} use_device_map={use_device_map} "
        f"tp_devices={args.tensor_parallel_devices} device_ids={tp_device_ids or requested_device_ids}"
    )
    # Keep the full multimodal module tree loaded even for text-only prompts.
    # The original Ming inference path assumes the complete model is present, and
    # aggressively pruning vision/audio for text requests noticeably degrades text quality.
    load_multimodal = True
    log_stage(
        f"text_only={not bool(modalities)} load_multimodal={load_multimodal} "
        f"requested_modalities={modalities or ['text']}"
    )
    log_stage("loading model")
    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        load_multimodal=load_multimodal,
        load_image_gen=False,
        load_talker=False,
        device_map=model_load_device_map,
        low_cpu_mem_usage=True,
    )
    if model_load_device_map is None:
        model = model.to(device=device, dtype=torch.bfloat16)
    if args.quantized_llm_path:
        log_stage("applying quantized llm weights")
        replaced = load_quantized_llm_weights(
            model,
            Path(args.quantized_llm_path).resolve(),
            device,
        )
        log_stage(f"quantized llm ready replaced_layers={replaced}")
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
    if args.quantized_llm_path:
        TimestepManager.set_timestep_idx(0)
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
    parser.add_argument("--quantized-llm-path")
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
