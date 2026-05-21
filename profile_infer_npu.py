import argparse
import json
import os
import statistics
import sys
import threading
import time
from contextlib import nullcontext
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


def log_stage(message: str):
    print(f"[stage] {message}", file=sys.stderr, flush=True)


def sync_device(device: torch.device):
    if device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def get_runtime_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        current_idx = torch.npu.current_device() if hasattr(torch.npu, "current_device") else 0
        return torch.device(f"npu:{current_idx}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def get_attn_implementation(device: torch.device) -> str:
    return "eager"


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
    if args.num_runs < 1:
        raise ValueError("--num-runs must be at least 1.")
    if args.profile_skip_first < 0 or args.profile_warmup_runs < 0 or args.profile_active_runs < 0:
        raise ValueError("Profiler schedule values must be non-negative.")
    if args.profile_output and (
        args.profile_skip_first + args.profile_warmup_runs + args.profile_active_runs > args.num_runs
    ):
        raise ValueError(
            "Profiler schedule exceeds total runs: "
            f"skip_first({args.profile_skip_first}) + warmup({args.profile_warmup_runs}) + "
            f"active({args.profile_active_runs}) must be <= num_runs({args.num_runs})."
        )
    return modalities


def build_profile_context(profile_output: str | None, device: torch.device, args):
    if not profile_output:
        return nullcontext()
    if device.type != "npu":
        raise RuntimeError("--profile-output currently only supports Ascend NPU runtime.")

    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError("torch_npu is required when --profile-output is enabled.") from exc

    profile_dir = Path(profile_output)
    profile_dir.mkdir(parents=True, exist_ok=True)
    profiler_level = {
        0: torch_npu.profiler.ProfilerLevel.Level0,
        1: torch_npu.profiler.ProfilerLevel.Level1,
        2: torch_npu.profiler.ProfilerLevel.Level2,
    }[args.profile_level]
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=profiler_level,
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        data_simplification=False,
        sys_interconnection=True,
    )

    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=args.profile_skip_first,
            warmup=args.profile_warmup_runs,
            active=args.profile_active_runs,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        experimental_config=experimental_config,
    )


class DeviceSampler:
    def __init__(self, devices: list[torch.device], interval_sec: float):
        self.devices = devices
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread = None
        self.samples = []

    def _normalize_device_arg(self, device: torch.device):
        if device.type in {"npu", "cuda"}:
            return device.index
        return device

    def _get_utilization(self, device: torch.device):
        if device.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "utilization"):
            try:
                return float(torch.npu.utilization(self._normalize_device_arg(device)))
            except Exception:
                return None
        return None

    def _get_mem_info(self, device: torch.device):
        if device.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "mem_get_info"):
            try:
                free_bytes, total_bytes = torch.npu.mem_get_info(self._normalize_device_arg(device))
                return {
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                    "used_bytes": int(total_bytes - free_bytes),
                }
            except Exception:
                return None
        return None

    def _run(self):
        while not self._stop.is_set():
            sample = {"timestamp": time.time(), "devices": {}}
            for device in self.devices:
                device_key = str(device)
                device_sample = {
                    "utilization": self._get_utilization(device),
                }
                mem_info = self._get_mem_info(device)
                if mem_info is not None:
                    device_sample.update(mem_info)
                sample["devices"][device_key] = device_sample
            self.samples.append(sample)
            self._stop.wait(self.interval_sec)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="device-sampler", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 2))


def maybe_reset_peak_memory(device: torch.device):
    if device.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "reset_peak_memory_stats"):
        try:
            torch.npu.reset_peak_memory_stats(device)
        except Exception:
            pass
    elif device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass


def current_memory_stats(device: torch.device):
    if device.type == "npu" and hasattr(torch, "npu"):
        return {
            "memory_allocated": int(torch.npu.memory_allocated(device)) if hasattr(torch.npu, "memory_allocated") else None,
            "memory_reserved": int(torch.npu.memory_reserved(device)) if hasattr(torch.npu, "memory_reserved") else None,
            "max_memory_allocated": int(torch.npu.max_memory_allocated(device)) if hasattr(torch.npu, "max_memory_allocated") else None,
            "max_memory_reserved": int(torch.npu.max_memory_reserved(device)) if hasattr(torch.npu, "max_memory_reserved") else None,
        }
    if device.type == "cuda" and torch.cuda.is_available():
        return {
            "memory_allocated": int(torch.cuda.memory_allocated(device)),
            "memory_reserved": int(torch.cuda.memory_reserved(device)),
            "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    return {
        "memory_allocated": None,
        "memory_reserved": None,
        "max_memory_allocated": None,
        "max_memory_reserved": None,
    }


def summarize_samples(samples):
    if not samples:
        return {}

    summary = {"num_samples": len(samples), "devices": {}}
    device_keys = sorted(
        {
            device_key
            for sample in samples
            for device_key in sample.get("devices", {}).keys()
        }
    )
    all_utils = []
    all_used_bytes = []

    for device_key in device_keys:
        utilizations = [
            sample["devices"][device_key]["utilization"]
            for sample in samples
            if device_key in sample.get("devices", {})
            and sample["devices"][device_key].get("utilization") is not None
        ]
        used_bytes = [
            sample["devices"][device_key]["used_bytes"]
            for sample in samples
            if device_key in sample.get("devices", {})
            and sample["devices"][device_key].get("used_bytes") is not None
        ]
        device_summary = {}
        if utilizations:
            device_summary.update(
                {
                    "utilization_avg": statistics.fmean(utilizations),
                    "utilization_max": max(utilizations),
                    "utilization_min": min(utilizations),
                }
            )
            all_utils.extend(utilizations)
        if used_bytes:
            device_summary.update(
                {
                    "device_used_bytes_avg": int(statistics.fmean(used_bytes)),
                    "device_used_bytes_max": int(max(used_bytes)),
                    "device_used_bytes_min": int(min(used_bytes)),
                }
            )
            all_used_bytes.extend(used_bytes)
        summary["devices"][device_key] = device_summary

    if all_utils:
        summary.update(
            {
                "utilization_avg": statistics.fmean(all_utils),
                "utilization_max": max(all_utils),
                "utilization_min": min(all_utils),
            }
        )
    if all_used_bytes:
        summary.update(
            {
                "device_used_bytes_avg": int(statistics.fmean(all_used_bytes)),
                "device_used_bytes_max": int(max(all_used_bytes)),
                "device_used_bytes_min": int(min(all_used_bytes)),
            }
        )
    return summary


def human_bytes(num_bytes):
    if num_bytes is None:
        return None
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def run_profile(args):
    modalities = validate_args(args)
    stage_timings = {}
    overall_start = time.time()

    def mark_stage(name: str, started_at: float):
        stage_timings[name] = time.time() - started_at

    stage_start = time.time()
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
    input_device = torch.device(f"{device.type}:{tp_device_ids[0]}") if use_device_map else device
    profiled_devices = (
        [torch.device(f"{device.type}:{device_id}") for device_id in tp_device_ids]
        if use_device_map and tp_device_ids
        else [input_device]
    )
    mark_stage("setup", stage_start)

    log_stage(
        f"runtime_device={device} use_device_map={use_device_map} "
        f"tp_devices={args.tensor_parallel_devices} device_ids={tp_device_ids or requested_device_ids}"
    )

    stage_start = time.time()
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
    mark_stage("model_load", stage_start)
    log_stage("model ready")

    stage_start = time.time()
    log_stage("loading processor")
    processor = AutoProcessor.from_pretrained(args.code_path, trust_remote_code=True)
    mark_stage("processor_load", stage_start)
    log_stage("processor ready")

    stage_start = time.time()
    messages = build_messages(args)
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
    mark_stage("input_prepare", stage_start)
    log_stage(
        "inputs ready "
        f"input_ids={tuple(inputs.input_ids.shape)} "
        f"attention_mask={tuple(inputs.attention_mask.shape)} "
        f"target_device={input_device}"
    )

    profiler_ctx = build_profile_context(args.profile_output, device, args)
    run_metrics = []
    last_output_text = ""
    last_output_tokens = 0

    with profiler_ctx as prof:
        for run_idx in range(args.num_runs):
            maybe_reset_peak_memory(input_device)
            sampler = DeviceSampler(profiled_devices, args.utilization_sample_interval)

            pre_mem = current_memory_stats(input_device)
            sync_device(input_device)
            sampler.start()
            run_start = time.time()
            log_stage(f"run={run_idx + 1}/{args.num_runs} generate start")
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    eos_token_id=processor.gen_terminator,
                    num_logits_to_keep=1,
                )
            sync_device(input_device)
            generate_elapsed = time.time() - run_start
            sampler.stop()

            decode_start = time.time()
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            decode_elapsed = time.time() - decode_start
            post_mem = current_memory_stats(input_device)

            output_tokens = int(generated_ids_trimmed[0].shape[0]) if generated_ids_trimmed else 0
            input_tokens = int(inputs.input_ids.shape[-1])
            total_elapsed = generate_elapsed + decode_elapsed
            token_throughput = output_tokens / generate_elapsed if generate_elapsed > 0 else 0.0
            sampler_summary = summarize_samples(sampler.samples)
            run_metrics.append(
                {
                    "run_index": run_idx + 1,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "generate_seconds": generate_elapsed,
                    "decode_seconds": decode_elapsed,
                    "total_seconds": total_elapsed,
                    "tokens_per_second": token_throughput,
                    "memory_pre": pre_mem,
                    "memory_post": post_mem,
                    "utilization": sampler_summary,
                }
            )

            last_output_text = output_text
            last_output_tokens = output_tokens
            log_stage(
                f"run={run_idx + 1}/{args.num_runs} generate done "
                f"tokens={output_tokens} tps={token_throughput:.2f}"
            )
            if prof is not None:
                prof.step()

    stable_runs = run_metrics[args.skip_summary_runs :]
    summary_source = stable_runs if stable_runs else run_metrics
    summary = {
        "generate_seconds_avg": statistics.fmean(item["generate_seconds"] for item in summary_source),
        "generate_seconds_min": min(item["generate_seconds"] for item in summary_source),
        "generate_seconds_max": max(item["generate_seconds"] for item in summary_source),
        "decode_seconds_avg": statistics.fmean(item["decode_seconds"] for item in summary_source),
        "tokens_per_second_avg": statistics.fmean(item["tokens_per_second"] for item in summary_source),
        "tokens_per_second_min": min(item["tokens_per_second"] for item in summary_source),
        "tokens_per_second_max": max(item["tokens_per_second"] for item in summary_source),
        "output_tokens_avg": statistics.fmean(item["output_tokens"] for item in summary_source),
    }

    peak_allocated = [
        item["memory_post"]["max_memory_allocated"]
        for item in summary_source
        if item["memory_post"].get("max_memory_allocated") is not None
    ]
    peak_reserved = [
        item["memory_post"]["max_memory_reserved"]
        for item in summary_source
        if item["memory_post"].get("max_memory_reserved") is not None
    ]
    util_avg = [
        item["utilization"]["utilization_avg"]
        for item in summary_source
        if item["utilization"].get("utilization_avg") is not None
    ]
    util_max = [
        item["utilization"]["utilization_max"]
        for item in summary_source
        if item["utilization"].get("utilization_max") is not None
    ]
    used_max = [
        item["utilization"]["device_used_bytes_max"]
        for item in summary_source
        if item["utilization"].get("device_used_bytes_max") is not None
    ]

    if peak_allocated:
        summary["peak_memory_allocated_avg"] = int(statistics.fmean(peak_allocated))
        summary["peak_memory_allocated_max"] = int(max(peak_allocated))
    if peak_reserved:
        summary["peak_memory_reserved_avg"] = int(statistics.fmean(peak_reserved))
        summary["peak_memory_reserved_max"] = int(max(peak_reserved))
    if util_avg:
        summary["npu_utilization_avg"] = statistics.fmean(util_avg)
    if util_max:
        summary["npu_utilization_peak"] = max(util_max)
    if used_max:
        summary["device_used_bytes_peak"] = int(max(used_max))

    result = {
        "device": str(device),
        "input_device": str(input_device),
        "device_map": device_map,
        "task": args.task,
        "modalities": modalities or ["text"],
        "attn_implementation": attn_implementation,
        "stage_timings": stage_timings,
        "num_runs": args.num_runs,
        "skip_summary_runs": args.skip_summary_runs,
        "summary": summary,
        "runs": run_metrics,
        "last_output_text": last_output_text,
        "elapsed_total": time.time() - overall_start,
        "profile_output": str(Path(args.profile_output).resolve()) if args.profile_output else None,
    }

    print(f"device: {device}")
    print(f"input_device: {input_device}")
    print(f"profiled_devices: {[str(item) for item in profiled_devices]}")
    print(f"device_map: {device_map}")
    print(f"task: {args.task}")
    print(f"modalities: {modalities or ['text']}")
    print(f"attn_implementation: {attn_implementation}")
    print(f"stage_setup: {stage_timings['setup']:.2f}s")
    print(f"stage_model_load: {stage_timings['model_load']:.2f}s")
    print(f"stage_processor_load: {stage_timings['processor_load']:.2f}s")
    print(f"stage_input_prepare: {stage_timings['input_prepare']:.2f}s")
    for item in run_metrics:
        print(
            f"run_{item['run_index']}: generate={item['generate_seconds']:.2f}s "
            f"decode={item['decode_seconds']:.2f}s output_tokens={item['output_tokens']} "
            f"tokens_per_second={item['tokens_per_second']:.2f}"
        )
    print(f"summary_generate_avg: {summary['generate_seconds_avg']:.2f}s")
    print(f"summary_tokens_per_second_avg: {summary['tokens_per_second_avg']:.2f}")
    if "peak_memory_allocated_max" in summary:
        print(f"summary_peak_memory_allocated_max: {human_bytes(summary['peak_memory_allocated_max'])}")
    if "peak_memory_reserved_max" in summary:
        print(f"summary_peak_memory_reserved_max: {human_bytes(summary['peak_memory_reserved_max'])}")
    if "device_used_bytes_peak" in summary:
        print(f"summary_device_used_peak: {human_bytes(summary['device_used_bytes_peak'])}")
    if "npu_utilization_avg" in summary:
        print(f"summary_npu_utilization_avg: {summary['npu_utilization_avg']:.2f}%")
    if "npu_utilization_peak" in summary:
        print(f"summary_npu_utilization_peak: {summary['npu_utilization_peak']:.2f}%")
    for device_key, device_summary in summary_source[-1]["utilization"].get("devices", {}).items():
        if "utilization_avg" in device_summary:
            print(
                f"last_run_{device_key}_utilization_avg: {device_summary['utilization_avg']:.2f}%"
            )
        if "device_used_bytes_max" in device_summary:
            print(
                f"last_run_{device_key}_used_peak: {human_bytes(device_summary['device_used_bytes_max'])}"
            )
    print(f"elapsed_total: {result['elapsed_total']:.2f}s")
    if args.profile_output:
        print(f"profile_saved: {result['profile_output']}")
    if args.metrics_output:
        metrics_output_path = Path(args.metrics_output)
        metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"metrics_saved: {metrics_output_path.resolve()}")
    print(last_output_text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile understanding inference on Ascend/CUDA/CPU."
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
    parser.add_argument("--device-ids")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--skip-summary-runs", type=int, default=1)
    parser.add_argument("--utilization-sample-interval", type=float, default=0.2)
    parser.add_argument("--metrics-output")
    parser.add_argument("--profile-output")
    parser.add_argument("--profile-skip-first", type=int, default=1)
    parser.add_argument("--profile-warmup-runs", type=int, default=1)
    parser.add_argument("--profile-active-runs", type=int, default=1)
    parser.add_argument("--profile-level", type=int, choices=(0, 1, 2), default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run_profile(parse_args())
