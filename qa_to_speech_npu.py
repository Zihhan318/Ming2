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
import torchaudio
import transformers.modeling_utils as modeling_utils
from transformers import AutoProcessor

from AudioVAE.modeling_audio_vae import AudioVAE
from configuration_bailingmm2 import BailingMM2Config
from modeling_bailing_talker import BailingTalker2
from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
from test_infer_npu import load_quantized_llm_weights


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


def get_runtime_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        current_idx = torch.npu.current_device() if hasattr(torch.npu, "current_device") else 0
        return torch.device(f"npu:{current_idx}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def set_runtime_device(device: torch.device):
    if device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(device)
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)


def sync_device(device: torch.device):
    if device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def empty_device_cache(device: torch.device):
    if device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def get_vae_device(runtime_device: torch.device) -> torch.device:
    if runtime_device.type == "npu":
        return torch.device("cpu")
    return runtime_device


def generate_answer(args) -> tuple[str, torch.device]:
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
    log_stage("loading understanding model")
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
    if args.quantized_llm_path:
        log_stage("applying quantized llm weights")
        replaced = load_quantized_llm_weights(
            model,
            Path(args.quantized_llm_path).resolve(),
            device,
        )
        log_stage(f"quantized llm ready replaced_layers={replaced}")
    model.eval()
    log_stage("understanding model ready")

    log_stage("loading processor")
    processor = AutoProcessor.from_pretrained(args.code_path, trust_remote_code=True)
    log_stage("processor ready")

    messages = [{"role": "HUMAN", "content": [{"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(
        messages,
        sys_prompt_exp=args.sys_prompt_exp,
        use_cot_system_prompt=args.use_cot_system_prompt,
    )
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    log_stage("tokenizing question")
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

    sync_device(input_device)
    start_time = time.time()
    log_stage(f"starting answer generation max_new_tokens={args.max_new_tokens}")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            num_logits_to_keep=1,
        )
    sync_device(input_device)
    log_stage("answer generation finished")

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    print(f"qa_device: {device}")
    print(f"qa_device_map: {device_map}")
    print(f"qa_elapsed: {time.time() - start_time:.2f}s")
    print(f"answer: {answer}")

    del generated_ids
    del generated_ids_trimmed
    del inputs
    del processor
    del model
    sync_device(input_device)
    empty_device_cache(input_device)

    return answer, device


def speak_answer(args, answer: str, runtime_device: torch.device):
    log_stage("loading talker")
    vae_device = get_vae_device(runtime_device)
    talker = BailingTalker2.from_pretrained(
        f"{args.model_path}/talker",
        torch_dtype=torch.bfloat16,
    ).eval().to(dtype=torch.bfloat16, device=runtime_device)
    talker.use_vllm = False

    vae = AudioVAE.from_pretrained(
        f"{args.model_path}/talker/vae",
        torch_dtype=torch.bfloat16,
    ).eval().to(dtype=torch.bfloat16, device=vae_device)
    log_stage("talker ready")

    start_time = time.time()
    wav_chunks = []
    with torch.no_grad():
        for tts_speech, text_span, text_position, duration in talker.omni_audio_generation(
            tts_text=answer,
            voice_name=args.voice_name,
            prompt_text=args.prompt_text,
            prompt_wav_path=args.prompt_wav_path,
            max_length=args.max_length,
            audio_detokenizer=vae,
            stream=args.stream,
        ):
            wav_chunks.append(tts_speech)
            print(
                f"chunk_text={text_span!r} position={text_position} "
                f"duration_ms={duration} samples={tts_speech.shape[-1]}"
            )

    if not wav_chunks:
        raise RuntimeError("Talker returned no audio chunks.")

    waveform = torch.cat(wav_chunks, dim=-1)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform, sample_rate=vae.config.sample_rate)

    if args.answer_output:
        answer_output_path = Path(args.answer_output)
        answer_output_path.parent.mkdir(parents=True, exist_ok=True)
        answer_output_path.write_text(answer + "\n", encoding="utf-8")

    elapsed = time.time() - start_time
    audio_duration = waveform.shape[-1] / vae.config.sample_rate
    print(f"talker_device: {runtime_device}")
    print(f"vae_device: {vae_device}")
    print(f"voice_name: {args.voice_name}")
    print(f"stream: {args.stream}")
    print(f"tts_elapsed: {elapsed:.2f}s")
    print(f"audio_duration: {audio_duration:.2f}s")
    print(f"rtf: {elapsed / audio_duration:.3f}")
    print(f"saved_audio: {output_path}")
    if args.answer_output:
        print(f"saved_answer: {args.answer_output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Question answering plus talker/TTS entry for Ascend/CUDA/CPU."
    )
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--code-path", default=".")
    parser.add_argument("--quantized-llm-path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="generated_audios/answer_tts.wav")
    parser.add_argument("--answer-output")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sys-prompt-exp")
    parser.add_argument("--use-cot-system-prompt", action="store_true")
    parser.add_argument("--tensor-parallel-devices", type=int, default=1)
    parser.add_argument("--device-ids")
    parser.add_argument("--voice-name", default="DB30")
    parser.add_argument("--prompt-text")
    parser.add_argument("--prompt-wav-path")
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    answer, runtime_device = generate_answer(args)
    speak_answer(args, answer, runtime_device)


if __name__ == "__main__":
    main()
