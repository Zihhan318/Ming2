import argparse
import gc
import os
import time
from pathlib import Path

DEFAULT_HF_CACHE = Path(__file__).resolve().parent / ".hf-cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE))
os.environ.setdefault("HF_MODULES_CACHE", str(DEFAULT_HF_CACHE / "modules"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

import torch
import transformers.modeling_utils as modeling_utils
import torchaudio
from transformers import AutoProcessor

from AudioVAE.modeling_audio_vae import AudioVAE
from modeling_bailing_talker import BailingTalker2
from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
from test_infer_npu import build_layer_split_device_map, move_inputs_to_device


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


def get_vae_device(runtime_device: torch.device) -> torch.device:
    if runtime_device.type == "npu":
        return torch.device("cpu")
    return runtime_device


def clear_runtime_cache():
    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_available():
        if hasattr(torch.npu, "empty_cache"):
            torch.npu.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_tts_audio(args, runtime_device: torch.device) -> Path:
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

    wav_chunks = []
    with torch.no_grad():
        for tts_speech, text_span, text_position, duration in talker.omni_audio_generation(
            tts_text=args.text,
            voice_name=args.voice_name,
            prompt_text=args.prompt_text,
            prompt_wav_path=args.prompt_wav_path,
            max_length=args.max_length,
            audio_detokenizer=vae,
            stream=False,
        ):
            wav_chunks.append(tts_speech)
            print(
                f"tts_chunk_text={text_span!r} position={text_position} "
                f"duration_ms={duration} samples={tts_speech.shape[-1]}"
            )

    if not wav_chunks:
        raise RuntimeError("Talker returned no audio chunks.")

    waveform = torch.cat(wav_chunks, dim=-1)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform, sample_rate=vae.config.sample_rate)
    print(f"tts_saved: {output_path}")
    print(f"tts_audio_duration: {waveform.shape[-1] / vae.config.sample_rate:.2f}s")
    print(f"tts_device: {runtime_device}")
    print(f"tts_vae_device: {vae_device}")
    del talker
    del vae
    del waveform
    clear_runtime_cache()
    return output_path


def build_asr_model(args, runtime_device: torch.device):
    use_device_map = args.tensor_parallel_devices and args.tensor_parallel_devices > 1
    device_map = None
    if use_device_map:
        device_map = build_layer_split_device_map(num_layers=32, num_devices=args.tensor_parallel_devices)

    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        load_image_gen=False,
        load_talker=False,
        device_map=device_map,
    )
    if not use_device_map:
        model = model.to(device=runtime_device, dtype=torch.bfloat16)
    else:
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    return model, device_map


def transcribe_audio(args, audio_path: Path, runtime_device: torch.device):
    processor = AutoProcessor.from_pretrained(args.code_path, trust_remote_code=True)
    model, device_map = build_asr_model(args, runtime_device)

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": args.asr_prompt},
                {"type": "audio", "audio": str(audio_path)},
            ],
        },
    ]
    text = processor.apply_chat_template(messages)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        audio_kwargs={"use_whisper_encoder": True},
    )
    input_device = torch.device("npu:0") if device_map else runtime_device
    inputs = move_inputs_to_device(inputs, input_device)

    if args.lang:
        language = torch.tensor([processor.tokenizer.encode(f"{args.lang}\t")], device=inputs["input_ids"].device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], language], dim=1)
        attention_mask = inputs["attention_mask"]
        inputs["attention_mask"] = torch.ones(
            inputs["input_ids"].shape,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.asr_max_new_tokens,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            num_logits_to_keep=1,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    transcript = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f"asr_elapsed: {time.time() - start_time:.2f}s")
    print(f"asr_device_map: {device_map}")
    return transcript


def parse_args():
    parser = argparse.ArgumentParser(
        description="Round-trip verification: TTS first, then ASR/transcription on the generated audio."
    )
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--code-path", default=".")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice-name", default="DB30")
    parser.add_argument("--prompt-text")
    parser.add_argument("--prompt-wav-path")
    parser.add_argument("--output", default="generated_audios/roundtrip_tts.wav")
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--tensor-parallel-devices", type=int, default=1)
    parser.add_argument(
        "--asr-prompt",
        default="Please recognize the language of this speech and transcribe it. Format: oral.",
    )
    parser.add_argument("--asr-max-new-tokens", type=int, default=128)
    parser.add_argument("--lang", default=None)
    return parser.parse_args()


def main():
    ensure_local_hf_cache()
    disable_allocator_warmup()
    runtime_device = get_runtime_device()
    print(f"runtime_device: {runtime_device}")
    audio_path = save_tts_audio(parse_args_result, runtime_device)
    transcript = transcribe_audio(parse_args_result, audio_path, runtime_device)
    print(f"source_text: {parse_args_result.text}")
    print(f"asr_text: {transcript}")


if __name__ == "__main__":
    parse_args_result = parse_args()
    main()
