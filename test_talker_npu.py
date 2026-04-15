import argparse
import os
import time
from pathlib import Path

DEFAULT_HF_CACHE = Path(__file__).resolve().parent / ".hf-cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE))
os.environ.setdefault("HF_MODULES_CACHE", str(DEFAULT_HF_CACHE / "modules"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torchaudio

from AudioVAE.modeling_audio_vae import AudioVAE
from modeling_bailing_talker import BailingTalker2


def ensure_local_hf_cache():
    cache_root = Path(os.environ["HF_HOME"])
    modules_root = cache_root / "modules"
    modules_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(modules_root)


def get_runtime_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        current_idx = torch.npu.current_device() if hasattr(torch.npu, "current_device") else 0
        return torch.device(f"npu:{current_idx}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def get_vae_device(runtime_device: torch.device) -> torch.device:
    # Keep the vocoder / ISTFT path off NPU by default. The current Ascend kernel
    # crashes in AudioVAE decoding, while CPU decoding remains valid.
    if runtime_device.type == "npu":
        return torch.device("cpu")
    return runtime_device


def run_talker(args):
    ensure_local_hf_cache()
    device = get_runtime_device()
    vae_device = get_vae_device(device)

    talker = BailingTalker2.from_pretrained(
        f"{args.model_path}/talker",
        torch_dtype=torch.bfloat16,
    ).eval().to(dtype=torch.bfloat16, device=device)
    talker.use_vllm = False

    vae = AudioVAE.from_pretrained(
        f"{args.model_path}/talker/vae",
        torch_dtype=torch.bfloat16,
    ).eval().to(dtype=torch.bfloat16, device=vae_device)

    start_time = time.time()
    wav_chunks = []
    text_spans = []

    with torch.no_grad():
        for tts_speech, text_span, text_position, duration in talker.omni_audio_generation(
            tts_text=args.text,
            voice_name=args.voice_name,
            prompt_text=args.prompt_text,
            prompt_wav_path=args.prompt_wav_path,
            max_length=args.max_length,
            audio_detokenizer=vae,
            stream=args.stream,
        ):
            wav_chunks.append(tts_speech)
            text_spans.append((text_span, text_position, duration))
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

    elapsed = time.time() - start_time
    audio_duration = waveform.shape[-1] / vae.config.sample_rate
    print(f"device: {device}")
    print(f"vae_device: {vae_device}")
    print(f"voice_name: {args.voice_name}")
    print(f"stream: {args.stream}")
    print(f"elapsed: {elapsed:.2f}s")
    print(f"audio_duration: {audio_duration:.2f}s")
    print(f"rtf: {elapsed / audio_duration:.3f}")
    print(f"saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Talker/TTS entry for Ascend/CUDA/CPU."
    )
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice-name", default="DB30")
    parser.add_argument("--prompt-text")
    parser.add_argument("--prompt-wav-path")
    parser.add_argument("--output", default="generated_audios/out_tts.wav")
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_talker(parse_args())
