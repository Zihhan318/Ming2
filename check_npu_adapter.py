import argparse
import os
import sys
from pathlib import Path

DEFAULT_HF_CACHE = Path(__file__).resolve().parent / ".hf-cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE))
os.environ.setdefault("HF_MODULES_CACHE", str(DEFAULT_HF_CACHE / "modules"))

import torch
from transformers import AutoProcessor


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


def require_ok(condition: bool, message: str):
    if not condition:
        print(f"[FAIL] {message}")
        sys.exit(1)


def run_checks(code_path: str, prompt: str, hf_home: str | None):
    ensure_local_hf_cache()
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_MODULES_CACHE"] = str(Path(hf_home) / "modules")

    print("[1/3] runtime")
    has_npu = hasattr(torch, "npu")
    npu_count = torch.npu.device_count() if has_npu else 0
    npu_available = torch.npu.is_available() if has_npu else False
    device = get_runtime_device()
    print(f"device={device}")
    print(f"has_npu={has_npu}")
    print(f"npu_device_count={npu_count}")
    print(f"npu_available={npu_available}")
    require_ok(has_npu, "torch.npu is not available in this environment")
    require_ok(npu_available, "NPU runtime is not available")

    print("[2/3] processor")
    processor = AutoProcessor.from_pretrained(code_path, trust_remote_code=True)
    messages = [{"role": "HUMAN", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages)
    print("processor_ok")
    print(text[:300])

    print("[3/3] input tensors")
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
    )
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    print(f"device_move_ok {device}")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} {value.device} {tuple(value.shape)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test NPU adapter readiness without loading model weights.")
    parser.add_argument("--code-path", default=".")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--hf-home", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_checks(**vars(parse_args()))
