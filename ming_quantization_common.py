#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
HF_CACHE_ROOT = REPO_ROOT / ".hf-cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_ROOT / "modules"))

STAGED_METADATA_SUFFIXES = (".json", ".py")
REMOTE_CODE_FILES = (
    "configuration_bailingmm2.py",
    "configuration_bailing_moe_v2.py",
    "configuration_audio.py",
    "configuration_whisper_encoder.py",
    "configuration_audio_vae.py",
    "configuration_bailing_talker.py",
    "modeling_bailingmm2.py",
    "modeling_bailing_moe_v2.py",
    "modeling_whisper_encoder.py",
    "modeling_utils.py",
    "image_processing_bailingmm2.py",
    "processing_bailingmm2.py",
    "audio_processing_bailingmm2.py",
    "tokenization_bailing.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "qwen3_moe_vit.py",
    "bailingmm_utils.py",
    "bailingmm_utils_video.py",
    "chat_format.py",
    "configuration_audio_vae.py",
    "AudioVAE",
    "talker_tn",
)


def create_model(model_type: str, model_path: Path, trust_remote_code: bool):
    from msmodelslim.model import ModelFactory
    from msmodelslim.model.base import BaseModelAdapter

    import msmodelslim_ming_adapter  # noqa: F401

    return ModelFactory.create(model_type, interface=BaseModelAdapter)(
        model_type=model_type,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
    )


def has_model_weights(model_path: Path) -> bool:
    patterns = (
        "*.safetensors",
        "*.safetensors.index.json",
        "pytorch_model*.bin",
        "model-*.bin",
    )
    return any(any(model_path.glob(pattern)) for pattern in patterns)


def ensure_remote_code_bundle(weight_path: Path, bundle_root: Path) -> Path:
    bundle_root.mkdir(parents=True, exist_ok=True)

    for item in weight_path.iterdir():
        target = bundle_root / item.name
        should_materialize = item.is_file() and item.suffix in STAGED_METADATA_SUFFIXES
        if should_materialize:
            if target.is_symlink() or target.exists():
                target.unlink()
            shutil.copy2(item, target)
            continue
        if target.exists() or target.is_symlink():
            continue
        os.symlink(item, target, target_is_directory=item.is_dir())

    for relative in REMOTE_CODE_FILES:
        source = REPO_ROOT / relative
        if not source.exists():
            continue
        target = bundle_root / relative
        if target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)

    return bundle_root


def _resolve_dataset_reference(dataset_ref: str, config_dir: Path) -> str:
    if not dataset_ref:
        return dataset_ref

    dataset_path = Path(dataset_ref)
    if dataset_path.is_absolute():
        return str(dataset_path)

    local_candidates = [
        REPO_ROOT / "data" / "calib" / dataset_path.name,
        config_dir / dataset_path,
        REPO_ROOT / dataset_path,
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return dataset_ref


def resolve_quant_config_path(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    spec = config.setdefault("spec", {})
    config_dir = config_path.parent

    for key in ("calib_dataset", "anti_dataset", "dataset"):
        if key in spec and isinstance(spec[key], str):
            spec[key] = _resolve_dataset_reference(spec[key], config_dir)

    resolved_dir = REPO_ROOT / ".cache" / "quant_configs"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / f"{config_path.stem}.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return resolved_path
