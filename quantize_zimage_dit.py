#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml

from msmodelslim.app.base import DeviceType
from msmodelslim.app.naive_quantization import NaiveQuantizationApplication
from msmodelslim.app.quant_service.proxy import QuantServiceProxy
from msmodelslim.cli.naive_quantization.__main__ import get_dataset_dir, get_practice_dir
from msmodelslim.infra.dataset_loader import FileDatasetLoader
from msmodelslim.infra.practice_manager import PracticeManager

from ming_quantization_common import create_model as _unused_create_model  # noqa: F401


def create_model(model_type: str, model_path: Path, trust_remote_code: bool):
    from msmodelslim.model import ModelFactory
    from msmodelslim.model.base import BaseModelAdapter

    import msmodelslim_zimage_dit_adapter  # noqa: F401

    return ModelFactory.create(model_type, interface=BaseModelAdapter)(
        model_type=model_type,
        model_path=model_path,
        trust_remote_code=trust_remote_code,
    )


def resolve_config_path(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = config.setdefault("spec", {}).setdefault("multimodal_sd_config", {}).setdefault("model_config", {})
    conditions_dir = model_config.get("conditions_dir")
    if conditions_dir:
        conditions_path = Path(conditions_dir)
        if not conditions_path.is_absolute():
            model_config["conditions_dir"] = str((config_path.parent / conditions_path).resolve())
    resolved_dir = Path(".cache/quant_configs").resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / f"{config_path.stem}.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return resolved_path


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize Ming ZImage DiT transformer blocks with msmodelslim.")
    parser.add_argument("--model-path", default=".", help="Model directory containing mlp/transformer/scheduler/vae.")
    parser.add_argument(
        "--save-path",
        default="quantized/ming-flash-omni-2.0-dit-w8a8",
        help="Output directory for quantized DiT weights.",
    )
    parser.add_argument(
        "--config-path",
        default="quant_zimage_dit_w8a8.yaml",
        help="msmodelslim multimodal quantization config yaml.",
    )
    parser.add_argument("--model-type", default="Ming-ZImage-DiT")
    parser.add_argument("--device", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = get_practice_dir()
    practice_manager = PracticeManager(official_config_dir=config_dir)
    dataset_dir = get_dataset_dir()
    dataset_loader = FileDatasetLoader(dataset_dir)
    quant_service = QuantServiceProxy(dataset_loader)
    app = NaiveQuantizationApplication(
        practice_manager=practice_manager,
        quant_service=quant_service,
        model_factory=create_model,
    )
    resolved_config_path = resolve_config_path(Path(args.config_path))
    app.quant(
        model_type=args.model_type,
        model_path=str(Path(args.model_path).resolve()),
        save_path=str(Path(args.save_path).resolve()),
        device=DeviceType(args.device),
        config_path=str(resolved_config_path),
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
