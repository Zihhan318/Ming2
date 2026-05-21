import argparse
import logging
import os
from pathlib import Path

import torch

from msmodelslim.app.base import DeviceType
from msmodelslim.app.naive_quantization import NaiveQuantizationApplication
from msmodelslim.app.quant_service.proxy import QuantServiceProxy
from msmodelslim.cli.naive_quantization.__main__ import get_dataset_dir, get_practice_dir
from msmodelslim.infra.dataset_loader import FileDatasetLoader
from msmodelslim.infra.practice_manager import PracticeManager
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantType
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_modules import LinearQuantizer
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_tools import Calibrator
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.save import ComplexQuantifier
from msmodelslim.pytorch.lowbit.quant_modules import LinearQuantizer as LowBitLinearQuantizer

from measure_moe_expert_coverage import measure_expert_coverage
from ming_quantization_common import (
    create_model,
    ensure_remote_code_bundle,
    has_model_weights,
    resolve_quant_config_path,
)

LOGGER = logging.getLogger("msmodelslim")
ATTENTION_FLOAT_SUFFIXES = (
    "attention.query_key_value",
    "attention.dense",
)


def install_msmodelslim_diagnostics() -> None:
    """Add clearer logs for modules that never produced activation quant params."""

    original_run_datafree_after_calib = Calibrator.run_datafree_after_calib
    original_generate_weight_of_linear_module = ComplexQuantifier.generate_weight_of_linear_module
    original_rollback_names_process = Calibrator.rollback_names_process

    def patched_rollback_names_process(self, model):
        protected_attention = [
            name
            for name, module in model.named_modules()
            if isinstance(module, (LinearQuantizer, LowBitLinearQuantizer))  # already quantized wrappers
        ]
        if protected_attention:
            # Defensive no-op for repeated patching after the model is already rewritten.
            protected_attention = []
        else:
            protected_attention = [
                name
                for name, module in model.named_modules()
                if isinstance(module, (torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear))
                and any(name.endswith(suffix) for suffix in ATTENTION_FLOAT_SUFFIXES)
            ]
        if protected_attention:
            merged = list(dict.fromkeys(list(getattr(self.cfg, "disable_names", [])) + protected_attention))
            self.cfg.disable_names = merged
            preview = ", ".join(protected_attention[:10])
            if len(protected_attention) > 10:
                preview += ", ..."
            LOGGER.warning(
                "Keeping %d attention linear layers in FLOAT for stability. Layers: %s",
                len(protected_attention),
                preview,
            )
        return original_rollback_names_process(self, model)

    def patched_run_datafree_after_calib(self):
        original_run_datafree_after_calib(self)
        missing_input_scale = []
        for name, module in self.model.named_modules():
            if not isinstance(module, (LinearQuantizer, LowBitLinearQuantizer)):
                continue
            quant_input = getattr(module, "quant_input", None)
            if quant_input is not None and getattr(quant_input, "input_scale", None) is None:
                missing_input_scale.append(name)

        if missing_input_scale:
            preview = ", ".join(missing_input_scale[:20])
            if len(missing_input_scale) > 20:
                preview += ", ..."
            self.logger.warning(
                "Detected %d quantized linear modules without input_scale after calibration. "
                "These modules were likely never executed during calibration. Modules: %s",
                len(missing_input_scale),
                preview,
            )

    def patched_generate_weight_of_linear_module(self, name, module, model_quant_type):
        if model_quant_type in (QuantType.W8A8, QuantType.W8A8S, QuantType.W8A8_MIX):
            quant_input = getattr(module, "quant_input", None)
            input_scale = getattr(quant_input, "input_scale", None) if quant_input is not None else None
            if input_scale is None:
                LOGGER.warning(
                    "Falling back to FLOAT export for quantized linear module '%s' because "
                    "its input_scale is missing after calibration.",
                    name,
                )
                yield name + ".weight", QuantType.FLOAT, module.weight.detach().cpu()
                if hasattr(module, "bias") and module.bias is not None:
                    yield name + ".bias", QuantType.FLOAT, module.bias.detach().cpu()
                return
        yield from original_generate_weight_of_linear_module(self, name, module, model_quant_type)

    Calibrator.run_datafree_after_calib = patched_run_datafree_after_calib
    ComplexQuantifier.generate_weight_of_linear_module = patched_generate_weight_of_linear_module
    Calibrator.rollback_names_process = patched_rollback_names_process
def parse_args():
    parser = argparse.ArgumentParser(description="Quantize Ming-flash-omni-2.0 LLM trunk with msmodelslim.")
    parser.add_argument("--model-path", default=".", help="Model directory containing config/tokenizer/weight files.")
    parser.add_argument(
        "--save-path",
        default="quantized/ming-flash-omni-2.0-llm-w8a8",
        help="Output directory for quantized weights.",
    )
    parser.add_argument(
        "--bundle-path",
        default=".cache/ming-flash-omni-2.0-quant-bundle",
        help="Workspace-local staging directory that combines weights with remote-code files.",
    )
    parser.add_argument(
        "--config-path",
        default="quant_ming_llm_w8a8.yaml",
        help="msmodelslim quantization config yaml.",
    )
    parser.add_argument(
        "--model-type",
        default="Ming-flash-omni-2.0-LLM",
        help="Registered adapter name.",
    )
    parser.add_argument(
        "--device",
        choices=("npu", "cpu"),
        default="npu",
        help="Quantization device.",
    )
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument(
        "--skip-expert-coverage",
        action="store_true",
        help="Skip pre-quantization MoE expert coverage measurement.",
    )
    parser.add_argument(
        "--expert-coverage-report",
        help="Optional JSON output path for MoE expert coverage statistics.",
    )
    parser.add_argument(
        "--expert-coverage-max-samples",
        type=int,
        default=256,
        help="Maximum number of calibration samples to use for the expert coverage pass.",
    )
    return parser.parse_args()


def main():
    install_msmodelslim_diagnostics()
    args = parse_args()
    weight_path = Path(args.model_path).resolve()
    if not has_model_weights(weight_path):
        raise FileNotFoundError(
            f"No Hugging Face weight files were found under {weight_path}. "
            "Download or symlink the Ming-flash-omni-2.0 weights into this directory before quantizing."
        )
    model_path = ensure_remote_code_bundle(weight_path, Path(args.bundle_path).resolve())
    resolved_config_path = resolve_quant_config_path(Path(args.config_path).resolve())

    if not args.skip_expert_coverage:
        if args.expert_coverage_report:
            coverage_report_path = Path(args.expert_coverage_report).resolve()
        else:
            save_path = Path(args.save_path).resolve()
            coverage_report_path = save_path.parent / f"{save_path.name}_expert_coverage.json"
        measure_expert_coverage(
            model_path=model_path,
            config_path=resolved_config_path,
            model_type=args.model_type,
            device=DeviceType(args.device),
            trust_remote_code=args.trust_remote_code,
            report_path=coverage_report_path,
            max_samples=args.expert_coverage_max_samples,
        )

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
    app.quant(
        model_type=args.model_type,
        model_path=str(model_path),
        save_path=str(Path(args.save_path).resolve()),
        device=DeviceType(args.device),
        config_path=str(resolved_config_path),
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
