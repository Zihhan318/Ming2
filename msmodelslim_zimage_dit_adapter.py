#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import torch
from safetensors.torch import load_file
from torch import distributed as dist, nn

from contextlib import nullcontext

from diffusion.zimage_loss import ZImageLoss
from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.app.base.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.graph.adapter_types import AdapterConfig, MappingConfig
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.common.layer_wise_forward import TransformersForwardBreak, generated_decoder_layer_visit_func
from msmodelslim.model.factory import ModelFactory
from msmodelslim.utils.cache import to_device
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.app.quant_service.multimodal_sd_v1.pipeline_interface import MultimodalPipelineInterface
from msmodelslim.quant.processor.anti_outlier.smooth_interface import FlexSmoothQuantInterface


def _resolve_device(device: DeviceType) -> str:
    if device == DeviceType.NPU and hasattr(torch, "npu") and torch.npu.is_available():
        current_idx = torch.npu.current_device() if hasattr(torch.npu, "current_device") else 0
        return f"npu:{current_idx}"
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


@ModelFactory.register("Ming-ZImage-DiT")
@logger_setter()
class MingZImageDiTAdapter(
    BaseModelAdapter,
    ModelInfoInterface,
    MultimodalPipelineInterface,
    FlexSmoothQuantInterface,
):
    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        self.diffusion_loss = None
        self.transformer = None
        self.model_args = {
            "conditions_dir": None,
            "seed": 42,
            "steps": 30,
            "height": 1024,
            "width": 1024,
            "max_cases": 8,
            "torch_dtype": "bfloat16",
            "dit_impl": "profile",
        }

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return "ming_zimage_dit"

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> Iterable[Any]:
        return dataset

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self.transformer

    def _iter_transformer_blocks(
        self,
        model: torch.nn.Module,
    ) -> List[Tuple[str, torch.nn.Module]]:
        return [(f"layers.{idx}", block) for idx, block in enumerate(model.layers)]

    def generate_model_forward(self, model: torch.nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        transformer_blocks = self._iter_transformer_blocks(model)
        first_block_input = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs)
            raise TransformersForwardBreak()

        hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)]
        try:
            if isinstance(inputs, (list, tuple)):
                model(*inputs)
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        finally:
            for hook in hooks:
                hook.remove()

        if first_block_input is None:
            raise InvalidModelError("Can't get first DiT block input.", action="Please check the captured calibration data.")

        current_inputs = to_device(first_block_input, "cpu")
        if dist.is_initialized():
            dist.barrier()

        for name, block in transformer_blocks:
            args, kwargs = current_inputs
            outputs = yield ProcessRequest(name, block, args, kwargs)
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
            current_inputs = ((hidden_states, *args[1:]), kwargs)

    def generate_model_visit(
        self,
        model: torch.nn.Module,
        transformer_blocks: Optional[List[Tuple[str, torch.nn.Module]]] = None,
    ) -> Generator[ProcessRequest, Any, None]:
        return generated_decoder_layer_visit_func(
            model,
            transformer_blocks=transformer_blocks or self._iter_transformer_blocks(model),
        )

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return None

    def get_adapter_config_for_subgraph(self) -> List[AdapterConfig]:
        adapter_config: List[AdapterConfig] = []
        if self.transformer is None:
            return adapter_config
        for layer_idx, _ in enumerate(self.transformer.layers):
            # Keep AntiOutlier conservative: only smooth the FFN input norm
            # into the two quantized expansion branches used by stage1.
            adapter_config.append(
                AdapterConfig(
                    subgraph_type="norm-linear",
                    mapping=MappingConfig(
                        source=f"layers.{layer_idx}.ffn_norm1",
                        targets=[
                            f"layers.{layer_idx}.feed_forward.w1",
                            f"layers.{layer_idx}.feed_forward.w3",
                        ],
                    ),
                )
            )
        return adapter_config

    def set_model_args(self, override_model_config: object):
        if not isinstance(override_model_config, dict):
            raise SchemaValidateError("model_config for Ming-ZImage-DiT must be a dict")
        merged = dict(self.model_args)
        unknown_keys = [key for key in override_model_config if key not in merged]
        if unknown_keys:
            raise SchemaValidateError(
                f"illegal config attributes: {unknown_keys}. supported config attributes: {sorted(merged.keys())}"
            )
        merged.update(override_model_config)
        if not merged["conditions_dir"]:
            raise SchemaValidateError("model_config.conditions_dir is required for Ming-ZImage-DiT quantization")
        self.model_args = merged

    def _load_diffusion_mlp_state(self) -> tuple[dict, dict]:
        mlp_path = self.model_path / "mlp" / "model.safetensors"
        mlp_config_path = self.model_path / "mlp" / "config.json"
        temp_state_dict = load_file(str(mlp_path))
        metax_config = json.loads(mlp_config_path.read_text(encoding="utf-8"))
        diffusion_mlp_state_dict = {
            key[len("mlp.") :]: temp_state_dict[key] for key in temp_state_dict if key.startswith("mlp.")
        }
        return diffusion_mlp_state_dict, metax_config

    def load_pipeline(self):
        diffusion_mlp_state_dict, metax_config = self._load_diffusion_mlp_state()
        torch_dtype_name = str(self.model_args["torch_dtype"]).lower()
        torch_dtype = torch.bfloat16 if torch_dtype_name == "bfloat16" else torch.float16
        os.environ["MING_ZIMAGE_TRANSFORMER_IMPL"] = str(self.model_args["dit_impl"]).strip().lower()
        device = _resolve_device(DeviceType.NPU)
        diffusion_c_input_dim = metax_config.get("diffusion_c_input_dim", 2048)
        self.diffusion_loss = ZImageLoss(
            model_path=str(self.model_path),
            scheduler_path=str(self.model_path),
            vision_dim=diffusion_c_input_dim,
            mlp_state_dict=diffusion_mlp_state_dict,
            torch_dtype=torch_dtype,
            device=device,
            use_identity_mlp=metax_config.get("use_identity_mlp", False),
            text_encoder_norm=metax_config.get("text_encoder_norm", False),
        )
        self.diffusion_loss.eval()
        self.transformer = self.diffusion_loss.train_model.transformer
        self.transformer.eval()

    def _iter_condition_files(self) -> List[Path]:
        conditions_dir = Path(self.model_args["conditions_dir"]).resolve()
        files = sorted(conditions_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No condition embed files found under {conditions_dir}")
        max_cases = int(self.model_args["max_cases"])
        return files[:max_cases] if max_cases > 0 else files

    def run_calib_inference(self):
        assert self.diffusion_loss is not None
        device = self.diffusion_loss.device
        seed = int(self.model_args["seed"])
        steps = int(self.model_args["steps"])
        height = int(self.model_args["height"])
        width = int(self.model_args["width"])
        for condition_path in self._iter_condition_files():
            payload = torch.load(condition_path, map_location="cpu")
            condition_embeds = payload["condition_embeds"].to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                self.diffusion_loss.sample(
                    encoder_hidden_states=condition_embeds,
                    steps=steps,
                    cfg=2.0,
                    seed=seed,
                    height=height,
                    width=width,
                    profile_transformer_layers=False,
                )

    def apply_quantization(self, process_model_func):
        assert self.transformer is not None
        runtime_device = _resolve_device(DeviceType.NPU)
        for name, module in self.transformer.named_children():
            if name == "layers":
                for block in module:
                    block.to("cpu")
            else:
                module.to(runtime_device)
        with torch.no_grad(), nullcontext():
            process_model_func()
