from pathlib import Path
import os
from typing import Any, Generator, List

HF_CACHE_ROOT = Path("/home/zzh/Ming2/.hf-cache")
os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_CACHE_ROOT / "modules"))

import torch.nn as nn

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
from msmodelslim.app.base.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func,
)
from msmodelslim.model.factory import ModelFactory
from msmodelslim.model.interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
)
from msmodelslim.model.transformers import TransformersModel
from msmodelslim.utils.logging import logger_setter


@ModelFactory.register("Ming-flash-omni-2.0")
@ModelFactory.register("Ming-flash-omni-2.0-LLM")
@logger_setter()
class MingFlashOmniLLMAdapter(
    TransformersModel,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV0,
    ModelSlimPipelineInterfaceV1,
):
    def _load_tokenizer(self, trust_remote_code: bool = False) -> PreTrainedTokenizerBase:
        kwargs = {
            "local_files_only": True,
            "trust_remote_code": trust_remote_code,
            "use_fast": True,
            "legacy": False,
        }
        tokenizer_file = self.model_path / "tokenizer.json"
        if tokenizer_file.exists():
            kwargs["tokenizer_file"] = str(tokenizer_file)
        return AutoTokenizer.from_pretrained(str(self.model_path), **kwargs)

    def get_model_type(self) -> str:
        return self.model_type

    def get_model_pedigree(self) -> str:
        return "ming_llm"

    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        device_map = "auto" if device == DeviceType.NPU else "cpu"
        return BailingMM2NativeForConditionalGeneration.from_pretrained(
            str(self.model_path),
            device_map=device_map,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=self.trust_remote_code,
            attn_implementation="eager",
            load_multimodal=False,
            load_image_gen=False,
            load_talker=False,
        )

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)

    def handle_dataset_by_batch(
        self,
        dataset: Any,
        batch_size: int,
        device: DeviceType = DeviceType.NPU,
    ) -> List[Any]:
        return self._get_batch_tokenized_data(calib_list=dataset, batch_size=batch_size, device=device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self.load_model(device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        model.model.config.use_cache = need_kv_cache
