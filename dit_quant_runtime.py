#!/usr/bin/env python3
import gc
import json
import os
from pathlib import Path

import torch
from mindiesd.quantization.layer import W8A8QuantLinear
from safetensors import safe_open


def _resolve_named_module(root_module: torch.nn.Module, qualified_name: str):
    current = root_module
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current, parts[-1]


def _set_named_module(root_module: torch.nn.Module, qualified_name: str, new_module: torch.nn.Module):
    parent, leaf = _resolve_named_module(root_module, qualified_name)
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def load_quantized_dit_weights(
    model: torch.nn.Module,
    quantized_dit_path: Path,
    device: torch.device,
) -> int:
    transformer = model.diffusion_loss.train_model.transformer
    desc_candidates = [
        quantized_dit_path / "quant_model_description_w8a8.json",
        quantized_dit_path / "quant_model_description.json",
        *sorted(quantized_dit_path.glob("quant_model_description_*.json")),
    ]
    desc_path = next((candidate for candidate in desc_candidates if candidate.exists()), None)
    if desc_path is None:
        raise FileNotFoundError(f"No quantized DiT description json found under {quantized_dit_path}.")
    quant_description = json.loads(desc_path.read_text())

    index_json_candidates = [
        quantized_dit_path / "quant_model_weight_w8a8.safetensors.index.json",
        quantized_dit_path / "quant_model_weight.safetensors.index.json",
        *sorted(quantized_dit_path.glob("quant_model_weight_*.safetensors.index.json")),
    ]
    index_json_path = next((candidate for candidate in index_json_candidates if candidate.exists()), None)
    if index_json_path is not None:
        shard_names = sorted(set(json.loads(index_json_path.read_text())["weight_map"].values()))
    else:
        shard_candidates = [
            quantized_dit_path / "quant_model_weight_w8a8.safetensors",
            quantized_dit_path / "quant_model_weight.safetensors",
            *sorted(quantized_dit_path.glob("quant_model_weight_*.safetensors")),
        ]
        shard_path = next((candidate for candidate in shard_candidates if candidate.exists()), None)
        if shard_path is None:
            raise FileNotFoundError(f"No quantized DiT safetensors found under {quantized_dit_path}.")
        shard_names = [shard_path.name]

    quantized_linear_name_to_type = {
        name[: -len(".weight")]: str(quant_type).upper()
        for name, quant_type in quant_description.items()
        if name.endswith(".weight") and str(quant_type).upper() in {"W8A8", "W8A8_DYNAMIC"}
    }
    quantized_linear_names = set(quantized_linear_name_to_type.keys())
    if not quantized_linear_names:
        raise RuntimeError(f"No W8A8/W8A8_DYNAMIC linear entries found under {quantized_dit_path}.")

    quantized_linears = {
        name: module
        for name, module in transformer.named_modules()
        if name in quantized_linear_names and isinstance(module, torch.nn.Linear)
    }
    if not quantized_linears:
        raise RuntimeError(f"No DiT linear modules matched the exported W8A8 names under {quantized_dit_path}.")

    replaced = 0
    replaced_names = set()
    for shard_name in shard_names:
        shard_path = quantized_dit_path / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            shard_module_names = sorted(
                {
                    tensor_name.rpartition(".")[0]
                    for tensor_name in shard.keys()
                    if tensor_name.rpartition(".")[0] in quantized_linears
                }
            )
            for module_name in shard_module_names:
                if module_name in replaced_names:
                    continue
                linear = quantized_linears[module_name]
                quant_module = W8A8QuantLinear(
                    linear.in_features,
                    linear.out_features,
                    bias=linear.bias is not None,
                    weights=shard,
                    prefix=module_name,
                    dtype=linear.weight.dtype,
                    is_dynamic=quantized_linear_name_to_type[module_name] == "W8A8_DYNAMIC",
                ).to(device=device)
                _set_named_module(transformer, module_name, quant_module)
                replaced_names.add(module_name)
                replaced += 1

    missing_modules = sorted(set(quantized_linears.keys()) - replaced_names)
    if missing_modules:
        raise RuntimeError(
            f"Some quantized DiT linear modules were not loaded from safetensors: {missing_modules[:8]}"
            + ("..." if len(missing_modules) > 8 else "")
        )

    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    return replaced
