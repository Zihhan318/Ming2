import argparse
import json
import math
import struct
from collections import defaultdict
from pathlib import Path


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def human_count(value: float) -> str:
    abs_value = abs(value)
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs_value >= scale:
            return f"{value / scale:.3f}{suffix}"
    return str(int(value))


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def numel(shape) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_len))


def add_stat(target: dict, key: str, param_count: int, byte_count: int):
    target[key]["params"] += param_count
    target[key]["bytes"] += byte_count


def classify_root_key(key: str) -> str:
    if key.startswith("audio."):
        return "omni.audio_encoder"
    if key.startswith("vision."):
        return "omni.vision_encoder"
    if key.startswith("linear_proj_audio."):
        return "omni.audio_projector"
    if key.startswith("linear_proj."):
        return "omni.image_projector"
    if key.startswith("model."):
        if ".mlp.experts." in key:
            return "llm.sparse_experts"
        if ".mlp.shared_experts." in key:
            return "llm.shared_experts"
        if ".mlp.image_gate." in key:
            return "llm.router_image"
        if ".mlp.audio_gate." in key:
            return "llm.router_audio"
        if ".mlp.gate." in key:
            return "llm.router_text"
        return "llm.dense"
    return "other"


def classify_talker_key(key: str) -> str:
    prefix = key.split(".", 1)[0]
    mapping = {
        "model": "talker.llm_backbone",
        "cfm": "talker.cfm",
        "aggregator": "talker.aggregator",
        "spk_head": "talker.spk_head",
        "stop_head": "talker.stop_head",
    }
    return mapping.get(prefix, f"talker.{prefix}")


def classify_vae_key(key: str) -> str:
    prefix = key.split(".", 1)[0]
    mapping = {
        "encoder": "vae.encoder",
        "decoder": "vae.decoder",
    }
    return mapping.get(prefix, f"vae.{prefix}")


def accumulate_header_stats(header: dict, classifier, stats: dict):
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        shape = meta["shape"]
        dtype = meta["dtype"]
        param_count = numel(shape)
        byte_count = param_count * DTYPE_BYTES[dtype]
        add_stat(stats, classifier(key), param_count, byte_count)


def scan_root_model(model_path: Path, stats: dict):
    index_path = model_path / "model.safetensors.index.json"
    weight_index = json.loads(index_path.read_text())
    shard_names = sorted(set(weight_index["weight_map"].values()))
    for shard_name in shard_names:
        header = read_safetensors_header(model_path / shard_name)
        accumulate_header_stats(header, classify_root_key, stats)


def scan_single_safetensors(path: Path, classifier, stats: dict):
    header = read_safetensors_header(path)
    accumulate_header_stats(header, classifier, stats)


def summarize_groups(stats: dict) -> dict:
    ordered = {}
    for key in sorted(stats.keys()):
        params = stats[key]["params"]
        byte_count = stats[key]["bytes"]
        ordered[key] = {
            "params": params,
            "params_human": human_count(params),
            "bytes": byte_count,
            "bytes_human": human_bytes(byte_count),
        }
    return ordered


def compute_active_estimates(model_path: Path, stats: dict) -> dict:
    root_config = json.loads((model_path / "config.json").read_text())
    llm_config = root_config["llm_config"]
    num_experts = llm_config["num_experts"]
    num_experts_per_tok = llm_config["num_experts_per_tok"]

    llm_dense = stats["llm.dense"]["params"]
    llm_shared = stats["llm.shared_experts"]["params"]
    llm_sparse = stats["llm.sparse_experts"]["params"]
    router_text = stats["llm.router_text"]["params"]
    router_image = stats["llm.router_image"]["params"]
    router_audio = stats["llm.router_audio"]["params"]

    sparse_active = llm_sparse * num_experts_per_tok / num_experts
    sparse_active_int = int(round(sparse_active))

    text_active = llm_dense + llm_shared + router_text + sparse_active_int
    image_active = (
        text_active
        - router_text
        + router_image
        + stats["omni.vision_encoder"]["params"]
        + stats["omni.image_projector"]["params"]
    )
    audio_active = (
        text_active
        - router_text
        + router_audio
        + stats["omni.audio_encoder"]["params"]
        + stats["omni.audio_projector"]["params"]
    )
    talker_total = (
        stats["talker.llm_backbone"]["params"]
        + stats["talker.cfm"]["params"]
        + stats["talker.aggregator"]["params"]
        + stats["talker.spk_head"]["params"]
        + stats["talker.stop_head"]["params"]
    )
    vae_total = stats["vae.encoder"]["params"] + stats["vae.decoder"]["params"]

    return {
        "moe": {
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "sparse_experts_total_params": llm_sparse,
            "sparse_experts_total_params_human": human_count(llm_sparse),
            "sparse_experts_active_params_per_token_est": sparse_active_int,
            "sparse_experts_active_params_per_token_est_human": human_count(sparse_active_int),
        },
        "active_params_estimates": {
            "text_understanding": {
                "params": text_active,
                "params_human": human_count(text_active),
            },
            "image_understanding": {
                "params": image_active,
                "params_human": human_count(image_active),
            },
            "audio_understanding": {
                "params": audio_active,
                "params_human": human_count(audio_active),
            },
            "talker_tts": {
                "params": talker_total,
                "params_human": human_count(talker_total),
            },
            "audio_vae_decode": {
                "params": vae_total,
                "params_human": human_count(vae_total),
            },
        },
        "matmul_flops_proxy_per_token": {
            "text_understanding": int(2 * text_active),
            "image_understanding": int(2 * image_active),
            "audio_understanding": int(2 * audio_active),
        },
        "notes": [
            "Parameter counts are exact, scanned from safetensors headers without loading weights.",
            "Active parameter counts are estimates. For MoE layers they assume num_experts_per_tok / num_experts of sparse expert weights are active per token.",
            "The FLOPs proxy is a first-order matmul proxy (about 2 * active params per token) and excludes attention quadratic terms, sequence length effects, convolutions, and cache reuse.",
        ],
    }


def build_output(model_path: Path) -> dict:
    stats = defaultdict(lambda: {"params": 0, "bytes": 0})
    scan_root_model(model_path, stats)
    scan_single_safetensors(model_path / "talker" / "model.safetensors", classify_talker_key, stats)
    scan_single_safetensors(model_path / "talker" / "vae" / "model.safetensors", classify_vae_key, stats)

    group_summary = summarize_groups(stats)
    total_params = sum(item["params"] for item in stats.values())
    total_bytes = sum(item["bytes"] for item in stats.values())

    return {
        "model_path": str(model_path),
        "totals": {
            "params": total_params,
            "params_human": human_count(total_params),
            "bytes": total_bytes,
            "bytes_human": human_bytes(total_bytes),
        },
        "groups": group_summary,
        "derived": compute_active_estimates(model_path, stats),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Ming model parameter counts and rough active-load estimates.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    result = build_output(Path(args.model_path))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
