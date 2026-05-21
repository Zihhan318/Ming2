import argparse
import json
import re
import subprocess
import time
from pathlib import Path


OVERALL_RE = re.compile(r"^overall:\s*([0-9.]+)s$", re.MULTILINE)
GENERATE_RE = re.compile(r"^generate:\s*([0-9.]+)s$", re.MULTILINE)
MODEL_LOAD_RE = re.compile(r"^model_load:\s*([0-9.]+)s$", re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser(description="Run image-generation eval suite through the stable NPU entrypoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--quantized-llm-path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-devices", type=int, default=6)
    parser.add_argument("--device-ids", default="1,2,3,4,5,6")
    return parser.parse_args()


def load_cases(path: str):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases file must be a non-empty JSON list")
    return cases


def extract_metric(pattern: re.Pattern[str], text: str):
    match = pattern.search(text)
    return float(match.group(1)) if match else None


def run_case(args, case: dict, output_root: Path):
    output_path = output_root / f"{case['id']}.png"
    cmd = [
        "python",
        "test_infer_imagegen_npu.py",
        "--model-path",
        args.model_path,
        "--code-path",
        args.code_path,
        "--prompt",
        case["prompt"],
        "--output",
        str(output_path),
        "--seed",
        str(args.seed),
        "--tensor-parallel-devices",
        str(args.tensor_parallel_devices),
        "--device-ids",
        args.device_ids,
        "--num-runs",
        "1",
    ]
    if args.quantized_llm_path:
        cmd.extend(["--quantized-llm-path", args.quantized_llm_path])
    if case.get("image"):
        cmd.extend(["--image", case["image"]])

    start = time.time()
    completed = subprocess.run(
        cmd,
        cwd=args.code_path,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.time() - start
    output_text = (completed.stdout or "") + (completed.stderr or "")
    return {
        "id": case["id"],
        "prompt": case["prompt"],
        "image": case.get("image"),
        "output": str(output_path),
        "returncode": completed.returncode,
        "wall": wall,
        "overall": extract_metric(OVERALL_RE, output_text),
        "generate": extract_metric(GENERATE_RE, output_text),
        "model_load": extract_metric(MODEL_LOAD_RE, output_text),
        "log": output_text,
    }


def main():
    args = parse_args()
    cases = load_cases(args.cases_file)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] running {case['id']}")
        result = run_case(args, case, output_root)
        print(
            f"[{idx}/{len(cases)}] rc={result['returncode']} "
            f"wall={result['wall']:.2f}s overall={result['overall']} generate={result['generate']}"
        )
        results.append(result)

    payload = {
        "model_path": args.model_path,
        "quantized_llm_path": args.quantized_llm_path,
        "tensor_parallel_devices": args.tensor_parallel_devices,
        "device_ids": args.device_ids,
        "seed": args.seed,
        "cases_file": args.cases_file,
        "results": results,
    }
    Path(args.json_output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
