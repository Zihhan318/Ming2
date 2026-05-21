# Quantization Routing 2026-05-18

## Verdict

`LLM full W8A8` is not the best primary route right now.

The current evidence supports switching the image-generation path to:

- `LLM` stays on the stable route: original `bf16` or `W8A16 + attention FLOAT`
- `DiT` becomes the next dedicated quantization target

## Why full LLM W8A8 is not worth prioritizing

1. The latest full-coverage attempt did not produce a usable exported artifact.
   - Coverage report exists at [quantized/ming-flash-omni-2.0-llm-w8a8-full_expert_coverage.json](/home/zzh/Ming2/quantized/ming-flash-omni-2.0-llm-w8a8-full_expert_coverage.json)
   - Export directory [quantized/ming-flash-omni-2.0-llm-w8a8-full](/home/zzh/Ming2/quantized/ming-flash-omni-2.0-llm-w8a8-full) is empty

2. MoE coverage improved, but is still not clean enough for a confident full-W8A8 recommendation.
   - `avg_layer_coverage_ratio = 0.9433`
   - `min_layer_coverage_ratio = 0.8984`
   - `layers_with_gaps = 30 / 31`
   - `uncovered_expert_slots = 450`

3. The previous W8A8 route already showed calibration failure symptoms.
   - [quantize_attnfloat.log](/home/zzh/Ming2/quantize_attnfloat.log) reports `747` quantized linear modules without `input_scale`
   - The same log contains many `Falling back to FLOAT export` warnings for MoE expert projections

4. The existing stable text/imagegen validation path already converged on `W8A16 + attention FLOAT`.
   - [quantized_eval_20_zh.json](/home/zzh/Ming2/quantized_eval_20_zh.json)
   - [quantized/ming-flash-omni-2.0-llm-w8a16-attnfloat-mix/eval_compare_20_zh.md](/home/zzh/Ming2/quantized/ming-flash-omni-2.0-llm-w8a16-attnfloat-mix/eval_compare_20_zh.md)

## Why DiT-only is the better next step

1. The image-generation stack is already separable.
   - [modeling_bailingmm2.py](/home/zzh/Ming2/modeling_bailingmm2.py) supports `image_gen_condition_embeds`
   - [test_infer_imagegen_npu.py](/home/zzh/Ming2/test_infer_imagegen_npu.py) already has `--extract-condition-only`

2. The pipeline already supports a practical cut between:
   - `LLM condition extraction`
   - `DiT denoising`
   - `VAE decode`

3. This makes DiT quantization easier to evaluate independently.
   - Text quality risk stays contained in the stable LLM path
   - Image quality and DiT latency can be measured directly

## New enablement added in this pass

1. [diffusion/pipeline_z_image.py](/home/zzh/Ming2/diffusion/pipeline_z_image.py) can now dump real DiT call inputs during denoising when:
   - `MING_DIT_CALIB_OUTPUT_DIR=/path/to/output`
   - `MING_DIT_CALIB_CAPTURE_STEPS=0,last`
   - `MING_DIT_CALIB_CASE_ID=...`

2. [build_dit_calib_dataset.py](/home/zzh/Ming2/build_dit_calib_dataset.py) builds DiT calibration bundles from imagegen cases by:
   - extracting stable condition embeddings once
   - rerunning image generation with precomputed condition embeddings
   - capturing the actual DiT tensors used at selected denoising steps

## Recommended next execution

1. Keep text validation on the current stable LLM path.
2. Build a DiT calibration bundle from `imagegen_eval_cases_10.json`.
3. Use the captured tensors to prototype DiT-only PTQ on the transformer submodule.
