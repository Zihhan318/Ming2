import argparse
import json
import os
import time
from bisect import bisect_left
from contextlib import nullcontext
from pathlib import Path

DEFAULT_HF_CACHE = Path(__file__).resolve().parent / ".hf-cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE))
os.environ.setdefault("HF_MODULES_CACHE", str(DEFAULT_HF_CACHE / "modules"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoProcessor

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration


def configure_npu_runtime():
    if not hasattr(torch, "npu") or not torch.npu.is_available():       # 没有npu则跳过
        return

    torch.npu.set_compile_mode(jit_compile=False)                       # 关闭在线动态编译JIT. 昇腾底层的 CANN喜欢把算子即时编译成最适合当前数据形状（Shape）的机器码
    torch.npu.config.allow_internal_format = False                      # 禁用npu私有数据格式。npu为了计算效率有时会把PyTorch 标准的张量格式（比如 NCHW）转换成它私有的 5HD 或 FRACTAL_NZ 格式。但这样会带来Host-Device 数据同步阻塞


def ensure_local_hf_cache():                                            # 为 HuggingFace 模型强制指定一个本地的独立缓存目录
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


def get_attn_implementation(device: torch.device) -> str:             
    # FlashAttention2 is only relevant on CUDA and should degrade cleanly elsewhere.
    if device.type != "cuda":
        return "eager"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "eager"
    return "flash_attention_2"

# 音频先过audio,再过linear_proj_audio变成和文本同维度的 audio_embeds. 图像经过linear_proj后变成和文本同维度的image_embeds. 文本直接过word_embeddings. 这些模块最后都会变成同一个hidden_size 空间里的表示，再送进LLM主干去统一推理
# LLM主干层平均切到其他卡

def build_imagegen_device_map(num_layers: int, num_devices: int):
    if num_devices <= 1:
        return None

    llm_devices = num_devices - 1
    device_map = {}
    layer_boundaries = [round(i * num_layers / llm_devices) - 1 for i in range(1, llm_devices + 1)]     # 计算每个设备负责的层范围边界
    for i in range(num_layers):
        device_id = bisect_left(layer_boundaries, i) + 1                                                # 用二分查找判断第i层落在哪个边界区间里，归属哪个设备。设备编号从1开始。
        if device_id > llm_devices:                                                                     # 兜底：万一卡号越界了，就用求余数的方式强行把层发出去
            device_id = i % llm_devices + 1
        device_map[f"model.model.layers.{i}"] = device_id                                               # 把结果写进字典里

    # Reserve device 0 for vision, heads and image-gen modules.
    device_map["vision"] = 0                                                                            # 视觉模块 
    device_map["audio"] = 0                                                                             # 音频模块
    device_map["linear_proj"] = 0
    device_map["linear_proj_audio"] = 0
    device_map["model.model.word_embeddings"] = 0                                                       # 入口：文字转向量 （词嵌入）
    device_map["model.model.word_embeddings.weight"] = 0
    device_map["model.model.norm"] = 0
    device_map["model.model.norm.weight"] = 0
    device_map["model.lm_head"] = 0                                                                     # 出口：向量转文字 （输出头）
    device_map["model.lm_head.weight"] = 0
    device_map[f"model.model.layers.{num_layers - 1}"] = 0                                              # 把最后一层（索引为 num_layers - 1）抓回0卡 （所以0卡需要等其他几张卡跑完收尾）
    return device_map                                                                                   # model.model.layers.{i}即Transformer / MoE 层，也就是LLM主干层


def move_inputs_to_device(inputs, device: torch.device):                                    
    for key, value in inputs.items():                                                                   # inputs 是一个字典，里面装着 Prompt 转换后的 input_ids、图片的 pixel_values 等
        if not isinstance(value, torch.Tensor):                                                         # 如果这个值不是 PyTorch 的张量（比如是个字符串或列表），直接跳过
            continue
        if value.is_floating_point() and key in {                                                       # 如果是浮点数，且是图像、视频或音频特征：
            "pixel_values",
            "pixel_values_videos",
            "audio_feats",
            "pixel_values_reference",
        }:
            inputs[key] = value.to(device=device, dtype=torch.bfloat16)                                 # 将这些多模态输入转成bfloat16精度 （npu原生支持的最好）
        else:
            inputs[key] = value.to(device=device)                                                       # 文字的ID (Long类型，整数)，不需要转精度，直接从cpu内存搬运到npu显存里
    return inputs


def validate_imagegen_assets(model_path: str):
    model_root = Path(model_path)
    mlp_config_path = model_root / "mlp" / "config.json"                                                # config.json里记录了模型的技术架构
    if not mlp_config_path.exists():
        raise FileNotFoundError(f"Missing image generation config: {mlp_config_path}")

    with mlp_config_path.open("r") as f:   
        metax_config = json.load(f)

    dit_type = metax_config.get("dit_type", "sd3")                                                       # 确定DiT类型 (Diffusion Transformer:目前最火的生成架构)。判断是兼容SD3 (stable diffusion 3)的标准架构，还是属于zimage的特殊架构
    required_subdirs = ["mlp", "connector", "byt5"]                                                      # 基础清单（所有模式都有）
    if dit_type == "zimage":                                                                             # zimage的增强清单： 
        required_subdirs.extend(["vae", "transformer", "scheduler"])                                     # vae：编解码器，负责把潜空间的向量还原成像素图片；treansformer：图像生成的核心模块；scheduler：调度器，控制去噪过程的步数。

    missing = [name for name in required_subdirs if not (model_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Image generation weights are incomplete for this model path. "
            f"Missing subdirectories: {missing}. Expected under {model_root}."
        )

    return {"dit_type": dit_type, "required_subdirs": required_subdirs}


def validate_reference_image(image_path: str | None):
    if not image_path:                                                                                    # 如果只通过文字生图（Text-to-Image），没有提供参考图路径，则返回None，跳过后续步骤 
        return None

    image_path = Path(image_path)                                                                         # 如果传入了路径，则把 image_path 转化成一个 Path 对象
    if not image_path.exists():
        raise FileNotFoundError(f"Missing reference image: {image_path}")
    return image_path.resolve()                                                                           # resolve: 将相对路径变为绝对路径，同时把路径中../或者./这种冗余抹掉 （防止不同卡同时工作但工作目录不同而出错）


def build_messages(args):
    content = []                                                                                          # 准备一个列表
    if args.image:                                                                                        # 如果命令带了--image参数：
        content.append({"type": "image", "image": args.image})                                            # 先往content里塞一个字典，标记是“图片”类型，并给出图片路径 （模型先看图，后读文）
    content.append({"type": "text", "text": args.prompt})                                                 # 把在命令行中输入的--prompt塞进content
    return [{"role": "HUMAN", "content": content}]                                                        # 打包成对话格式：把content塞进带有role角色标签的列表里


def parse_args():                                                                                         # “参数解析器”
    parser = argparse.ArgumentParser(
        description="Image generation or editing entry for Ascend/CUDA/CPU."
    )
    parser.add_argument("--model-path", default=".")                                                      # 指定权重文件放在哪。如果不传则默认是当前目录（.）
    parser.add_argument("--code-path", default=".")                                                       # 指定代码库的路径，通常用于加载自定义的算子或模块
    parser.add_argument("--prompt", required=True)                                                        # 唯一的必填项
    parser.add_argument("--image", help="Optional reference image path for editing mode.")                # 可选，若提供了图片路径，程序会进入“编辑模式”或“图生图模式”。这就是 build_messages 里判断是否加入图片信息的来源
    parser.add_argument("--output", default="generated_imgs/output.png")                                  # 结果存在哪
    parser.add_argument("--seed", type=int, default=42)                                                   # 设置随机数种子，默认值为42
    parser.add_argument("--tensor-parallel-devices", type=int, default=1)                                 # 张量并行参数，需要是整数，若无输入默认用0卡。但实际我们在build_imagegen_device_map执行的是Pipeline Parallelism（流水线并行/层切分）
    parser.add_argument(
        "--num-runs",                                                                                     # 在同一个程序里连续跑几次生图任务。默认 4 次：1 次 skip，1 次 warmup，2 次 active
        type=int,
        default=4,
        help="Number of consecutive generate() runs to execute in the same process.",
    )
    parser.add_argument(
        "--profile-output",                                                                               # 性能分析的开关：若传了目录路径，程序就会启动Ascend PyTorch Profiler
        help="Optional output directory for a single-step Ascend PyTorch Profiler trace.",
    )

    # profiler的时间控制台：
    parser.add_argument(
        "--profile-skip-first",                                                                             # 跳过前N次运行。默认 1 次，用于避开第一次 generate 的冷启动
        type=int,
        default=1,
        help="Number of initial runs to skip before profiler starts recording.",
    )
    parser.add_argument(
        "--profile-warmup-runs",                                                                            # 预热N次。默认 1 次，让 profiler 和 runtime 进入稳定状态
        type=int,
        default=1,
        help="Number of warmup runs inside the profiler schedule before active collection.",
    )
    parser.add_argument(
        "--profile-active-runs",                                                                            # profiler记录的次数。默认 2 次，抓两轮稳态 generate
        type=int,
        default=2,
        help="Number of consecutive runs to collect after skip/warmup.",
    )
    parser.add_argument(
        "--profile-level",                                                                                  # profiler等级，默认 Level 1，先观察稳态推理和互连数据
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="Ascend profiler level. Use 1 or 2 when you need richer device-side details.",
    )
    return parser.parse_args()


def build_profile_context(profile_output: str | None, device: torch.device, args):
    if not profile_output:                                                                                  # 如果没传--profile-output，则什么都不干
        return nullcontext()
    if device.type != "npu":                                                                                # 如果不是npu则无法开启profiler监控
        raise RuntimeError("--profile-output currently only supports Ascend NPU runtime.")

    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu is required when --profile-output is enabled."
        ) from exc

    profile_dir = Path(profile_output)
    profile_dir.mkdir(parents=True, exist_ok=True)
    profiler_level = {                                                                                       #  把在命令行输入的数字（0, 1, 2）翻译成了npu底层的配置枚举值
        0: torch_npu.profiler.ProfilerLevel.Level0,
        1: torch_npu.profiler.ProfilerLevel.Level1,
        2: torch_npu.profiler.ProfilerLevel.Level2,
    }[args.profile_level]
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=profiler_level,
        export_type=torch_npu.profiler.ExportType.Text,                                                      # 要求profiler在导出结果时额外生成一份文本/表格数据 （kernel_details.csv，operator_details.csv）
        sys_interconnection=True,                                                                            # 打开系统互连采集，补充 PCIe/HCCS 链路数据
    )

    return torch_npu.profiler.profile(                                                                       # 调用昇腾的torch_npu.profiler.profile接口，拉起性能记录
        activities=[                                                                                         # 记录CPU下发指令和NPU计算矩阵的活动
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(                                                                # profiler排班：
            wait=args.profile_skip_first,                                                                    # 先冷跑几张图
            warmup=args.profile_warmup_runs,                                                                 # 热身几张图
            active=args.profile_active_runs,                                                                 # 开始做profiling
            repeat=1,                                                                                        # 这个循环跑几次
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),                       # 当profiler结束时，将二进制数据转换成JSON 格式（用于 Chrome 查看器）和 TensorBoard 格式。存在之前指定的 --profile-output 目录里
        record_shapes=True,                                                                                  # 记录张量大小
        profile_memory=True,                                                                                 # 记录显存怎么变多变少的
        with_stack=True,                                                                                     # 记录npu上每一个色块对应的python代码行号
        experimental_config=experimental_config,
    )


def build_output_path(base_output: str, run_idx: int, num_runs: int) -> Path:                                # 生成路径对象。若生成多张图则连续编号
    output_path = Path(base_output)
    if num_runs == 1:
        return output_path
    return output_path.with_name(f"{output_path.stem}_run{run_idx + 1}{output_path.suffix}")


def run_imagegen(args):
    if args.num_runs < 1:                                                                                    # 检查参数合法性
        raise ValueError("--num-runs must be at least 1.")
    if args.profile_skip_first < 0 or args.profile_warmup_runs < 0 or args.profile_active_runs < 0:
        raise ValueError("Profiler schedule values must be non-negative.")
    if args.profile_output and (args.profile_skip_first + args.profile_warmup_runs + args.profile_active_runs > args.num_runs):
        raise ValueError(
            "Profiler schedule exceeds total runs: "
            f"skip_first({args.profile_skip_first}) + warmup({args.profile_warmup_runs}) + "
            f"active({args.profile_active_runs}) must be <= num_runs({args.num_runs})."
        )

    stage_timings = {}                 # 字典，记录阶段耗时
    run_timings = []                   # 列表，记录每轮生成任务的具体耗时
    saved_outputs = []                 # 列表，存放生成图片的路径

    def mark_stage(stage_name: str, start_time: float):                                                      # 核心计时器
        stage_timings[stage_name] = time.time() - start_time                                                 # 计算每一步花了多久

    overall_start_time = time.time()

    stage_start = time.time()
    ensure_local_hf_cache()                                                                                  # 确保HuggingFace 的模型缓存路径是对的
    configure_npu_runtime()                                                                                  # 针对npu的核心设置，比如设置 jit_compile=False 或内存申请策略
    asset_info = validate_imagegen_assets(args.model_path)
    reference_image = validate_reference_image(args.image)
    device = get_runtime_device()
    attn_implementation = get_attn_implementation(device)                                                    # 选择注意力实现。在 NPU 上，它通常会尝试使用 FlashAttention 来提速
    use_device_map = args.tensor_parallel_devices and args.tensor_parallel_devices > 1                       # 如果 args.tensor_parallel_devices 有值且值>1 ，则 use_device_map 为true ，后续进行层切分
    device_map = None
    if use_device_map:
        device_map = build_imagegen_device_map(num_layers=32, num_devices=args.tensor_parallel_devices)      # 指定模型总共有32层，按照 build_imagegen_device_map 指定的方式去切分
    mark_stage("setup", stage_start)

    print(f"mode: {'edit' if reference_image else 'text-to-image'}")                                         # 将配置参数打印在终端
    print(f"runtime_device: {device}")
    print(f"attn_implementation: {attn_implementation}")
    print(f"tensor_parallel_devices: {args.tensor_parallel_devices}")
    print(f"num_runs: {args.num_runs}")
    print(f"dit_type: {asset_info['dit_type']}")
    if reference_image is not None:
        print(f"reference_image: {reference_image}")

    stage_start = time.time()                                                                                # 开始计时
    model = BailingMM2NativeForConditionalGeneration.from_pretrained(                                        # 从 args.model_path 加载一个带图像生成能力的预训练模型，并按指定配置初始化
        args.model_path,
        torch_dtype=torch.bfloat16,                                                                          # 昇腾 NPU 最喜欢bfloat16（BF16）格式，它比 float16（FP16）动态范围更广，不容易出现数值溢出
        attn_implementation=attn_implementation,                                                             # 如果是flash_attention，NPU 会调用专门优化的算子；如果是 sdpa，则是标准算子
        load_image_gen=True,                                                                                 # 只生图
        load_talker=False,                                                                                   # 关掉语音模块
        device_map=device_map,
    )
    if not use_device_map:                                                                                   # 单卡模式则把模型送到指定的device （通常是npu:0）
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(dtype=torch.bfloat16)                                                               # 多卡模式只转换数据类型
    model.eval()                                                                                             # 把模型设置为评估模式，会关闭掉那些只有训练时才需要的“随机干扰项”
    mark_stage("model_load", stage_start)                                                                    # model_load: 调用前面的 mark_stage 函数，记录下从 from_pretrained 开始到 model.eval() 结束总共花了多少秒。

    stage_start = time.time()
    processor = AutoProcessor.from_pretrained(args.code_path, trust_remote_code=True)                        # AutoProcessor：通用翻译官。包含了 Image Processor（图片预处理器）和 Tokenizer（文本分词器）
    messages = build_messages(args)                                                                          # 构建对话模版

    text = processor.apply_chat_template(messages, add_generation_prompt=True)                               # 把原始的对话列表（Role: Human, Content: ...）套入特定的模板中 （模型训练时见过特定的格式）
    image_inputs, _, _ = processor.process_vision_info(messages)                                             # 处理 messages 里的普通图片信息 （把图片像素转成 NPU 认识的四维张量 [Batch, Channel, Height, Width]）
    ref_image_inputs = processor.process_reference_vision_info(messages) if args.image else None             # 如果传入了--image参数（即args.image为真）才执行，调用专门的 process_reference_vision_info 函数处理

    inputs = processor(                                                                                      # 统一封装
        text=[text],                                                                                         # 包装好的文本 Token 序列
        images=image_inputs,                                                                                 # 处理过的视觉张量
        return_tensors="pt",                                                                                 # 要求返回 PyTorch 格式的 Tensor
        image_gen_ref_images=ref_image_inputs,                                                               # 如果是编辑模式，带上参考图张量
    )
    # The processor derives image_gen_text from the chat-formatted prompt, which can
    # collapse to an empty string for plain text prompts without quoted substrings.
    # Override it with the original user prompt so the ByT5 text-conditioning path
    # receives the actual prompt content.
    inputs["image_gen_text"] = [args.prompt]                                                                 # 将“prompt”强行塞入inputs字典
    input_device = torch.device("npu:0") if use_device_map else device                                       # 多卡模式强制指定npu:0接收原始输入（Vision/Text Embedding）
    inputs = move_inputs_to_device(inputs, input_device)                                                     # 将inputs字典里所有Tensor从系统内存（RAM）拷贝到 NPU 显存里 （吃PCle带宽）
    mark_stage("input_prepare", stage_start)                                                                 # 记录名称：input_prepare 阶段的耗时

    # 原来的实现保留，不直接删除。
    # stage_start = time.time()
    # profiler_ctx = build_profile_context(args.profile_output, device, args)
    # with profiler_ctx as prof:                                                                               # 创建并启动昇腾的 Profiler 环境。触发msprof工具
    #     for run_idx in range(args.num_runs):                                                                 # 根据设置的 num_runs 重复执行模型生成
    #         run_start = time.time()
    #         with torch.no_grad():                                                                            # 告诉 PyTorch 不需要计算梯度，从而节省显存并提速
    #             image = model.generate(                                                                      # 开始推理。调用模型生成内容的函数，通常是扩散模型：把之前 inputs 里的文本特征，经过多次去噪迭代，最终转变成一张图像
    #                 **inputs,
    #                 image_gen=True,
    #                 image_gen_seed=args.seed,                                                                # 随机性的种子：扩散模型生成图片本质上是从一堆“随机噪声”开始去噪的。通过固定seed（比如42）可以保证每次生成的图片是一样的，否则可能会有差别
    #             )
    #         if prof is not None:                                                                             # 在profiler的active范围内分析npu算子耗时
    #             prof.step()
    #         run_timings.append(time.time() - run_start)                                                      # 记录当前时间并计算执行时间
    #
    #         output_path = build_output_path(args.output, run_idx, args.num_runs)
    #         output_path.parent.mkdir(parents=True, exist_ok=True)
    #         image.save(output_path)
    #         saved_outputs.append(output_path)
    # mark_stage("generate", stage_start)
    # stage_timings["save_output"] = 0.0

    stage_start = time.time()
    generated_images = []
    count = args.num_runs

    if args.profile_output:
        try:
            import torch_npu
        except ImportError as exc:
            raise RuntimeError("torch_npu is required for profiling.") from exc

        if count != 5:
            print(
                f"warning: current profiler schedule is wait=2, warmup=2, active=1, "
                f"so num_runs最好设成5；当前是 {count}"
            )

        profile_dir = Path(args.profile_output)
        profile_dir.mkdir(parents=True, exist_ok=True)

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,
            data_simplification=False,
        )

        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=True,
            record_shapes=True,
            profile_memory=True,
            schedule=torch_npu.profiler.schedule(wait=2, warmup=2, active=1, repeat=1),
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        ) as prof:
            for run_idx in range(count):
                run_start = time.time()
                with torch.no_grad():
                    image = model.generate(
                        **inputs,
                        image_gen=True,
                        image_gen_seed=args.seed,
                    )
                prof.step()
                run_timings.append(time.time() - run_start)
                generated_images.append((run_idx, image))
    else:
        for run_idx in range(count):
            run_start = time.time()
            with torch.no_grad():
                image = model.generate(
                    **inputs,
                    image_gen=True,
                    image_gen_seed=args.seed,
                )
            run_timings.append(time.time() - run_start)
            generated_images.append((run_idx, image))

    mark_stage("generate", stage_start)

    stage_start = time.time()
    for run_idx, image in generated_images:
        output_path = build_output_path(args.output, run_idx, args.num_runs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        saved_outputs.append(output_path)
    mark_stage("save_output", stage_start)
    print(f"device: {device}")
    print(f"device_map: {device_map}")
    print(f"attn_implementation: {attn_implementation}")
    print(f"mode: {'edit' if reference_image else 'text-to-image'}")
    print(f"stage_setup: {stage_timings['setup']:.2f}s")
    print(f"stage_model_load: {stage_timings['model_load']:.2f}s")
    print(f"stage_input_prepare: {stage_timings['input_prepare']:.2f}s")
    print(f"stage_generate: {stage_timings['generate']:.2f}s")
    print(f"stage_save_output: {stage_timings['save_output']:.2f}s")
    for run_idx, elapsed in enumerate(run_timings, 1):
        print(f"run_{run_idx}_generate: {elapsed:.2f}s")
    print(f"elapsed: {time.time() - overall_start_time:.2f}s")
    if args.profile_output:
        print(f"profile_saved: {Path(args.profile_output).resolve()}")
    for output_path in saved_outputs:
        print(f"saved: {output_path}")


if __name__ == "__main__":
    run_imagegen(parse_args())
