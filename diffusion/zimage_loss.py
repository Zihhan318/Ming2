from accelerate import scheduler
from transformers import CLIPTokenizer, CLIPTextModel, SiglipModel
import torch
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from tqdm import tqdm
from typing import Any, Mapping

import math
import copy
# import atorch

import torchvision
from diffusers import AutoencoderKL

try:
    from IPython import embed
except ImportError:
    embed = None
import argparse
import gc
import json
import os
import random
import threading

from collections import OrderedDict

import diffusers
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
)
#from .sd3_transformer import SD3Transformer2DModel
from .pipeline_z_image import ZImagePipeline
import torch.nn.functional as F
import torch.nn as nn

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_zimage_transformer_cls():
    impl = os.environ.get("MING_ZIMAGE_TRANSFORMER_IMPL", "profile").strip().lower()
    if impl == "profile":
        from .transformer_z_image import ZImageTransformer2DModel

        return ZImageTransformer2DModel
    if impl == "noprof":
        from .transformer_z_image_noprof import ZImageTransformer2DModel

        return ZImageTransformer2DModel
    if impl == "fusion":
        from .transformer_z_image_fusion import ZImageTransformer2DModel

        return ZImageTransformer2DModel
    raise ValueError(
        "Unsupported MING_ZIMAGE_TRANSFORMER_IMPL value: "
        f"{impl!r}. Expected 'profile', 'noprof' or 'fusion'."
    )

class ToClipMLP(nn.Module):                                 # 由两个线性层组成的MLP(多层感知机). LayerNorm:用于多模态对齐（比如把图像特征对齐到文本特征），以及特征流向下一个模块时数据分布一致
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(input_dim, 2048)               # 将输入的维度扩展或压缩到 2048 维。这是一个较宽的隐层，用于捕获更复杂的特征
        self.layer_norm1 = nn.LayerNorm(2048)               # 层归一化。这在 Transformer 架构中非常常见，它能保证神经元的输出不会因为数值过大而导致梯度消失或爆炸，让训练更稳定
        self.relu = nn.ReLU()                               # 非线性激活函数，给模型提供学习非线性特征的能力
        self.fc2 = nn.Linear(2048, output_dim)              # 将隐层的2048维输出映射到最终的输出维度。这通常是模型需要预测的特征维度 （如果这个 output_dim 是 512 或 768，通常就是为了对齐 CLIP 的特征空间）
        self.layer_norm2 = nn.LayerNorm(output_dim)         # 对最终输出再次进行归一化，确保输出特征在数值分布上是整齐的，方便后续计算相似度（如余弦相似度）

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states

class ZImageModel_withMLP(nn.Module):
    def __init__(self, transformer, vision_dim=1152, use_identity_mlp=False, text_encoder_norm=False):
        super().__init__()
        self.transformer = transformer
        self.dtype = torch.bfloat16
        self.mlp = ToClipMLP(vision_dim, 2560) if not use_identity_mlp else nn.Identity()       # 张量进入ToClipMLP为1152维，输出为2560维。如果use_identity_mlp为True，则直接使用nn.Identity()，即不对输入进行任何变换，直接传递给transformer
        # self.mlp_pool = ToClipMLP(vision_dim, 768)
        self.config = self.transformer.config
        self.in_channels = self.transformer.in_channels
        self.text_encoder_norm = text_encoder_norm              # 布尔开关，决定是否对文本编码器的输出作归一化

        # 需要搭配使用
        #if text_encoder_norm or use_identity_mlp:
        #    assert use_identity_mlp and text_encoder_norm

    
    def forward(self, hidden_states,
                    timestep,
                    encoder_hidden_states,
                    return_dict,
                    encoder_attention_mask=None,
                    extra_vit_input=None,
                    ref_hidden_states=None,
                     **kargs):

        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)    # 如果传入的是一串列表（比如多张图片或多段文本的特征），先用 torch.stack 把它们在第 0 维（Batch 维度）堆叠起来，变成一个整齐的矩阵 （统一格式）

        if self.text_encoder_norm:
            encoder_hidden_states = F.normalize(encoder_hidden_states, dim=-1) * 1000.0     # 归一化并放大
        
        encoder_hidden_states = self.mlp(encoder_hidden_states)                 # 强制变维 （1152->2560）
         
        # from IPython import embed
        # if torch.distributed.get_rank() == 0:     # 只允许主卡进入调试
        #     embed()
        # torch.distributed.barrier()
        if extra_vit_input is not None:
            encoder_hidden_states = torch.cat((encoder_hidden_states, extra_vit_input), dim=1)  # 将刚才 MLP 处理完的 encoder_hidden_states（维度已经是 2560）与额外的视觉输入 extra_vit_input 在长度维度（dim=1，序列长度）上拼起来     

        encoder_hidden_states = list(encoder_hidden_states.unbind(dim=0))       # 将堆叠好的 encoder_hidden_states 在第 0 维（Batch 维度）上拆分成一个个单独的张量，变成一个列表。每个元素都是一个样本的特征表示，方便后续逐个处理
        hidden_states = self.transformer(                       # 进入transformer核心模块，进行实际的特征变换和生成。传入的参数包括：
                    x=hidden_states,                            # 输入的隐藏状态，通常是当前的噪声图像特征
                    cap_feats=encoder_hidden_states,            # 处理过的文本或视觉特征，已经经过 MLP 变换和可能的拼接 （既融合了视觉模块、又被拆成了单个 Batch 样本的列表）
                    t=timestep,                                 # 当前的时间步
                    return_dict=False,                          # 是否返回字典格式的输出，这里设置为 False，表示直接返回张量    
                    ref_x=ref_hidden_states,                    # 参考图像的隐藏状态（如果有的话），可以用来引导生成过程
                     **kargs                                    # 其他可能的参数，比如注意力掩码等
                )
        return hidden_states

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()


class ZImageLoss(torch.nn.Module):
    def __init__(self, 
            model_path,                         # 预训练模型的主路径（包含 VAE 和 Transformer）
            vision_dim=2560,                    # 图像特征的输入维度，默认 2560（对应之前分析的 MLP 输入）
            scheduler_path=None,                # 调度器的预训练路径，通常包含噪声调度器的配置和权重
            mlp_state_dict=None,                # 必须提供的 MLP（投影层）权重字典
            torch_dtype=torch.float32,          # 模型权重的数据类型，默认为 float32，可以根据需要调整为 float16 或 bfloat16 以节省显存和加速计算
            device='cpu',                       # 设备设置，默认为 'cpu'
            use_identity_mlp=False,             # 是否使用恒等映射的 MLP（即不进行任何变换），如果为 True，则直接使用 nn.Identity()，否则使用 ToClipMLP 进行维度变换
            text_encoder_norm=False,            # 是否对文本编码器的输出进行归一化处理，如果为 True，则在 MLP 之前对 encoder_hidden_states 进行 L2 归一化并放大（乘以 1000.0），以便更好地对齐 CLIP 的特征空间
        ):
        super(ZImageLoss, self).__init__()      # 初始化父类的构造函数，确保模块能够正确地注册参数和子模块

        if device is not None:                          # 如果用户指定了设备（比如 'cuda:0' 或 'cpu'），就直接使用这个设备；否则，默认使用当前可用的 CUDA 设备（如果有的话），如果没有 CUDA，则回退到 CPU
            self.device = torch.device(device)   
        else:
            self.device = torch.device(torch.cuda.current_device())    

        self.scheduler_path = scheduler_path           
        self.vae = AutoencoderKL.from_pretrained(       # 从预训练模型中加载 VAE 模块，路径由 model_path 和子文件夹 "vae" 指定，权重的数据类型由 torch_dtype 指定
            model_path,
            subfolder="vae",
            torch_dtype=torch_dtype,
        )
        
        # self.vae.to(self.torch_type).to(self.device)
        self.vae.requires_grad_(False)                  # 冻结 VAE 的参数，确保在训练过程中不会更新 VAE 的权重，这样可以节省显存和计算资源，同时也保持 VAE 的预训练能力不受干扰

        transformer_cls = resolve_zimage_transformer_cls()
        logger.info("loading diffusion transformer implementation: %s", transformer_cls.__module__)
        self.train_model = transformer_cls.from_pretrained(        # 从预训练模型中加载 Transformer 模块，路径由 model_path 和子文件夹 "transformer" 指定，权重的数据类型由 torch_dtype 指定
            model_path, subfolder="transformer",
            torch_dtype=torch_dtype,
        )

        self.train_model = ZImageModel_withMLP(self.train_model, vision_dim=vision_dim, use_identity_mlp=use_identity_mlp, text_encoder_norm=text_encoder_norm)     # 将加载好的 Transformer 模块包装在 ZImageModel_withMLP 中，这个包装类添加了一个可选的 MLP 层（用于特征变换）和一些额外的功能（比如文本编码器归一化）。参数 vision_dim、use_identity_mlp 和 text_encoder_norm 用于配置这个包装类的行为

        assert mlp_state_dict is not None               # 确保用户提供了 MLP 的权重字典
        self.train_model.mlp.load_state_dict(mlp_state_dict, strict=True)           # 将提供的 MLP 权重字典加载到 train_model 的 MLP 层中，确保权重完全匹配（strict=True），如果不匹配会抛出错误

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.scheduler_path, subfolder="scheduler")              # 从预训练模型中加载噪声调度器，路径由 scheduler_path 和子文件夹 "scheduler" 指定，这个调度器负责在扩散过程中控制噪声的添加和去除策略
        self.noise_scheduler.config['use_dynamic_shifting'] = True              # 在调度器的配置中启用动态偏移（dynamic shifting），这可能是一个特定的功能，用于在扩散过程中动态调整噪声的添加方式，以提高生成质量或稳定性

        self.pipelines = ZImagePipeline(                # 初始化 ZImagePipeline，这个管道类负责将 VAE、Transformer 和调度器整合在一起，提供一个统一的接口来进行图像生成。传入的参数包括：
            vae=self.vae,                               # VAE 模块，用于编码和解码图像特征
            transformer=self.train_model,               # Transformer 模块，负责根据输入的特征进行图像生成
            text_encoder=None,                          # text_encoder 和 tokenizer 全都被设为了 None，
            tokenizer=None,                             # 证明它完全依赖 encoder_hidden_states 传入的特征 （包括从图片提取出来的特征）
            scheduler=self.noise_scheduler,             # 噪声调度器，控制扩散过程中的噪声添加和去除策略
        ).to(self.device)

    def set_trainable_params(self, trainable_params):                   # 这个方法用于设置哪些参数是可训练的，哪些是冻结的。参数 trainable_params 可以是一个字符串 'all'（表示所有参数都可训练），或者是一个包含参数名称片段的列表（表示只有名称中包含这些片段的参数可训练）。这个方法会根据用户的选择来调整模型中各个模块的 requires_grad 属性，从而控制训练过程中哪些参数会被更新。
        
        self.vae.requires_grad_(False)                                  # 无论用户选择什么，VAE 的参数都被固定为不可训练（冻结），确保 VAE 的预训练能力不受干扰，同时节省显存和计算资源

        if trainable_params == 'all':                                   # 如果用户选择 'all'，表示所有参数都可训练，那么直接将 train_model 的 requires_grad 设置为 True，允许所有参数在训练过程中被更新
            self.train_model.requires_grad_(True)
        else:
            self.train_model.requires_grad_(False)                      # 否则，先将 train_model 的所有参数设置为不可训练（冻结），然后根据用户提供的 trainable_params 列表，逐个检查 train_model 中的模块名称，如果模块名称中包含 trainable_params 中的任何一个片段，就将该模块的参数设置为可训练（requires_grad = True）。这样可以实现更细粒度的控制，允许用户只训练模型中的特定部分（比如只训练 Transformer 的某些层，或者只训练 MLP 层等）
            for name, module in self.train_model.named_modules():
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        for params in module.parameters():
                            params.requires_grad = True

        num_parameters_trainable = 0                                    # 统计可训练参数的数量，并打印出总参数数量和可训练参数数量的日志信息，帮助用户了解当前模型的训练状态和资源需求
        num_parameters = 0
        name_parameters_trainable = []
        for n, p in self.train_model.named_parameters():                # 遍历 train_model 中的所有参数，统计总参数数量（num_parameters）和可训练参数数量（num_parameters_trainable）。如果参数 p 的 requires_grad 属性为 True，说明它是可训练的，就将它的元素数量（p.data.nelement()）加到 num_parameters_trainable 中，并将参数名称 n 添加到 name_parameters_trainable 列表中；无论参数是否可训练，都将它的元素数量加到 num_parameters 中，以统计总参数数量。最后，打印出总参数数量和可训练参数数量的日志信息，帮助用户了解当前模型的训练状态和资源需求。
            num_parameters += p.data.nelement()
            if not p.requires_grad:
                continue  # frozen weights
            name_parameters_trainable.append(n)
            num_parameters_trainable += p.data.nelement()
        logger.info(f"number of all Diffusion parameters: {num_parameters}, trainable: {num_parameters_trainable}")
    

    def sample(self, encoder_hidden_states, steps=20, cfg=7.0, image_cfg=1.0, cfg_mode=1, seed=42, height=512, width=512, use_dynamic_shifting=False, extra_vit_input=None, ref_x=None, negative_encoder_hidden_states=None, profile_transformer_layers=False):       # 这个方法用于根据输入的 encoder_hidden_states（通常是从文本或图像提取的特征）进行图像采样生成。参数包括：
        
        encoder_hidden_states = list(encoder_hidden_states.unbind(dim=0))           # 首先将输入的 encoder_hidden_states 在第 0 维（Batch 维度）上拆分成一个个单独的张量，变成一个列表。每个元素都是一个样本的特征表示，方便后续逐个处理
        self.dit_total = 0.0                                                        # 初始化一个属性 dit_total，用于记录 DIT（Diffusion Inference Time，扩散推理时间）的总和，初始值为 0.0。这个属性可能会在后续的采样过程中被更新，用于统计整个采样过程中的平均推理时间
        self.vae_total = 0.0                                                        # 初始化一个属性 vae_total，用于记录 VAE（Variational Autoencoder，变分自编码器）的总和，初始值为 0.0。这个属性可能会在后续的采样过程中被更新，用于统计整个采样过程中的平均 VAE 处理时间
        self.ming_profile = {}

        pipeline_output = self.pipelines(                                           # 调用 ZImagePipeline 的 __call__ 方法，进行图像生成采样。传入的参数包括：
            prompt_embeds=encoder_hidden_states,                                    # 输入的特征表示，已经被拆成了单个样本的列表，每个元素都是一个样本的特征张量，这些特征通常是从文本或图像提取出来的，用于指导生成过程
            negative_prompt_embeds=[en*0 for en in encoder_hidden_states],          # 负面提示的特征表示，这里通过将 encoder_hidden_states 中的每个元素乘以 0 来创建一个新的列表，表示没有负面提示（即所有特征都被置零），这可能是为了实现某种形式的对比学习或消除不必要的特征影响
            guidance_scale=cfg,                                                     # 分类引导的强度，通常用于控制生成过程中的条件引导程度，数值越大表示引导越强，生成的图像会更倾向于符合输入的特征表示 
            #image_guidance_scale=image_cfg,
            #guidance_scale_mode=cfg_mode,
            generator=torch.manual_seed(seed),                                      # 设置随机种子，确保生成过程的可重复性。通过 torch.manual_seed(seed) 来固定随机数生成器的状态，使得每次运行时生成的图像都相同，方便调试和比较不同配置的效果
            num_inference_steps=steps,                                              # 采样的步骤数，控制生成过程中的迭代次数，通常步骤数越多，生成的图像质量越高，但计算时间也会增加。这里通过 num_inference_steps 参数来指定采样的总步数
            height=height,                                                          
            width=width,
            max_sequence_length=512,                                                # 最大序列长度，可能用于控制输入特征的长度，确保它们不会超过模型的处理能力，避免内存溢出或计算错误
            device=self.device,                                                     # 指定设备，确保所有的计算都在同一个设备上进行，避免跨设备的数据传输带来的性能损失和错误
            #extra_vit_input=extra_vit_input,   
            ref_hidden_states=ref_x,                                                # 参考图像的隐藏状态，如果提供了这个参数，生成过程可能会利用这些参考特征来引导生成，产生更符合参考图像风格或内容的结果
            profile_transformer_layers=profile_transformer_layers,
            #use_dynamic_shifting=use_dynamic_shifting
        )

        image = pipeline_output.images                                                                    # 从管道的输出中提取生成的图像，通常是一个 PIL Image 对象或一个张量，具体取决于 ZImagePipeline 的实现细节
        self.ming_profile = dict(getattr(self.pipelines, "_ming_profile", {}))
        self.dit_total = getattr(self.pipelines, "_ming_profile", {}).get("dit_total", 0.0)               # 从管道的性能分析配置（_ming_profile）中提取 DIT（Diffusion Inference Time，扩散推理时间）的总和，如果没有这个配置项，则默认为 0.0。这个值可能是在采样过程中通过某种性能分析工具（比如 Ming）记录的，用于统计整个采样过程中的平均推理时间
        self.vae_total = getattr(self.pipelines, "_ming_profile", {}).get("vae_total", 0.0)               # 从管道的性能分析配置（_ming_profile）中提取 VAE（Variational Autoencoder，变分自编码器）的总和，如果没有这个配置项，则默认为 0.0。这个值可能是在采样过程中通过某种性能分析工具（比如 Ming）记录的，用于统计整个采样过程中的平均 VAE 处理时间

        return image  
