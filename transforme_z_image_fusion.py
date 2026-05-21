# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of the implementations are adapted from https://github.com/Tongyi-MAI/Z-Image/blob/main/src/zimage/transformer.py. 
# Based on this code, we made modifications and extensions, including adding Image-Editing functionality, to better support training for Ming-Omni image generation. 
# All rights and credit for the original implementation remain with the original authors and contributors, and this project complies with the applicable open-source license terms of the referenced repository.

import math
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_dispatch import dispatch_attention_fn
#from .attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class TimestepEmbedder(nn.Module):                                                # 时间步编码器
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):       # 定义一个两层MLP
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),                   # 先把时间步的频率编码从 frequency_embedding_size 映射到 mid_size
            nn.SiLU(),                                                                  # 经过SiLU非线性激活
            nn.Linear(mid_size, out_size, bias=True),                                   # 再映射到最终的out_size
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod                                                                    # 时间步编码的核心，把标量时间步变成一串 cos/sin 特征      
    def timestep_embedding(t, dim, max_period=10000):                                   # 定义一个函数，把时间步 t 编码成一个长度为 dim 的向量
        with torch.amp.autocast("cuda", enabled=False):                                 # 显示关闭 CUDA autocast
            half = dim // 2                                                             # 把维度 dim 分成两半。 前一半留给 cos，后一半留给 sin
            freqs = torch.exp(                                                          # 取指数 【最终得到的频率按对数尺度递减：前面的维度频率高，后面的频率维度低】
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half        # arange/half: 生成[0,1)按固定步长增加的序列。再乘以频率的对数
            )
            args = t[:, None].float() * freqs[None]                                     # 把时间步和频率组合起来。每个样本的时间步，在每个频率上都算一个相位值
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)           # 分别对args做 cos 和 sin, 再拼接起来
            if dim % 2:                                                                 # 如果 dim 是奇数，前面 dim//2 拼出来的长度 2*half 可能比原始维度少1，所以在最后一列补全0， 让维度对齐到 dim
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)              # 先把时间步t变成一份固定的 sin/cos 编码
        weight_dtype = self.mlp[0].weight.dtype                                         # 确保时间步编码的数据类型和MLP的权重数据类型一致（通常是float32或float16）
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)                                                        # 送入mlp，得到最终时间嵌入
        return t_emb


class ZSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention that adapts the existing Attention class to match the behavior of the
    original Z-ImageAttention module.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ZSingleStreamAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        self.profiling_enabled = True

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        profile_parts: Optional[dict] = None,
    ) -> torch.Tensor:
        if not self.profiling_enabled:
            profile_parts = None

        def sync_profile_device():
            if profile_parts is None:
                return
            if hidden_states.device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.synchronize(hidden_states.device.index if hidden_states.device.index is not None else None)
            elif hidden_states.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(hidden_states.device.index if hidden_states.device.index is not None else None)

        sync_profile_device()
        qkv_start = time.perf_counter() if profile_parts is not None else None

        sync_profile_device()
        q_start = time.perf_counter() if profile_parts is not None else None
        query = attn.to_q(hidden_states)                            # 把输入 hidden states 线性投影成 Q / K / V，供后面的 attention 使用
        sync_profile_device()
        if profile_parts is not None:
            profile_parts["q_proj_total"] += time.perf_counter() - q_start

        sync_profile_device()
        k_start = time.perf_counter() if profile_parts is not None else None
        key = attn.to_k(hidden_states)
        sync_profile_device()
        if profile_parts is not None:
            profile_parts["k_proj_total"] += time.perf_counter() - k_start

        sync_profile_device()
        v_start = time.perf_counter() if profile_parts is not None else None
        value = attn.to_v(hidden_states)
        sync_profile_device()
        if profile_parts is not None:
            profile_parts["v_proj_total"] += time.perf_counter() - v_start

        #                                                           把线性层输出从“最后一维的大向量”拆成“多头注意力格式“
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms                                                对 Q 和 K 做归一化，稳定注意力数值分布
        sync_profile_device()
        qk_norm_start = time.perf_counter() if profile_parts is not None else None
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        sync_profile_device()
        if profile_parts is not None:
            profile_parts["qk_norm_total"] += time.perf_counter() - qk_norm_start

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:          # 旋转位置编码
            @torch.amp.autocast("npu", enabled=False)
            def rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
                cos = freqs[..., 0]
                sin = freqs[..., 1]
                if cos.ndim == 3:
                    cos = cos[0]
                    sin = sin[0]
                if cos.ndim == 2 and cos.shape[-1] * 2 == x.shape[-1]:
                    cos = cos.repeat_interleave(2, dim=-1)
                    sin = sin.repeat_interleave(2, dim=-1)
                return rotary_position_embedding(
                    x,
                    cos,
                    sin,
                    rotated_mode="rotated_interleaved",
                    head_first=False,
                    fused=True,
                )

            if rotary_position_embedding is not None and x_in.device.type == "npu":
                return rope_apply(x_in, freqs_cis)

            with torch.amp.autocast("cuda", enabled=False):
                # Keep this path real-valued for Ascend/NPU compatibility.
                # The original complex64 formulation is mathematically equivalent,
                # but indexing complex tensors triggers unsupported kernels on torch_npu.
                x_parts = x_in.float().reshape(*x_in.shape[:-1], -1, 2)                                     # 先将输入张量 x_in（通常是 fp16 或 bf16）强制转换为 float32。用reshape将最后一个维度（Head Dimension）切分成大小为 2 的组
                cos = freqs_cis[..., 0].unsqueeze(2).unsqueeze(-1)                                          # 从 freqs_cis 中提取 cos 和 sin 分量，并调整维度以便后续广播。unsqueeze：在索引为2的位置和最后一个位置插入一个新的维度
                sin = freqs_cis[..., 1].unsqueeze(2).unsqueeze(-1)                                          # 使其形状能够与前面的 x_parts 广播对齐（Broadcast）
                x_rotated = torch.stack((-x_parts[..., 1], x_parts[..., 0]), dim=-1)                        # 将最后一维的[x1,x2]旋转成[-x2,x1]
                x_out = (x_parts * cos + x_rotated * sin).flatten(3)                                        # RoPE 的标准实数计算公式
                return x_out.type_as(x_in)                                                                  # 将计算完的 float32 张量重新转换回输入时的数据类型（例如 float16）
            
        @torch.amp.autocast('npu', enabled=False)
        def rope_apply(x, grid_sizes, freqs_list):
            cos, sin = freqs_list[0]
            return rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", fused=True)



        sync_profile_device()
        rope_start = time.perf_counter() if profile_parts is not None else None
        if freqs_cis is not None:                                                                   # 检查当前模块是否接收到了计算好的旋转复数张量（freqs_cis） 
            query = apply_rotary_emb(query, freqs_cis)                                                       # 把输入的 query 张量送入旋转位置编码的底层算子中，赋予 Query 当前 Token 所在的绝对位置 m 的空间属性
            key = apply_rotary_emb(key, freqs_cis)                                                           # 用完全相同的位置编码频率，对 key 张量进行旋转。赋予 Key 其自身所在绝对位置 n 的空间属性
        sync_profile_device()
        if profile_parts is not None:
            profile_parts["rope_total"] += time.perf_counter() - rope_start

        # Cast to correct dtype    确保 Q/K 数据类型一致，方便后面的 attention kernel
        dtype = query.dtype                                                                                  # 记录当前 query 张量的数据类型
        query, key = query.to(dtype), key.to(dtype)                                                          # 强制确保 query 和 key 的数据类型完全一致
        sync_profile_device()
        if profile_parts is not None:
            profile_parts["qkv_proj_total"] += time.perf_counter() - qkv_start

        # Ascend FlashAttention does not accept broadcast-only masks like [B, 1, 1, S].
        # Materialize a [B, 1, S, S] bool mask instead.
        if attention_mask is not None and attention_mask.ndim == 2:                                           # 检查传入的 attention_mask 是否存在，并且是不是一个二维张量 [Batch, Seq_len]
            attention_mask = attention_mask[:, None, None, :]                                                 # 利用 None（即 np.newaxis）进行升维，将形状从 [B, S] 强行撑开变成 [B, 1, 1, S]
            attention_mask = attention_mask.expand(-1, 1, query.shape[1], -1)                                 # 进一步扩展成 [B, 1, S_query, S_key] 的形状，其中 S_query 是 query 的序列长度，S_key 是 key 的序列长度。这样每个 query token 都有一个对应的 key mask

        # Compute joint attention  计算注意力权重并聚合value（用 Q/K/V 做真正的注意力计算）
        sync_profile_device()
        attention_core_start = time.perf_counter() if profile_parts is not None else None
        hidden_states = dispatch_attention_fn(                                                      # 真正执行 $Softmax(QK^T)V$ 的地方。代码并没有自己写繁琐的矩阵相乘，而是调用了一个底层的路由分发函数 dispatch_attention_fn。attention 计算后的结果就是 hidden_states
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,                                                                                    # 表明这是双向注意力（通常用于图像模型 DiT 或 ViT，可以看到全部上下文），而不是自回归模型（如 GPT，需要因果掩盖未来信息）
            backend=self._attention_backend,                                                                    # 根据硬件环境（CUDA 或 Ascend）选择attention的实现（比如 FlashAttention V2 或者是 NPU 专属的 NPUFlashAttention）。此步执行$Softmax(QK^T)V$
            parallel_config=self._parallel_config,
        )
        sync_profile_device()
        if profile_parts is not None:
            profile_parts["attention_core_total"] += time.perf_counter() - attention_core_start

        # Reshape back      把attention输出再投影回模型隐藏维度，注意力结果投影回主干隐藏空间
        sync_profile_device()
        out_proj_start = time.perf_counter() if profile_parts is not None else None
        hidden_states = hidden_states.flatten(2, 3)                                                             # FlashAttention 出来的结果形状是 [Batch, Seq_Len, Num_Heads, Head_Dim]。flatten(2, 3) 将多头特征重新拼接合并，把形状变回主干网络所需的 [Batch, Seq_Len, Hidden_Dim]（即 Num_Heads * Head_Dim）
        hidden_states = hidden_states.to(dtype)                                                                 # 防御性编程，确保从 FlashAttention 算子出来的结果类型，完全吻合模型预设的精度标准（比如 fp16 或 bf16）
        output = attn.to_out[0](hidden_states)                                                                  # 把 attention 计算后的结果 hidden_states 送入输出线性层，投影回模型主干隐藏维度
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)                                                                     # 若 attn.to_out 有第2层，则dropout
        sync_profile_device()
        if profile_parts is not None:
            profile_parts["out_proj_total"] += time.perf_counter() - out_proj_start

        return output


class FeedForward(nn.Module):                                                                       # FFN变体：SwiGLU。相较于传统 Transformer 里的两层 Linear 加一个激活函数，它巧妙地引入了乘法门控机制
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)                                                        # 没有偏置项的线性全连接层。w1 和 w3：负责把输入的低维特征映射到高维空间
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)                                                        # 把高维特征重新“压缩”投影回原来的维度，以便出 FFN 后能够顺利进行残差连接
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3                                                                                  # 对 x1 分支应用 SiLU 激活函数。SiLU 的数学定义为 $x \cdot \sigma(x)$，它比传统的 ReLU 曲线更平滑，允许微小的负值通过，在深层网络中梯度表现更好 （SwiGLU 的灵魂）
                                                                                                                # 将激活后的 x1 与线性的 x3 进行逐元素相乘，形成一个门控机制。只有当 x1 的值较大时，x3 的信息才能有效通过，这有助于模型学习更复杂的特征交互
    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))                                       # 先让同一个输入 x 分别通过 self.w1(x) 和 self.w3(x)，生成两份并行的、各自独立的高维特征。将两份特征送入门控函数，w1 负责提供非线性的“门控权重”，去筛选 w3 携带的“线性信息”。
                                                                                                                # 将筛选融合后的高维结果，送入最外层的 self.w2(...)，重新映射回原本的 dim 维度

@maybe_allow_in_graph
class ZImageTransformerBlock(nn.Module):                                                            # 基于 DiT（Diffusion Transformer）架构的单层 Transformer Block 的初始化部分
    def __init__(
        self,
        layer_id: int,                                                                                          # 当前是第几层 Block
        dim: int,                                                                                               # 模型的主干隐藏层维度（特征通道数）
        n_heads: int,                                                                                           # 多头注意力机制的 Query 头数
        n_kv_heads: int,                                                                                        # Key 和 Value 的头数
        norm_eps: float,
        qk_norm: bool,                                                                                          # 布尔值，决定是否对 Query 和 Key 进行归一化
        modulation=True,                                                                                        # 表明该网络使用我们在之前代码里见过的 AdaLN（自适应层归一化）来接收外部条件注入
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        # Refactored to use diffusers Attention with custom processor
        # Original Z-Image params: dim, n_heads, n_kv_heads, qk_norm
        self.attention = Attention(                                                                             # 实例化 diffusers 库提供的标准 Attention 模块
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=ZSingleStreamAttnProcessor(),                                                             # 下发到ZSingleStreamAttnProcessor.__call__()
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))                                   # 实例化 FeedFoward 类（带乘法门控的 SwiGLU 模块）
        self.layer_id = layer_id                                                                                # 记录当前 Block 是整个模型里的第几层

        # 为 Attention 模块和 FFN 模块分别准备了两套归一化层（norm1 和 norm2）
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)                                                       # RMSNorm 归一化：它抛弃了传统 LayerNorm 里的“减去均值（Mean Centering）”操作，只做均方根缩放，能省下计算开销
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation                                                                            # 记录是否开启条件调制（modulation）
        self.profiling_enabled = True
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))     # 如果开启了调制，就构建一个 AdaLN（Adaptive Layer Normalization）的线性映射网络。这个线性层把输入的条件信号，强行放大并映射成了 4 份（总长度是 4 * dim）。（调制层需要依靠 bias 提供一套“默认的控制阀门初始值”）

    def set_internal_profiling(self, enabled: bool):
        self.profiling_enabled = enabled
        processor = getattr(self.attention, "processor", None)                                                  # 利用 Python 的内建函数 getattr，去探查 self.attention 有没有挂载一个叫 processor 的属性。如果没有，就安全地返回 None
        if processor is not None and hasattr(processor, "profiling_enabled"):                                   # 如果有，且它身上也有 profiling_enabled 这个开关，就把控制指令继续往下传递，把它的开关也拨到对应状态
            processor.profiling_enabled = enabled

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        profile_parts: Optional[dict] = None,
    ):
        if not self.profiling_enabled:
            profile_parts = None

        def sync_profile_device():
            if profile_parts is None:
                return
            if x.device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.synchronize(x.device.index if x.device.index is not None else None)
            elif x.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(x.device.index if x.device.index is not None else None)

        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)      # AdaLN调制
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()                                                           # tanh()：把控制信息流的 gate 强行压缩到 [-1, 1] 之间，防止残差相加时数值爆炸
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            sync_profile_device()
            attn_start = time.perf_counter() if profile_parts is not None else None
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,                                                                        # 图像特征 x 先经过第一层 RMSNorm 洗掉极端数值，乘上刚才生成的 scale_msa（听从上级指令改变特征方差）
                attention_mask=attn_mask,                                                                                   # 送入 diffusers 的 Attention 模块（在这个模块的底层，ZSingleStreamAttnProcessor 会挂载 freqs_cis 旋转位置编码，处理 attn_mask，并调用 NPU 算子完成计算）
                freqs_cis=freqs_cis,
                profile_parts=profile_parts,
            )
            sync_profile_device()
            if profile_parts is not None:
                profile_parts["attention_total"] += time.perf_counter() - attn_start
            x = x + gate_msa * self.attention_norm2(attn_out)                                                               # 出来后的结果经过第二层 RMSNorm，乘上 gate_msa 控制流量，最后通过 x = x + ... 融合回主干道

            # FFN block
            sync_profile_device()     
            ffn_start = time.perf_counter() if profile_parts is not None else None
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))                             # 逻辑与 Attention 完全对称：特征再次归一化，乘上 scale_mlp，送入 SwiGLU 门控前馈网络（self.feed_forward）。经过提纯后，再次归一化，由 gate_mlp 把控流量，加回残差主干道
            sync_profile_device()
            if profile_parts is not None:
                profile_parts["ffn_total"] += time.perf_counter() - ffn_start
        else:                                                                                                               # 无调制分支：没有任何 scale 和 gate 参与的裸 Transformer 运算
            # Attention block
            sync_profile_device()
            attn_start = time.perf_counter() if profile_parts is not None else None
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
                profile_parts=profile_parts,
            )
            sync_profile_device()
            if profile_parts is not None:
                profile_parts["attention_total"] += time.perf_counter() - attn_start
            x = x + self.attention_norm2(attn_out)

            # FFN block
            sync_profile_device()
            ffn_start = time.perf_counter() if profile_parts is not None else None
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
            sync_profile_device()
            if profile_parts is not None:
                profile_parts["ffn_total"] += time.perf_counter() - ffn_start

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)                                       # 对于hidden states（图像特征序列）做一个层归一化 （LayerNorm：只盯着每一个单独的 Token 内部那 Hidden_Size（比如 1024）个数值，把这 1024 个数变成一个均值为 0、方差为 1 的标准分布）
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)                                                         # 线性层 降维：把 Transformer 认识的高维抽象特征（hidden_size），强行压缩映射成具体的图像像素/噪声维度（out_channels）

        self.adaLN_modulation = nn.Sequential(                                                                                # adaLN调制
            nn.SiLU(),                                                                                                        # 在最后输出前增加一点非线性表达能力
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),                                             # linear：y = xW^T + b
        )

    def forward(self, x, c):                                                                                                   # x 是从最后一个 Transformer Block 走出来的图像特征序列；c 是外部条件（也就是 adaln_input，比如时间步或文本特征）
        scale = 1.0 + self.adaLN_modulation(c)                                                                                 # 恒等映射（Identity）初始化技巧：1.0 + 0 保证了特征一开始能够原封不动地通过这道关卡，极大地稳定了初期的梯度 （此为逐元素相乘，Mul，计算量小）
        x = self.norm_final(x) * scale.unsqueeze(1)                                                                            # unsqueeze(1) ：在scale索引为 1 的位置（即中间）强行塞入一个大小为 1 的新维度，对齐self.norm_final(x)的维度 （linear中特征与权重为矩阵乘，MatMul，吃算力）
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,                                                                                                   # 旋转基数（Base frequency），决定波长的跨度
        axes_dims: List[int] = (16, 56, 56),                                                                                    # 每一个维度（如时间、高度、宽度）分配到的特征维度
        axes_lens: List[int] = (64, 128, 128),                                                                                  # 每一个维度的最大长度（如最大帧数、最大宽高）
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"                            # 确保维度分配和长度设置是匹配的（比如 3D 特征必须有 3 个维度的参数）
        self.freqs_cis = None                                                                                                   # 初始化存放频率缓存的变量

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):                                 # 核心数学预计算, 负责提前算出所有的 $\cos$ 和 $\sin$ 值，避免在推理时重复计算
        with torch.device("cpu"):                                                                                                # 预计算通常在 CPU 上完成，节省显存
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):                                                                           # 遍历每一个轴（例如第 0 轴是时间，第 1 轴是高度）
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))                          # 生成一系列递减的频率：序列开头的频率较大（旋转快），负责捕捉短距离的相对位置关系；序列末尾的频率极小（旋转慢），负责捕捉长距离的全局位置关系
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)                                             # torch.arange(e)：生成一个从 $0$ 到 $e-1$ 的整数序列
                freqs = torch.outer(timestep, freqs).float()                                                                     # 外积：为每一个位置（timestep）的每一个维度（freqs），分配一个专属的旋转相位
                # Store RoPE frequencies as explicit cos/sin pairs instead of complex64
                # so the same tensors can be indexed on NPU safely.
                freqs_cis_i = torch.stack((torch.cos(freqs), torch.sin(freqs)), dim=-1)                                          # 这个相位构成了一个复数平面的旋转因子 $e^{i\phi} = \cos\phi + i\sin\phi$
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:                                                                                               # 第一次运行：在 CPU 上计算频率表，然后搬运到 NPU
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:                                                                                                                    # 后续运行：检查设备。如果 ids 换到了另一个 NPU，则自动搬运缓存的频率表
            # Ensure freqs_cis are on the same device as ids                            
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):                                                                                      # 遍历每一个轴（T, H, W）
            index = ids[:, i]                                                                                                     # 取出所有 Token 在当前轴的具体坐标
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=1)                                                                                           # 将各个轴提取出的旋转编码在“特征维度”上拼接起来


class ZImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True                                                                                       # 支持梯度检查点，通过时间换空间，减少显存占用
    _no_split_modules = ["ZImageTransformerBlock"]                                                                                # 在分布式加载（如 DeepSpeed 或 Accelerate）时，指示 ZImageTransformerBlock 作为一个整体，不要被拆分到不同显卡上
    _repeated_blocks = ["ZImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]  # precision sensitive layers

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,                           # Transformer 的隐层维度（Embedding Dimension），3840 是非常大的规模，接近原版 DiT-XL
        n_layers=30,                        # 主体 Transformer 层的数量
        n_refiner_layers=2,
        n_heads=30,                         # 多头注意力的头数。每个头的维度为 $3840 \div 30 = 128$
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,                       # 在计算 Attention 前对 Query 和 Key 进行归一化，有助于超大规模模型训练的稳定性
        cap_feat_dim=2560,
        rope_theta=256.0,                   #  RoPE 旋转位置编码 的基数 $\theta$
        t_scale=1000.0,
        axes_dims=[32, 48, 48],             # 定义了 3D RoPE 的参数。[32, 48, 48] 对应 T, H, W 三个轴的维度分配，总和为 128
        axes_lens=[1024, 512, 512],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False
        self.profiling_enabled = True

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)                    # 1. 定义 Patch 投影层：将图像块平铺后投影到隐层维度 dim
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder                                                     # 2. 以 "patch-frame" 的尺寸作为 Key，存入字典

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)                       # 3. 定义输出层：将 dim 维特征还原回像素级别的 patch 尺寸
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)                                                                 # 将 Python 字典转化为 PyTorch 的 ModuleDict，确保参数能被模型识别并移动到 NPU
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(                                                                                 # 初始化“噪声细化层”：用于对生成过程中的噪声特征进行精细化处理
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(                                                                               # 初始化“上下文细化层”：用于处理文本或条件信息，不包含某些调制逻辑
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)                                            # 时间步嵌入：将扩散模型的 timestep 转化为向量
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True))         # 文本嵌入：对输入的 Caption 特征进行归一化并投影到 dim 维度

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))                                                                  # 定义两个可学习的 Padding Token，用于填充序列长度不一致的情况
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(                                                                                            # 构建主体的 Transformer 堆栈，共 n_layers 层（如 30 层）
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads                                                                                               # 验证多头注意力的每个头的维度是否与 RoPE 的维度分配一致
        assert head_dim == sum(axes_dims)                                                                                       # 必须相等，否则旋转位置编码无法应用
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)                           # 初始化 RoPE 编码器
        self.set_internal_profiling(True)

    def set_internal_profiling(self, enabled: bool):
        self.profiling_enabled = enabled
        for block_group in (self.noise_refiner, self.context_refiner, self.layers):
            for block in block_group:
                block.set_internal_profiling(enabled)

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:     # 拼图：将被patchify拆成的图像碎片Token，按照正确的空间和时间顺序，重新缝合成一张完整的图像（或一段视频）
        pH = pW = patch_size                                                                                                    # 设置空间方向（高、宽）的分块大小  【patch_size 定义了图像被切割的方块大小。如果 patch_size=2，意味着我们要把 1 个 Token 还原回 $2 \times 2$ 的像素区域】
        pF = f_patch_size                                                                                                       # 设置时间/帧方向（Frame）的分块大小
        bsz = len(x)                                                                                                            # 获取 Batch Size（批大小）
        assert len(size) == bsz                                                                                                 # 确保传入的尺寸信息与数据数量匹配
        for i in range(bsz):
            F, H, W = size[i]                                                                                                   # 提取当前样本的目标尺寸：F（帧数/时间）、H（高度）、W（宽度）
            ori_len = (F // pF) * (H // pH) * (W // pW)                                                                         # 计算原始有效 Token 的数量
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]                                                                                                  # x[i][:ori_len]：截取有效长度，剔除为了对齐而补齐的 Padding Token
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)                                                 # 1. 视图重构：将一维的 Token 序列拆解为 3D 空间坐标 + 3D 块内像素坐标
                .permute(6, 0, 3, 1, 4, 2, 5)                                                                                   # 2. 维度置换（Permute）。通常对应 aclnnPermute 等算子
                .reshape(self.out_channels, F, H, W)                                                                            # 3. 形状还原：将“格子坐标”和“块内像素坐标”合并，形成连续的图像/视频
            )
            x[i] = x[i][:,:1,:,:]                                                                                               # 这一行比较特殊：只取了时间轴上的第一帧 [:, :1, :, :]。如果是纯图像生成，F 通常为 1，这行确保了输出形状的严谨
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:                                                                                                       # 如果用户没有提供 start，则默认从全零开始（例如 (0, 0, 0)）
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]                 # 为每一个维度创建一个一维的索引序列
        grids = torch.meshgrid(axes, indexing="ij")                                                                             # 将多个一维序列组合成一个覆盖所有维度的坐标矩阵。indexing="ij" 确保了矩阵排列遵循矩阵坐标系（行、列），这与图像处理中的 $(H, W)$ 逻辑一致
        return torch.stack(grids, dim=-1)                                                                                       # 在最后一个维度上将这些坐标矩阵堆叠起来

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],                                                                                          # 输入的图像列表（通常是 VAE 空间下的 Latent）
        all_cap_feats: List[torch.Tensor],                                                                                      # 对应的文本特征列表
        patch_size: int,                                                                                                        # 空间维度的分块大小（如 2）
        f_patch_size: int,                                                                                                      # 时间/帧维度的分块大小（如 1）
        all_image_ref: List[torch.Tensor] = None,                                                                               # 可选的参考图像（用于图像补全或编辑任务）
    ):
        pH = pW = patch_size                                                                                                    # 将 patch_size 赋值给高度和宽度方向
        pF = f_patch_size                                                                                                       # 赋值给时间方向（Frame）
        device = all_image[0].device                                                                                            # 获取输入数据所在的设备（如 NPU）

        # 初始化容器（用于存储处理后的结果）
        all_image_out = []                                                                                                      # 存储投影后的图像 Patch Token
        all_image_size = []                                                                                                     # 存储原始图像的尺寸 (F, H, W)，用于之后的 unpatchify
        all_image_pos_ids = []                                                                                                  # 存储图像 Patch 在 3D 空间中的坐标 ID（传给 RoPE）
        all_image_pad_mask = []                                                                                                 # 图像的 Padding 掩码（区分真实像素和填充位）
        all_cap_pos_ids = []                                                                                                    # 文本特征的位置 ID
        all_cap_pad_mask = []                                                                                                   # 文本的 Padding 掩码
        all_cap_feats_out = []                                                                                                  # 处理后的文本特征向量

        if all_image_ref is None:                                                                                               # 如果没有提供参考图，则创建一个与图像列表等长的空列表，方便后续统一循环处理
            all_image_ref = [None]*len(all_image)

        for i, (image, cap_feat, image_ref) in enumerate(zip(all_image, all_cap_feats, all_image_ref)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids 为填充后的序列创建坐标网格
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask 创建 Mask：0 表示真实数据（False），1 表示填充部分（True）
            cap_pad_mask = torch.cat(
                [
                    torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),                                               # 有效部分
                    torch.ones((cap_padding_len,), dtype=torch.bool, device=device),                                            # 填充部分
                ],
                dim=0,
            )
            all_cap_pad_mask.append(                                                                                            # 如果 需要填充长度 > 0
                cap_pad_mask if cap_padding_len > 0 else torch.zeros((cap_ori_len,), dtype=torch.bool, device=device)           # 使用刚才拼接好的、包含 True 的掩码。否则（即原始长度正好对齐，无需填充），生成一个全是 False 的掩码
            )

            # padded featurea
            cap_padded_feat = torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)], dim=0)                            # 提取 cap_feat 序列中的最后一个 Token，将这最后一个 Token 沿着序列维度复制 cap_padding_len 次，将原始的文本特征和刚刚生成的重复特征在第 0 维（序列长度维度）拼接起来
            all_cap_feats_out.append(cap_padded_feat)                                                                           # 将处理好的填充特征存入列表，准备送往 Transformer

            ### Process Imagea
            if image_ref is not None:                                                                                           # 如果存在参考图
                image = torch.cat([image, image_ref], dim=1)                                                                    # 在通道维度（dim=1）将原始图像和参考图像拼接

            C, F, H, W = image.size()                                                                                           # 获取 通道、帧数、高度、宽度
            all_image_size.append((F, H, W))                                                                                    # 保存原始尺寸，用于后续 unpatchify 还原
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW                                                            # 计算在 时间、高度、宽度 三个维度上分别能切出多少个 Patch

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)                                                     # 将张量拆解，暴露出每个 Patch 内部的微观维度 (pF, pH, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)                # 空间转序列

            image_ori_len = len(image)                                                                                          # 获取当前 Token 序列的长度
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF                                                                 # 计算需要 Padding 的长度，使序列总长符合 SEQ_MULTI_OF 的倍数

            image_ori_pos_ids = self.create_coordinate_grid(                                                                    # 给真实图像 patch 生成坐标
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padded_pos_ids = torch.cat(                                                                                   # 如果有 image padding token，给 padding token 补上占位坐标
                [
                    image_ori_pos_ids,
                    self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                    .flatten(0, 2)
                    .repeat(image_padding_len, 1),
                ],
                dim=0,
            )
            all_image_pos_ids.append(image_padded_pos_ids if image_padding_len > 0 else image_ori_pos_ids)
            # pad mask
            image_pad_mask = torch.cat(                                                                                         # 构造 image padding mask
                [
                    torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_image_pad_mask.append(
                image_pad_mask
                if image_padding_len > 0
                else torch.zeros((image_ori_len,), dtype=torch.bool, device=device)
            )
            # padded feature
            image_padded_feat = torch.cat(                                                                                      # 构造 padding 后的 image feature
                [image, image[-1:].repeat(image_padding_len, 1)],
                dim=0,
            )
            all_image_out.append(image_padded_feat if image_padding_len > 0 else image)

        return (
            all_image_out,                                                                                                     # 每个样本 padding 后的image feature
            all_cap_feats_out,                                                                                                 # 每个样本 padding 后的文本特征 (caption/ text feature)
            all_image_size,                                                                                                    # 每个样本的原始图像尺寸 (F, H, W)，用于 unpatchify 还原            
            all_image_pos_ids,                                                                                                 # 每个样本的图像位置编码 ID
            all_cap_pos_ids,                                                                                                   # 每个样本的文本位置编码 ID
            all_image_pad_mask,                                                                                                # 每个样本的图像 Padding 掩码
            all_cap_pad_mask,                                                                                                  # 每个样本的文本 Padding 掩码
        )

    def forward(
        self,
        x: List[torch.Tensor],             # 噪声图像输入
        t: torch.Tensor,                   # 时间步输入 
        cap_feats: List[torch.Tensor],     # 文本特征输入
        patch_size=2,                      # 图像分块大小
        f_patch_size=1,                    # 帧/频率维度的 Patch 大小（如果是视频或多帧）
        ref_x=None,                        # 可选的参考图像输入（用于图像编辑任务）
        return_dict: bool = True,          # 是否以字典形式返回输出
        profile_transformer_layers: bool = False,
    ):
        profile_transformer_layers = profile_transformer_layers and self.profiling_enabled

        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        def sync_device():
            if not self.profiling_enabled:
                return
            if device.type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.synchronize(device.index if device.index is not None else None)
            elif device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(device.index if device.index is not None else None)

        stage_timings = {
            "patchify_and_embed": 0.0,
            "image_branch_total": 0.0,
            "text_branch_total": 0.0,
            "unified_total": 0.0,
            "unified_prepare": 0.0,
            "transformer_blocks_main": 0.0,
            "transformer_attention_total": 0.0,
            "transformer_ffn_total": 0.0,
            "qkv_proj_total": 0.0,
            "q_proj_total": 0.0,
            "k_proj_total": 0.0,
            "v_proj_total": 0.0,
            "qk_norm_total": 0.0,
            "rope_total": 0.0,
            "attention_core_total": 0.0,
            "out_proj_total": 0.0,
            "transformer_blocks_layer_times": [],
            "decode_total": 0.0,
        }

        bsz = len(x)                       # 获取 Batch Size（批大小）
        device = x[0].device               # 获取当前张量所在的设备

        sync_device()
        patchify_start = time.perf_counter()
        (
            x,                          # 转换后的图像序列（Tokens）
            cap_feats,                  # 转换后的文本特征序列（Tokens）
            x_size,                     # 每个图像的原始尺寸（帧数/频率维度、图像高度、图像宽度）（用于后期恢复图片）
            x_pos_ids,                  # 图像的位置编码 ID（告诉模型每个 Token 在图上的坐标）
            cap_pos_ids,                # 文本的位置编码 ID
            x_inner_pad_mask,           # 图像的 Padding 掩码（处理不同尺寸图片时忽略空白区域）
            cap_inner_pad_mask,         # 文本的 Padding 掩码
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size, ref_x)          # 把三维图片(C, H, W)拉平（Flatten）成了一维的 Tokens 序列
        sync_device()
        stage_timings["patchify_and_embed"] = time.perf_counter() - patchify_start

        # image branch: x embed/prepare + noise refiner
        sync_device()
        image_branch_start = time.perf_counter()
        t = t * self.t_scale               # 对时间步进行缩放（通常是为了匹配预训练时的分布）
        t = self.t_embedder(t)             # 将标量时间步映射成高维向量（Time Embedding），让模型理解现在是去噪的哪个阶段
        x_item_seqlens = [len(_) for _ in x]                                # x 是一个列表（List），里面每个元素代表一张图片转换后的 Patch 序列。这里计算每个样本的序列长度
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)           # 确保每个序列长度都是 SEQ_MULTI_OF 的倍数（可能是未来适配适配 Flash Attention 或 NPU 上的高性能算子。许多加速算子要求序列长度必须对齐）
        x_max_item_seqlen = max(x_item_seqlens)                             # 找出这一组图片中最长的那一个序列

        x = torch.cat(x, dim=0)                                             # 将列表中的多个 Tensor 在第 0 维度（Sequence 维度）拼接成一个连续的长张量
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)          # 根据传入的 patch_size，从一个预设的字典（字典里存了不同尺寸的 Linear 或 Conv 层）中找到对应的 Embedding 层，对 x 进行线性映射 （将原始的像素值（或 Latent 值）映射到了模型能够理解的 隐空间维度）

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.type_as(x)                                          # 将时间步嵌入 t 的数据类型（如 float32 或 float16）转换为与图像特征 x 完全一致
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token                   # 根据之前的 x_inner_pad_mask，在图像序列中所有属于“填充区域（Padding）”的位置，统一替换为模型预设的 x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))                            # 将之前拼接成一个大张量的图像序列，根据每个样本的原始序列长度，再次拆分回一个列表（List）形式，每个元素对应一张图片的 Patch 序列
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0))           # 最终得到一个列表，每个元素对应一个样本的旋转位置编码

        x = pad_sequence(x, batch_first=True, padding_value=0.0)                                    # 将之前拆散的列表 x 和 x_freqs_cis 重新打包成一个形状为 [Batch, Max_Seq_Len, Hidden_Dim] 的大张量
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)                # 由于每张图切出的 Token 数可能不同，短序列会被补 0 达到当前 Batch 的最大长度。batch_first=True 确保第 0 维是 Batch Size，这是适配大多数主流算子的格式
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        x_freqs_cis = x_freqs_cis[:, : x.shape[1]]                                                  # 显式裁剪旋转位置编码的长度，使其与图像特征 x 的长度完全一致

        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)        # 创建注意力掩码
        for i, seq_len in enumerate(x_item_seqlens):                                                # 生成一个布尔矩阵，标记哪些位置是真实的图像 Token（1/True），哪些是填充的 0（0/False）
            x_attn_mask[i, :seq_len] = 1                                                            # 在计算 Self-Attention 时，模型会根据这个 Mask 忽略掉 Padding 区域，防止“无效的 0”干扰图像特征的提取

        if torch.is_grad_enabled() and self.gradient_checkpointing:                                             # 如果在训练模式且开启了梯度检查点（Gradient Checkpointing），则
            for layer in self.noise_refiner:
                x = self._gradient_checkpointing_func(layer, x, x_attn_mask, x_freqs_cis, adaln_input)          # 不存储中间层的激活值，而是在反向传播时重新计算 （节省显存）
        else:
            for layer in self.noise_refiner:                                                                    # 在推理阶段或未开启检查点时，直接按顺序将特征送入 noise_refiner（由多个 Transformer Block 组成）
                x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)                                             # 图像特征、掩码、位置编码和时间嵌入（adaln_input）在这里深度融合
        sync_device()
        stage_timings["image_branch_total"] = time.perf_counter() - image_branch_start

        # text branch: cap embed/prepare + context refiner
        sync_device()
        text_branch_start = time.perf_counter()
        cap_item_seqlens = [len(_) for _ in cap_feats]                      # 计算文本特征序列的长度
        cap_max_item_seqlen = max(cap_item_seqlens)                         # 找出文本特征序列中最长的那个（因为不同样本的文本长度可能不同）

        cap_feats = torch.cat(cap_feats, dim=0)                         # 将文本特征列表在第0维拼接成一个大张量，准备送入线性层进行映射
        cap_feats = self.cap_embedder(cap_feats)                        # 将拼接后的张量送入 cap_embedder（通常是一个线性映射层），把原始的文本特征维度映射到与模型主体（Transformer）一致的隐空间维度（Hidden Dimension）
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token   # 根据之前的 cap_inner_pad_mask，在文本特征序列中所有属于“填充区域（Padding）”的位置，统一替换为模型预设的 cap_pad_token （防止无效的零值或随机噪声干扰模型对文本语义的理解）
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))      # 使用 split 操作和之前记录的长度，将长张量重新切分回原来的列表结构
        cap_freqs_cis = list(                                           # 对文本特征同样生成对应的旋转位置编码，确保文本和图像在 Transformer 中能够正确地融合位置信息
            self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(_) for _ in cap_pos_ids], dim=0)      # 先将所有文本的位置 ID 拼接起来，统一计算 RoPE 的复数旋转因子（cis），然后再按照每个样本原本的长度切分回列表。这赋予了模型感知文本词汇前后顺序（序列相对位置）的能力
        )

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)                                            # 将文本特征列表重新打包成一个形状为 [Batch, Max_Cap_Seq_Len, Hidden_Dim] 的大张量，短序列会被补 0 达到当前 Batch 的最大长度
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)                                    # 同样地，将文本的旋转位置编码也打包成一个大张量，确保它与文本特征的形状完全匹配
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        cap_freqs_cis = cap_freqs_cis[:, : cap_feats.shape[1]]                                                              # 显式裁剪文本的旋转位置编码的长度，使其与文本特征 cap_feats 的长度完全一致

        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)                            # 创建文本的注意力掩码，标记哪些位置是真实的文本 Token（1/True），哪些是填充的 0（0/False）。在计算 Self-Attention 时，模型会根据这个 Mask 忽略掉 Padding 区域，防止“无效的 0”干扰模型对文本特征的提取
        for i, seq_len in enumerate(cap_item_seqlens):                                                                      # 生成一个布尔矩阵，标记哪些位置是真实的文本 Token（1/True），哪些是填充的 0（0/False）
            cap_attn_mask[i, :seq_len] = 1                                                                                  # 这里的 cap_attn_mask 将被传入 context_refiner 中的 Transformer Block，用于指导模型正确地处理文本特征序列  
        if torch.is_grad_enabled() and self.gradient_checkpointing:                                                         # 如果在训练模式且开启了梯度检查点（Gradient Checkpointing），则
            for layer in self.context_refiner:                                                                              # 不存储中间层的激活值，而是在反向传播时重新计算 （节省显存）
                cap_feats = self._gradient_checkpointing_func(layer, cap_feats, cap_attn_mask, cap_freqs_cis)
        else:
            for layer in self.context_refiner:                                                                              # 在推理阶段或未开启检查点时，直接按顺序将文本特征送入 context_refiner（由多个 Transformer Block 组成）
                cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)
        sync_device()
        stage_timings["text_branch_total"] = time.perf_counter() - text_branch_start

        # unified branch: join image/text streams + main transformer layers
        sync_device()
        unified_start = time.perf_counter()
        unified_prepare_start = unified_start
        unified = []                                                                                                        # 将图像特征和文本特征在序列维度上进行拼接，形成一个统一的输入序列，送入后续的 Transformer Block 进行深度融合。每个样本的图像 Token 序列和文本 Token 序列被简单地连接在一起，模型通过位置编码和注意力机制来区分它们并学习它们之间的关系
        unified_freqs_cis = []                                                                                              # 同样地，将图像和文本的旋转位置编码也进行拼接，确保它们在后续的 Transformer Block 中能够正确地提供位置信息
        for i in range(bsz):                                                                                                # 遍历当前批次
            x_len = x_item_seqlens[i]                                                                                       # 从之前记录的长度列表中，取出当前第 i 个样本真实的图像 Token 数量和文本 Token 数量               
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))                                               # x[i][:x_len] 和 cap_feats[i][:cap_len]把刚才 pad_sequence 补上去的 0.0 全部切掉了，只保留有效的图像和文本数据。使用 torch.cat 将真实的图像序列和文本序列连在一起
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))                       # 对 RoPE（旋转位置编码）执行与特征完全相同的“剥离+拼接”操作，确保位置编码与特征 Token 一一对应
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]                                    # 将文本的有效长度和图像的有效长度逐元素相加（a + b），得到每一个联合序列的最新总长度，并存入一个新的列表
        assert unified_item_seqlens == [len(_) for _ in unified]                                                            # 再次验证一下，确保我们计算的联合序列长度和实际拼接后的长度完全一致（这是一个重要的 sanity check，防止后续处理时出现维度不匹配的错误）
        unified_max_item_seqlen = max(unified_item_seqlens)                                                                 # 在所有拼接完的 unified 序列中，找出最长的那一根。由于去掉了旧的 Padding，拼接后的序列长度又是参差不齐的了。 这里需要重新计算一个新的 max_item_seqlen 来为后续的 Transformer Block 创建正确大小的注意力掩码和进行必要的 Padding

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)                                                # 再次使用 pad_sequence，用 0.0 将所有联合序列补齐到当前批次中的最大长度（unified_max_item_seqlen）。形成最终的 [Batch, Seq_Len, Hidden_Dim] 规整张量
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)                    # 初始化一个全 0（False）的布尔矩阵，然后通过循环，把每个样本真实有效的部分（即图像 Token + 文本 Token 的总长度 seq_len）全部标记为 1（True）
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1
        sync_device()
        stage_timings["unified_prepare"] = time.perf_counter() - unified_prepare_start

        sync_device()
        transformer_blocks_start = time.perf_counter()
        layer_times = [0.0] * len(self.layers) if profile_transformer_layers else None
        transformer_parts = (
            {
                "attention_total": 0.0,
                "ffn_total": 0.0,
                "qkv_proj_total": 0.0,
                "q_proj_total": 0.0,
                "k_proj_total": 0.0,
                "v_proj_total": 0.0,
                "qk_norm_total": 0.0,
                "rope_total": 0.0,
                "attention_core_total": 0.0,
                "out_proj_total": 0.0,
            }
            if profile_transformer_layers
            else None
        )
        if torch.is_grad_enabled() and self.gradient_checkpointing:                                                          # 如果在训练模式且开启了梯度检查点（Gradient Checkpointing），则
            for layer_idx, layer in enumerate(self.layers):                                           # 数据在这个分支下进入 self.layers（联合 Transformer 块）。通过包裹 _gradient_checkpointing_func，框架不会在显存中保存每一层的中间激活值。只有在计算反向传播（Backward Pass）需要用到这些值去算梯度时，它才会重新前向计算一次。
                if layer_times is not None:
                    sync_device()
                    layer_start = time.perf_counter()
                    unified = self._gradient_checkpointing_func(                    # “以时间换空间”策略。牺牲大约 20%-30% 的计算时间，换取显存占用的大幅度下降，是训练超大模型或处理超长图文序列的标配                                      
                        layer, unified, unified_attn_mask, unified_freqs_cis, adaln_input, transformer_parts
                    )
                    sync_device()
                    layer_times[layer_idx] = time.perf_counter() - layer_start
                else:
                    unified = self._gradient_checkpointing_func(
                        layer, unified, unified_attn_mask, unified_freqs_cis, adaln_input
                    )
        else:                                                                   # 如果是推理阶段（Inference，比如用户在运行生成任务），或者训练时未开启梯度检查点，就走这条最直接的高速通道
            for layer_idx, layer in enumerate(self.layers):                                            # 一个简单的 for 循环，将联合张量 unified、注意力掩码、联合位置编码、以及时间步特征（adaln_input）一层一层地穿透 self.layers。每一层的输出直接作为下一层的输入，最终完成图像与文本在隐空间深度的 Cross-Modal 和 Intra-Modal 注意力交互
                if layer_times is not None:
                    sync_device()
                    layer_start = time.perf_counter()
                    unified = layer(
                        unified,
                        unified_attn_mask,
                        unified_freqs_cis,
                        adaln_input,
                        profile_parts=transformer_parts,
                    )
                    sync_device()
                    layer_times[layer_idx] = time.perf_counter() - layer_start
                else:
                    unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)
        sync_device()
        stage_timings["transformer_blocks_main"] = time.perf_counter() - transformer_blocks_start
        if transformer_parts is not None:
            stage_timings["transformer_attention_total"] = transformer_parts["attention_total"]
            stage_timings["transformer_ffn_total"] = transformer_parts["ffn_total"]
            stage_timings["qkv_proj_total"] = transformer_parts["qkv_proj_total"]
            stage_timings["q_proj_total"] = transformer_parts["q_proj_total"]
            stage_timings["k_proj_total"] = transformer_parts["k_proj_total"]
            stage_timings["v_proj_total"] = transformer_parts["v_proj_total"]
            stage_timings["qk_norm_total"] = transformer_parts["qk_norm_total"]
            stage_timings["rope_total"] = transformer_parts["rope_total"]
            stage_timings["attention_core_total"] = transformer_parts["attention_core_total"]
            stage_timings["out_proj_total"] = transformer_parts["out_proj_total"]
        if layer_times is not None:
            stage_timings["transformer_blocks_layer_times"] = layer_times
        stage_timings["unified_total"] = time.perf_counter() - unified_start

        # decode branch: final layer + unpatchify
        sync_device()
        decode_start = time.perf_counter()
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)        # 最终层投影（Final Layer Projection）：将融合后的特征序列送入最终的处理块。
        unified = list(unified.unbind(dim=0))                                                       # unbind(dim=0) 是 torch.split 或切片的另一种高效写法。它沿着第 0 维（Batch 维度）将这个大矩阵“解绑”拆开成一个列表，每个元素对应一个样本的输出序列。这样做是为了后续根据每个样本原始的图像尺寸，单独处理每个样本的输出，恢复成图像的形状
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)                              # 最后一步，unpatchify（反 Patch 化）。根据之前记录的每个样本的原始图像尺寸 x_size，以及 patch_size 和 f_patch_size 的信息，将每个样本的输出序列重新组合成对应尺寸的图像张量。这个过程涉及到复杂的 reshape 和 permute 操作，把线性的 Token 序列转换回三维的图像格式（C, H, W）
        sync_device()
        stage_timings["decode_total"] = time.perf_counter() - decode_start
        self._ming_profile = stage_timings if self.profiling_enabled else {}

        if not return_dict:
            return (x,)

        return Transformer2DModelOutput(sample=x)       # Hugging Face diffusers 库的标准输出类。通过封装成这个对象，你的模型就能完美接入 diffusers 的 StableDiffusionPipeline 等高级流水线中，直接复用现成的调度器（Scheduler）进行推理采样
