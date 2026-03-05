import math

from transformers import PretrainedConfig


# hugging face的类，PretrainedConfig是一个基类，用于定义预训练模型的配置。它包含了模型的各种超参数和设置，可以通过继承这个类来创建特定模型的配置类。在这个代码中，MokioMindConfig继承了PretrainedConfig，并定义了MokioMind模型的特定配置参数，如dropout率、隐藏层大小、注意力头数等。这些参数可以在实例化MokioMindConfig时进行设置，以便在训练或推理过程中使用。
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import funcitonal as F
import math
from .activation_function import ACT2FN
# msnorm


# 继承nn.Module类
class RMSNorm(nn.Module):
    # __init__初始化
    def __init__(self, dim: int, eps: float = 1e-5):
        # dim:int是输入张量的维度，eps:float是一个小的常数，用于防止除以零的情况。
        # dim：是python中的类型提示，表示dim参数应该是一个整数。eps：也是类型提示，表示eps参数应该是一个浮点数。
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    # _norm
    def _norm(self, x):
        # keepdim=True保持维度不变，mean(-1)在最后一个维度上求平均值，pow(2)对每个元素进行平方运算，rsqrt开方求倒数。
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # forward前向传播
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

    # precomputer_freqs_cis函数用于预计算频率的复数表示。它首先计算频率的实部和虚部，然后根据ROPE缩放的配置进行调整，最后返回一个复数张量。


def precomputer_freqs_cis(
    dim: int, end: int(32 * 1024), rope_base, rope_scaling: Optional[dict] = None
):
    # dum:int是一个整数参数，表示维度的大小。end:int是一个整数参数，表示频率的结束值，默认为32*1024。rope_base是一个参数，表示ROPE（Relative Positional Encoding）的基数。rope_scaling:Optional[dict]是一个可选的字典参数，表示ROPE缩放的配置。

    # 初始化RoPE频率
    freqs, attn_factor = 1.0 / (
        rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim),
        1.0,
    )
    # //2表示整数除法，torch.arange(0,dim,2)生成一个从0到dim的步长为2的整数序列，[:(dim//2)]表示取前dim//2个元素。float()将整数转换为浮点数。最后得到的freqs是一个包含频率值的张量，attn_factor是一个标量值。

    if rope_scaling is not None:
        # 如果rope_scaling不为None,用yarn。
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
        )

        # 推断长度大于训练长度，使用缩放
        if end > orig_max:
            # 波长b到i的映射
            inv_dim = lambda b: (  # noqa: E731
                (dim * math.log(orig_max / (b * 2 * math.pi)))
                / (2 * math.log(rope_base))
            )
            # lambda函数是一种匿名函数，inv_dim是一个函数对象，接受一个参数b，并返回一个计算结果。这个函数的作用是根据输入的b值计算出对应的维度大小。

            # 划分高低维度
            # low :不需要缩放的高频部分
            # high:需要缩放的低频部分

            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 计算缩放因子
            # low之前，ramp为0，再high之后，ramp为1，在low和high之间，ramp在0和1之间线性变化，先行过度。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                # arange生成一个从0到dim//2的整数序列，clamp函数将输入的值限制在0和1之间，确保ramp的值在合理范围内。max(high-low,0.001)用于防止除以零的情况。
                # (dim//2 - low)表示需要缩放的维度范围，除以这个范围可以将ramp的值归一化到0和1之间。
                # (device = freqs.device)确保生成的张量与freqs在同一设备上。
                0,
                1,
            )
            # 当ramp = 0（高频）：系数为1，保持原频率不变
            # 当ramp = 1（低频）：系数为1/factor，频率缩小factor倍
            # 当0 < ramp < 1（过渡区域）：系数在1和1/factor之间线性变化，逐渐缩小频率
            freqs = freqs * (1 - ramp + ramp / factor)
        # 根据end，生成位置索引t
        t = torch.arange(end, device=freqs.device).float()

        # 计算外积，将t和freq相乘，得到每个位置的旋转角度
        # t是温度，freqs是频率，外积得到每个位置的旋转角度。
        freqs = torch.outer(t, freqs)

        freqs_cos = (
            torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).float()
            * attn_factor
        )

        freqs_sin = (
            torch.cat([torch.sin(freqs), -torch.cos(freqs)], dim=-1).float()
            * attn_factor
        )

        return freqs_cos, freqs_sin


# 编写RoPE
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # [a,b] - > [-b,a]
    def rotate_half(x):
        # x.shape[-1]取最后一个维度的大小
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    # x_rotated = x * cos + rotate_half(x) * sin

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape

    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
        # expand函数用于扩展张量的维度，reshape函数用于改变张量的形状。通过这两步操作，可以将输入张量x按照指定的方式重复n_rep次，并返回一个新的张量。
    )


class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads is None
            else args.num_attention_heads
        )
        # num_key_value_heads是一个整数参数，表示键值头的数量。如果args.num_key_value_heads不为None，则使用args.num_key_value_heads的值；否则，使用args.num_attention_heads的值。这样可以确保键值头的数量与注意力头的数量一致，或者根据需要进行调整。

        assert args.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_heads must be divisible by num_key_value_heads"
        )
        # asssert语句用于检查条件是否满足，如果条件不满足，则会抛出一个AssertionError异常，并显示指定的错误消息。在这里，它检查num_attention_heads是否能够被num_key_value_heads整除，如果不能整除，则会抛出异常并提示错误消息。

        self.n_local_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.num_key_value_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)  # 残差连接的dropout
        self.dropout = args.dropout

        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )
        # flash_attention是一个布尔参数，表示是否使用Flash Attention机制。Flash Attention是一种优化的注意力计算方法，可以提高计算效率和内存使用效率。如果torch.nn.functional中存在scaled_dot_product_attention函数，并且args.flash_attention为True，则self.flash将被设置为True，表示可以使用Flash Attention机制；否则，self.flash将被设置为False，表示不使用Flash Attention机制。

    def forward(
        self,
        x: torch.Tensor,
        positon_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # 投影，计算qkv
        bsz, seq_len = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 把输入拆分成多个头，用view,8个头
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        # q和k，使用rope
        cos, sin = positon_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
        # 对于k和v，使用repeat（注意kv cache）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_kv = (xk, xv) if use_cache else None
        # if else None表示如果use_cache为False，则past_kv将被设置为None。这意味着在这种情况下，不会返回任何缓存的键值对。反正则，如果use_cache为True，则past_kv将包含当前的键和值，以便在后续的前向传播中使用。

        xq, xk, xv = (
            xq.transpose(1, 2),
            # [bsz,n_local_heads*head_dim,seq_len]
            # [bsz,n_local_heads,seq_len,head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
        # 进行attention计算，q@k^T/sqrt(d)

        if (
            self.flash
            and seq_len > 1
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1)
                .expand(bsz, self.n_local_heads, seq_len, -1)
                .bool()
            )
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            """
self.training: 这是一个开关。

当你在训练模式（Training）时，它是 True。

当你在测试/推理模式（Evaluation）时，它是 False。           
if self.training == True:
    dropout_p = self.dropout
else:
    dropout_p = 0.0
参数拆解 (Parameter Breakdown)
我们可以用“搜寻知识”的过程来类比这三个关键矩阵：
xq (Query/查询): 你想知道什么（比如：问题“苹果是什么？”）。
xk (Key/键): 库里所有资料的索引（比如：书架上的标签“水果”、“科技”）。
xv (Value/值): 资料的具体内容（比如：关于苹果的详细描述）。
其他参数：
attn_mask: 掩码。告诉模型哪些地方不该看（比如填充的无效位）。
dropout_p: 随机“关掉”一部分神经元，防止模型过度依赖某些特定特征（防止过拟合）。
is_causal = True: 因果掩码。确保模型在预测下一个词时，不能偷看后面的答案。
           """

            # 缩放点积注意力机制 (Scaled Dot-Product Attention)。
        else:
            # 手敲的attention计算
            scores = (xq @ xk.transpose(-1, -1)) / math.sqert(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                # -inf表示负无穷，triu函数用于生成一个上三角矩阵，full函数用于创建一个指定形状的张量，并用指定的值填充。这里生成了一个大小为(seq_len, seq_len)的矩阵，并用负无穷填充。这个矩阵将被添加到scores中，以实现因果掩码的效果，确保模型在计算注意力分数时只能关注当前词之前的位置。
                # 后面会置为0，前面为-inf，保证了模型只能关注当前词之前的位置。
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            # unsqueeze函数用于在指定位置插入一个新的维度。这里的uns

            # 最后拼接头，输出投影，返回

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as
            scores = self.attn_dropout(scores)
            output = scores @ xv
        # [bsz, n_local_heads,seq_len,head_dim] -> [bsz,seq_len,n_local_heads*head_dim]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    # 初始化
    # 升维
    # 降维
    # 门控
    # dropout
    # 激活函数
    def __init__(self, args: MokioMindConfig):
        super().__init__()
        # 升维度
        # 计算模型前馈网络（FFN）中中间层的宽度，并进行内存对齐。
        # intermediate_size是前馈网络中间层的宽度，如果在配置中没有指定，则根据hidden_size计算一个默认值，并进行内存对齐处理。
        # 同时也是升维维度大小
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            # 比例缩放 (The 8/3 Ratio)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            # 内存对齐 (Memory Alignment)，64倍数

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_sizem, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=False
        )  # 门控
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]  # 激活函数

    def forwrad(self, x):
        return self.dropout(
            self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        )


class MokioMindBlock(nn.Module):
    # Transformer Layer k
    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        # self attention
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = residual + hidden_states

        # mlp
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value

class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.vocab_size,self.num_hidden_layers=(
            config.vocab_size,
            config.num_hidden_layers,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        #vocab_size是词汇表的大小，hidden_size是每个词向量的维度。这个嵌入层将输入的词索引转换为对应的词向量表示。
        # nn.Embedding是PyTorch中的一个类，用于创建一个嵌入层。它接受两个参数：vocab_size和hidden_size。vocab_size表示词汇表的大小，即模型可以处理的不同词汇的数量；hidden_size表示每个词向量的维度，即每个词被表示为一个hidden_size维的向量。通过这个嵌入层，输入的词索引将被转换为对应的词向量表示，这些词向量将作为模型后续层的输入。
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [MokioMindBlock(i,config) for i in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #RopE预计算
        freqs_cos,freqs_sin = precomputer_freqs_cis(
            dim = config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling,
        )
        """
        指针数量、表盘的最大刻度、齿轮的转速，以及表盘是否需要为了长文本进行弹性拉伸。
        单头维度 (Head Dimension)每个“经理”（注意力头）分到了多少根**“时钟指针”（维度）。这个值是通过将模型的隐藏层大小除以注意力头的数量来计算的。
        最大位置嵌入 (Maximum Position Embeddings)表盘上最多能画多少个**“时间刻度”**。也就是你允许模型一口气读多少个字

        旋转基础底数 (RoPE Base / Theta)这个参数决定了频率的计算方式，影响了位置编码的周期性。较大的值会导致频率更密集，适合处理长文本；较小的值则适合短文本。

        旋转缩放因子 (RoPE Scaling Factor)代码就会动态地把刻度挤一挤，让模型不至于“看花眼”。
        """

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        #register_buffer函数用于注册一个持久化的缓冲区，这个缓冲区不会被视为模型的参数，也不会在训练过程中更新。这里注册了两个缓冲区freqs_cos和freqs_sin，分别存储预计算的频率的余弦和正弦值。这些缓冲区将被保存在模型的状态字典中，并且在模型保存和加载时会被正确处理。

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            **kwargs,
    ):
        """
        输入标识符 (Input Token IDs)
        维度公式： 它的形状通常是二维矩阵 。(Batch Size)：批次大小（同时处理几句话）。 (Sequence Length)：序列长度（这句话有几个字）。
        注意力掩码 (Attention Mask)
        防止作弊 (Causal Mask)： 在预测第 3 个字时，必须把第 4、5 个字遮住（变成 $-\infty$），防止模型偷看未来。忽略空白 (Padding Mask)： 如果一句话只有 3 个字，另一句有 5 个字，为了凑齐矩阵，短句后面会补上“空白占位符”。掩码会告诉模型：“这些空白没有意义，开会时不要理它们”。

        历史键值缓存 (Past Key-Values Cache)
        KV Cache,保留特征

        使用缓存开关 (Use Cache Flag)
        当你训练 (Training) 模型时，这个开关是关闭的（False），因为训练时是一次性看完全文，不需要逐字缓存。
        当你推理 (Inference / Chat) 与模型聊天时，这个开关会打开（True），告诉模型：“算完这个字后，把新的笔记存下来，等下一个字进来时再拿给我用”

        关键字参数 (Keyword Arguments) 防止程序因为不认识多余的参数而报错
        """
        batch_size,seq_len = input_ids.shape

        if hasattr(past_key_values,"layers"):
            past_key_values=None
        # hasattr 就是用来检查有没有这个按钮的，如果有就用它，没有就算了。这里检查past_key_values是否有属性"layers"，如果有，就把past_key_values设置为None。这可能是为了兼容不同版本的输入格式，确保后续代码能够正确处理past_key_values。

        past_key_values = past_key_values or [None] * len(self.layers)
        # or（或者）： Python 的魔法逻辑。如果左边有东西（比如正在连续对话），就直接用左边的；如果左边是空的（None），就立刻执行右边的动作。
        # [None] * len(self.layers)（右边）：[None, None, None, ..., None]

        start_pos= (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )
        # [批次大小, 历史字数, 头数, 维度]。索引 [1] 拿到的正好是第二个维度——也就是历史字数
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents=[]

        for layer_idx,(layer,past_key_values) in enumerate(zip(self.layers,past_key_values)):
            hidden_states,present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            # presents是一个列表（List），用于收集并保存大模型在当前计算步骤中，每一层 Transformer 计算出的最新键（Key）和值（Value）缓存
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states,presents
