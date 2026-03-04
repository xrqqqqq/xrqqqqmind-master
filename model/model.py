import math

from transformers import PretrainedConfig

#hugging face的类，PretrainedConfig是一个基类，用于定义预训练模型的配置。它包含了模型的各种超参数和设置，可以通过继承这个类来创建特定模型的配置类。在这个代码中，MokioMindConfig继承了PretrainedConfig，并定义了MokioMind模型的特定配置参数，如dropout率、隐藏层大小、注意力头数等。这些参数可以在实例化MokioMindConfig时进行设置，以便在训练或推理过程中使用。
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
#msnorm

#继承nn.Module类
class RMSNorm(nn.Module):
#__init__初始化
    def __init__(self,dim:int,eps:float=1e-5):
        #dim:int是输入张量的维度，eps:float是一个小的常数，用于防止除以零的情况。
        #dim：是python中的类型提示，表示dim参数应该是一个整数。eps：也是类型提示，表示eps参数应该是一个浮点数。
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
#_norm
    def _norm(self,x):
        #keepdim=True保持维度不变，mean(-1)在最后一个维度上求平均值，pow(2)对每个元素进行平方运算，rsqrt开方求倒数。
        return torch.rsqrt(x.pow(2).mean(-1,keepdim = True) + self.eps)
#forward前向传播
    def forward(self,x):
        return self.weight * self._norm(x.float()).type_as(x)

    # precomputer_freqs_cis函数用于预计算频率的复数表示。它首先计算频率的实部和虚部，然后根据ROPE缩放的配置进行调整，最后返回一个复数张量。


def precomputer_freqs_cis(dim:int,end:int(32*1024),rope_base,rope_scaling:Optional[dict]=None):
    #dum:int是一个整数参数，表示维度的大小。end:int是一个整数参数，表示频率的结束值，默认为32*1024。rope_base是一个参数，表示ROPE（Relative Positional Encoding）的基数。rope_scaling:Optional[dict]是一个可选的字典参数，表示ROPE缩放的配置。

    #初始化RoPE频率
    freqs,attn_factor = (1.0/(rope_base ** (torch.arange(0,dim,2)[:(dim//2)].float()/dim),1.0))
    #//2表示整数除法，torch.arange(0,dim,2)生成一个从0到dim的步长为2的整数序列，[:(dim//2)]表示取前dim//2个元素。float()将整数转换为浮点数。最后得到的freqs是一个包含频率值的张量，attn_factor是一个标量值。

    if rope_scaling is not None:
        #如果rope_scaling不为None,用yarn。
        orig_max,factor,beta_fast,beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"],
        )

        #推断长度大于训练长度，使用缩放
        if end > orig_max:
            # 波长b到i的映射
            inv_dim = lambda b: (dim*math.log(orig_max/(b*2*math.pi)))/(2*math.log(rope_base))
            #lambda函数是一种匿名函数，inv_dim是一个函数对象，接受一个参数b，并返回一个计算结果。这个函数的作用是根据输入的b值计算出对应的维度大小。

            #划分高低维度
            #low :不需要缩放的高频部分
            #high:需要缩放的低频部分

            low,high =  max(math.floor(inv_dim(beta_fast)),0),min(math.ceil(inv_dim(beta_slow)),dim//2 - 1)

            #计算缩放因子
            #low之前，ramp为0，再high之后，ramp为1，在low和high之间，ramp在0和1之间线性变化，先行过度。
            ramp = torch.clamp(
                (torch.arange(dim//2,device = freqs.device).float() - low)/max(high-low,0.001),
                #arange生成一个从0到dim//2的整数序列，clamp函数将输入的值限制在0和1之间，确保ramp的值在合理范围内。max(high-low,0.001)用于防止除以零的情况。
                #(dim//2 - low)表示需要缩放的维度范围，除以这个范围可以将ramp的值归一化到0和1之间。
                #(device = freqs.device)确保生成的张量与freqs在同一设备上。
                0,
                1,
            )
            #当ramp = 0（高频）：系数为1，保持原频率不变
            #当ramp = 1（低频）：系数为1/factor，频率缩小factor倍
            #当0 < ramp < 1（过渡区域）：系数在1和1/factor之间线性变化，逐渐缩小频率
            freqs = freqs *(1-ramp+ramp/factor)
        #根据end，生成位置索引t
        t = torch.arange(end,device = freqs.device).float()

        #计算外积，将t和freq相乘，得到每个位置的旋转角度
        #t是温度，freqs是频率，外积得到每个位置的旋转角度。
        freqs = torch.outer(t,freqs)

        freqs_cos = (
            torch.cat([torch.cos(freqs),torch.sin(freqs)],dim = -1).float()*attn_factor

        ) 

        freqs_sin = (
            torch.cat([torch.sin(freqs),-torch.cos(freqs)],dim = -1).float()*attn_factor
        ) 

        return freqs_cos,freqs_sin
    

