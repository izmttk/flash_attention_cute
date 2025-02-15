import torch
from .load_cpp_extention import load_extension

flash_attention_cuda = load_extension()

@torch.library.custom_op("flash_attention::forward", mutates_args=())
def flash_attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float = None, causal: bool = False) -> torch.Tensor:
    # q: [batch_size, n_heads, q_seq_len,  d]
    # k: [batch_size, n_heads, kv_seq_len, d]
    # v: [batch_size, n_heads, kv_seq_len, d]
    # softmax_scale: float
    # causal: bool
    print("Flash Attention only support cuda now, fallback to pytorch implementation.")
    attn = torch.functional.F.scaled_dot_product_attention(q, k, v, scale=softmax_scale, is_causal=causal)
    return attn

@torch.library.register_kernel("flash_attention::forward", "cuda")
def flash_attention_forward_cuda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float = None, causal: bool = False) -> torch.Tensor:
    # q: [batch_size, n_heads, q_seq_len,  d]
    # k: [batch_size, n_heads, kv_seq_len, d]
    # v: [batch_size, n_heads, kv_seq_len, d]
    # softmax_scale: float
    # causal: bool
    head_dim = q.size(3)

    need_padding = False
    if head_dim % 8 != 0:
        need_padding = True
        q = torch.nn.functional.pad(q, [0, 8 - head_dim % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_dim % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_dim % 8])
    q = q.contiguous() if q.stride(3) != 1 else q
    k = k.contiguous() if k.stride(3) != 1 else k
    v = v.contiguous() if v.stride(3) != 1 else v
    attn = flash_attention_cuda.flash_attention_fwd(q, k, v, softmax_scale, causal)
    if need_padding:
        attn = attn[:, :, :, :head_dim]
    return attn

@torch.library.register_fake("flash_attention::forward")
def flash_attention_forward_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float = None, causal: bool = False) -> torch.Tensor:
    attn = torch.empty_like(q)
    return attn


def flash_attn_func(q, k, v, softmax_scale=None, causal=False):
    # q: [batch_size, n_heads, q_seq_len,  d]
    # k: [batch_size, n_heads, kv_seq_len, d]
    # v: [batch_size, n_heads, kv_seq_len, d]
    # softmax_scale: float
    # causal: bool
    softmax_scale = (q.size(-1) ** -0.5) if softmax_scale is None else softmax_scale
    return torch.ops.flash_attention.forward(q, k, v, softmax_scale, causal)
