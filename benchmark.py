import torch
import time 
from flash_attn import flash_attn_func as flash_attn_official
from flash_attention import flash_attn_func as flash_attn_custom

torch.set_grad_enabled(False)

def flash_attention_custom(q, k, v, softmax_scale=None, is_causal=False):
    return flash_attn_custom(q, k, v, softmax_scale=softmax_scale, causal=is_causal)

def flash_attention_official(q, k, v, softmax_scale=None, is_causal=False):
    # reshape from [batch_size, n_heads, seq_len, d] to [batch_size, seq_len, n_heads, d]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    attn = flash_attn_official(q, k, v, softmax_scale=softmax_scale, causal=is_causal)
    # reshape back to [batch_size, n_heads, seq_len, d]
    attn = attn.permute(0, 2, 1, 3)
    return attn

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
def sdpa(query, key, value, softmax_scale=None, is_causal=False):
    # scale_factor = (query.size(-1) ** -0.5) if softmax_scale is None else softmax_scale
    # scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    # # print(select_thrCreg_SM80_16x8x16_F16F16F16F16_TN(torch.matmul(query, key.transpose(-2, -1))[0, 0, :, :], 0))
    # if mask is not None:
    #     scores = scores.masked_fill(mask == 0, float('-inf'))
    # attn = torch.softmax(scores, dim=-1, dtype=torch.float)
    # attn = torch.matmul(attn.to(value.dtype), value)
    # return attn
    with sdpa_kernel(SDPBackend.MATH):
        attn = scaled_dot_product_attention(query, key, value, scale=softmax_scale, is_causal=is_causal)
    return attn

def mse(a, b):
    a = a.float()
    b = b.float()
    return torch.mean((a - b) ** 2)

def benchmark(iter = 100):
    bs = 16
    num_heads_q = 64
    # num_heads_kv = 64 # Case for MHA
    num_heads_kv = 8 # Case for GQA
    # num_heads_kv = 1 # Case for MQA
    seqlen_q = 1024 # Case for prefilling stage
    # seqlen_q = 1 # Case for decoding stage
    seqlen_kv = 1024
    dim = 128
    is_causal = True

    q = torch.randn((bs, num_heads_q, seqlen_q, dim), dtype=torch.half, device="cuda")
    k = torch.randn((bs, num_heads_kv, seqlen_kv, dim), dtype=torch.half, device="cuda")
    v = torch.randn((bs, num_heads_kv, seqlen_kv, dim), dtype=torch.half, device="cuda")
    # q = torch.ones((bs, num_heads, seq_len, dim), dtype=torch.half, device="cuda")
    # k = torch.ones((bs, num_heads, seq_len, dim), dtype=torch.half, device="cuda")
    # v = torch.ones((bs, num_heads, seq_len, dim), dtype=torch.half, device="cuda")
    # for i in range(v.size(-1)):
    #     v[:, :, :, i] = i
    # q = torch.arange(0, bs * num_heads * seq_len * dim, dtype=torch.half, device="cuda").reshape(bs, num_heads, seq_len, dim)
    # k = torch.arange(0, bs * num_heads * seq_len * dim, dtype=torch.half, device="cuda").reshape(bs, num_heads, seq_len, dim)
    # v = torch.arange(0, bs * num_heads * seq_len * dim, dtype=torch.half, device="cuda").reshape(bs, num_heads, seq_len, dim)
    # q = torch.ones((bs, num_heads, seq_len, dim), dtype=torch.half, device="cuda")
    # k = torch.ones((bs, num_heads, seq_len, dim), dtype=torch.half, device="cuda") * 2
    # v = torch.ones((bs, num_heads, seq_len, dim), dtype=torch.half, device="cuda") * 3

    print("Benchmarking...")
    print("Q:", q.size(), q.stride())
    print("K:", k.size(), k.stride())
    print("V:", v.size(), v.stride())
    o_ours = flash_attention_custom(q, k, v, is_causal=is_causal)
    print("O(ours):", o_ours.size(), o_ours.stride())
    # o_pytorch = sdpa(q, k, v, is_causal=is_causal)
    # print("O(pytorch):", o_pytorch.size(), o_pytorch.stride())
    o_official = flash_attention_official(q, k, v, is_causal=is_causal)
    print("O(official):", o_official.size(), o_official.stride())

    # print("MSE(ours, pytorch):", mse(o_ours, o_pytorch).item())
    # print("Allclose(ours, pytorch):", torch.allclose(o_ours, o_pytorch, atol=1e-3))

    print("MSE(ours, official):", mse(o_ours, o_official).item())
    print("Allclose(ours, official):", torch.allclose(o_ours, o_official, atol=1e-3))

    # print("MSE(pytorch, official):", mse(o_pytorch, o_official).item())
    # print("Allclose(pytorch, official):", torch.allclose(o_pytorch, o_official, atol=1e-3))

    # print(o_ours)
    # print(o_pytorch)
    # print(o_official)

    start = time.time()
    for _ in range(iter):
        o_ours = flash_attention_custom(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    end = time.time()
    elapsed = (end - start) * 1e3
    print(f"Elapsed time (ours): {elapsed:.3f} ms")
    print(f"Time per iteration (ours): {elapsed / iter:.3f} ms")

    # start = time.time()
    # for _ in range(iter):
    #     o_pytorch = sdpa(q, k, v, is_causal=is_causal)
    # torch.cuda.synchronize()
    # end = time.time()
    # elapsed = (end - start) * 1e3
    # print(f"Elapsed time (pytorch): {elapsed:.3f} ms")
    # print(f"Time per iteration (pytorch): {elapsed / iter:.3f} ms")


    start = time.time()
    for _ in range(iter):
        o_official = flash_attention_official(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    end = time.time()
    elapsed = (end - start) * 1e3
    print(f"Elapsed time (official): {elapsed:.3f} ms")
    print(f"Time per iteration (official): {elapsed / iter:.3f} ms")

if __name__ == "__main__":
    benchmark()