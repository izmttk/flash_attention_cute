import torch
import time 
from flash_attn import flash_attn_func as flash_attn_official
from flash_attention import flash_attn_func as flash_attn_custom

torch.set_grad_enabled(False)


def flash_attention_custom(q, k, v, softmax_scale=None):
    return flash_attn_custom(q, k, v, softmax_scale=softmax_scale)

def flash_attention_official(q, k, v, softmax_scale=None):
    # reshape from [batch_size, n_heads, seq_len, d] to [batch_size, seq_len, n_heads, d]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    attn = flash_attn_official(q, k, v, softmax_scale=softmax_scale)
    # reshape back to [batch_size, n_heads, seq_len, d]
    attn = attn.permute(0, 2, 1, 3)
    return attn

def sdpa(query, key, value, softmax_scale=None, mask=None):
    # query: [batch_size, n_heads, seq_len, d_k]
    query = query.float()
    key = key.float()
    value = value.float()
    scale_factor = (query.size(-1) ** -0.5) if softmax_scale is None else softmax_scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    # print(select_thrCreg_SM80_16x8x16_F16F16F16F16_TN(torch.matmul(query, key.transpose(-2, -1))[0, 0, :, :], 0))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.matmul(attn, value)
    return attn.half()

def mse(a, b):
    a = a.float()
    b = b.float()
    return torch.mean((a - b) ** 2)

def benchmark(iter = 100):
    bs = 16
    num_heads = 16
    q_seqlen = 1024
    kv_seqlen = 1024
    dim = 128
    q = torch.randn((bs, num_heads, q_seqlen, dim), dtype=torch.half, device="cuda")
    k = torch.randn((bs, num_heads, kv_seqlen, dim), dtype=torch.half, device="cuda")
    v = torch.randn((bs, num_heads, kv_seqlen, dim), dtype=torch.half, device="cuda")
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

    # print(torch.matmul(q.float(), k.transpose(-2, -1).float())[0, 0, 0, :2])
    print("Q stride:", q.stride())
    print("K stride:", k.stride())
    print("V stride:", v.stride())

    print("Benchmarking...")
    print("Q shape:", q.shape)
    print("K shape:", k.shape)
    print("V shape:", v.shape)
    o_ours = flash_attention_custom(q, k, v)
    print("O shape(ours):", o_ours.shape)
    o_pytorch = sdpa(q, k, v)
    print("O shape(pytorch):", o_pytorch.shape)
    o_official = flash_attention_official(q, k, v)
    print("O shape(official):", o_official.shape)

    print("MSE(ours, pytorch):", mse(o_ours, o_pytorch).item())
    print("Allclose(ours, pytorch):", torch.allclose(o_ours, o_pytorch, atol=1e-3))

    print("MSE(ours, official):", mse(o_ours, o_official).item())
    print("Allclose(ours, official):", torch.allclose(o_ours, o_official, atol=1e-3))

    print("MSE(pytorch, official):", mse(o_pytorch, o_official).item())
    print("Allclose(pytorch, official):", torch.allclose(o_pytorch, o_official, atol=1e-3))

    # print(o_ours)
    # print(o_pytorch)
    # print(o_official)

    start = time.time()
    for _ in range(iter):
        o_ours = flash_attention_custom(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    elapsed = (end - start) * 1e3
    print(f"Elapsed time (ours): {elapsed:.3f} ms")
    print(f"Time per iteration (ours): {elapsed / iter:.3f} ms")

    start = time.time()
    for _ in range(iter):
        o_pytorch = sdpa(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    elapsed = (end - start) * 1e3
    print(f"Elapsed time (pytorch): {elapsed:.3f} ms")
    print(f"Time per iteration (pytorch): {elapsed / iter:.3f} ms")


    start = time.time()
    for _ in range(iter):
        o_official = flash_attention_official(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    elapsed = (end - start) * 1e3
    print(f"Elapsed time (official): {elapsed:.3f} ms")
    print(f"Time per iteration (official): {elapsed / iter:.3f} ms")

if __name__ == "__main__":
    benchmark()