import os
import torch
import time 
from torch.utils.cpp_extension import load
from pathlib import Path
from flash_attn import flash_attn_func

torch.set_grad_enabled(False)

ROOT = Path(__file__).parent.parent
CSRC = ROOT / "src"

# 这应该是 pytorch cpp extension load 的一个 bug，不会检查 cflags 是否已经提供 arch 信息
# 然后会导致一个 warning，不影响编译
if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

# Load the CUDA kernel as a python module
flash_attention_ext = load(
    name="flash_attention",
    sources=[
        str(CSRC / "flash_attention_api.cpp"),
        str(CSRC / "flash_attention_hdim128_fp16.cu"),
    ],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        # "-Xcompiler", "/w",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math", # used for expf
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        
        # "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        # "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        # "-U__CUDA_NO_BFLOAT162_OPERATORS__",

        # "-gencode=arch=compute_80,code=sm_80",
    ],
    extra_include_paths=[
        str(ROOT / "include"),
        str(ROOT / "3rd/cutlass/include"),
        str(ROOT / "3rd/cutlass/tools/util/include"),
    ],
    # verbose=True
)


def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float = None) -> torch.Tensor:
    # q: [batch_size, n_heads, q_seq_len,  d]
    # k: [batch_size, n_heads, kv_seq_len, d]
    # v: [batch_size, n_heads, kv_seq_len, d]
    # scale: float
    softmax_scale = (q.size(-1) ** -0.5) if softmax_scale is None else softmax_scale
    head_dim = q.size(3)

    need_padding = False
    if head_dim % 8 != 0:
        need_padding = True
        q = torch.nn.functional.pad(q, [0, 8 - head_dim % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_dim % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_dim % 8])
    q = q.contiguous() if q.stride(3) != -1 else q
    k = k.contiguous() if k.stride(3) != -1 else k
    v = v.contiguous() if v.stride(3) != -1 else v
    attn = flash_attention_ext.flash_attention_v2_cute(q, k, v, softmax_scale)
    if need_padding:
        attn = attn[:, :, :, :head_dim]
    return attn

def flash_attention_official(q, k, v, softmax_scale=None):
    # reshape from [batch_size, n_heads, seq_len, d] to [batch_size, seq_len, n_heads, d]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    attn = flash_attn_func(q, k, v, softmax_scale=softmax_scale)
    # reshape back to [batch_size, n_heads, seq_len, d]
    attn = attn.permute(0, 2, 1, 3)
    return attn

def select_thrCreg_SM80_16x8x16_F16F16F16F16_TN(C: torch.Tensor, thridx):
    C = torch.nn.functional.pad(C, [0, (8 - C.size(1) % 8) % 8, 0, (16 - C.size(0) % 16) % 16])
    print("C shape:", C.shape)
    atom_m, atom_n = C.size(0) // 16, C.size(1) // 8
    thr_m, thr_n = thridx // 4, thridx % 4
    reg = torch.zeros((atom_m * 2, atom_n * 2), device=C.device, dtype=C.dtype)
    for m in range(atom_m):
        for n in range(atom_n):
            reg[m * 2,     n * 2    ] = C[m * 16 + thr_m,     n * 8 + thr_n * 2    ]
            reg[m * 2,     n * 2 + 1] = C[m * 16 + thr_m,     n * 8 + thr_n * 2 + 1]
            reg[m * 2 + 1, n * 2    ] = C[m * 16 + thr_m + 8, n * 8 + thr_n * 2    ]
            reg[m * 2 + 1, n * 2 + 1] = C[m * 16 + thr_m + 8, n * 8 + thr_n * 2 + 1]
    return reg

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
    o_ours = flash_attn(q, k, v)
    print("O shape(ours):", o_ours.shape)
    o_pytorch = sdpa(q, k, v)
    print("O shape(pytorch):", o_pytorch.shape)
    o_official = flash_attention_official(q, k, v)
    print("O shape(official):", o_official.shape)

    print("MSE(ours, pytorch):", mse(o_ours, o_pytorch).item())
    print("Allclose(ours, pytorch):", torch.allclose(o_ours, o_pytorch, atol=1e-3))

    print("MSE(ours, official):", mse(o_ours, o_official).item())
    print("Allclose(ours, official):", torch.allclose(o_ours, o_official, atol=1e-3))

    # print(o_ours)
    # print(o_pytorch)
    # print(o_official)

    start = time.time()
    for _ in range(iter):
        o_ours = flash_attn(q, k, v)
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