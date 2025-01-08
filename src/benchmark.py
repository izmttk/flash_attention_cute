import torch
import time 
from torch.utils.cpp_extension import load
from pathlib import Path

torch.set_grad_enabled(False)

ROOT = Path(__file__).parent.parent
CSRC = ROOT / "src"

# Load the CUDA kernel as a python module
flash_attention_ext = load(
    name="flash_attention",
    sources=[
        str(CSRC / "flash_attention_api.cpp"),
        str(CSRC / "flash_attention.cu"),
    ],
    extra_cuda_cflags=[
        "-O2",
        "-std=c++17",
        "-Xcompiler", "/w",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        # "-U__CUDA_NO_HALF2_OPERATORS__",
        # "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        # "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    extra_include_paths=[
        str(ROOT / "include"),
        str(ROOT / "3rd/cutlass/include"),
        str(ROOT / "3rd/cutlass/tools/util/include"),
    ]
    # verbose=True
)


def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float = None) -> torch.Tensor:
    # q: [batch_size, n_heads, q_seq_len,  d]
    # k: [batch_size, n_heads, kv_seq_len, d]
    # v: [batch_size, n_heads, kv_seq_len, d]
    # scale: float
    scale = (q.size(-1) ** -0.5) if scale is None else scale
    head_dim = q.size(3)
    if head_dim % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_dim % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_dim % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_dim % 8])
    q = q.contiguous() if q.stride(3) != -1 else q
    k = k.contiguous() if k.stride(3) != -1 else k
    v = v.contiguous() if v.stride(3) != -1 else v
    attn = flash_attention_ext.flash_attention_v2_cute(q, k, v, scale)
    return attn


def select_thrCreg_SM80_16x8x16_F16F16F16F16_TN(C, thridx):
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

def sdpa(query, key, value, scale=None, mask=None):
    # query: [batch_size, n_heads, seq_len, d_k]
    query = query.float()
    key = key.float()
    value = value.float()
    scale_factor = (query.size(-1) ** -0.5) if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
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
    bs = 2
    num_heads = 16
    seq_len = 512
    dim = 128
    q = torch.randn((bs, num_heads, seq_len, dim), dtype=torch.half, device="cuda")
    k = torch.randn((bs, num_heads, seq_len*2, dim), dtype=torch.half, device="cuda")
    v = torch.randn((bs, num_heads, seq_len*2, dim), dtype=torch.half, device="cuda")
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
    o_cpp = flash_attn(q, k, v)
    print("O shape(cpp):", o_cpp.shape)
    o_py = sdpa(q, k, v)
    print("O shape(py):", o_py.shape)
    print("MSE:", mse(o_cpp, o_py).item())
    print("Allclose:", torch.allclose(o_cpp, o_py, atol=1e-3))

    # print(o_cpp)
    # print(o_py)

    start = time.time()
    for _ in range(iter):
        o_cpp = flash_attn(q, k, v)
    end = time.time()
    elapsed = (end - start) * 1e3
    print(f"Elapsed time (cpp): {elapsed:.3f} ms")
    print(f"Time per iteration (cpp): {elapsed / iter:.3f} ms")

    start = time.time()
    for _ in range(iter):
        o_py = sdpa(q, k, v)
    end = time.time()
    elapsed = (end - start) * 1e3
    print(f"Elapsed time (py): {elapsed:.3f} ms")
    print(f"Time per iteration (py): {elapsed / iter:.3f} ms")

if __name__ == "__main__":
    benchmark()