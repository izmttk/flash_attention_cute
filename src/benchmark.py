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


def select_reg_SM80_16x8x16_F16F16F16F16_TN(Q):
    n_atoms = Q.size(-1) // 8
    reg = torch.zeros((2, n_atoms * 2), device="cuda")
    for i in range(n_atoms):
        reg[0, i*2] = Q[0, 0, 0, i*8]
        reg[0, i*2+1] = Q[0, 0, 0, i*8 + 1]
        reg[1, i*2] = Q[0, 0, 8, i*8]
        reg[1, i*2+1] = Q[0, 0, 8, i*8 + 1]
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
    # print(select_reg_SM80_16x8x16_F16F16F16F16_TN(attn))
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
    k = torch.randn((bs, num_heads, seq_len, dim), dtype=torch.half, device="cuda")
    v = torch.randn((bs, num_heads, seq_len, dim), dtype=torch.half, device="cuda")
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
    scale = dim ** -0.5
    o_cpp = flash_attention_ext.flash_attention_v2_cute(q, k, v, scale)
    print("O shape(cpp):", o_cpp.shape)
    o_py = sdpa(q, k, v, scale=scale)
    print("O shape(py):", o_py.shape)
    print("MSE:", mse(o_cpp, o_py).item())
    print("Allclose:", torch.allclose(o_cpp, o_py, atol=1e-3))

    # print(o_cpp)
    # print(o_py)

    start = time.time()
    for _ in range(iter):
        o_cpp = flash_attention_ext.flash_attention_v2_cute(q, k, v, scale)
    end = time.time()
    elapsed = (end - start) * 1e3
    print(f"Elapsed time (cpp): {elapsed:.3f} ms")
    print(f"Time per iteration (cpp): {elapsed / iter:.3f} ms")

    start = time.time()
    for _ in range(iter):
        o_py = sdpa(q, k, v, scale=scale)
    end = time.time()
    elapsed = (end - start) * 1e3
    print(f"Elapsed time (py): {elapsed:.3f} ms")
    print(f"Time per iteration (py): {elapsed / iter:.3f} ms")

if __name__ == "__main__":
    benchmark()