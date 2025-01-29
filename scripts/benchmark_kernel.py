import torch
import time
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT.resolve()))

from flash_attn import flash_attn_func as flash_attn_official
from flash_attention import flash_attn_func as flash_attn_custom

torch.set_grad_enabled(False)

def eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float = None,
    is_causal=False,
):
    scale_factor = 1 / (query.size(-1) ** 0.5) if softmax_scale is None else softmax_scale
    L, S = query.size(-2), key.size(-2)
    
    origin_dtype = query.dtype
    query = query.float()
    key = key.float()
    value = value.float()
    
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal:
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        
    key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
    value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    attn_weights += attn_bias
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.to(origin_dtype)

def flash_attention_custom(q, k, v, softmax_scale=None, is_causal=False):
    return flash_attn_custom(q, k, v, softmax_scale=softmax_scale, causal=is_causal)

def flash_attention_official(q, k, v, softmax_scale=None, is_causal=False):
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    attn = flash_attn_official(q, k, v, softmax_scale=softmax_scale, causal=is_causal)
    attn = attn.permute(0, 2, 1, 3)
    return attn

def mse(a, b):
    a = a.float()
    b = b.float()
    return torch.mean((a - b) ** 2)

def run_benchmark(
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    iter: int,
    is_causal: bool,
    dtype: torch.dtype,
    device: torch.device,
):
    # 初始化张量
    q = torch.randn((batch_size, num_heads_q, seqlen_q, dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, num_heads_kv, seqlen_kv, dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, num_heads_kv, seqlen_kv, dim), dtype=dtype, device=device)

    print("\nBenchmark Configuration:")
    print(f"Batch size: {batch_size}")
    print(f"Query heads: {num_heads_q}, Key/Value heads: {num_heads_kv}")
    print(f"Query length: {seqlen_q}, Key/Value length: {seqlen_kv}")
    print(f"Dimension: {dim}, Causal: {is_causal}")
    print(f"Data type: {dtype}, Device: {device}")
    print(f"Iterations: {iter}\n")

    # 运行基准测试
    print("Running benchmarks...")
    
    # 自定义实现
    start = time.time()
    for _ in range(iter):
        o_custom = flash_attention_custom(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1e3
    print(f"[Custom] Total: {elapsed:.3f}ms | Per iter: {elapsed/iter:.3f}ms")

    # 官方FA2实现
    start = time.time()
    for _ in range(iter):
        o_official = flash_attention_official(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1e3
    print(f"[Official] Total: {elapsed:.3f}ms | Per iter: {elapsed/iter:.3f}ms")

    # Pytorch eager 实现
    start = time.time()
    for _ in range(iter):
        o_eager = eager_attention(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1e3
    print(f"[Eager] Total: {elapsed:.3f}ms | Per iter: {elapsed/iter:.3f}ms")

    # 精度验证
    print("\nChecking accuracy...")
    with torch.no_grad():
        o_custom = flash_attention_custom(q, k, v, is_causal=is_causal)
        o_official = flash_attention_official(q, k, v, is_causal=is_causal)
        o_eager = eager_attention(q, k, v, is_causal=is_causal)
        print(f"MSE between implementations (custom, official): {mse(o_custom, o_official):.4e}")
        print(f"MSE between implementations (custom, eager): {mse(o_custom, o_eager):.4e}")
        print(f"AllClose check (custom, official): {torch.allclose(o_custom, o_official, atol=1e-3)}")
        print(f"AllClose check (custom, eager): {torch.allclose(o_custom, o_eager, atol=1e-3)}")
        
def main():
    parser = argparse.ArgumentParser(description='Flash Attention Benchmark Tool')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-heads-q', type=int, default=64, help='Number of query heads')
    parser.add_argument('--num-heads-kv', type=int, default=8, help='Number of key/value heads (for GQA/MQA)')
    parser.add_argument('--seqlen-q', type=int, default=1024, help='Query sequence length')
    parser.add_argument('--seqlen-kv', type=int, default=1024, help='Key/Value sequence length')
    parser.add_argument('--dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--iter', type=int, default=100, help='Number of iterations')
    parser.add_argument('--dtype', type=str, default='half', help='Data type')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--causal', action='store_true', help='Use causal attention')
    
    def get_torch_dtype(dtype: str) -> torch.dtype:
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)
        return dtype
    
    def get_torch_device(device: str) -> torch.device:
        return torch.device(device)
    
    args = parser.parse_args()
    batch_size = args.batch_size
    num_heads_q = args.num_heads_q
    num_heads_kv = args.num_heads_kv
    seqlen_q = args.seqlen_q
    seqlen_kv = args.seqlen_kv
    dim = args.dim
    iter = args.iter
    is_causal = args.causal
    dtype = get_torch_dtype(args.dtype)
    device = get_torch_device(args.device)

    run_benchmark(
        batch_size,
        num_heads_q,
        num_heads_kv,
        seqlen_q,
        seqlen_kv,
        dim,
        iter,
        is_causal,
        dtype,
        device,
    )

if __name__ == "__main__":
    main()

"""
Usage Examples:

GQA config in Llama 2 7B model in prefilling stage with seqlen_q = 1024:
python benchmark_kernel.py --batch-size 16 --num-heads-q 64 --num-heads-kv 8 --seqlen-q 1024 --seqlen-kv 1024 --dim 128 --dtype half --device cuda

decoding stage with seqlen_q = 1
python benchmark_kernel.py --batch-size 16 --num-heads-q 64 --num-heads-kv 8 --seqlen-q 1 --seqlen-kv 1024 --dim 128  --dtype half --device cuda

dim = 64:
python benchmark_kernel.py --batch-size 16 --num-heads-q 64 --num-heads-kv 8 --seqlen-q 1024 --seqlen-kv 1024 --dim 64 --dtype half --device cuda

with causal mask:
python benchmark_kernel.py --batch-size 16 --num-heads-q 64 --num-heads-kv 8 --seqlen-q 1024 --seqlen-kv 1024 --dim 128 --causal --dtype half --device cuda

bfloat16 dtype:
python benchmark_kernel.py --batch-size 16 --num-heads-q 64 --num-heads-kv 8 --seqlen-q 1024 --seqlen-kv 1024 --dim 128 --dtype bfloat16 --device cuda

MHA, aka num_heads_q == num_heads_kv:
python benchmark_kernel.py --batch-size 16 --num-heads-q 64 --num-heads-kv 64 --seqlen-q 1024 --seqlen-kv 1024 --dim 128 --dtype half --device cuda
"""