import os
from torch.utils.cpp_extension import load
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
CSRC = ROOT / "csrc"

IS_WINDOWS = sys.platform == 'win32'

def load_extension():
    # 这应该是 pytorch cpp extension load 的一个 bug，不会检查 cflags 是否已经提供 arch 信息
    # 然后会导致一个 warning，不影响编译
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

    if IS_WINDOWS:
        cflags = ["/O2", "/std:c++17"]
    else:
        cflags = ["-O3", "-std=c++17"]

    # Load the CUDA kernel as a python module
    flash_attention_ext = load(
        name="flash_attention_extension",
        sources=[
            str(CSRC / "flash_attention_api.cpp"),
            str(CSRC / "flash_attention_hdim128_fp16.cu"),
            str(CSRC / "flash_attention_hdim128_bf16.cu"),
        ],
        extra_cflags=cflags,
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
            str(CSRC / "3rd/cutlass/include"),
            str(CSRC / "3rd/cutlass/tools/util/include"),
        ],
        # verbose=True
    )
    return flash_attention_ext