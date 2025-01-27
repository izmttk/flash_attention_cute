#include "flash_attention.h"
#include "utils.h"
#include "flash_attention_template.cuh"

namespace flash_attention {

template <typename T, bool IsCausal>
void run_flash_attention_hdim128(FlashAttentionParams &params, cudaStream_t stream) {
    // kBlockM x kBlockN = 128x32 在 hdim128 下相比 64x64 性能更高一点
    // 猜测可能的原因是这两种配置需要的 smem 大小相同，48KB
    // 但是 64x64 会启动 q_seqlen / 64 个 block，而 128x32 只需要启动 q_seqlen / 128 个 block
    // 128x32 的线程块数量比 64x64 少一半
    // 又因为每个 block 都需要重复加载一次 kv，所以 128x32 的访存压力更小
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x
    // compute capability   8.0      8.6      8.7      8.9
    // shared memory / SM   168 KB   100 KB   168 KB   100 KB
    if (cc_minor == 0 || cc_minor == 7) {
        // 128x64 需要 smem 大小为 64KB，适合 sm80、sm87
        launch_flash_attention<T, /* kBlockM */ 128, /* kBlockN */ 64, /* kHeaddim */ 128, /* kNWarps */ 4, /* IsCausal */ IsCausal>(params, stream);
    } else {
        // 128x32 需要 smem 大小为 48KB，适合 sm86、sm89
        launch_flash_attention<T, /* kBlockM */ 128, /* kBlockN */ 32, /* kHeaddim */ 128, /* kNWarps */ 4, /* IsCausal */ IsCausal>(params, stream);
    }
}

// hdim=128 函数模板具体化

template<>
void run_flash_attention<cutlass::half_t, 128, false>(FlashAttentionParams &params, cudaStream_t stream) {
    run_flash_attention_hdim128<cutlass::half_t, false>(params, stream);
}

template<>
void run_flash_attention<cutlass::half_t, 128, true>(FlashAttentionParams &params, cudaStream_t stream) {
    run_flash_attention_hdim128<cutlass::half_t, true>(params, stream);
}


template<>
void run_flash_attention<cutlass::bfloat16_t, 128, false>(FlashAttentionParams &params, cudaStream_t stream) {
    run_flash_attention_hdim128<cutlass::bfloat16_t, false>(params, stream);
}

template<>
void run_flash_attention<cutlass::bfloat16_t, 128, true>(FlashAttentionParams &params, cudaStream_t stream) {
    run_flash_attention_hdim128<cutlass::bfloat16_t, true>(params, stream);
}

} // namespace flash_attention
