#include "flash_attention.h"
#include "flash_attention_template.cuh"

namespace flash_attention {

template<>
void run_flash_attention<cutlass::half_t, 128, false>(FlashAttentionParams &params) {
    // kBlockM x kBlockN = 128x32 在 hdim128 下相比 64x64 性能更高一点
    // 猜测可能的远因是这两种配置需要的 smem 大小相同，48KB
    // 但是 64x64 会启动 q_seqlen / 64 个 block，而 128x32 只需要启动 q_seqlen / 128 个 block
    // 128x32 的线程块数量比 64x64 少一半
    // 又因为每个 block 都需要重复加载一次 kv，所以 128x32 的访存压力更小
    launch_flash_attention<cutlass::half_t, /* kBlockM */ 128, /* kBlockN */ 32, /* kHeaddim */ 128, /* kNWarps */ 4, /* IsCausal */ false>(params);
}

template<>
void run_flash_attention<cutlass::half_t, 128, true>(FlashAttentionParams &params) {
    launch_flash_attention<cutlass::half_t, /* kBlockM */ 128, /* kBlockN */ 32, /* kHeaddim */ 128, /* kNWarps */ 4, /* IsCausal */ true>(params);
}

} // namespace flash_attention
