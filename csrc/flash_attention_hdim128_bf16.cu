#include "flash_attention.h"
#include "flash_attention_template.cuh"

namespace flash_attention {

template<>
void run_flash_attention<cutlass::bfloat16_t, 128, false>(FlashAttentionParams &params) {
    launch_flash_attention<cutlass::bfloat16_t, /* kBlockM */ 128, /* kBlockN */ 32, /* kHeaddim */ 128, /* kNWarps */ 4, /* IsCausal */ false>(params);
}

template<>
void run_flash_attention<cutlass::bfloat16_t, 128, true>(FlashAttentionParams &params) {
    launch_flash_attention<cutlass::bfloat16_t, /* kBlockM */ 128, /* kBlockN */ 32, /* kHeaddim */ 128, /* kNWarps */ 4, /* IsCausal */ true>(params);
}

} // namespace flash_attention
