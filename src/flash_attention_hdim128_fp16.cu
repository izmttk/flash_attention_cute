#include "flash_attention_template.cuh"

template<>
void run_flash_attention<cutlass::half_t, 128>(FlashAttentionParams params) {
    constexpr int kHeaddim = 128;
    constexpr int kBlockM = 32;
    constexpr int kBlockN = 128;
    constexpr int kNWarps = 4;
    launch_flash_attention<cutlass::half_t, kHeaddim, kBlockM, kBlockN, kNWarps>(params);
}