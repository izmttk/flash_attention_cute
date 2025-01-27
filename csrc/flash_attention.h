#pragma once

namespace flash_attention {

struct FlashAttentionParams {
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrict
    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;
    void* __restrict__ o_ptr;

    int64_t batch_size;
    int64_t num_heads_q;
    int64_t num_heads_kv;
    int64_t seqlen_q;
    int64_t seqlen_kv;
    int64_t headdim;

    int64_t head_q_per_group;

    int64_t q_batch_stride;
    int64_t k_batch_stride;
    int64_t v_batch_stride;
    int64_t o_batch_stride;

    int64_t q_head_stride;
    int64_t k_head_stride;
    int64_t v_head_stride;
    int64_t o_head_stride;

    int64_t q_seqlen_stride;
    int64_t k_seqlen_stride;
    int64_t v_seqlen_stride;
    int64_t o_seqlen_stride;

    float softmax_scale;
};

// Need to be implemented in flash_attention_impl.cu
template <typename T, int kHeaddim, bool IsCausal>
void run_flash_attention(FlashAttentionParams &params, cudaStream_t stream);

} // namespace flash_attention
