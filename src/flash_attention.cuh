#pragma once

struct FlashAttentionParams {
    void* q_ptr;
    void* k_ptr;
    void* v_ptr;
    void* o_ptr;

    int batch_size;
    int num_heads;
    int q_seqlen;
    int kv_seqlen;
    int headdim;

    int q_batch_stride;
    int k_batch_stride;
    int v_batch_stride;
    int o_batch_stride;

    int q_head_stride;
    int k_head_stride;
    int v_head_stride;
    int o_head_stride;

    int q_seqlen_stride;
    int k_seqlen_stride;
    int v_seqlen_stride;
    int o_seqlen_stride;

    float softmax_scale;
};

template <typename T, int kHeaddim>
void run_flash_attention(FlashAttentionParams params);

#define CUDA_ERROR_CHECK(condition)                                            \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      printf("CUDA_CHECK error in line %d of file %s \
              : %s \n",                                                        \
             __LINE__, __FILE__, cudaGetErrorString(error));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)