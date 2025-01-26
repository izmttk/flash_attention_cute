#include <torch/extension.h>
#include <torch/types.h>
#include <cutlass/numeric_types.h>
#include "flash_attention.h"
#include "kernel_dispatcher.h"

namespace flash_attention {

torch::Tensor flash_attention_fwd(
    torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, float softmax_scale, bool causal) {

    // Check input shape
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "q, k, v must have the same batch size");
    TORCH_CHECK(k.size(1) == v.size(1),
                "k, v must have the same number of heads");
    TORCH_CHECK(k.size(2) == v.size(2),
                "k, v must have the same sequence length");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3),
                "q, k, v must have the same hidden dimension");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 && q.size(3) > 0,
                "q, k, v must have at least one element");
    TORCH_CHECK(q.size(1) >= k.size(1),
                "number of heads in q must be greater or equal to number of heads in k and v");
    TORCH_CHECK(q.size(1) % k.size(1) == 0,
                "number of heads in q must be multiple of number of heads in k and v");

    // Check input data type
    TORCH_CHECK(q.dtype() == k.dtype() && q.dtype() == v.dtype(),
                "q, k, v must have the same data type");
    TORCH_CHECK(q.dtype() == torch::kHalf || q.dtype() == torch::kBFloat16,
                "q, k, v only support fp16 or bf16 data type");

    // Check input memory contiguity
    TORCH_CHECK(q.stride(3) == 1,
                "q must be contiguous in the last dimension");
    TORCH_CHECK(k.stride(3) == 1,
                "k must be contiguous in the last dimension");
    TORCH_CHECK(v.stride(3) == 1,
                "v must be contiguous in the last dimension");

    // Check kernel constraints
    TORCH_CHECK(q.size(3) % 8 == 0,
                "hidden dimension must be multiple of 8");
    TORCH_CHECK(q.size(3) <= 128,
                "only support hidden dimension <= 128");
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "q, k, v must be on CUDA device");

    int64_t bs = q.size(0);
    int64_t head_q = q.size(1);
    int64_t head_kv = k.size(1);
    int64_t seqlen_q = q.size(2);
    int64_t seqlen_kv = k.size(2);
    int64_t headdim = q.size(3);
    int64_t head_q_per_group = head_q / head_kv;

    bool is_pack_head_q = seqlen_q == 1; // 或许应该判断 seqlen_q 是否小于 Tensor Core 的 M/2 尺寸
    // 这种方法目的是在 seqlen_q 很短时，充分利用 Tensor Core
    // 对于 seqlen_q == 1，理论上 head_q_per_group 每提高一倍，计算时间减少一倍
    // 直到 head_q_per_group == tiled_mma_m（在这里是 64），此时一个 block 调用的所有 Tensor Core 的输入被填满
    // 然后 head_q_per_group 再翻倍也会减少时间，但是不会是成倍减少，这里的提升来自于充分利用一个 block 的 reg
    // 直到 head_q_per_group == kBlockM，此时等同于 seqlen_q == kBlockM 的情况
    if (is_pack_head_q) {
        head_q = head_kv;
        seqlen_q = seqlen_q * head_q_per_group;
        causal = false;
        q = q.reshape({bs, head_q, seqlen_q, headdim});
    }

    auto o = torch::empty({bs, head_q, seqlen_q, headdim}, q.options());

    softmax_scale *= M_LOG2E;
    FlashAttentionParams params = {
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        o.data_ptr(),
        bs,
        head_q,
        head_kv,
        seqlen_q,
        seqlen_kv,
        headdim,
        is_pack_head_q ? 1 : head_q_per_group, // 注意 pack_head_q 时，q 和 kv 注意力头数量相同，实际上执行的是 MHA
        q.stride(0),
        k.stride(0),
        v.stride(0),
        o.stride(0),
        q.stride(1),
        k.stride(1),
        v.stride(1),
        o.stride(1),
        q.stride(2),
        k.stride(2),
        v.stride(2),
        o.stride(2),
        softmax_scale
    };

    type_dispatcher(q.scalar_type(), [&](auto type_wrapper) {
        headdim_dispatcher(headdim, [&](auto headdim_wrapper) {
            bool_dispatcher(causal, [&](auto causal_wrapper) {
                using kDtype = typename decltype(type_wrapper)::type;
                constexpr auto kHeaddim = decltype(headdim_wrapper)::value;
                constexpr auto IsCausal = decltype(causal_wrapper)::value;
                run_flash_attention<kDtype, kHeaddim, IsCausal>(params);
            });
        });
    });

    if (is_pack_head_q) {
        head_q = head_kv * head_q_per_group;
        seqlen_q = seqlen_q / head_q_per_group;
        o = o.reshape({bs, head_q, seqlen_q, headdim});
        q = q.reshape({bs, head_q, seqlen_q, headdim});
    }
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("flash_attention_fwd", &flash_attention::flash_attention_fwd,
          "Flash attention v2 implement in cutlass cute");
}

} // namespace flash_attention
