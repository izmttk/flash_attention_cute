#include <torch/extension.h>
#include <torch/types.h>
#include <cutlass/numeric_types.h>
#include "flash_attention.h"

namespace flash_attention {

template <typename T>
struct ConstantType {
    using type = T;
};

template <typename T, T val>
struct ConstantValue {
    static constexpr T value = val;
};

template <class Callback>
void headdim_dispatch(int headdim, Callback func) {
    if (headdim <= 128) {
        constexpr int kHeaddim = 128;
        func(ConstantValue<int, kHeaddim>{});
    }
}

template <class Callback>
void dtype_dispatch(torch::ScalarType dtype, Callback func) {
    if (dtype == torch::kHalf) {
        func(ConstantType<cutlass::half_t>{});
    }
}

torch::Tensor flash_attention_fwd(
    torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, float softmax_scale) {

    // Check input shape
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "q, k, v must have the same batch size");
    TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1),
                "q, k, v must have the same number of heads");
    TORCH_CHECK(k.size(2) == v.size(2),
                "k, v must have the same sequence length");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3),
                "q, k, v must have the same hidden dimension");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 && q.size(3) > 0,
                "q, k, v must have at least one element");

    // Check input data type
    TORCH_CHECK(q.dtype() == k.dtype() && q.dtype() == v.dtype(),
                "q, k, v must have the same data type");
    TORCH_CHECK(q.dtype() == torch::kHalf,
                "q, k, v only support fp16 data type");

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
    int64_t head = q.size(1);
    int64_t q_seqlen = q.size(2);
    int64_t kv_seqlen = k.size(2);
    int64_t headdim = q.size(3);

    auto o = torch::empty({bs, head, q_seqlen, headdim}, q.options());

    softmax_scale *= M_LOG2E;
    FlashAttentionParams params = {
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        o.data_ptr(),
        bs,
        head,
        q_seqlen,
        kv_seqlen,
        headdim,
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

    dtype_dispatch(q.scalar_type(), [&](auto dtype) {
        headdim_dispatch(headdim, [&](auto headdim) {
            using kDtype = decltype(dtype)::type;
            constexpr auto kHeaddim = decltype(headdim)::value;
            run_flash_attention<kDtype, kHeaddim>(params);
        });
    });
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("flash_attention_fwd", &flash_attention::flash_attention_fwd,
          "Flash attention v2 implement in cutlass cute");
}

} // namespace flash_attention
