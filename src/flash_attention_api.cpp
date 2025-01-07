#include <torch/extension.h>
#include <torch/types.h>

#include <vector>
#include "flash_attention.cuh"

torch::Tensor flash_attention_v2_cute(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, float softmax_scale) {

    int bs = q.size(0);
    int head = q.size(1);
    int q_seqlen = q.size(2);
    int kv_seqlen = k.size(2);
    int headdim = q.size(3);

    auto o = torch::empty({bs, head, q_seqlen, headdim}, q.options());

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

    AT_DISPATCH_SWITCH(q.scalar_type(), "flash_attention_v2_cute",
        AT_DISPATCH_CASE(c10::kHalf, [&] {
            launch_flash_attention_v2(params);
        })
    );
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("flash_attention_v2_cute", &flash_attention_v2_cute,
          "Flash attention v2 implement in cutlass cute");
}