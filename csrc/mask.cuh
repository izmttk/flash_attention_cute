#pragma once
#include <cute/tensor.hpp>
#include <cfloat>

namespace flash_attention {

template<class TiledMMA, class TVCrd, class MNCrd>
CUTE_HOST_DEVICE constexpr auto tv2crd_C(
    TiledMMA const& tiled_mma,
    TVCrd const& tv_crd,
    MNCrd const& mn_crd = cute::Coord<cute::_0, cute::_0>{}
) {
    using namespace cute;
    using AtomShapeMNK = typename TiledMMA::AtomShape_MNK;
    auto ref_C = make_layout(make_shape(
        size<1>(tiled_mma.get_thr_layout_vmnk()) * get<0>(AtomShapeMNK{}),
        size<2>(tiled_mma.get_thr_layout_vmnk()) * get<1>(AtomShapeMNK{})
    ));
    auto layoutC_TV = tiled_mma.thrfrg_C(ref_C);
    auto idx = layoutC_TV(tv_crd);
    auto inner_crd = ref_C.get_flat_coord(idx);
    auto crd = make_coord(
        get<0>(inner_crd) + get<0>(mn_crd) * shape<0>(ref_C),
        get<1>(inner_crd) + get<1>(mn_crd) * shape<1>(ref_C)
    );
    return crd;
}


template<int kBlockM, int kBlockN>
struct Mask {
    const int seqlen_q;
    const int seqlen_k;

    __forceinline__ __device__ Mask(int seqlen_q, int seqlen_k) : seqlen_q(seqlen_q), seqlen_k(seqlen_k) {}

    __forceinline__ __device__ bool is_coord_masked(int m, int n) {
        return seqlen_q - m > seqlen_k - n;
    }

    __forceinline__ __device__ bool is_coord_OOB(int m, int n) {
        return m >= seqlen_q || n >= seqlen_k;
    }

    __forceinline__ __device__ bool is_block_masked(int block_m, int block_n) {
        using namespace cute;
        const int n_left_padding = seqlen_k - seqlen_q; // 开始 Causal Mask 前 padding 列的长度
        const int n_causal_len = (block_m + 1) * kBlockM; // 当前块最长不被 causal mask 的列数，也就是最后一行所在的行数
        // n_block_max 是最大需要计算的块的数量，考虑被 causal mask 完全覆盖的块不需要计算
        const int n_block_max = ceil_div(n_left_padding + n_causal_len, kBlockN); // n_left_padding + n_causal_len 当前块最多能有多少列
        return block_n >= n_block_max;
    }

    template<bool OOB_Mask=true, bool CausalMask=true, typename TiledMMA, typename Tensor>
    __forceinline__ __device__ void apply(TiledMMA tiled_mma, Tensor &tensor, int block_m, int block_n, int tid) {
        // tensor 形状为 (MMA, MMA_M, MMA_N)
        using namespace cute;
        CUTE_UNROLL
        for (int v = 0; v < size<0>(tensor); v++) {
            CUTE_UNROLL
            for (int m = 0; m < size<1>(tensor); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<2>(tensor); n++) {
                    auto crd = tv2crd_C(tiled_mma, make_coord(tid, v), make_coord(m, n));
                    const int real_m = block_m * kBlockM + get<0>(crd);
                    const int real_n = block_n * kBlockN + get<1>(crd);
                    if constexpr (CausalMask) {
                        // if (thread0()) {
                        //     printf("v, %d, m, %d, n, %d, real_m: %d, real_n: %d, seqlen_q: %d, seqlen_k: %d, mask: %d\n", 
                        //         v, m, n,
                        //         real_m, real_n,
                        //         seqlen_q, seqlen_k,
                        //         seqlen_q - real_m > seqlen_k - real_n);
                        // }
                        if (is_coord_masked(real_m, real_n)) {
                            tensor(v, m, n) = -FLT_MAX;
                        }
                    }
                    if constexpr (OOB_Mask) {
                        if (is_coord_OOB(real_m, real_n)) {
                            tensor(v, m, n) = -FLT_MAX;
                        }
                    }
                }
            }
        }
    }
};

} // namespace flash_attention
