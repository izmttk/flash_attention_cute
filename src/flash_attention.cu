#include <cuda.h>
#include <cuda_runtime.h>

#include "flash_attention.cuh"
#include <cute/tensor.hpp>
#define WARP_SIZE 32

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(cute::Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return cute::make_tensor(cute::make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// #include <cuda_fp16.h>
// template <typename STensor>
// __forceinline__ __device__ auto convert_type_f322f16(STensor const &acc_fp32) {
//     using namespace cute;
//     auto acc_fp16 = make_tensor_like<half_t>(acc_fp32);
//     {
//         auto acc_fp32x2 = recast<float2>(acc_fp32);
//         auto acc_fp16x2 = recast<half2>(acc_fp16);
//         CUTE_UNROLL
//         for (int i = 0; i < size(acc_fp32x2); ++i) { acc_fp16x2(i) = __float22half2_rn(acc_fp32x2(i)); }
//     }
//     return acc_fp16;
// }

template <bool IsClearOOB = true, class TiledCopy, class IdentityTensor, class MaxMN, class  STensor, class DTensor>
__forceinline__ __device__ void copy(
    TiledCopy const& tiled_copy,
    IdentityTensor const& identity, // (CPY, CPY_M, CPY_N)
    MaxMN const& max_mn,            // (MAX_M, MAX_N)
    STensor const& src,             // (CPY, CPY_M, CPY_N)
    DTensor& dst                    // (CPY, CPY_M, CPY_N)
) {
    using namespace cute;
    CUTE_UNROLL
    for (int m = 0; m < size<1>(src); m++) {
        if (get<0>(identity(0, m, 0)) < get<0>(max_mn)) {
            CUTE_UNROLL
            for (int n = 0; n < size<2>(src); n++) {
                if (get<1>(identity(0, m, n)) < get<1>(max_mn)) {
                    copy(tiled_copy, src(_, m, n), dst(_, m, n));
                } else if (IsClearOOB) {
                    clear(dst(_, m, n));
                }
            }
        } else if (IsClearOOB) {
            clear(dst(_, m, _));
        }
    }
}

template<class MMA_Atom>
CUTE_HOST_DEVICE constexpr auto get_thr_C_shape(MMA_Atom const& mma_atom) {
    using namespace cute;
    using AtomLayoutC_TV = typename MMA_Atom::LayoutC_TV;
    using AtomShapeMNK = typename MMA_Atom::Shape_MNK;
    auto V = layout<1>(AtomLayoutC_TV{});
    auto C = select<0, 1>(AtomShapeMNK{});
    auto R = complement(V, C);
    auto val_shape = shape(R);
    auto thr_shape = make_shape(
        ceil_div(get<0>(C), get<0>(val_shape)),
        ceil_div(get<1>(C), get<1>(val_shape))
    );
    return thr_shape;
}

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
//  query: [batch_size, num_heads,  q_seqlen, qk_headdim]
//    key: [batch_size, num_heads, kv_seqlen, qk_headdim]
//  score: [batch_size, num_heads,  q_seqlen,  kv_seqlen]
//  value: [batch_size, num_heads, kv_seqlen,  v_headdim]
// output: [batch_size, num_heads,  q_seqlen,  v_headdim]
// rowmax: [batch_size, num_heads,  q_seqlen]
// rowsum: [batch_size, num_heads,  q_seqlen]

// cuda 分配的内存默认 256 字节对齐，所以不需要担心指针不对齐向量访存大小的问题
// 向量化访存 128bit 是沿着 head dim 方向进行的，所以只需要确保 head_dim 大小是 8 的倍数即可（对于半精度数据而言）
// 同时在内存连续性上，只要求最后一维，即 head dim 连续即可，即 stride 为 1，其他的维度通过参数中的各自 stride 来控制
__global__ void flash_attention_v2(FlashAttentionParams params) {
    using namespace cute;

    constexpr int kBlockM = 64; // Br
    constexpr int kBlockN = 64; // Bc
    constexpr int kHeadDim = 128;
    constexpr int kNWarps = 4;
    constexpr int kNThreads = kNWarps * WARP_SIZE;
    
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int q_tile = blockIdx.x;

    // Tensor mQ = make_tensor(
    //     make_gmem_ptr(reinterpret_cast<half_t*>(params.q_ptr)),
    //     make_shape(params.batch_size, params.num_heads, params.q_seqlen, params.headdim),
    //     make_stride(params.q_batch_stride, params.q_head_stride, params.q_seqlen_stride, Int<1>{})
    // );
    // Tensor mK = make_tensor(
    //     make_gmem_ptr(reinterpret_cast<half_t*>(params.k_ptr)),
    //     make_shape(params.batch_size, params.num_heads, params.kv_seqlen, params.headdim),
    //     make_stride(params.k_batch_stride, params.k_head_stride, params.k_seqlen_stride, Int<1>{})
    // );
    // Tensor mV = make_tensor(
    //     make_gmem_ptr(reinterpret_cast<half_t*>(params.v_ptr)),
    //     make_shape(params.batch_size, params.num_heads, params.kv_seqlen, params.headdim),
    //     make_stride(params.v_batch_stride, params.v_head_stride, params.v_seqlen_stride, Int<1>{})
    // );
    // Tensor mO = make_tensor(
    //     make_gmem_ptr(reinterpret_cast<half_t*>(params.o_ptr)),
    //     make_shape(params.batch_size, params.num_heads, params.q_seqlen, params.headdim),
    //     make_stride(params.o_batch_stride, params.o_head_stride, params.o_seqlen_stride, Int<1>{})
    // );

    // // 1. gmem, smem Tensor 的定义，需要：
    // // gmem global Tensor
    // // gmem tiled Tensor
    // // smem Tensor

    // // Q: (q_seqlen, qk_headdim)
    // Tensor Q = mQ(batch, head, _, _);
    // // K: (kv_seqlen, qk_headdim)
    // Tensor K = mK(batch, head, _, _);
    // // V: (kv_seqlen, v_headdim)
    // Tensor V = mV(batch, head, _, _);
    // // O: (q_seqlen, v_headdim)
    // Tensor O = mO(batch, head, _, _);

    half_t *q_offset = reinterpret_cast<half_t*>(params.q_ptr) + batch * params.q_batch_stride + head * params.q_head_stride;
    half_t *k_offset = reinterpret_cast<half_t*>(params.k_ptr) + batch * params.k_batch_stride + head * params.k_head_stride;
    half_t *v_offset = reinterpret_cast<half_t*>(params.v_ptr) + batch * params.v_batch_stride + head * params.v_head_stride;
    half_t *o_offset = reinterpret_cast<half_t*>(params.o_ptr) + batch * params.o_batch_stride + head * params.o_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(q_offset), make_layout(make_shape(params.q_seqlen, params.headdim), make_stride(params.q_seqlen_stride, Int<1>{})));
    Tensor mK = make_tensor(make_gmem_ptr(k_offset), make_layout(make_shape(params.kv_seqlen, params.headdim), make_stride(params.k_seqlen_stride, Int<1>{})));
    Tensor mV = make_tensor(make_gmem_ptr(v_offset), make_layout(make_shape(params.kv_seqlen, params.headdim), make_stride(params.v_seqlen_stride, Int<1>{})));
    Tensor mO = make_tensor(make_gmem_ptr(o_offset), make_layout(make_shape(params.q_seqlen, params.headdim), make_stride(params.o_seqlen_stride, Int<1>{})));

    // gQ 是固定的 Q 的第 q_tile 个分块
    // gQ: (kBlockM, qk_headdim, num_tile_qk_headdim) = (64, qk_headdim, 1)
    Tensor gQ = local_tile(mQ, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(q_tile, _));
    // gK, gV 会在 tile 迭代时更新，这里预取第一个 Q、K tile
    // gK: (kBlockN, qk_headdim, num_tile_qk_headdim) = (64, qk_headdim, 1)
    Tensor gK = local_tile(mK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
    // gV: (kBlockN, v_headdim, num_tile_v_headdim) = (64, v_headdim, 1)
    Tensor gV = local_tile(mV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
    // gO: (kBlockM, v_headdim, num_tile_v_headdim) = (64, v_headdim, 1)
    Tensor gO = local_tile(mO, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(q_tile, _));

    // TODO: 需要 Swizzle 优化
    using SmemLayoutQ = decltype(make_layout(make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})));
    using SmemLayoutK = decltype(make_layout(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})));
    using SmemLayoutV = decltype(make_layout(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})));
    using SmemLayoutVt = decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_stride(Int<1>{}, Int<kHeadDim>{})));
    using SmemLayoutO = decltype(make_layout(make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_stride(Int<kHeadDim>{}, Int<1>{})));

    __shared__ half_t q_smem[cosize(SmemLayoutQ{})];
    __shared__ half_t k_smem[cosize(SmemLayoutK{})];
    __shared__ half_t v_smem[cosize(SmemLayoutV{})];

    Tensor sQ = make_tensor(make_smem_ptr(q_smem), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(k_smem), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(v_smem), SmemLayoutV{});
    Tensor sVt = make_tensor(make_smem_ptr(v_smem), SmemLayoutVt{});
    // 复用 sQ 的存储
    Tensor sO = make_tensor(sQ.data(), SmemLayoutO{});

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    // 2. gmem to smem 的定义，需要：
    // 单线程负责的 gmem tiled Tensor
    // 单线程负责的 smem tiled Tensor
    constexpr int kBlockKSmem = 64;
    constexpr int kGmemElemsPerLoad = sizeof(uint128_t) / sizeof(half_t);
    constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    // 一个 TiledCopy 处理 16x64 的矩阵
    TiledCopy gmem_tiled_copy_QKV = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>{}, // thread layout
        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{} // value layout
    );
    ThrCopy gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(threadIdx.x);
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    // 3. smem to reg 的定义，需要：
    // 单线程负责的 smem tiled Tensor
    // 单线程负责的 reg Tensor
    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
    using MMA_Atom_Shape = MMA_Atom_Arch::Shape_MNK;
    using MMA_Atom_M = decltype(get<0>(MMA_Atom_Shape{}));
    using MMA_Atom_N = decltype(get<1>(MMA_Atom_Shape{}));
    using MMA_Atom_K = decltype(get<2>(MMA_Atom_Shape{}));

    TiledMMA tiled_mma = make_tiled_mma(
        MMA_Atom_Arch{}, // 32 线程处理 m16n8k16 mma
        Layout<Shape<Int<kNWarps>, _1, _1>>{}, // 重复线程，这意味着处理这个 mma 需要 4x1x1x32 = 128 线程
        // Cutlass 3.4 之后，第 3 个参数变为 PermutationsMNK (之前是 ValLayoutMNK)
        // PermutationsMNK 是 3 个分别应用于 MNK 维度的 Tiler
        // Tile<m, n, k> 等价于 Tile<Layout<Shape<m>, Stride<1>>, Layout<Shape<n>, Stride<1>>, Layout<Shape<k>, Stride<1>>>
        // 直接扩展 mnk 大小会使 mma atom 在每个线程上重复计算值
        // 如果自定义了一个维度的 layout，那么这个维度会按照这个 layout 的 1D 坐标到 index 的映射来置换该维度
        // 也就是说，这个 layout 是从旧坐标到新坐标的映射
        Tile<Int<MMA_Atom_M{} * kNWarps>, Int<MMA_Atom_N{} * 2>, MMA_Atom_K>{} // 这意味着一个线程重复 kNWarps * 2 * 1 = 8 遍 mma atom
    );
    // 总的来说，一个 TiledMMA 用 128 个线程处理 64x16x16 的矩阵乘，每个 warp 处理 16x16x16 的矩阵乘（一个 mma atom），一共 4 个 warp

    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    // 注意 MMAValLayout/Permutations 不会影响 partition 的划分，因为 partition MMA_Atom 和 MMAThrLayout 进行分区的
    // patititon 划分出来一个线程处理的数据块 (MMA, MMA_M, MMA_K)，其中 MMA 是一个 mma atom 在该线程上消耗的数据
    // https://github.com/NVIDIA/cutlass/discussions/1345
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    // 划分寄存器 C 的时候不需要一个已经定义好的 Tensor，因为 C 不会向其他内存写入
    Tensor rAccS = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));

    Tensor tOrVt = thr_mma.partition_fragment_B(sVt);
    Tensor rAccO = partition_fragment_C(tiled_mma, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));

    // 定义 smem 到 reg 的 tild copy 对象
    // 按照 tiled_mma 划分，所以划分出来的块对应一个线程中 mma 消耗的数据
    TiledCopy smem_tiled_copy_Q = make_tiled_copy_A(
        Copy_Atom<DefaultCopy, half_t>{},
        tiled_mma
    );
    ThrCopy smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(threadIdx.x);
    // (CPY, CPY_M, CPY_K) = ((1, (2, 2, 2)), 1, 8)
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    TiledCopy smem_tiled_copy_K = make_tiled_copy_B(
        Copy_Atom<DefaultCopy, half_t>{},
        tiled_mma
    );
    ThrCopy smem_thr_copy_K = smem_tiled_copy_K.get_slice(threadIdx.x);
    // 根据 TiledMMA 划分不同的 TiledCopy，显然这里就会考虑 MMAValLayout/Permutations
    // (CPY, CPY_N, CPY_K) = ((1, (2, 2, 2)), 4, 8)
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    TiledCopy smem_tiled_copy_V = make_tiled_copy_B(
        Copy_Atom<DefaultCopy, half_t>{},
        tiled_mma
    );
    ThrCopy smem_thr_copy_V = smem_tiled_copy_V.get_slice(threadIdx.x);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    copy(gmem_tiled_copy_QKV, tQcQ, make_tuple(params.q_seqlen - q_tile * kBlockM, params.headdim), tQgQ, tQsQ);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    copy(gmem_tiled_copy_QKV, tKVcKV, make_tuple(params.kv_seqlen - 0 * kBlockN, params.headdim), tKgK, tKsK);
    cp_async_fence();

    clear(rAccO);

    auto thr_C_row = size<1>(get_thr_C_shape(MMA_Atom_Arch{}));
    Tensor scores_max = make_tensor<float>(make_shape(size<1>(rAccS) * thr_C_row));
    // 这里 x2 是因为 SM80_16x8x16_F32F16F16F32_TN 的 LayoutC_TV 是 ((4, 8), (2, 2)):((32, 1), (16, 8))
    // 意味着一共使用 32 个线程，每个线程处理 (2, 2) 个值，也就是每个线程有 2 行需要计算 max 和 sum
    Tensor scores_sum = make_tensor<float>(make_shape(size<1>(rAccS) * thr_C_row));

    // 记得初始化这里
    fill(scores_max, -FLT_MAX);
    fill(scores_sum, 0.0f);

    const int n_blocks = ceil_div(params.kv_seqlen, kBlockN);
    for (int block = 0; block < n_blocks; block++) {
        clear(rAccS);


        // 等待 K gmem 到 smem 的拷贝完成
        cp_async_wait<0>();
        __syncthreads();

        // 复制接下来将要用到的 V 到 smem
        gV = local_tile(mV, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(block, _));
        tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));

        copy(gmem_tiled_copy_QKV, tKVcKV, make_tuple(params.kv_seqlen - block * kBlockN, params.headdim), tVgV, tVsV);

        cp_async_fence();

        // S = Q * K^T
        copy(smem_tiled_copy_Q, tSsQ(_, _, 0), tSrQ(_, _, 0));
        copy(smem_tiled_copy_K, tSsK(_, _, 0), tSrK(_, _, 0));

        CUTE_UNROLL
        for (int sub_block = 0; sub_block < size<2>(tSrQ); sub_block ++) {
            if (sub_block < size<2>(tSrQ) - 1) {
                copy(smem_tiled_copy_Q, tSsQ(_, _, sub_block + 1), tSrQ(_, _, sub_block + 1));
                copy(smem_tiled_copy_K, tSsK(_, _, sub_block + 1), tSrK(_, _, sub_block + 1));
            }
            gemm(tiled_mma, tSrQ(_, _, sub_block), tSrK(_, _, sub_block), rAccS);
        }

        // 等待 V gmem 到 smem 的拷贝完成
        cp_async_wait<0>();
        __syncthreads();

        // 复制下一个 block 的 K 到 smem
        if (block != n_blocks - 1) {
            // 更新 gK tKgK 指向下一个 K block，并进行复制
            gK = local_tile(mK, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(block + 1, _));
            tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));

            copy(gmem_tiled_copy_QKV, tKVcKV, make_tuple(params.kv_seqlen - (block + 1) * kBlockN, params.headdim), tKgK, tKsK);
            cp_async_fence();
        }

        // mask rAccS，越界的值置为 -FLT_MAX
        // 这种使用 layout 计算的方法比之前纯手写要慢 2% 左右
        CUTE_UNROLL
        for (int v = 0; v < size<0>(rAccS); v++) {
            CUTE_UNROLL
            for (int m = 0; m < size<1>(rAccS); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<2>(rAccS); n++) {
                    auto crd = tv2crd_C(tiled_mma, make_coord(threadIdx.x, v), make_coord(m, n));
                    const int real_m = q_tile * kBlockM + get<0>(crd);
                    const int real_n = block * kBlockN + get<1>(crd);
                    if (real_m >= params.q_seqlen || real_n >= params.kv_seqlen) {
                        rAccS(v, m, n) = -FLT_MAX;
                    }
                }
            }
        }

        // softmax
        // rAccS: (MMA, MMA_M, MMA_N) = ((2, 2), 1, 8)
        // scores: (2, 16)
        // 就是把 tSrS 重新视为二维矩阵
        // 有办法直接获得一个线程处理矩阵的行数吗？现在是根据 SM80_16x8x16_F32F16F16F32_TN 的 LayoutC_TV 看出来一个线程 2 行
        //
        // rAccS layout from:                                   to:
        // V0 V2 V0 V2 V0 V2 V0 V2 V0 V2 V0 V2 V0 V2 V0 V2      V0 V1 V0 V1 V0 V1 V0 V1 V0 V1 V0 V1 V0 V1 V0 V1
        // V1 V3 V1 V3 V1 V3 V1 V3 V1 V3 V1 V3 V1 V3 V1 V3      V2 V3 V2 V3 V2 V3 V2 V3 V2 V3 V2 V3 V2 V3 V2 V3
        //
        // 一个 mma atom 消耗 V0 V1 V2 V3 4 个值，其中 V0 V1 为一行，V2 V3 为一行
        // 但是在 rAccS 的 第一个维度上，布局是 col-major 的，我们把它转换为 row-major，更符合直觉方便后续按行遍历，也就是：
        // from col-major: V0 V3    to row-major: V0 V1
        //                 V1 V2                  V2 V3
        // 注意这里仅重新将值按照新布局解释，即更改了遍历顺序，并不涉及数据移动
        auto flat2d_layoutS = zipped_divide(layout(rAccS), Tile<Layout<Shape<_2>, Stride<_2>>>{});
        Tensor rS_fp32 = make_tensor(rAccS.data(), flat2d_layoutS);
        auto flat2d_layoutO = zipped_divide(layout(rAccO), Tile<Layout<Shape<_2>, Stride<_2>>>{});
        Tensor rO_fp32 = make_tensor(rAccO.data(), flat2d_layoutO);

        CUTE_UNROLL
        for (int row = 0; row < size<0>(rS_fp32); row++) {
            float& prev_rowmax = scores_max(row);
            float new_rowmax = prev_rowmax;
            CUTE_UNROLL
            for (int col = 0; col < size<1>(rS_fp32); col++) {
                new_rowmax = max(new_rowmax, rS_fp32(row, col));
            }
            constexpr int num_threads_per_col = 4;
            CUTE_UNROLL
            for (int offset = num_threads_per_col / 2; offset > 0; offset /= 2) {
                new_rowmax = max(new_rowmax, __shfl_xor_sync(0xffffffff, new_rowmax, offset));
            }
            // 因为求一行的最大值的时候，S没有进行缩放，所以这里要缩放一下
            float scores_scale = expf((prev_rowmax - new_rowmax) * params.softmax_scale);
            prev_rowmax = new_rowmax;

            // 缩放 O
            CUTE_UNROLL
            for (int col = 0; col < size<1>(rO_fp32); col++) {
                rO_fp32(row, col) *= scores_scale;
            }

            CUTE_UNROLL
            for (int col = 0; col < size<1>(rS_fp32); col++) {
                rS_fp32(row, col) = rS_fp32(row, col) == -FLT_MAX ? 0 : expf((rS_fp32(row, col) - new_rowmax) * params.softmax_scale);
            }

            float& prev_rowsum = scores_sum(row);
            float new_rowsum = 0.0f;
            CUTE_UNROLL
            for (int col = 0; col < size<1>(rS_fp32); col++) {
                new_rowsum += rS_fp32(row, col);
            }
            CUTE_UNROLL
            for (int offset = num_threads_per_col / 2; offset > 0; offset /= 2) {
                new_rowsum += __shfl_xor_sync(0xffffffff, new_rowsum, offset);
            }
            prev_rowsum = scores_scale * prev_rowsum + new_rowsum;
        }

        // O = S * V
        // 由于 S 计算出来是 fp32，参与下一个 mma 计算前，先转换为 fp16
        // 这里很奇怪，当你去 print_tensor 的时候，会发现 rP 转换结果是错误的，但是最终计算出来的结果是正确的
        // 暂时没想到原因
        Tensor rP = convert_type<half_t>(rS_fp32);
        // Tensor rP = convert_type_f322f16(rS_fp32);

        // layout C:((2, 2), MMA_M, MMA_N) -> layout A:((2, 2, 2), MMA_M, MMA_N / 2)
        // 即 ((2, 2), 1, 8) -> ((2, 2, 2), 1, 4)
        // ((2, 2), 1, 8)
        auto mmaC_layoutP = layout(rAccS);
        // ((2, 2), 1, (2, 4))
        auto mmaC_split_layoutP = logical_divide(mmaC_layoutP, make_shape(_, _, Int<2>{}));
        // ((2, 2, 2), 1, 4)
        auto mmaA_layoutP = make_layout(
            append(get<0>(mmaC_split_layoutP), get<0>(get<2>(mmaC_split_layoutP))),
            get<1>(mmaC_split_layoutP),
            get<1>(get<2>(mmaC_split_layoutP))
        );
        Tensor tOrP = make_tensor(rP.data(), mmaA_layoutP);
        // tOrVt 用 mma atom 进行 partition
        // tOsVt 用 copy atom 进行 partition
        // 这两个 atom 的 value layout 可能不同，用 retile 按照 copy atom 的划分模式重新划分
        // retile 会假设参数 tensor 第一个元素是 ValLayout
        // 不过这里 tOrVt_view 和 tOrVt 的 1D 坐标到 index 的映射是一样的，所以应该可以不需要 retile
        // Tensor tOrVt_view = smem_thr_copy_V.retile_D(tOrVt);
        // 注意 tOrP 已经全部在寄存器中了，所以只 copy V 从 smem 到 reg 即可
        copy(smem_tiled_copy_V, tOsVt(_, _, 0), tOrVt(_, _, 0));
        CUTE_UNROLL
        for (int sub_block = 0; sub_block < size<2>(tOrP); sub_block++) {
            if (sub_block < size<2>(tOrP) - 1) {
                copy(smem_tiled_copy_V, tOsVt(_, _, sub_block + 1), tOrVt(_, _, sub_block + 1));
            }
            gemm(tiled_mma, tOrP(_, _, sub_block), tOrVt(_, _, sub_block), rAccO);
        }
    }
    auto flat2d_layoutO = zipped_divide(layout(rAccO), Tile<Layout<Shape<_2>, Stride<_2>>>{});
    Tensor rO_fp32 = make_tensor(rAccO.data(), flat2d_layoutO);
    CUTE_UNROLL
    for (int row = 0; row < size<0>(rO_fp32); row++) {
        float scale = scores_sum(row) == 0 ? 1 : 1 / scores_sum(row);
        CUTE_UNROLL
        for (int col = 0; col < size<1>(rO_fp32); col++) {
            rO_fp32(row, col) *= scale;
        }
    }
    // write back
    Tensor rO = convert_type<half_t>(rO_fp32);
    // Tensor rO = convert_type_f322f16(rO_fp32);

    Tensor tAccOrO = make_tensor(rO.data(), layout(rAccO));
    // 先复制到 smem，再写回 gmem
    // 因为寄存器是分散再各个线程上的，先写回 smem 可以使用向量化指令，实现更高带宽？？？
    TiledCopy smem_tiled_copy_O = make_tiled_copy_C(
        Copy_Atom<DefaultCopy, half_t>{},
        tiled_mma
    );
    ThrCopy smem_thr_copy_O = smem_tiled_copy_O.get_slice(threadIdx.x);
    // Tensor tOrO_view = smem_thr_copy_O.retile_S(tOrO);
    Tensor tAccOsO = smem_thr_copy_O.partition_D(sO);

    copy(smem_tiled_copy_O, tAccOrO, tAccOsO);

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    TiledCopy gmem_tiled_copy_O = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
        Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>{}, // thread layout
        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{} // value layout
    );
    ThrCopy gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(threadIdx.x);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));
    __syncthreads();

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);

    // 最后复制 O 到 gmem 时，要关闭 IsClearOOB，因为 mO 的尺寸不一定等于 gO，会有值被覆盖为 0 的情况
    copy<false>(gmem_tiled_copy_O, tOcO, make_tuple(params.q_seqlen - q_tile * kBlockM, params.headdim), tOsO, tOgO);
}


void launch_flash_attention_v2(FlashAttentionParams params) {
    constexpr int kBlockM = 64; // Br
    constexpr int kNWarps = 4;
    constexpr int kNThreads = kNWarps * WARP_SIZE;
    int num_q_tiles = cute::ceil_div(params.q_seqlen, kBlockM);
    dim3 grid(num_q_tiles, params.num_heads, params.batch_size);
    dim3 block(kNThreads);
    flash_attention_v2<<<grid, block>>>(params);
    CUDA_ERROR_CHECK(cudaGetLastError());
}