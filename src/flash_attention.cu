#include <cuda.h>
#include <cuda_runtime.h>

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

struct AttentionFeatureDim {
    int batch_size;
    int num_heads;
    int q_seqlen;
    int kv_seqlen;
    int headdim;
};

//  query: [batch_size, num_heads,  q_seqlen, qk_headdim]
//    key: [batch_size, num_heads, kv_seqlen, qk_headdim]
//  score: [batch_size, num_heads,  q_seqlen,  kv_seqlen]
//  value: [batch_size, num_heads, kv_seqlen,  v_headdim]
// output: [batch_size, num_heads,  q_seqlen,  v_headdim]
// rowmax: [batch_size, num_heads,  q_seqlen]
// rowsum: [batch_size, num_heads,  q_seqlen]

__global__ void flash_attention_v2(
    cute::half_t* query,
    cute::half_t* key,
    cute::half_t* value,
    cute::half_t* output,
    AttentionFeatureDim dim,
    float softmax_scale,
    int head_stride
) {
    using namespace cute;

    constexpr int kBlockM = 64; // Br
    constexpr int kBlockN = 64; // Bc
    constexpr int kHeadDim = 128;
    constexpr int kNWarps = 4;
    constexpr int kNThreads = kNWarps * WARP_SIZE;
    
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int q_tile = blockIdx.x;

    const int bs_head_offset = batch * head * head_stride;

    // 1. gmem, smem Tensor 的定义，需要：
    // gmem global Tensor
    // gmem tiled Tensor
    // smem Tensor

    // Q: (q_seqlen, qk_headdim)
    Tensor Q = make_tensor(
        make_gmem_ptr(query + bs_head_offset), 
        make_shape(dim.q_seqlen, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, Int<1>{}) // row-major
    );
    // K: (kv_seqlen, qk_headdim)
    Tensor K = make_tensor(
        make_gmem_ptr(key + bs_head_offset), 
        make_shape(dim.kv_seqlen, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, Int<1>{})
    );
    // V: (kv_seqlen, v_headdim)
    Tensor V = make_tensor(
        make_gmem_ptr(value + bs_head_offset), 
        make_shape(dim.kv_seqlen, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, Int<1>{})
    );
    // O: (q_seqlen, v_headdim)
    Tensor O = make_tensor(
        make_gmem_ptr(output + bs_head_offset), 
        make_shape(dim.q_seqlen, Int<kHeadDim>{}),
        make_stride(Int<kHeadDim>{}, Int<1>{})
    );

    // gQ 是固定的 Q 的第 q_tile 个分块
    // gQ: (kBlockM, qk_headdim, num_tile_qk_headdim) = (64, qk_headdim, 1)
    Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(q_tile, _));

    // gK, gV 会在 tile 迭代时更新，这里预取第一个 Q、K tile
    // gK: (kBlockN, qk_headdim, num_tile_qk_headdim) = (64, qk_headdim, 1)
    Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
    // gV: (kBlockN, v_headdim, num_tile_v_headdim) = (64, v_headdim, 1)
    Tensor gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
    // gO: (kBlockM, v_headdim, num_tile_v_headdim) = (64, v_headdim, 1)
    Tensor gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(q_tile, _));

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
    
    copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    cp_async_fence();

    clear(rAccO);
    Tensor scores_max = make_tensor<float>(make_shape(Int<size<1>(rAccS) * 2>{}));
    // 这里 x2 是因为 SM80_16x8x16_F32F16F16F32_TN 的 LayoutC_TV 是 ((4, 8), (2, 2)):((32, 1), (16, 8))
    // 意味着一共使用 32 个线程，每个线程处理 (2, 2) 个值，也就是每个线程有 2 行需要计算 max 和 sum
    Tensor scores_sum = make_tensor<float>(make_shape(Int<size<1>(rAccS) * 2>{}));

    // 记得初始化这里
    fill(scores_max, -FLT_MAX);
    fill(scores_sum, 0.0f);

    const int n_blocks = ceil_div(dim.kv_seqlen, kBlockN);
    for (int block = 0; block < n_blocks; block++) {
        clear(rAccS);


        // 等待 K gmem 到 smem 的拷贝完成
        cp_async_wait<0>();
        __syncthreads();

        // 复制接下来将要用到的 V 到 smem
        gV = local_tile(V, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(block, _));
        tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
        copy(gmem_tiled_copy_QKV, tVgV, tVsV);
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
            gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(block + 1, _));
            tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
            copy(gmem_tiled_copy_QKV, tKgK, tKsK);
            cp_async_fence();
        }

        // softmax
        // tSrS: (MMA, MMA_M, MMA_N) = ((2, 2), 1, 8)
        // scores: (2, 16)
        // 就是把 tSrS 重新视为二维矩阵
        // 有办法直接获得一个线程处理矩阵的行数吗？现在是根据 SM80_16x8x16_F32F16F16F32_TN 的 LayoutC_TV 看出来一个线程 2 行
        Tensor rS_fp32 = make_tensor(rAccS.data(), make_shape(
            Int<get<0>(shape<0>(rAccS)) * shape<1>(rAccS)>{},
            Int<get<1>(shape<0>(rAccS)) * shape<2>(rAccS)>{}
        ));
        Tensor rO_fp32 = make_tensor(rAccO.data(), make_shape(
            Int<get<0>(shape<0>(rAccO)) * shape<1>(rAccO)>{},
            Int<get<1>(shape<0>(rAccO)) * shape<2>(rAccO)>{}
        ));

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
            float scores_scale = expf((prev_rowmax - new_rowmax) * softmax_scale);
            prev_rowmax = new_rowmax;

            // 缩放 O
            CUTE_UNROLL
            for (int col = 0; col < size<1>(rO_fp32); col++) {
                rO_fp32(row, col) *= scores_scale;
            }

            CUTE_UNROLL
            for (int col = 0; col < size<1>(rS_fp32); col++) {
                rS_fp32(row, col) = expf((rS_fp32(row, col) - new_rowmax) * softmax_scale);
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
        Tensor rP = convert_type<half_t>(rS_fp32);
        // layout C:((2, 2), MMA_M, MMA_N) -> layout A:((2, 2, 2), MMA_M, MMA_N / 2)
        // 即 ((2, 2), 1, 8) -> ((2, 2, 2), 1, 4)
        // ((2, 2), 1, 8)
        auto scores_layoutC = layout(rAccS);
        // ((2, 2), 1, (2, 4))
        auto scores_layoutC_N_div_2 = logical_divide(scores_layoutC, make_shape(_, _, Int<2>{}));
        // ((2, 2, 2), 1, 4)
        auto scores_layoutA = make_layout(
            append(get<0>(scores_layoutC_N_div_2), get<0>(get<2>(scores_layoutC_N_div_2))),
            get<1>(scores_layoutC_N_div_2),
            get<1>(get<2>(scores_layoutC_N_div_2))
        );
        Tensor tOrP = make_tensor(rP.data(), scores_layoutA);
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
    Tensor rO_fp32 = make_tensor(rAccO.data(), make_shape(
        Int<get<0>(shape<0>(rAccO)) * shape<1>(rAccO)>{},
        Int<get<1>(shape<0>(rAccO)) * shape<2>(rAccO)>{}
    ));
    CUTE_UNROLL
    for (int row = 0; row < size<0>(rO_fp32); row++) {
        float scale = 1 / scores_sum(row);
        CUTE_UNROLL
        for (int col = 0; col < size<1>(rO_fp32); col++) {
            rO_fp32(row, col) *= scale;
        }
    }
    // write back
    Tensor rO = convert_type<half_t>(rO_fp32);


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

    TiledCopy gmem_tiled_copy_O = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
        Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>{}, // thread layout
        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{} // value layout
    );
    ThrCopy gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(threadIdx.x);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));
    __syncthreads();
    copy(gmem_tiled_copy_O, tOsO, tOgO);
}


void launch_flash_attention_v2(
    cute::half_t* query,
    cute::half_t* key,
    cute::half_t* value,
    cute::half_t* output,
    AttentionFeatureDim dim,
    float softmax_scale,
    int head_stride
) {
    constexpr int kBlockM = 64; // Br
    constexpr int kNWarps = 4;
    constexpr int kNThreads = kNWarps * WARP_SIZE;
    int num_q_tiles = cute::ceil_div(dim.q_seqlen, kBlockM);
    dim3 grid(num_q_tiles, dim.num_heads, dim.batch_size);
    dim3 block(kNThreads);
    flash_attention_v2<<<grid, block>>>(query, key, value, output, dim, softmax_scale, head_stride);
}

int main() {
    return 0;
}