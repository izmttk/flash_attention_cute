#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <chrono>

__global__ void gemm_device(
    float *Aptr,
    float *Bptr,
    float *Cptr,
    int M,
    int N,
    int K
) {
    constexpr int kTileM = 128;
    constexpr int kTileN = 128;
    constexpr int kTileK = 8;
    constexpr int kBlockSz = 256;
    using namespace cute;
    // GEMM NT: gemm no-transpose A, transpose B, 其中 A(M, K) B^T(K, N) C(M, N)
    // GEMM TN: gemm transpose A, no-transpose B, 其中 A^T(M, K) B(K, N) C(M, N)
    // cuBLAS 默认 A B 矩阵都是 col-major 参与计算, 结果 C 也是 col-major
    // NT 表示 A 是 col-major, B 是 row-major, 这样 B^T 就是 col-major 了
    // 以下代码实现的是 GEMM NT
    // Aptr 中存储了 col-major A(M, K)
    // Bptr 中存储了 row-major B(N, K), 也就是 col-major B^T(K, N)
    // C = A * B^T, col-major C(M, N)
    // 这一部分反映 矩阵 A B 在 global memory 中的原始存储方式
    Tensor mA = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(Int<1>{}, M));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(Int<1>{}, N));
    Tensor mC = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(Int<1>{}, M));
    // 如果是 GEMM TN, A 是 row-major, B 是 col-major, C 是 col-major
    // Tensor mA = make_tensor(make_gmem_ptr(Aptr), make_shape(K, M), make_stride(M, Int<1>{}));
    // Tensor mB = make_tensor(make_gmem_ptr(Bptr), make_shape(K, N), make_stride(N, Int<1>{}));
    // Tensor mC = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(Int<1>{}, M));

    // if (thread(0)) {
    //     print("mA: ");print_tensor(mA);print("\n");
    //     print("mB: ");print_tensor(mB);print("\n");
    //     print("mC: ");print_tensor(mC);print("\n");
    // }

    // 每个 block 负责 A B 的 k 个 tile, C 的一个 tile
    // gA(kTileM, kTileK, k)
    // gB(kTileN, kTileK, k)
    // gC(kTileM, kTileN)
    // k = K / kTileK, 就是沿 K 维度切分的块数
    Tensor gA = local_tile(mA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(blockIdx.y, _));
    Tensor gB = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(blockIdx.x, _));
    Tensor gC = local_tile(mC, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(blockIdx.y, blockIdx.x));

    // if (thread(0)) {
    //     print("gA: ");print_tensor(gA);print("\n");
    //     print("gB: ");print_tensor(gB);print("\n");
    //     print("gC: ");print_tensor(gC);print("\n");
    // }
    
    // identity layout for mA, mB
    // make_identity_tensor 产生的坐标 (i, j) 的元素值是 tuple(i, j)
    // ??? cute 3.5.0/3.5.1 在 windows 下编译会出错
    // make_identity_tensor 产生错误的结果，每个元素都是 (0, 44)，不明白为什么
    Tensor idA = make_identity_tensor(shape(mA));
    Tensor idB = make_identity_tensor(shape(mB));
    Tensor idC = make_identity_tensor(shape(mC));
    // 对 idA idB 采取相同的切分方式
    Tensor cA = local_tile(idA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(blockIdx.y, _));
    Tensor cB = local_tile(idB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(blockIdx.x, _));
    Tensor cC = local_tile(idC, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(blockIdx.y, blockIdx.x));

    // if (thread(0)) {
    //     print("idA: ");print_tensor(idA);print("\n");
    //     print("idB: ");print_tensor(idB);print("\n");
    //     print("idC: ");print_tensor(idC);print("\n");
    // }


    using ABlockLayout = decltype(make_layout(make_shape(Int<kTileM>{}, Int<kTileK>{}))); // m-major, 即 stride m = 1, 即 col-major
    using BBlockLayout = decltype(make_layout(make_shape(Int<kTileN>{}, Int<kTileK>{}))); // n-major, 即 stride m = 1, 即 col-major
    using CBlockLayout = decltype(make_layout(make_shape(Int<kTileM>{}, Int<kTileN>{}))); // m-major, 即 stride m = 1, 即 col-major
    

    // cosize 相当于 layout 最大映射到的 index + 1, 即 A(size(A)) + 1
    __shared__ float smemA[cosize_v<ABlockLayout>];
    __shared__ float smemB[cosize_v<BBlockLayout>];

    ABlockLayout sA_layout;
    BBlockLayout sB_layout;
    CBlockLayout sC_layout;

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);
    
    // tA 以 (32, 8) 和 col-major 的方式将坐标映射到线程的线性索引
    // 也就是决定了在这个范围内，线程对应哪些坐标的数据
    // tA tB 用于读取 gA gB 的数据
    auto tA = make_layout(make_shape(Int<32>{}, Int<kTileK>{}));
    auto tB = make_layout(make_shape(Int<32>{}, Int<kTileK>{}));
    // 最终用于每个线程计算 8x8 的 partition
    // tC 用于计算 gemm
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

    // kThrM = kTileM / 32, kThrN = kTileN / 32, kThrK = kTileK / 8
    // 表示每个线程在 M N K 维度上需要遍历读取的数据数量
    Tensor tAgA = local_partition(gA, tA, threadIdx.x); // (kThrM, kThrK, k)
    Tensor tBgB = local_partition(gB, tB, threadIdx.x); // (kThrN, kThrK, k)
    
    // tAsA: Partitioning pattern tA applied to tensor sA
    Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (kThrM, kThrK)
    Tensor tBsB = local_partition(sB, tB, threadIdx.x); // (kThrN, kThrK)
    
    // 对 cA cB 采取相同的切分方式
    Tensor tAcA = local_partition(cA, tA, threadIdx.x); // (kThrM, kThrK)
    Tensor tBcB = local_partition(cB, tB, threadIdx.x); // (kThrN, kThrK)

    constexpr int kThrM = size<0>(tAgA);
    constexpr int kThrN = size<0>(tBgB);
    constexpr int kThrK = size<1>(tAgA);

    // predicate tensor
    // 表示tAgA tBgB 中的数据是否越界，如果 tApA(i, j) 为 true, 则 tAgA(i, j) 是有效数据
    Tensor tApA = make_tensor<bool>(make_shape(Int<kThrM>{}, Int<kThrK>{}));
    Tensor tBpB = make_tensor<bool>(make_shape(Int<kThrN>{}, Int<kThrK>{}));

    // 计算每个数据是否越界，elem_less 指 a tuple 中对应位置的元素都小于 b tuple 中对应位置的元素
    CUTE_UNROLL
    for (int i = 0; i < size(tApA); i++) {
        tApA(i) = elem_less(tAcA(i), make_coord(M, K));
    }
    CUTE_UNROLL
    for (int i = 0; i < size(tBpB); i++) {
        tBpB(i) = elem_less(tBcB(i), make_coord(N, K));
    }

    // if (thread(0)) {
    //     print_tensor(tApA);print("\n");
    // }

    // kThrM = kTileM / 16, kThrN = kTileN / 16
    // 这里列大小是 1 的原因是，使用 local_partition 后能够按完整一行分割 sA sB
    auto AThreadLayout = make_layout(make_shape(tC.shape<0>(), Int<1>{}));
    auto BThreadLayout = make_layout(make_shape(tC.shape<1>(), Int<1>{}));
    // 每个线程要读取的 sA sB 矩阵 partition, 以及写入的 sC 矩阵 partition
    Tensor tCsA = local_partition(sA, AThreadLayout, threadIdx.x); // (kThrM, kTileK)
    Tensor tCsB = local_partition(sB, BThreadLayout, threadIdx.x); // (kThrN, kTileK)
    Tensor tCgC = local_partition(gC, tC, threadIdx.x); // (kThrM, kThrN)
    // 对 cC 采取相同的切分方式
    Tensor tCcC = local_partition(cC, tC, threadIdx.x); // (kThrM, kThrN)

    // 沿 K 维度累加用的临时矩阵 (kThrM, kThrN)
    Tensor tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    auto K_TILE_MAX = size<2>(tAgA);
    CUTE_UNROLL
    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++) {
        // 读取 A B 数据到 smem
        // copy(tAgA(_, _, k_tile), tAsA);
        // copy(tBgB(_, _, k_tile), tBsB);
        // 这个实现每个线程处理 (4, 1) 个数据，因此不需要判断 K 维度是否越界
        // 因此不需要重置 smem，因为 tCgC 在写入时还会进行一次 M N 越界判断
        // clear(tAsA);
        // clear(tBsB);
        copy_if(tApA, tAgA(_, _, k_tile), tAsA);
        copy_if(tBpB, tBgB(_, _, k_tile), tBsB);
        cp_async_fence();
        cp_async_wait<0>(); // wait all
        __syncthreads(); // 等待写入 smem 完成

        // gemm
        gemm(tCsA, tCsB, tCrC);
        __syncthreads();
    }
    // 写回结果
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); i++) {
        if (elem_less(tCcC(i), make_coord(M, N))) {
            tCgC(i) = tCrC(i); // 没用 axpby, 因为直接计算的 C = A * B
        }
    }
    // if (thread(0)) {
    //     print_tensor(tCgC);print("\n");
    // }
}

void launch_gemm(
    float *Aptr,
    float *Bptr,
    float *Cptr,
    int M,
    int N,
    int K
) {
    auto cdiv = [](int a, int b) { return (a + b - 1) / b; };
    dim3 grid(cdiv(N, 128), cdiv(M, 128));
    dim3 block(256);
    gemm_device<<<grid, block>>>(Aptr, Bptr, Cptr, M, N, K);
    // using namespace cute;
    // auto id = make_identity_tensor(make_shape(Int<4>{}, Int<3>{}));
    // print_tensor(id);print("\n");
    // print(sizeof(id));print("\n");
    // print(id(5, 5));print("\n");
}

int main() {
    const int h = 3, w=4, k=2;
    float a[h][k] = {
        {1, 2},
        {3, 4},
        {5, 6}
    };
    float b[k][w] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8}
    };
    float *out_d, *out_h, *a_d, *b_d;
    cudaMalloc(&out_d, sizeof(float) * h * w);
    cudaMalloc(&a_d, sizeof(float) * h * k);
    cudaMalloc(&b_d, sizeof(float) * k * w);
    cudaMemcpy(a_d, a, sizeof(float) * h * k, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(float) * k * w, cudaMemcpyHostToDevice);

    auto cdiv = [](int a, int b) { return (a + b - 1) / b; };

    namespace chrono = std::chrono;
    chrono::time_point<chrono::high_resolution_clock> start, end;
    chrono::duration<double, std::milli> elapsed;

    start = chrono::high_resolution_clock::now();
    launch_gemm(a_d, b_d, out_d, h, w, k);
    cudaDeviceSynchronize();
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Calculation time: " << elapsed.count() << " ms" << std::endl;

    out_h = (float *)malloc(sizeof(float) * h * w);
    cudaMemcpy(out_h, out_d, sizeof(float) * h * w, cudaMemcpyDeviceToHost);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%f ", out_h[i * w + j]);
        }
        printf("\n");
    }
    return 0;
}
