#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <chrono>
#include <random>

template <class Layout>
__host__ __device__ void my_print_layout(Layout const& layout)  // (m,n) -> idx
{
  using namespace cute;
//   CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  int idx_width = num_digits(cosize(layout)) + 2;

  print(layout); print("\n");

  // Column indices
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) { printf("  %*d ", idx_width-2, n); }
  printf("\n");

  // Print out A m-by-n
  for (int m = 0; m < size<0>(layout); ++m) {
    // Header
    print("    ");
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("+");
      for (int i = 0; i < idx_width; ++i) { printf("-"); }
    }
    printf("+\n");
    // Values
    printf("%2d  ", m);  // Row indices
    for (int n = 0; n < size<1>(layout); ++n) { printf("| %*d ", idx_width-2, int(layout(m,n))); }
    printf("|\n");
  }
  // Footer
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) {
    printf("+");
    for (int i = 0; i < idx_width; ++i) { printf("-"); }
  }
  printf("+\n");
}

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
    using namespace cute;

    // mA: M x K col-major
    // mB: N x K col-major
    // mC: M x N col-major
    Tensor mA = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(Int<1>{}, M));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(Int<1>{}, N));
    Tensor mC = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(Int<1>{}, M));


    // gA: (kTileM, kTileK, k) col-major
    // gB: (kTileN, kTileK, k) col-major
    // gC: (kTileM, kTileN) col-major
    Tensor gA = local_tile(mA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(blockIdx.y, _));
    Tensor gB = local_tile(mB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(blockIdx.x, _));
    Tensor gC = local_tile(mC, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(blockIdx.y, blockIdx.x));

    // predicate tensor, 元素值是坐标元组，用来检查 gmem 坐标是否越界
    Tensor idA = make_identity_tensor(shape(mA));
    Tensor idB = make_identity_tensor(shape(mB));
    Tensor idC = make_identity_tensor(shape(mC));
    // 对 idA idB 采取相同的切分方式
    Tensor cA = local_tile(idA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(blockIdx.y, _));
    Tensor cB = local_tile(idB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(blockIdx.x, _));
    Tensor cC = local_tile(idC, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(blockIdx.y, blockIdx.x));


    auto sA_layout = make_layout(make_shape(Int<kTileM>{}, Int<kTileK>{})); // m-major, 即 stride m = 1, 即 col-major
    auto sB_layout = make_layout(make_shape(Int<kTileN>{}, Int<kTileK>{})); // n-major, 即 stride m = 1, 即 col-major
    
    __shared__ float smemA[cosize(sA_layout)];
    __shared__ float smemB[cosize(sB_layout)];
    
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    // 这意味着 (32, 8) 个线程，每个线程 copy (4, 1) 个元素，并且最小操作单位是 uint128_t，即 4 个 float
    // TiledCopy 规定了执行一次 TiledCopy 操作处理的数据布局/规模，包括每个线程需要负责哪些数据
    TiledCopy copy_A = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{}, // copy_atom(copy_op, copy_type)
        make_layout(make_shape(Int<32>{}, Int<8>{})), // Thread Layout (32, 8)
        make_layout(make_shape(Int<4>{}, Int<1>{}))   // Value Layout (4, 1)
    );
    // 根据 Thread Layout 通过 threadIdx.x 索引到线程负责的数据块
    // ThrCopy 规定了当前线程需要负责的数据
    ThrCopy thr_copy_A = copy_A.get_slice(threadIdx.x);

    // kThrM = 4, kThrN = 4, kThrK = 1
    // partition_S|D 的结果是 ((ValLayout...), 按 TiledCopy 将 gA|B 分块的坐标...)
    // TiledCopy 的分块坐标可能少于 gA|B 的维度，只分块前几个维度，后面维度尺寸会保持不变
    Tensor tAgA = thr_copy_A.partition_S(gA); // ((kThrM, kThrM), 1, 1, k) = ((4, 1), 1, 1, k)
    Tensor tAsA = thr_copy_A.partition_D(sA); // ((kThrN, kThrM), 1, 1)
    Tensor tAcA = thr_copy_A.partition_S(cA); // ((kThrM, kThrM), 1, 1, k)
    Tensor tApA = make_tensor_like<bool>(tAsA(0, _, _));


    TiledCopy copy_B = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{}, // copy_atom(copy_op, copy_type)
        make_layout(make_shape(Int<32>{}, Int<8>{})), // Thread Layout (32, 8)
        make_layout(make_shape(Int<4>{}, Int<1>{}))   // Value Layout (4, 1)
    );
    ThrCopy thr_copy_B = copy_B.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_B.partition_S(gB); // ((kThrM, kThrM), 1, 1, k)
    Tensor tBsB = thr_copy_B.partition_D(sB); // ((kThrN, kThrM), 1, 1)
    Tensor tBcB = thr_copy_B.partition_S(cB); // ((kThrM, kThrM), 1, 1, k)
    Tensor tBpB = make_tensor_like<bool>(tBsB(0, _, _));

    // 1x1x1 FMA atom, repeat 16x16x1
    // TiledMMA 规定了执行一次 TiledMMA 操作处理的数据布局/规模，包括每个线程需要负责哪些数据
    TiledMMA mma = make_tiled_mma(
        MMA_Atom<UniversalFMA<float>>{},  // mma_atom(mma_op) 1x1x1
        make_layout(make_shape(Int<16>{}, Int<16>{}, Int<1>{})) // Thread Layout (16, 16, 1)
    );
    // 这意味着我们一个 block 中所有线程一次要处理的问题规模为 MxNxK = 16x16x1
    // 子问题 C = A * B, C(16, 16), A(16, 1), B(1, 16)
    // make_tiled_mma 还可以有第三个参数 PermutationMNK, 用于表示对 M N K 维度的置换
    // 用这个参数可以实现一个线程多次计算 MMA_Atom，如何做？？？

    // 别的教程可能这里会使用 get_thread_slice, 其实是一个函数
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    // MMA 表示 一次 MMA_Atom 消耗的数据量/布局，MMA_M/N 就是被分解的子问题布局了
    // TiledMMA 定义的子问题规模为 MNK = 16x16x1，那么 sA 被 16x1 分块，sB 被 1x16 分块，gC 被 16x16 分块
    // 每个线程需要在这些分块上重复计算，具体来说 sA|B(128, 8) 被分为 (8, 8) 块，gC(128, 128) 被分为 (8, 8) 块
    // 那么每个线程需要串行循环计算 8x8x8 个 MMA_Atom
    // partition_A|B|C 的结果是 ((单个MMA_Op消耗的数据布局...), 按 TiledMMA 将 sA|sB|gC 分块的坐标...)
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA, MMA_M, MMA_K)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA, MMA_N, MMA_K)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)
    Tensor tCcC = thr_mma.partition_C(cC);


    Tensor tArA = make_fragment_like(tAsA);
    Tensor tBrB = make_fragment_like(tBsB);

    // 这里是按照 tCgC 的布局来创建一个 Tensor
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA ,MMA_M, MMA_N)

    // SM80 以下的 GPU，从 gmem 复制到 smem 实际上是 LDG + STS，需要使用中间寄存器

    // 未使用 pipeline 的指令流
    // +---------------------+--------+-----------------+---------------------+--------+-----------------+
    // | LDG T1              | STS T1 | LDS T1 | MMA T1 | LDG T2              | STS T2 | LDS T2 | MMA T2 | ...
    // +---------------------+--------+-----------------+---------------------+--------+-----------------+

    // 使用 pipeline(two-stage, double-buffer) 的指令流
    // +---------------------+--------+---------------------+--------+---------------------+--------+
    // | LDG T1              | STS T1 | LDG T2              | STS T2 | LDG T3              | STS T3 | ...
    // +---------------------+--------+--------+--------+---+--------+--------+--------+---+--------+
    //                                | LDS T1 | MMA T1 |            | LDS T2 | MMA T2 |
    //                                +--------+--------+            +--------+--------+
    //                       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^ 循环节
    //                                ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^ 或者这种循环节也可以，但是不高效


    // 先将 k_tile = 0 的块复制到寄存器
    // 用于 compute 和 gmem read 的 pipeline，通过交替
    // 1. 读取 gmem 数据到寄存器
    // 2. 从寄存器加载到 smem，计算 gemm
    copy(copy_A, tAgA(_, _, _, 0), tArA);
    copy(copy_B, tBgB(_, _, _, 0), tBrB);
    
    clear(tCrC);

    auto K_TILE_MAX = size<3>(tAgA);
    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++) {
        // CUTE_UNROLL
        // for (int i = 0; i < size(tApA); i++) {
        //     tApA(i) = elem_less(tAcA(_, _, _, k_tile)(i), make_coord(M, K));
        // }
        // CUTE_UNROLL
        // for (int i = 0; i < size(tBpB); i++) {
        //     tBpB(i) = elem_less(tBcB(_, _, _, k_tile)(i), make_coord(N, K));
        // }
        // if (thread(0)) {
        //     print("k_tile: ");print(k_tile);print("\n");
        //     print("tAgA: ");print_tensor(tAgA(_, _, _, k_tile));print("\n");
        //     print("tBgB: ");print_tensor(tBgB(_, _, _, k_tile));print("\n");
        // }
        __syncthreads();
        // clear(tAsA);
        // clear(tBsB);
        // copy_if(copy_A, tApA, tAgA(_, _, _, k_tile), tAsA);
        // copy_if(copy_B, tBpB, tBgB(_, _, _, k_tile), tBsB);
        // 读取当前数据 从寄存器到 smem
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();
        if (k_tile + 1 < K_TILE_MAX) {
            copy(copy_A, tAgA(_, _, _, k_tile + 1), tArA);
            copy(copy_B, tBgB(_, _, _, k_tile + 1), tBrB);
        }
        // gemm
        gemm(mma, tCsA, tCsB, tCrC);
    }

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); i++) {
        if (elem_less(tCcC(i), make_coord(M, N))) {
            tCgC(i) = tCrC(i); // 没用 axpby, 因为直接计算的 C = A * B
        }
    }
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
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void gemm_host(
    float *Aptr,
    float *Bptr,
    float *Cptr,
    int M,
    int N,
    int K
) {
    // gemm nt, A(M, K) col-major, B(N, K) col-major, C(M, N) col-major
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += Aptr[k * M + m] * Bptr[k * N + n];
            }
            Cptr[n * M + m] = sum;
        }
    }
}


float *generate_data(int n) {
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    auto distribution = std::uniform_real_distribution<float>(1.0, 10.0);
    float *data = new float[n];
    for (int i = 0; i < n; i++) {
        data[i] = distribution(generator);
    }
    return data;
}

int main() {
    // 注意由于使用了向量化访存，数据应当对齐 16 字节，即 M 是 4 的倍数
    const int M = 4, N = 4, K = 17;
    float *a_h = generate_data(M * K);
    float *b_h = generate_data(K * N);

    float *out_d, *a_d, *b_d;
    cudaMalloc(&out_d, sizeof(float) * M * N);
    cudaMalloc(&a_d, sizeof(float) * M * K);
    cudaMalloc(&b_d, sizeof(float) * K * N);
    cudaMemcpy(a_d, a_h, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    namespace chrono = std::chrono;
    chrono::time_point<chrono::high_resolution_clock> start, end;
    chrono::duration<double, std::milli> elapsed;

    start = chrono::high_resolution_clock::now();
    launch_gemm(a_d, b_d, out_d, M, N, K);
    cudaDeviceSynchronize();
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Calculation time: " << elapsed.count() << " ms" << std::endl;

    float* out_h = (float *)malloc(sizeof(float) * M * N);
    cudaMemcpy(out_h, out_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    float* out_hh = (float *)malloc(sizeof(float) * M * N);
    gemm_host(a_h, b_h, out_hh, M, N, K);

    printf("out_h: \n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", out_h[j * M + i]);
        }
        printf("\n");
    }

    printf("out_hh: \n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", out_hh[j * M + i]);
        }
        printf("\n");
    }
    return 0;
}
