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
    constexpr int kStages = 4;
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

    // if (thread(0)) {
    //     print_tensor(gA(_, _, 0));
    // }

    // predicate tensor, 元素值是坐标元组，用来检查 gmem 坐标是否越界
    // Tensor idA = make_identity_tensor(shape(mA));
    // Tensor idB = make_identity_tensor(shape(mB));
    Tensor idC = make_identity_tensor(shape(mC));
    // 对 idA idB 采取相同的切分方式
    // Tensor cA = local_tile(idA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(blockIdx.y, _));
    // Tensor cB = local_tile(idB, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(blockIdx.x, _));
    Tensor cC = local_tile(idC, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(blockIdx.y, blockIdx.x));


    auto sA_layout = make_layout(make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStages>{})); // m-major, 即 stride m = 1, 即 col-major
    auto sB_layout = make_layout(make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStages>{})); // n-major, 即 stride m = 1, 即 col-major
    
    __shared__ float smemA[cosize(sA_layout)];
    __shared__ float smemB[cosize(sB_layout)];
    
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (kTileM, kTileK, kStages)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (kTileN, kTileK, kStages)

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
    Tensor tAsA = thr_copy_A.partition_D(sA); // ((kThrN, kThrM), 1, 1, kStages)
    // Tensor tAcA = thr_copy_A.partition_S(cA); // ((kThrM, kThrM), 1, 1, k)
    // Tensor tApA = make_tensor_like<bool>(tAsA(0, _, _, 0));

    TiledCopy copy_B = make_tiled_copy(
        Copy_Atom<UniversalCopy<uint128_t>, float>{}, // copy_atom(copy_op, copy_type)
        make_layout(make_shape(Int<32>{}, Int<8>{})), // Thread Layout (32, 8)
        make_layout(make_shape(Int<4>{}, Int<1>{}))   // Value Layout (4, 1)
    );
    ThrCopy thr_copy_B = copy_B.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_B.partition_S(gB); // ((kThrM, kThrM), 1, 1, k)
    Tensor tBsB = thr_copy_B.partition_D(sB); // ((kThrN, kThrM), 1, 1, kStages)
    // Tensor tBcB = thr_copy_B.partition_S(cB); // ((kThrM, kThrM), 1, 1, k)
    // Tensor tBpB = make_tensor_like<bool>(tBsB(0, _, _, 0));

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

    // 本来 tCrA|B 可以只开辟 MMA_K = 2 的空间，因为 LDS 和 MMA 流水线可以交替复用，可以节省寄存器数量
    // 但是 nvcc 似乎已经实现了寄存器的复用，修改后并没有 live register 数量减少
    Tensor tCrA = thr_mma.make_fragment_A(tCsA(_, _, _, 0)); // (MMA, MMA_M, MMA_K)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB(_, _, _, 0)); // (MMA, MMA_N, MMA_K)
    // 这里是按照 tCgC 的布局来创建一个 Tensor
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // (MMA ,MMA_M, MMA_N)

    // sm70 two-stage pipeline:
    //                v sync    v v sync    v v sync    v v sync    v v sync    v v sync
    // LDG ░░░░░░░░░   ░░░░░░░░░   ░░░░░░░░░   ░░░░░░░░░   ░░░░░░░░░   ░░░░░░░░░
    // STS          ██          ██          ██          ██          ██          ██
    // LDS            ▒█▒█▒█▒█    ░█░█░█░█    ▒█▒█▒█▒█    ░█░█░█░█    ▒█▒█▒█▒█    ░█░█░█░█
    // MMA             ▒█▒█▒█▒    █░█░█░█░    █▒█▒█▒█▒    █░█░█░█░    █▒█▒█▒█▒    █░█░█░█░█
    //                 ^~~~~~~~~~~^ 循环节

    // 在 SM80（Ampere 架构）及以上的 GPU 上，可以使用 cp.async 指令实现 gmem 复制的多级流水线，
    // cp.async 指令实际上会被编译为 LDGSTS 指令，LDGSTS 从 gmem 异步读取数据到 smem，并且不需要中间寄存器
    // 如果当前tile 的 LDG 时间远大于完成 MMA 所有计算需要的时间，那么我们可以在 LDG 上再次进行 pipeline 优化，此时需要 LDGSTS 指令
    // two-stage 指的是 gmem to smem 和 smem to reg 两个阶段
    // multi-stage 指的是多个 gmem to smem 阶段和一个 smem to reg 阶段
    // 为实现 multi-stage，我们需要在共享内存开辟 stage * tile size 的空间
    // multi-stage 的目的是尽量充分利用 compute unit 的资源
    //                    |       |       |       |       |       |       |       |       |       |       |       |
    // LDGSTS T1 ░░░░░░░░░|       |░░░░░░░░░      |       |       |░░░░░░░░░      |       |       |       |       |
    // LDGSTS T2 ░░░░░░░░░|       |       |░░░░░░░░░      |       |       |░░░░░░░░░      |       |       |       |
    // LDGSTS T3 ░░░░░░░░░|       |       |       |░░░░░░░░░      |       |       |░░░░░░░░░      |       |       |
    // LDGSTS T4          |░░░░░░░░░      |       |       |░░░░░░░░░      |       |       |░░░░░░░░░      |       |
    //       LDS          ▒█▒█▒█▒█░█░█░█░█▒█▒█▒█▒█░█░█░█░█▒█▒█▒█▒█░█░█░█░█▒█▒█▒█▒█░█░█░█░█▒█▒█▒█▒█░█░█░█░█▒█▒█▒█▒█░█░█░█░█
    //       MMA          |▒█▒█▒█▒█░█░█░█░█▒█▒█▒█▒█░█░█░█░█▒█▒█▒█▒█░█░█░█░█▒█▒█▒█▒█░█░█░█░█▒█▒█▒█▒█░█░█░█░█▒█▒█▒█▒█░█░█░█░█
    //                    |^~~~~~~^loop   |       |       |       |       |       |       |       |       |       |
    //                    ^waitT1 ^waitT2 ^waitT3 ^waitT4 ^waitT1 ^waitT2 ^waitT3 ^waitT4 ^waitT1 ^waitT2 ^waitT3 ^waitT4

    auto K_PIPE_MAX = size<3>(tAsA);
    auto K_TILE_MAX = size<3>(tAgA);

    int k_tile_next = 0;

    // 在正式开始循环之前，启动前 kStages - 1 个 gmem to smem
    for (int k_pipe = 0; k_pipe < min(K_PIPE_MAX - 1, K_TILE_MAX); k_pipe++) {
        copy(copy_A, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        copy(copy_B, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();
        k_tile_next++;
    }

    
    clear(tCrC); // 放在这里是为了可以与上面的 copy 重叠


    int smem_pipe_read = 0; // 将要被读取的 smem 块
    int smem_pipe_write = K_PIPE_MAX - 1; // 将要被写入的 smem 块
    // 小循环中，需要读取 smem 的并计算 MMA 的切片
    Tensor tCsA_p = tCsA(_, _, _, smem_pipe_read);
    Tensor tCsB_p = tCsB(_, _, _, smem_pipe_read);

    auto K_BLOCK_MAX = size<2>(tCrC);

    // 在正式开始之前，还需要从 smem 读出第一个 tile 第一个块的数据到寄存器
    if (K_BLOCK_MAX > 1) {
        // 这里的 K_PIPE_MAX - 2 是因为已经启动了 K_PIPE_MAX - 1 个 copy 操作，这里需要等待最旧的那个 copy 操作完成
        // 因此保留 K_PIPE_MAX - 1 - 1 个 copy 操作，也就是 K_PIPE_MAX - 2 个 copy 操作
        cp_async_wait<K_PIPE_MAX - 2>();
        __syncthreads();
        
        // 读取第一个 tile 的第一个 k_block 的数据到寄存器
        copy(tCsA_p(_, _, 0), tCrA(_, _, 0));
        copy(tCsB_p(_, _, 0), tCrB(_, _, 0));
    }

    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++) {

            // 如果是当前 tile 的最后一块，切换需要读取的 smem 的分块
            if (k_block == K_BLOCK_MAX - 1) {
                tCsA_p = tCsA(_, _, _, smem_pipe_read);
                tCsB_p = tCsB(_, _, _, smem_pipe_read);
                cp_async_wait<K_PIPE_MAX - 2>();
                // 为什么这里是 K_PIPE_MAX - 2
                // 一次 gmem 读取到一个 smem pipe块是一个 async copy 的 group
                // 最后一个 k_block 处理时，pipeline 里有 K_PIPE_MAX - 1 个 copy 操作
                // 因为当前 tile 执行前有 K_PIPE_MAX - 1 个 copy 操作，执行到最后一个 k_block 时，
                // 消耗掉了往前数第 K_PIPE_MAX - 1 个 copy 操作，也就是最旧的那个 smem 块，
                // 这里准备下一个 tile 的第一个 k_block，因此需要等待往前数第 K_PIPE_MAX - 2 个 copy 操作

                // 注意 cp_async_wait 只是线程内的同步，而 __syncthreads 是线程间的同步
                // 如果不同线程需要读写同一个 smem 块，该加上还是要加上
                __syncthreads();
            }

            // 读取下一个块的数据到寄存器
            int k_block_next = (k_block + 1) % K_BLOCK_MAX;
            copy(tCsA_p(_, _, k_block_next), tCrA(_, _, k_block_next));
            copy(tCsB_p(_, _, k_block_next), tCrB(_, _, k_block_next));

            // 如果是当前 tile 的第一块，开始下一个 tile gmem 读取到 smem
            if (k_block == 0) {
                if (k_tile_next < K_TILE_MAX) {
                    copy(copy_A, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, smem_pipe_write));
                    copy(copy_B, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, smem_pipe_write));
                    cp_async_fence();
                }
                // 准备一下个 tile 的读取
                k_tile_next++;

                // 在这里修改是因为当前 tile 第 1 个块的前一个块使用了 smem_pipe_read，第 1 个块使用了 smem_pipe_write
                // 后面不会再用到旧值，因此可以在使用完后立刻更新为下一个 tile 将要从 smem 读取 到 reg 的 smem 区域，
                // 和将要从 gmem 写入 smem 的 smem 区域
                smem_pipe_write = smem_pipe_read; // 上一个被读取的 smem 块完成了计算
                smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX; // 循环复用 smem
            }

            // gemm
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
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
    const int M = 4, N = 4, K = 16;
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

    float* out_h = new float[M * N];
    cudaMemcpy(out_h, out_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    float* out_hh = new float[M * N];
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
