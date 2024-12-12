#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <iostream>
#include <chrono>

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
    // GEMM NT: gemm no-transpose A, transpose B, 其中 A(M, K) B^T(K, N) C(M, N)
    // GEMM TN: gemm transpose A, no-transpose B, 其中 A^T(M, K) B(K, N) C(M, N)
    // cuBLAS 默认 A B 矩阵都是 col-major 参与计算, 结果 C 也是 col-major
    // 以下代码实现的是 GEMM NT
    // Aptr 中存储了 col-major A(M, K)
    // Bptr 中存储了 col-major B(N, K), 这样 B^T 就是 row-major B^T(K, N)
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
    // NOTE: 需要添加 flag --expt-relaxed-constexpr
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
    

    // cosize 相当于 layout 最大映射到的 index + 1, 即 A(size(A)) + 1
    __shared__ float smemA[cosize_v<ABlockLayout>];
    __shared__ float smemB[cosize_v<BBlockLayout>];

    ABlockLayout sA_layout;
    BBlockLayout sB_layout;

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
    
    // if (thread(0)) {
    //     print("tC: ");my_print_layout(tC);print("\n");
    // }

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

    // local_partition(tensor, tile, index) 首先计算 index 在 tile 中的逻辑坐标，因此 tile 是一个 Layout 而不只是 Shape
    // 然后把 tensor 切块成 ((TM, TN), (M / TM, N / TN)) 的形式，上取整，其中 (TM, TN) 就是 tile 的形状
    // 并根据得到的 tile 内的逻辑坐标索引出 tile 内，即 (TM, TN) 维度上的块，得到 (M / TM, N / TN) 分块
    // 最终切片的形状为 (M / TM, N / TN)

    // threadIdx.x 每一个线程对应 tCgC 的一个元素，也对应一个（或多个）逻辑坐标
    // 我们希望逻辑坐标 (i, j) 下的线程，tCsA 对应 sA 的 (i, _) 切片，tCsB 对应 sB 的 (j, _) 切片
    // 根据 local_partition 的计算方法，我们可以知道最终切片的形状为 (kThrM, kTileK) 和 (kThrN, kTileK)
    // 也就是 tiled tensor 形状的第一个维度的形状 (TM, TN)
    // sA 要从 (kTileM, kTileK) 被切分为 ((TM, TN), (kThrM, kTileK)) 的形式，可知 (TM, TN) = (kTileM / kThrM, 1)
    // sB 要从 (kTileN, kTileK) 被切分为 ((TM, TN), (kThrN, kTileK)) 的形式，可知 (TM, TN) = (kTileN / kThrN, 1)
    // 我们就知道了形状 TilerA (kTileM / kThrM, 1) 和 TilerB (kTileN / kThrN, 1)
    // 其中 kThrM = kTileM / tC_M, kThrN = kTileN / tC_N,
    // 因此，TilerA 的 shape = (tC_M, 1)，TilerB 的 shape = (tC_N, 1)
    // 上面有些复杂了，其实想分割出连续的一行，只需要让 tile 的第二维度为 1 即可，结论是一样的
    // 只知道形状还不够，如何让 tCsA tCsB 对应 sA sB 的正确切片呢？这个与 tiler 的 stride 有关
    // 由于 TilerA TilerB 的第二维度为 1，因此我们就设定它们的 strideY = 0
    // 在 local_partition 的计算方法中，threadIdx.x 会被转换为 tile 内的逻辑坐标，也就是每个线程一个逻辑坐标
    // idx2crd: coordX = (idx / strideX) % shapeX, ...
    // 在这里，idx = j * tC_M + i，因为 tC 是 col-major，此时
    // coordX = ((j * tC_M + i) / strideX) % shapeX
    // 对于 TilerA，我们想让在 tCgC 下处理逻辑坐标 (i, j) 的线程， 对应 tCsA 的 (i, 0) 切片
    // 也就是 tCgC 下同一行 i 的线程，TilerA 的布局能够将 threadIdx.x 转换为 (i, 0) 的逻辑坐标
    // j = 0, coordX = i, 可解得 strideX = 1
    // 对于 TilerB，我们想让在 tCgC 下处理逻辑坐标 (i, j) 的线程， 对应 tCsB 的 (j, 0) 切片
    // 也就是 tCgC 下同一列 j 的线程，TilerB 的布局能够将 threadIdx.x 转换为 (j, 0) 的逻辑坐标
    // i = 0, coordX = j, 可解得 strideX = tC_M
    // 因此，TilerA 的 stride = (1, 0)，TilerB 的 stride = (tC_M, 0)

    auto TilerA = make_layout(make_shape(size<0>(tC), Int<1>{}), make_stride(Int<1>{}, Int<0>{}));
    auto TilerB = make_layout(make_shape(size<1>(tC), Int<1>{}), make_stride(size<1>(tC), Int<0>{}));
    Tensor tCsA = local_partition(sA, TilerA, threadIdx.x);
    Tensor tCsB = local_partition(sB, TilerB, threadIdx.x);
    // 下面这种方式也能做到，不过中间使用的是 vector layout
    // Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, Underscore>{});
    // Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<Underscore, _1>{});
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
        clear(tAsA);
        clear(tBsB);
        copy_if(tApA, tAgA(_, _, k_tile), tAsA);
        copy_if(tBpB, tBgB(_, _, k_tile), tBsB);
        cp_async_fence();
        cp_async_wait<0>(); // wait all
        __syncthreads(); // 等待写入 smem 完成
        // gemm
        gemm(tCsA, tCsB, tCrC);
        __syncthreads();
        // if (thread(2)) {
        //     print("tCsA: ");print_tensor(tCsA);print("\n");
        //     print("tCsB: ");print_tensor(tCsB);print("\n");
        //     print("tCrC: ");print_tensor(tCrC);print("\n");
        // }
    }

    // 写回结果
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); i++) {
        if (elem_less(tCcC(i), make_coord(M, N))) {
            tCgC(i) = tCrC(i); // 没用 axpby, 因为直接计算的 C = A * B
        }
    }
    // if (thread(0)) {
    //     print("tCcC: ");print_tensor(tCcC);print("\n");
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
    // auto a = make_layout(make_shape(Int<4>{}), make_stride(Int<4>{}));
    // for (int i = 0; i < 4*size(a); i++) {
    //     print(a.get_flat_coord(i));print(" ");
    // }
    // print("\n");
    // auto a = make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<1>{}, Int<0>{}));
    // my_print_layout(a);
    // auto b = make_layout(make_shape(Int<4>{}, Int<4>{}), make_stride(Int<0>{}, Int<1>{}));
    // my_print_layout(b);
    // for (int i = 0; i < 4 * size(b); i++) {
    //     print(b.get_flat_coord(i));print(" ");
    // }
    // print("\n");
    // print("dice: ");print(dice(Step<Underscore, _1>{}, a));print("\n");
    // my_print_layout(a);
    // // print_layout 在 device 上输出不完整的格式，这个与 cuda 对 device printf 的支持有关？
    // print_layout(a);print("\n");
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
    // C = A * B
    // [11, 14, 17, 20]
    // [23, 30, 37, 44]
    // [35, 46, 57, 68]

    // row-major to col-major
    float *a_h = new float[h * k];
    float *b_h = new float[w * k];
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < k; j++) {
            a_h[j * h + i] = a[i][j];
        }
    }
    printf("a_h: ");
    for (int i = 0; i < h * k; i++) {
        printf("%f ", a_h[i]);
    }
    printf("\n");
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < w; j++) {
            b_h[i * w + j] = b[i][j];
        }
    }
    printf("b_h: ");
    for (int i = 0; i < w * k; i++) {
        printf("%f ", b_h[i]);
    }
    printf("\n");

    float *out_d, *out_h, *a_d, *b_d;
    cudaMalloc(&out_d, sizeof(float) * h * w);
    cudaMalloc(&a_d, sizeof(float) * h * k);
    cudaMalloc(&b_d, sizeof(float) * k * w);
    cudaMemcpy(a_d, a_h, sizeof(float) * h * k, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeof(float) * k * w, cudaMemcpyHostToDevice);

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

    printf("out_h: \n");
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%f ", out_h[j * h + i]);
        }
        printf("\n");
    }
    return 0;
}
