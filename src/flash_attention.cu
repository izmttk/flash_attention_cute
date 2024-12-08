#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include <iostream>
#include <random>
#include <chrono>

struct AttentionFeatureDim {
    int batch_size;  // N
    int num_heads;   // H
    int q_seqlen;    // L
    int kv_seqlen;   // S
    int qk_headdim;  // E
    int v_headdim;   // Ev
};

//  query: [batch_size, num_heads,  q_seqlen, qk_headdim]
//    key: [batch_size, num_heads, kv_seqlen, qk_headdim]
//  score: [batch_size, num_heads,  q_seqlen,  kv_seqlen]
//  value: [batch_size, num_heads, kv_seqlen,  v_headdim]
// output: [batch_size, num_heads,  q_seqlen,  v_headdim]
// rowmax: [batch_size, num_heads,  q_seqlen]
// rowsum: [batch_size, num_heads,  q_seqlen]

__global__ void flash_attention_v2(
    float* query,
    float* key,
    float* value,
    float* output,
    AttentionFeatureDim dim
) {

}


void launch_flash_attention_v2(
    float* query,
    float* key,
    float* value,
    float* output,
    AttentionFeatureDim dim
) {

}


void attention_cpu(
    float *query,
    float *key,
    float *value,
    float *output,
    AttentionFeatureDim dim
) {
    float scale = 1.0f / sqrt(static_cast<float>(dim.qk_headdim));
    float *score = new float[dim.batch_size * dim.num_heads * dim.q_seqlen * dim.kv_seqlen];
    for (int batch = 0; batch < dim.batch_size; batch++) {
        for (int head = 0; head < dim.num_heads; head++) {
            float* Q = query + (batch * dim.num_heads + head) * (dim.q_seqlen * dim.qk_headdim);
            // K: [kv_seqlen, qk_headdim]
            float* K = key + (batch * dim.num_heads + head) * (dim.kv_seqlen * dim.qk_headdim);
            // V: [kv_seqlen, v_headdim]
            float* V = value + (batch * dim.num_heads + head) * (dim.kv_seqlen * dim.v_headdim);
            // O: [q_seqlen, v_headdim]
            float* O = output + (batch * dim.num_heads + head) * (dim.q_seqlen * dim.v_headdim);
            // S: [q_seqlen, kv_seqlen]
            float* S = score + (batch * dim.num_heads + head) * (dim.q_seqlen * dim.kv_seqlen);
            // S = (Q * K^T) / sqrt(d_k)
            for (int i = 0; i < dim.q_seqlen; i++) {
                for (int j = 0; j < dim.kv_seqlen; j++) {
                    float sum = 0.0f;
                    for (int h = 0; h < dim.qk_headdim; h++) {
                        sum += Q[i * dim.qk_headdim + h] * K[j * dim.qk_headdim + h];
                    }
                    S[i * dim.kv_seqlen + j] = sum * scale;
                }
            }
            // softmax
            for (int i = 0; i < dim.q_seqlen; i++) {
                float max_val = -FLT_MAX;
                float sum = 0.0f;
                for (int j = 0; j < dim.kv_seqlen; j++) {
                    max_val = max(max_val, S[i * dim.kv_seqlen + j]);
                }
                for (int j = 0; j < dim.kv_seqlen; j++) {
                    sum += exp(S[i * dim.kv_seqlen + j] - max_val);
                }
                for (int j = 0; j < dim.kv_seqlen; j++) {
                    S[i * dim.kv_seqlen + j] = exp(S[i * dim.kv_seqlen + j] - max_val) / sum;
                }
            }
            // O = S * V
            for (int i = 0; i < dim.q_seqlen; i++) {
                for (int j = 0; j < dim.v_headdim; j++) {
                    float sum = 0.0f;
                    for (int h = 0; h < dim.kv_seqlen; h++) {
                        sum += S[i * dim.kv_seqlen + h] * V[h * dim.v_headdim + j];
                    }
                    O[i * dim.v_headdim + j] = sum;
                }
            }
        }
    }
    delete[] score;
}

float *generate_data(int n) {
    std::default_random_engine generator;
    auto distribution = std::uniform_real_distribution<float>(1.0, 10.0);
    float *data = new float[n];
    for (int i = 0; i < n; i++) {
        data[i] = distribution(generator);
    }
    return data;
}

float mse(float *a, float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum / n;
}

void benchmark(
    AttentionFeatureDim dim,
    int iter=100
) {
    printf("benchmark preparing\n");

    namespace chrono = std::chrono;
    chrono::time_point<chrono::high_resolution_clock> start, end;
    chrono::duration<double, std::milli> elapsed;
    
    int qn = dim.batch_size * dim.num_heads * dim.q_seqlen * dim.qk_headdim;
    int kn = dim.batch_size * dim.num_heads * dim.kv_seqlen * dim.qk_headdim;
    int vn = dim.batch_size * dim.num_heads * dim.kv_seqlen * dim.v_headdim;
    int on = dim.batch_size * dim.num_heads * dim.q_seqlen * dim.v_headdim;
    float *q = generate_data(qn);
    float *k = generate_data(kn);
    float *v = generate_data(vn);
    float *o = new float[on];
    memset(o, 0, on * sizeof(float));

    float *q_d, *k_d, *v_d, *o_d;
    cudaMalloc(&q_d, qn * sizeof(float));
    cudaMalloc(&k_d, kn * sizeof(float));
    cudaMalloc(&v_d, vn * sizeof(float));
    cudaMalloc(&o_d, on * sizeof(float));
    cudaMemcpy(q_d, q, qn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, k, kn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v, vn * sizeof(float), cudaMemcpyHostToDevice);


    // warm up
    // for (int i = 0; i < 3; i++) {
    //     launch_flash_attention_v2(q_d, k_d, v_d, o_d, dim);
    // }
    printf("benchmark start\n");
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++) {
        launch_flash_attention_v2(q_d, k_d, v_d, o_d, dim);
    }
    cudaDeviceSynchronize();
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "CUDA calculation time: " << elapsed.count() / iter << " ms" << std::endl;

    cudaMemcpy(o, o_d, on * sizeof(float), cudaMemcpyDeviceToHost);

    float *o_cpu = new float[on];
    start = chrono::high_resolution_clock::now();
    attention_cpu(q, k, v, o_cpu, dim);
    end = chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "CPU calculation time: " << elapsed.count() << " ms" << std::endl;

    // for (int batch = 0; batch < dim.batch_size; batch++) {
    //     for (int head = 0; head < dim.num_heads; head++) {
    //         float* O_d = o + (batch * dim.num_heads + head) * (dim.q_seqlen * dim.v_headdim);
    //         float* O_cpu = o_cpu + (batch * dim.num_heads + head) * (dim.q_seqlen * dim.v_headdim);
    //         for (int s = 0; s < dim.q_seqlen; s++) {
    //             for (int h = 0; h < dim.v_headdim; h++) {
    //                 float O_d_val = O_d[s * dim.v_headdim + h];
    //                 float O_cpu_val = O_cpu[s * dim.v_headdim + h];
    //                 if (abs(O_d_val - O_cpu_val) > 1e-4) {
    //                     printf("(%d, %d, %d, %d): cpu %f, gpu %f, diff: %f\n",
    //                         batch, head, s, h,
    //                         O_cpu_val, O_d_val, O_cpu_val - O_d_val
    //                     );
    //                 }
    //                 // printf("(%d, %d, %d, %d): cpu %f, gpu %f\n", batch, head, s, h, O_cpu_val, O_d_val);
    //             }
    //         }
    //     }
    // }

    float error = mse(o, o_cpu, on);
    printf("error: %f\n", error);
    
    delete[] o_cpu;

    delete[] q;
    delete[] k;
    delete[] v;
    delete[] o;
    cudaFree(q_d);
    cudaFree(k_d);
    cudaFree(v_d);
    cudaFree(o_d);
}

int main() {
    AttentionFeatureDim dim;
    dim.batch_size = 2;
    dim.num_heads = 8;
    dim.q_seqlen = 1024;
    dim.kv_seqlen = 1024;
    dim.qk_headdim = 64;
    dim.v_headdim = 64;
    benchmark(dim, 1);
    return 0;
}