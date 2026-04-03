#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " -> " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = (call);                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__      \
                      << " -> status code " << status << std::endl;             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Multiplicación CPU en column-major: C = A * B
void matrixMultCpuFloat(const float* A, const float* B, float* C, int N) {
    for (int col = 0; col < N; ++col) {
        for (int row = 0; row < N; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row + k * N] * B[k + col * N];
            }
            C[row + col * N] = sum;
        }
    }
}

double maxAbsDiffFloat(const float* ref, const float* test, int size) {
    double maxDiff = 0.0;
    for (int i = 0; i < size; ++i) {
        maxDiff = std::max(maxDiff, std::abs(static_cast<double>(ref[i]) - static_cast<double>(test[i])));
    }
    return maxDiff;
}

double maxAbsDiffDouble(const double* ref, const double* test, int size) {
    double maxDiff = 0.0;
    for (int i = 0; i < size; ++i) {
        maxDiff = std::max(maxDiff, std::abs(ref[i] - test[i]));
    }
    return maxDiff;
}

int main() {
    const int N = 1024;
    const int numElements = N * N;

    const size_t sizeF = numElements * sizeof(float);
    const size_t sizeD = numElements * sizeof(double);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Host float
    std::vector<float> h_Af(numElements);
    std::vector<float> h_Bf(numElements);
    std::vector<float> h_C_cpu_f(numElements, 0.0f);
    std::vector<float> h_C_gpu_s(numElements, 0.0f);

    // Host double
    std::vector<double> h_Ad(numElements);
    std::vector<double> h_Bd(numElements);
    std::vector<double> h_C_cpu_d(numElements, 0.0);
    std::vector<double> h_C_gpu_d(numElements, 0.0);

    // Inicialización
    for (int i = 0; i < numElements; ++i) {
        h_Af[i] = dist(rng);
        h_Bf[i] = dist(rng);

        h_Ad[i] = static_cast<double>(h_Af[i]);
        h_Bd[i] = static_cast<double>(h_Bf[i]);
    }

    // Referencia CPU float
    auto cpuStart = std::chrono::high_resolution_clock::now();
    matrixMultCpuFloat(h_Af.data(), h_Bf.data(), h_C_cpu_f.data(), N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = cpuEnd - cpuStart;

    // Referencia CPU double usando la misma lógica
    // Para no duplicar código, convertimos resultado CPU float a double solo como referencia aproximada.
    // Si quieres una referencia CPU double exacta, se puede hacer otra función.
    for (int i = 0; i < numElements; ++i) {
        h_C_cpu_d[i] = static_cast<double>(h_C_cpu_f[i]);
    }

    // Device pointers float
    float *d_Af = nullptr, *d_Bf = nullptr, *d_Cf = nullptr;

    // Device pointers double
    double *d_Ad = nullptr, *d_Bd = nullptr, *d_Cd = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Af, sizeF));
    CHECK_CUDA(cudaMalloc(&d_Bf, sizeF));
    CHECK_CUDA(cudaMalloc(&d_Cf, sizeF));

    CHECK_CUDA(cudaMalloc(&d_Ad, sizeD));
    CHECK_CUDA(cudaMalloc(&d_Bd, sizeD));
    CHECK_CUDA(cudaMalloc(&d_Cd, sizeD));

    CHECK_CUDA(cudaMemcpy(d_Af, h_Af.data(), sizeF, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Bf, h_Bf.data(), sizeF, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_Ad, h_Ad.data(), sizeD, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Bd, h_Bd.data(), sizeD, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Eventos para medir SGEMM
    cudaEvent_t sStart, sStop;
    CHECK_CUDA(cudaEventCreate(&sStart));
    CHECK_CUDA(cudaEventCreate(&sStop));

    const float alphaF = 1.0f;
    const float betaF = 0.0f;

    CHECK_CUDA(cudaEventRecord(sStart));
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alphaF,
        d_Af, N,
        d_Bf, N,
        &betaF,
        d_Cf, N
    ));
    CHECK_CUDA(cudaEventRecord(sStop));
    CHECK_CUDA(cudaEventSynchronize(sStop));

    float sgemmTimeMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&sgemmTimeMs, sStart, sStop));

    CHECK_CUDA(cudaMemcpy(h_C_gpu_s.data(), d_Cf, sizeF, cudaMemcpyDeviceToHost));

    // Eventos para medir DGEMM
    cudaEvent_t dStart, dStop;
    CHECK_CUDA(cudaEventCreate(&dStart));
    CHECK_CUDA(cudaEventCreate(&dStop));

    const double alphaD = 1.0;
    const double betaD = 0.0;

    CHECK_CUDA(cudaEventRecord(dStart));
    CHECK_CUBLAS(cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alphaD,
        d_Ad, N,
        d_Bd, N,
        &betaD,
        d_Cd, N
    ));
    CHECK_CUDA(cudaEventRecord(dStop));
    CHECK_CUDA(cudaEventSynchronize(dStop));

    float dgemmTimeMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&dgemmTimeMs, dStart, dStop));

    CHECK_CUDA(cudaMemcpy(h_C_gpu_d.data(), d_Cd, sizeD, cudaMemcpyDeviceToHost));

    // Validación
    double diffS = maxAbsDiffFloat(h_C_cpu_f.data(), h_C_gpu_s.data(), numElements);
    double diffD = maxAbsDiffDouble(h_C_cpu_d.data(), h_C_gpu_d.data(), numElements);

    // FLOPs de GEMM: 2 * N^3
    double flops = 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    double sgemmGflops = flops / (sgemmTimeMs * 1e6);
    double dgemmGflops = flops / (dgemmTimeMs * 1e6);
    double cpuGflops   = flops / (cpuTime.count() * 1e6);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "N = " << N << "\n\n";

    std::cout << "CPU float GEMM time: " << cpuTime.count() << " ms\n";
    std::cout << "CPU float GEMM throughput: " << cpuGflops << " GFLOP/s\n\n";

    std::cout << "cuBLAS SGEMM time: " << sgemmTimeMs << " ms\n";
    std::cout << "cuBLAS SGEMM throughput: " << sgemmGflops << " GFLOP/s\n";
    std::cout << "Max abs diff CPU(float) vs SGEMM: " << diffS << "\n\n";

    std::cout << "cuBLAS DGEMM time: " << dgemmTimeMs << " ms\n";
    std::cout << "cuBLAS DGEMM throughput: " << dgemmGflops << " GFLOP/s\n";
    std::cout << "Max abs diff CPU(float->double ref aprox.) vs DGEMM: " << diffD << "\n\n";

    std::cout << "Speedup SGEMM vs CPU: " << cpuTime.count() / sgemmTimeMs << "x\n";
    std::cout << "Speedup DGEMM vs CPU: " << cpuTime.count() / dgemmTimeMs << "x\n";

    // Limpieza
    CHECK_CUDA(cudaEventDestroy(sStart));
    CHECK_CUDA(cudaEventDestroy(sStop));
    CHECK_CUDA(cudaEventDestroy(dStart));
    CHECK_CUDA(cudaEventDestroy(dStop));

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaFree(d_Af));
    CHECK_CUDA(cudaFree(d_Bf));
    CHECK_CUDA(cudaFree(d_Cf));

    CHECK_CUDA(cudaFree(d_Ad));
    CHECK_CUDA(cudaFree(d_Bd));
    CHECK_CUDA(cudaFree(d_Cd));

    return 0;
}
