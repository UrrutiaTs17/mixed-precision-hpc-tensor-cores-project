#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <algorithm>

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

struct Stats {
    float avg_ms = 0.0f;
    float min_ms = 0.0f;
    float max_ms = 0.0f;
};

Stats computeStats(const std::vector<float>& times) {
    Stats s;
    if (times.empty()) return s;

    float sum = 0.0f;
    s.min_ms = std::numeric_limits<float>::max();
    s.max_ms = std::numeric_limits<float>::lowest();

    for (float t : times) {
        sum += t;
        s.min_ms = std::min(s.min_ms, t);
        s.max_ms = std::max(s.max_ms, t);
    }

    s.avg_ms = sum / static_cast<float>(times.size());
    return s;
}

// Multiplicación CPU en column-major:
// C = alpha * A * B + beta * C
void gemmCpuFloat(
    const float* A, const float* B, float* C,
    int N, float alpha, float beta
) {
    for (int col = 0; col < N; ++col) {
        for (int row = 0; row < N; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row + k * N] * B[k + col * N];
            }
            C[row + col * N] = alpha * sum + beta * C[row + col * N];
        }
    }
}

void gemmCpuDouble(
    const double* A, const double* B, double* C,
    int N, double alpha, double beta
) {
    for (int col = 0; col < N; ++col) {
        for (int row = 0; row < N; ++row) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[row + k * N] * B[k + col * N];
            }
            C[row + col * N] = alpha * sum + beta * C[row + col * N];
        }
    }
}

double maxAbsDiffFloat(const float* ref, const float* test, int size) {
    double maxDiff = 0.0;
    for (int i = 0; i < size; ++i) {
        double diff = std::abs(static_cast<double>(ref[i]) - static_cast<double>(test[i]));
        maxDiff = std::max(maxDiff, diff);
    }
    return maxDiff;
}

double maxAbsDiffDouble(const double* ref, const double* test, int size) {
    double maxDiff = 0.0;
    for (int i = 0; i < size; ++i) {
        double diff = std::abs(ref[i] - test[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    return maxDiff;
}

int main() {
    // -----------------------------
    // Parámetros del experimento
    // -----------------------------
    const int N = 1024;
    const int repetitions = 10;

    // Cambia estos para experimentar:
    const float  alphaF = 1.25f;
    const float  betaF  = 0.50f;

    const double alphaD = 1.25;
    const double betaD  = 0.50;

    const int numElements = N * N;
    const size_t sizeF = numElements * sizeof(float);
    const size_t sizeD = numElements * sizeof(double);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distA(0.0f, 1.0f);
    std::uniform_real_distribution<float> distB(0.0f, 1.0f);
    std::uniform_real_distribution<float> distC(0.0f, 1.0f);

    // -----------------------------
    // Host float
    // -----------------------------
    std::vector<float> h_Af(numElements);
    std::vector<float> h_Bf(numElements);
    std::vector<float> h_Cf_initial(numElements);

    std::vector<float> h_Cf_cpu(numElements);
    std::vector<float> h_Cf_gpu(numElements);

    // -----------------------------
    // Host double
    // -----------------------------
    std::vector<double> h_Ad(numElements);
    std::vector<double> h_Bd(numElements);
    std::vector<double> h_Cd_initial(numElements);

    std::vector<double> h_Cd_cpu(numElements);
    std::vector<double> h_Cd_gpu(numElements);

    // Inicialización
    for (int i = 0; i < numElements; ++i) {
        h_Af[i] = distA(rng);
        h_Bf[i] = distB(rng);
        h_Cf_initial[i] = distC(rng);

        h_Ad[i] = static_cast<double>(h_Af[i]);
        h_Bd[i] = static_cast<double>(h_Bf[i]);
        h_Cd_initial[i] = static_cast<double>(h_Cf_initial[i]);
    }

    // -----------------------------------
    // Referencia CPU FLOAT
    // -----------------------------------
    h_Cf_cpu = h_Cf_initial;
    auto cpuStartF = std::chrono::high_resolution_clock::now();
    gemmCpuFloat(h_Af.data(), h_Bf.data(), h_Cf_cpu.data(), N, alphaF, betaF);
    auto cpuEndF = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTimeF = cpuEndF - cpuStartF;

    // -----------------------------------
    // Referencia CPU DOUBLE
    // -----------------------------------
    h_Cd_cpu = h_Cd_initial;
    auto cpuStartD = std::chrono::high_resolution_clock::now();
    gemmCpuDouble(h_Ad.data(), h_Bd.data(), h_Cd_cpu.data(), N, alphaD, betaD);
    auto cpuEndD = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTimeD = cpuEndD - cpuStartD;

    // -----------------------------
    // Device pointers float
    // -----------------------------
    float *d_Af = nullptr, *d_Bf = nullptr, *d_Cf = nullptr;

    // -----------------------------
    // Device pointers double
    // -----------------------------
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

    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    std::vector<float> sgemmTimes;
    std::vector<float> dgemmTimes;
    sgemmTimes.reserve(repetitions);
    dgemmTimes.reserve(repetitions);

    // -----------------------------------
    // Warm-up SGEMM
    // -----------------------------------
    CHECK_CUDA(cudaMemcpy(d_Cf, h_Cf_initial.data(), sizeF, cudaMemcpyHostToDevice));
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
    CHECK_CUDA(cudaDeviceSynchronize());

    // -----------------------------------
    // Warm-up DGEMM
    // -----------------------------------
    CHECK_CUDA(cudaMemcpy(d_Cd, h_Cd_initial.data(), sizeD, cudaMemcpyHostToDevice));
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
    CHECK_CUDA(cudaDeviceSynchronize());

    // -----------------------------------
    // Benchmark SGEMM
    // -----------------------------------
    for (int rep = 0; rep < repetitions; ++rep) {
        // Restablecer C inicial en cada repetición
        CHECK_CUDA(cudaMemcpy(d_Cf, h_Cf_initial.data(), sizeF, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(startEvent));
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
        CHECK_CUDA(cudaEventRecord(stopEvent));
        CHECK_CUDA(cudaEventSynchronize(stopEvent));

        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        sgemmTimes.push_back(elapsedMs);
    }

    // Copiar último resultado SGEMM para validación
    CHECK_CUDA(cudaMemcpy(h_Cf_gpu.data(), d_Cf, sizeF, cudaMemcpyDeviceToHost));

    // -----------------------------------
    // Benchmark DGEMM
    // -----------------------------------
    for (int rep = 0; rep < repetitions; ++rep) {
        // Restablecer C inicial en cada repetición
        CHECK_CUDA(cudaMemcpy(d_Cd, h_Cd_initial.data(), sizeD, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaEventRecord(startEvent));
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
        CHECK_CUDA(cudaEventRecord(stopEvent));
        CHECK_CUDA(cudaEventSynchronize(stopEvent));

        float elapsedMs = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        dgemmTimes.push_back(elapsedMs);
    }

    // Copiar último resultado DGEMM para validación
    CHECK_CUDA(cudaMemcpy(h_Cd_gpu.data(), d_Cd, sizeD, cudaMemcpyDeviceToHost));

    // -----------------------------------
    // Estadísticas
    // -----------------------------------
    Stats sStats = computeStats(sgemmTimes);
    Stats dStats = computeStats(dgemmTimes);

    // FLOPs de GEMM general: 
    // multiplicación + suma interna = 2*N^3
    // y alpha/beta añaden trabajo adicional pequeño, pero
    // normalmente se reporta GEMM como 2*N^3 FLOPs
    const double flops = 2.0 * static_cast<double>(N) * N * N;

    const double cpuGFLOPSf = flops / (cpuTimeF.count() * 1e6);
    const double cpuGFLOPSd = flops / (cpuTimeD.count() * 1e6);

    const double sgemmGFLOPSavg = flops / (sStats.avg_ms * 1e6);
    const double dgemmGFLOPSavg = flops / (dStats.avg_ms * 1e6);

    // Validación
    const double diffS = maxAbsDiffFloat(h_Cf_cpu.data(), h_Cf_gpu.data(), numElements);
    const double diffD = maxAbsDiffDouble(h_Cd_cpu.data(), h_Cd_gpu.data(), numElements);

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "========================================\n";
    std::cout << "Benchmark GEMM con cuBLAS\n";
    std::cout << "========================================\n";
    std::cout << "N = " << N << "\n";
    std::cout << "Repeticiones = " << repetitions << "\n\n";

    std::cout << "Parametros SGEMM: alpha = " << alphaF << ", beta = " << betaF << "\n";
    std::cout << "Parametros DGEMM: alpha = " << alphaD << ", beta = " << betaD << "\n\n";

    std::cout << "----------- CPU -----------\n";
    std::cout << "CPU float time   = " << cpuTimeF.count() << " ms\n";
    std::cout << "CPU float GFLOPS = " << cpuGFLOPSf << "\n";
    std::cout << "CPU double time  = " << cpuTimeD.count() << " ms\n";
    std::cout << "CPU double GFLOPS= " << cpuGFLOPSd << "\n\n";

    std::cout << "---------- SGEMM ----------\n";
    std::cout << "Tiempo promedio = " << sStats.avg_ms << " ms\n";
    std::cout << "Tiempo minimo   = " << sStats.min_ms << " ms\n";
    std::cout << "Tiempo maximo   = " << sStats.max_ms << " ms\n";
    std::cout << "Rendimiento     = " << sgemmGFLOPSavg << " GFLOPS\n";
    std::cout << "Max abs diff    = " << diffS << "\n";
    std::cout << "Speedup vs CPU float = " << (cpuTimeF.count() / sStats.avg_ms) << "x\n\n";

    std::cout << "---------- DGEMM ----------\n";
    std::cout << "Tiempo promedio = " << dStats.avg_ms << " ms\n";
    std::cout << "Tiempo minimo   = " << dStats.min_ms << " ms\n";
    std::cout << "Tiempo maximo   = " << dStats.max_ms << " ms\n";
    std::cout << "Rendimiento     = " << dgemmGFLOPSavg << " GFLOPS\n";
    std::cout << "Max abs diff    = " << diffD << "\n";
    std::cout << "Speedup vs CPU double = " << (cpuTimeD.count() / dStats.avg_ms) << "x\n\n";

    std::cout << "Tiempos individuales SGEMM:\n";
    for (int i = 0; i < repetitions; ++i) {
        std::cout << "  Rep " << i + 1 << ": " << sgemmTimes[i] << " ms\n";
    }

    std::cout << "\nTiempos individuales DGEMM:\n";
    for (int i = 0; i < repetitions; ++i) {
        std::cout << "  Rep " << i + 1 << ": " << dgemmTimes[i] << " ms\n";
    }

    // Limpieza
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaFree(d_Af));
    CHECK_CUDA(cudaFree(d_Bf));
    CHECK_CUDA(cudaFree(d_Cf));

    CHECK_CUDA(cudaFree(d_Ad));
    CHECK_CUDA(cudaFree(d_Bd));
    CHECK_CUDA(cudaFree(d_Cd));

    return 0;
}
