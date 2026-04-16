#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cblas.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                  << " -> status code " << status << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

struct Options {
    int m = 2048;
    int n = 2048;
    int k = 2048;
    int iters = 20;
    bool use_double = false;
};

struct Metrics {
    double ms = 0.0;
    double gflops = 0.0;
    double tflops = 0.0;
};

struct ErrorMetrics {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
};

static void print_usage(const char* prog) {
    std::cout << "Uso:\n"
              << "  " << prog << " [--m M] [--n N] [--k K] [--iters I] [--double]\n\n"
              << "Descripcion:\n"
              << "  Compara CPU BLAS, GPU cuBLAS clasico y GPU cuBLAS con Tensor Cores\n"
              << "  para GEMM. El modo Tensor Core usa entradas FP16 y acumulacion/salida FP32.\n\n"
              << "Ejemplos:\n"
              << "  " << prog << "\n"
              << "  " << prog << " --m 4096 --n 4096 --k 4096 --iters 10\n"
              << "  " << prog << " --double --m 2048 --n 2048 --k 2048 --iters 5\n";
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) opt.m = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) opt.n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) opt.k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) opt.iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--double") == 0) opt.use_double = true;
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Argumento no reconocido: " << argv[i] << "\n\n";
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0 || opt.iters <= 0) {
        std::cerr << "Todos los parametros deben ser positivos." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return opt;
}

static void print_gpu_info() {
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    int gpuClockKHz = 0, memClockKHz = 0, memBusWidth = 0;
    cudaError_t err1 = cudaDeviceGetAttribute(&gpuClockKHz, cudaDevAttrClockRate, dev);
    cudaError_t err2 = cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, dev);
    cudaError_t err3 = cudaDeviceGetAttribute(&memBusWidth, cudaDevAttrGlobalMemoryBusWidth, dev);

    std::cout << "================ CARACTERISTICAS DE LA GPU ================\n";
    std::cout << "Dispositivo activo         : " << dev << "\n";
    std::cout << "Nombre                     : " << prop.name << "\n";
    std::cout << "Compute Capability         : " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memoria global             : "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)
              << " GiB\n";
    std::cout << "SMs                        : " << prop.multiProcessorCount << "\n";
    std::cout << "Max hilos por bloque       : " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Warp size                  : " << prop.warpSize << "\n";
    std::cout << "Reloj GPU                  : " << (err1 == cudaSuccess ? gpuClockKHz / 1000.0 : 0.0) << (err1 == cudaSuccess ? " MHz\n" : " no disponible\n");
    std::cout << "Reloj memoria              : " << (err2 == cudaSuccess ? memClockKHz / 1000.0 : 0.0) << (err2 == cudaSuccess ? " MHz\n" : " no disponible\n");
    std::cout << "Bus de memoria             : " << (err3 == cudaSuccess ? std::to_string(memBusWidth) + " bits" : "no disponible") << "\n";
    std::cout << "Memoria compartida/bloque  : " << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
    std::cout << "===========================================================\n\n";
}

static double gemm_flops_standard(int m, int n, int k) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
}

static void initialize_matrix_float(std::vector<float>& mat) {
    for (size_t i = 0; i < mat.size(); ++i) mat[i] = static_cast<float>((i % 101) - 50) / 25.0f;
}

static void initialize_matrix_double(std::vector<double>& mat) {
    for (size_t i = 0; i < mat.size(); ++i) mat[i] = static_cast<double>((i % 101) - 50) / 25.0;
}

static Metrics run_cpu_blas_float(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int m, int n, int k, int iters) {
    const float alpha = 1.0f, beta = 0.0f;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m);
    }
    auto end = std::chrono::high_resolution_clock::now();
    Metrics out;
    out.ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    out.gflops = gemm_flops_standard(m, n, k) / (out.ms * 1e6);
    out.tflops = out.gflops / 1000.0;
    return out;
}

static Metrics run_cpu_blas_double(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int m, int n, int k, int iters) {
    const double alpha = 1.0, beta = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m);
    }
    auto end = std::chrono::high_resolution_clock::now();
    Metrics out;
    out.ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    out.gflops = gemm_flops_standard(m, n, k) / (out.ms * 1e6);
    out.tflops = out.gflops / 1000.0;
    return out;
}

static Metrics run_gpu_cublas_float(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int m, int n, int k, int iters) {
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * A.size()));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * B.size()));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * C.size()));
    CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeof(float) * B.size(), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < 3; ++i)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(C.data(), dC, sizeof(float) * C.size(), cudaMemcpyDeviceToHost));

    Metrics out;
    out.ms = total_ms / iters;
    out.gflops = gemm_flops_standard(m, n, k) / (out.ms * 1e6);
    out.tflops = out.gflops / 1000.0;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return out;
}

static Metrics run_gpu_cublas_double(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int m, int n, int k, int iters) {
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(double) * A.size()));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(double) * B.size()));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(double) * C.size()));
    CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const double alpha = 1.0, beta = 0.0;

    for (int i = 0; i < 3; ++i)
        CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, m, dB, k, &beta, dC, m));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(C.data(), dC, sizeof(double) * C.size(), cudaMemcpyDeviceToHost));

    Metrics out;
    out.ms = total_ms / iters;
    out.gflops = gemm_flops_standard(m, n, k) / (out.ms * 1e6);
    out.tflops = out.gflops / 1000.0;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return out;
}

__global__ static void convert_float_to_half_kernel(const float* src, __half* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dst[idx] = __float2half(src[idx]);
}

static Metrics run_gpu_tensor_cores_fp16_accum_fp32(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int m, int n, int k, int iters) {
    float *dA_fp32 = nullptr, *dB_fp32 = nullptr, *dC = nullptr;
    __half *dA_fp16 = nullptr, *dB_fp16 = nullptr;

    CHECK_CUDA(cudaMalloc(&dA_fp32, sizeof(float) * A.size()));
    CHECK_CUDA(cudaMalloc(&dB_fp32, sizeof(float) * B.size()));
    CHECK_CUDA(cudaMalloc(&dA_fp16, sizeof(__half) * A.size()));
    CHECK_CUDA(cudaMalloc(&dB_fp16, sizeof(__half) * B.size()));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * C.size()));

    CHECK_CUDA(cudaMemcpy(dA_fp32, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_fp32, B.data(), sizeof(float) * B.size(), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocksA = (A.size() + threads - 1) / threads;
    int blocksB = (B.size() + threads - 1) / threads;
    convert_float_to_half_kernel<<<blocksA, threads>>>(dA_fp32, dA_fp16, (int)A.size());
    convert_float_to_half_kernel<<<blocksB, threads>>>(dB_fp32, dB_fp16, (int)B.size());
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < 3; ++i)
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  &alpha,
                                  dA_fp16, CUDA_R_16F, m,
                                  dB_fp16, CUDA_R_16F, k,
                                  &beta,
                                  dC, CUDA_R_32F, m,
                                  CUBLAS_COMPUTE_32F_FAST_16F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  &alpha,
                                  dA_fp16, CUDA_R_16F, m,
                                  dB_fp16, CUDA_R_16F, k,
                                  &beta,
                                  dC, CUDA_R_32F, m,
                                  CUBLAS_COMPUTE_32F_FAST_16F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(C.data(), dC, sizeof(float) * C.size(), cudaMemcpyDeviceToHost));

    Metrics out;
    out.ms = total_ms / iters;
    out.gflops = gemm_flops_standard(m, n, k) / (out.ms * 1e6);
    out.tflops = out.gflops / 1000.0;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA_fp32));
    CHECK_CUDA(cudaFree(dB_fp32));
    CHECK_CUDA(cudaFree(dA_fp16));
    CHECK_CUDA(cudaFree(dB_fp16));
    CHECK_CUDA(cudaFree(dC));
    return out;
}

static ErrorMetrics compare_float_vectors(const std::vector<float>& ref, const std::vector<float>& test) {
    ErrorMetrics out;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = (double)ref[i] - (double)test[i];
        out.max_abs = std::max(out.max_abs, std::abs(diff));
        num += diff * diff;
        den += (double)ref[i] * (double)ref[i];
    }
    out.rel_l2 = den > 0.0 ? std::sqrt(num / den) : 0.0;
    return out;
}

static ErrorMetrics compare_double_vectors(const std::vector<double>& ref, const std::vector<double>& test) {
    ErrorMetrics out;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = ref[i] - test[i];
        out.max_abs = std::max(out.max_abs, std::abs(diff));
        num += diff * diff;
        den += ref[i] * ref[i];
    }
    out.rel_l2 = den > 0.0 ? std::sqrt(num / den) : 0.0;
    return out;
}

static void print_header(const Options& opt) {
    std::cout << "================ CONFIGURACION DEL EXPERIMENTO ================\n";
    std::cout << "M                         : " << opt.m << "\n";
    std::cout << "N                         : " << opt.n << "\n";
    std::cout << "K                         : " << opt.k << "\n";
    std::cout << "Iteraciones               : " << opt.iters << "\n";
    std::cout << "Precision base            : " << (opt.use_double ? "FP64" : "FP32") << "\n";
    std::cout << "Tensor Core path          : " << (opt.use_double ? "no ejecutado en modo --double" : "FP16 entradas + acumulacion/salida FP32") << "\n";
    std::cout << "Conteo FLOPs usado        : " << std::scientific << std::setprecision(6) << gemm_flops_standard(opt.m, opt.n, opt.k) << "\n";
    std::cout << std::fixed;
    if ((opt.m % 8 != 0 || opt.n % 8 != 0 || opt.k % 8 != 0) && !opt.use_double)
        std::cout << "Aviso                     : para FP16 en Tensor Cores conviene usar M,N,K multiplos de 8\n";
    std::cout << "===============================================================\n\n";
}

static void run_experiment_float(const Options& opt) {
    std::vector<float> A((size_t)opt.m * opt.k), B((size_t)opt.k * opt.n), C_cpu((size_t)opt.m * opt.n, 0.0f), C_gpu((size_t)opt.m * opt.n, 0.0f), C_tc((size_t)opt.m * opt.n, 0.0f);
    initialize_matrix_float(A);
    initialize_matrix_float(B);

    Metrics cpu = run_cpu_blas_float(A, B, C_cpu, opt.m, opt.n, opt.k, opt.iters);
    Metrics gpu = run_gpu_cublas_float(A, B, C_gpu, opt.m, opt.n, opt.k, opt.iters);
    Metrics tc = run_gpu_tensor_cores_fp16_accum_fp32(A, B, C_tc, opt.m, opt.n, opt.k, opt.iters);

    ErrorMetrics err_gpu = compare_float_vectors(C_cpu, C_gpu);
    ErrorMetrics err_tc = compare_float_vectors(C_cpu, C_tc);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================ RESULTADOS GEMM FP32 =================\n";
    std::cout << "CPU BLAS  - tiempo medio   : " << cpu.ms << " ms\n";
    std::cout << "CPU BLAS  - rendimiento    : " << cpu.gflops << " GFLOP/s (" << cpu.tflops << " TFLOP/s)\n\n";
    std::cout << "GPU cuBLAS clasico - tiempo: " << gpu.ms << " ms\n";
    std::cout << "GPU cuBLAS clasico - rend. : " << gpu.gflops << " GFLOP/s (" << gpu.tflops << " TFLOP/s)\n";
    std::cout << "Speedup GPU/CPU           : " << cpu.ms / gpu.ms << "x\n";
    std::cout << "Error max abs vs CPU      : " << err_gpu.max_abs << "\n";
    std::cout << "Error relativo L2 vs CPU  : " << err_gpu.rel_l2 << "\n\n";
    std::cout << "GPU Tensor Core - tiempo   : " << tc.ms << " ms\n";
    std::cout << "GPU Tensor Core - rend.    : " << tc.gflops << " GFLOP/s (" << tc.tflops << " TFLOP/s)\n";
    std::cout << "Speedup TC/CPU            : " << cpu.ms / tc.ms << "x\n";
    std::cout << "Speedup TC/GPU clasico    : " << gpu.ms / tc.ms << "x\n";
    std::cout << "Error max abs vs CPU      : " << err_tc.max_abs << "\n";
    std::cout << "Error relativo L2 vs CPU  : " << err_tc.rel_l2 << "\n";
    std::cout << "=======================================================\n";
}

static void run_experiment_double(const Options& opt) {
    std::vector<double> A((size_t)opt.m * opt.k), B((size_t)opt.k * opt.n), C_cpu((size_t)opt.m * opt.n, 0.0), C_gpu((size_t)opt.m * opt.n, 0.0);
    initialize_matrix_double(A);
    initialize_matrix_double(B);

    Metrics cpu = run_cpu_blas_double(A, B, C_cpu, opt.m, opt.n, opt.k, opt.iters);
    Metrics gpu = run_gpu_cublas_double(A, B, C_gpu, opt.m, opt.n, opt.k, opt.iters);
    ErrorMetrics err_gpu = compare_double_vectors(C_cpu, C_gpu);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================ RESULTADOS GEMM FP64 =================\n";
    std::cout << "CPU BLAS  - tiempo medio   : " << cpu.ms << " ms\n";
    std::cout << "CPU BLAS  - rendimiento    : " << cpu.gflops << " GFLOP/s (" << cpu.tflops << " TFLOP/s)\n\n";
    std::cout << "GPU cuBLAS clasico - tiempo: " << gpu.ms << " ms\n";
    std::cout << "GPU cuBLAS clasico - rend. : " << gpu.gflops << " GFLOP/s (" << gpu.tflops << " TFLOP/s)\n";
    std::cout << "Speedup GPU/CPU           : " << cpu.ms / gpu.ms << "x\n";
    std::cout << "Error max abs vs CPU      : " << err_gpu.max_abs << "\n";
    std::cout << "Error relativo L2 vs CPU  : " << err_gpu.rel_l2 << "\n\n";
    std::cout << "Tensor Cores              : no se ejecutan en esta implementacion con --double.\n";
    std::cout << "Motivo                    : esta ruta usa FP16 + acumulacion FP32 mediante cublasGemmEx.\n";
    std::cout << "=======================================================\n";
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    print_gpu_info();
    print_header(opt);
    if (opt.use_double) run_experiment_double(opt);
    else run_experiment_float(opt);
    return 0;
}
