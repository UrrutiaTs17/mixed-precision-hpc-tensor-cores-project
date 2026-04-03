// nvcc gemm_compare_balanced.cu -o gemm_compare_balanced -I/usr/include/openblas -lcublas -lopenblas
// ./gemm_compare_balanced --double --m 4096 --n 4096 --k 4096 --iters 30


#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cblas.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err__) << std::endl;       \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#define CHECK_CUBLAS(call)                                                        \
    do {                                                                          \
        cublasStatus_t status__ = (call);                                         \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__     \
                      << " -> status code " << static_cast<int>(status__)         \
                      << std::endl;                                               \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

struct Options {
    int m = 2048;
    int n = 2048;
    int k = 2048;
    int iters = 10;
    bool use_double = false;
};

struct Metrics {
    double milliseconds = 0.0;
    double gflops = 0.0;
};

void print_usage(const char* prog) {
    std::cout << "Uso: " << prog << " [--m M] [--n N] [--k K] [--iters I] [--double]\n";
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
            opt.m = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            opt.n = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            opt.k = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            opt.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--double") == 0) {
            opt.use_double = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Argumento no reconocido: " << argv[i] << "\n";
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return opt;
}

void print_gpu_info() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::cerr << "No se detectaron GPUs CUDA." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

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
    std::cout << "Bus de memoria             : " << prop.memoryBusWidth << " bits\n";
    std::cout << "Memoria compartida/bloque  : " << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
    std::cout << "===========================================================\n\n";
}

double gemm_operations(int m, int n, int k) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
}

void initialize_matrix_float(std::vector<float>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dist(gen);
    }
}

void initialize_matrix_double(std::vector<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dist(gen);
    }
}

double max_abs_diff_float(const std::vector<float>& a, const std::vector<float>& b) {
    double max_err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (diff > max_err) {
            max_err = diff;
        }
    }
    return max_err;
}

double max_abs_diff_double(const std::vector<double>& a, const std::vector<double>& b) {
    double max_err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::abs(a[i] - b[i]);
        if (diff > max_err) {
            max_err = diff;
        }
    }
    return max_err;
}

double rel_l2_error_float(const std::vector<float>& ref, const std::vector<float>& test) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double r = static_cast<double>(ref[i]);
        double t = static_cast<double>(test[i]);
        double d = r - t;
        num += d * d;
        den += r * r;
    }
    return std::sqrt(num) / (std::sqrt(den) + 1e-30);
}

double rel_l2_error_double(const std::vector<double>& ref, const std::vector<double>& test) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = ref[i] - test[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num) / (std::sqrt(den) + 1e-30);
}

Metrics run_cpu_blas_float(int m, int n, int k,
                           const std::vector<float>& A,
                           const std::vector<float>& B,
                           std::vector<float>& C,
                           int iters) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha,
                A.data(), m,
                B.data(), k,
                beta,
                C.data(), m);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha,
                    A.data(), m,
                    B.data(), k,
                    beta,
                    C.data(), m);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double gflops = gemm_operations(m, n, k) / (ms * 1e12);
    return {ms, gflops};
}

Metrics run_cpu_blas_double(int m, int n, int k,
                            const std::vector<double>& A,
                            const std::vector<double>& B,
                            std::vector<double>& C,
                            int iters) {
    const double alpha = 1.0;
    const double beta = 0.0;

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha,
                A.data(), m,
                B.data(), k,
                beta,
                C.data(), m);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha,
                    A.data(), m,
                    B.data(), k,
                    beta,
                    C.data(), m);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double gflops = gemm_operations(m, n, k) / (ms * 1e12);
    return {ms, gflops};
}

Metrics run_gpu_cublas_float(int m, int n, int k,
                             const std::vector<float>& A,
                             const std::vector<float>& B,
                             std::vector<float>& C,
                             int iters) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    size_t sizeA = static_cast<size_t>(m) * k * sizeof(float);
    size_t sizeB = static_cast<size_t>(k) * n * sizeof(float);
    size_t sizeC = static_cast<size_t>(m) * n * sizeof(float);

    CHECK_CUDA(cudaMalloc(&dA, sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeB));
    CHECK_CUDA(cudaMalloc(&dC, sizeC));

    CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeB, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             dA, m,
                             dB, k,
                             &beta,
                             dC, m));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, k,
                                 &alpha,
                                 dA, m,
                                 dB, k,
                                 &beta,
                                 dC, m));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(C.data(), dC, sizeC, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    double ms = static_cast<double>(total_ms) / iters;
    double gflops = gemm_operations(m, n, k) / (ms * 1e12);
    return {ms, gflops};
}

Metrics run_gpu_cublas_double(int m, int n, int k,
                              const std::vector<double>& A,
                              const std::vector<double>& B,
                              std::vector<double>& C,
                              int iters) {
    const double alpha = 1.0;
    const double beta = 0.0;

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    size_t sizeA = static_cast<size_t>(m) * k * sizeof(double);
    size_t sizeB = static_cast<size_t>(k) * n * sizeof(double);
    size_t sizeC = static_cast<size_t>(m) * n * sizeof(double);

    CHECK_CUDA(cudaMalloc(&dA, sizeA));
    CHECK_CUDA(cudaMalloc(&dB, sizeB));
    CHECK_CUDA(cudaMalloc(&dC, sizeC));

    CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeB, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             dA, m,
                             dB, k,
                             &beta,
                             dC, m));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, k,
                                 &alpha,
                                 dA, m,
                                 dB, k,
                                 &beta,
                                 dC, m));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(C.data(), dC, sizeC, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    double ms = static_cast<double>(total_ms) / iters;
    double gflops = gemm_operations(m, n, k) / (ms * 1e12);
    return {ms, gflops};
}

void print_experiment_header(const Options& opt, const char* precision_name) {
    std::cout << "Configuracion del experimento\n";
    std::cout << "Precision                 : " << precision_name << "\n";
    std::cout << "Dimensiones GEMM          : C(" << opt.m << "x" << opt.n << ") = A("
              << opt.m << "x" << opt.k << ") * B(" << opt.k << "x" << opt.n << ")\n";
    std::cout << "Iteraciones promedio      : " << opt.iters << "\n";
    std::cout << "Disposicion de memoria    : Column-major (igual para BLAS y cuBLAS)\n";
}

void run_experiment_float(const Options& opt) {
    int m = opt.m;
    int n = opt.n;
    int k = opt.k;

    std::vector<float> A(static_cast<size_t>(m) * k);
    std::vector<float> B(static_cast<size_t>(k) * n);
    std::vector<float> C_cpu(static_cast<size_t>(m) * n, 0.0f);
    std::vector<float> C_gpu(static_cast<size_t>(m) * n, 0.0f);

    initialize_matrix_float(A);
    initialize_matrix_float(B);
    print_experiment_header(opt, "FP32");

    Metrics cpu = run_cpu_blas_float(m, n, k, A, B, C_cpu, opt.iters);
    Metrics gpu = run_gpu_cublas_float(m, n, k, A, B, C_gpu, opt.iters);

    double max_err = max_abs_diff_float(C_cpu, C_gpu);
    double rel_err = rel_l2_error_float(C_cpu, C_gpu);

    std::cout << std::fixed << std::setprecision(7);
    std::cout << "---------------- RESULTADOS ----------------\n";
    std::cout << "CPU BLAS  - tiempo medio  : " << cpu.milliseconds << " ms\n";
    std::cout << "CPU BLAS  - rendimiento   : " << cpu.gflops << " TFLOP/s\n";
    std::cout << "GPU cuBLAS- tiempo medio  : " << gpu.milliseconds << " ms\n";
    std::cout << "GPU cuBLAS- rendimiento   : " << gpu.gflops << " TFLOP/s\n";
    std::cout << "Speedup GPU/CPU           : " << (cpu.milliseconds / gpu.milliseconds) << "x\n";
    std::cout << "Error max abs             : " << max_err << "\n";
    std::cout << "Error relativo L2         : " << rel_err << "\n";
    std::cout << "--------------------------------------------\n\n";
}

void run_experiment_double(const Options& opt) {
    int m = opt.m;
    int n = opt.n;
    int k = opt.k;

    std::vector<double> A(static_cast<size_t>(m) * k);
    std::vector<double> B(static_cast<size_t>(k) * n);
    std::vector<double> C_cpu(static_cast<size_t>(m) * n, 0.0);
    std::vector<double> C_gpu(static_cast<size_t>(m) * n, 0.0);

    initialize_matrix_double(A);
    initialize_matrix_double(B);
    print_experiment_header(opt, "FP64");

    Metrics cpu = run_cpu_blas_double(m, n, k, A, B, C_cpu, opt.iters);
    Metrics gpu = run_gpu_cublas_double(m, n, k, A, B, C_gpu, opt.iters);

    double max_err = max_abs_diff_double(C_cpu, C_gpu);
    double rel_err = rel_l2_error_double(C_cpu, C_gpu);

    std::cout << std::fixed << std::setprecision(7);
    std::cout << "---------------- RESULTADOS ----------------\n";
    std::cout << "CPU BLAS  - tiempo medio  : " << cpu.milliseconds << " ms\n";
    std::cout << "CPU BLAS  - rendimiento   : " << cpu.gflops << " TFLOP/s\n";
    std::cout << "GPU cuBLAS- tiempo medio  : " << gpu.milliseconds << " ms\n";
    std::cout << "GPU cuBLAS- rendimiento   : " << gpu.gflops << " TFLOP/s\n";
    std::cout << "Speedup GPU/CPU           : " << (cpu.milliseconds / gpu.milliseconds) << "x\n";
    std::cout << "Error max abs             : " << max_err << "\n";
    std::cout << "Error relativo L2         : " << rel_err << "\n";
    std::cout << "--------------------------------------------\n\n";
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    print_gpu_info();

    if (opt.use_double) {
        run_experiment_double(opt);
    } else {
        run_experiment_float(opt);
    }

    return 0;
}
