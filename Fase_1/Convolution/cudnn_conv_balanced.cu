#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <algorithm>

#include <cblas.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CHECK_CUDNN(call)                                                     \
    do {                                                                      \
        cudnnStatus_t status = (call);                                        \
        if (status != CUDNN_STATUS_SUCCESS) {                                 \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(status)       \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

struct ConvConfig {
    int N = 1;
    int C = 3;
    int H = 1024;
    int W = 1024;
    int K = 16;
    int R = 5;
    int S = 5;
    int pad_h = 1;
    int pad_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int iters = 10;
    bool use_double = false;
};

struct ConvOutputDims {
    int outN;
    int outC;
    int outH;
    int outW;
};

struct Metrics {
    double cpu_ms = 0.0;
    double gpu_ms = 0.0;
    double cpu_gflops = 0.0;
    double gpu_gflops = 0.0;
    double speedup = 0.0;
    double max_abs_error = 0.0;
    double rel_l2_error = 0.0;
};

void print_usage(const char* prog) {
    std::cout << "Uso: " << prog << " [opciones]\n"
              << "  --double            usar FP64 (por defecto FP32)\n"
              << "  --n <int>           batch size\n"
              << "  --c <int>           canales de entrada\n"
              << "  --h <int>           alto de entrada\n"
              << "  --w <int>           ancho de entrada\n"
              << "  --k <int>           canales de salida / filtros\n"
              << "  --r <int>           alto del filtro\n"
              << "  --s <int>           ancho del filtro\n"
              << "  --pad_h <int>       padding vertical\n"
              << "  --pad_w <int>       padding horizontal\n"
              << "  --stride_h <int>    stride vertical\n"
              << "  --stride_w <int>    stride horizontal\n"
              << "  --dilation_h <int>  dilatacion vertical\n"
              << "  --dilation_w <int>  dilatacion horizontal\n"
              << "  --iters <int>       iteraciones para promedio\n";
}

int get_int_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Falta valor para " << argv[i] << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return std::atoi(argv[++i]);
}

ConvConfig parse_args(int argc, char** argv) {
    ConvConfig cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--double") == 0) cfg.use_double = true;
        else if (std::strcmp(argv[i], "--n") == 0) cfg.N = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--c") == 0) cfg.C = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--h") == 0) cfg.H = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--w") == 0) cfg.W = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--k") == 0) cfg.K = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--r") == 0) cfg.R = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--s") == 0) cfg.S = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--pad_h") == 0) cfg.pad_h = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--pad_w") == 0) cfg.pad_w = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--stride_h") == 0) cfg.stride_h = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--stride_w") == 0) cfg.stride_w = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--dilation_h") == 0) cfg.dilation_h = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--dilation_w") == 0) cfg.dilation_w = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--iters") == 0) cfg.iters = get_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Argumento no reconocido: " << argv[i] << std::endl;
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return cfg;
}

ConvOutputDims get_output_dims(const ConvConfig& cfg) {
    ConvOutputDims d;
    d.outN = cfg.N;
    d.outC = cfg.K;
    d.outH = (cfg.H + 2 * cfg.pad_h - cfg.dilation_h * (cfg.R - 1) - 1) / cfg.stride_h + 1;
    d.outW = (cfg.W + 2 * cfg.pad_w - cfg.dilation_w * (cfg.S - 1) - 1) / cfg.stride_w + 1;
    if (d.outH <= 0 || d.outW <= 0) {
        std::cerr << "Dimensiones de salida invalidas. Revisa padding/stride/dilation/filtro." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return d;
}

void print_gpu_info() {
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    int gpuClockKHz = 0;
    int memClockKHz = 0;
    int memBusWidth = 0;

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
    if (err1 == cudaSuccess) std::cout << "Reloj GPU                  : " << gpuClockKHz / 1000.0 << " MHz\n";
    if (err2 == cudaSuccess) std::cout << "Reloj memoria              : " << memClockKHz / 1000.0 << " MHz\n";
    if (err3 == cudaSuccess) std::cout << "Bus de memoria             : " << memBusWidth << " bits\n";
    std::cout << "Memoria compartida/bloque  : " << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
    std::cout << "===========================================================\n\n";
}

double conv_flop_count(const ConvConfig& cfg, const ConvOutputDims& d) {
    return 2.0 * static_cast<double>(cfg.N) * static_cast<double>(d.outH) *
           static_cast<double>(d.outW) * static_cast<double>(cfg.K) *
           static_cast<double>(cfg.C) * static_cast<double>(cfg.R) *
           static_cast<double>(cfg.S);
}

void initialize_vector_float(std::vector<float>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < v.size(); ++i) v[i] = dist(rng);
}

void initialize_vector_double(std::vector<double>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < v.size(); ++i) v[i] = dist(rng);
}

inline size_t x_index(int n, int c, int h, int w, int C, int H, int W) {
    return static_cast<size_t>(((n * C + c) * H + h) * W + w);
}

inline size_t y_index(int n, int k, int h, int w, int K, int outH, int outW) {
    return static_cast<size_t>(((n * K + k) * outH + h) * outW + w);
}

void im2col_float_single_image(const float* x, float* col, const ConvConfig& cfg, const ConvOutputDims& d) {
    const int rows = cfg.C * cfg.R * cfg.S;
    //const int cols = d.outH * d.outW;
    (void)rows;
    for (int c = 0; c < cfg.C; ++c) {
        for (int r = 0; r < cfg.R; ++r) {
            for (int s = 0; s < cfg.S; ++s) {
                const int row = (c * cfg.R + r) * cfg.S + s;
                for (int oh = 0; oh < d.outH; ++oh) {
                    for (int ow = 0; ow < d.outW; ++ow) {
                        const int ih = oh * cfg.stride_h - cfg.pad_h + r * cfg.dilation_h;
                        const int iw = ow * cfg.stride_w - cfg.pad_w + s * cfg.dilation_w;
                        const int col_idx = oh * d.outW + ow;
                        if (ih >= 0 && ih < cfg.H && iw >= 0 && iw < cfg.W) {
                            col[row + col_idx * (cfg.C * cfg.R * cfg.S)] = x[(c * cfg.H + ih) * cfg.W + iw];
                        } else {
                            col[row + col_idx * (cfg.C * cfg.R * cfg.S)] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void im2col_double_single_image(const double* x, double* col, const ConvConfig& cfg, const ConvOutputDims& d) {
    for (int c = 0; c < cfg.C; ++c) {
        for (int r = 0; r < cfg.R; ++r) {
            for (int s = 0; s < cfg.S; ++s) {
                const int row = (c * cfg.R + r) * cfg.S + s;
                for (int oh = 0; oh < d.outH; ++oh) {
                    for (int ow = 0; ow < d.outW; ++ow) {
                        const int ih = oh * cfg.stride_h - cfg.pad_h + r * cfg.dilation_h;
                        const int iw = ow * cfg.stride_w - cfg.pad_w + s * cfg.dilation_w;
                        const int col_idx = oh * d.outW + ow;
                        if (ih >= 0 && ih < cfg.H && iw >= 0 && iw < cfg.W) {
                            col[row + col_idx * (cfg.C * cfg.R * cfg.S)] = x[(c * cfg.H + ih) * cfg.W + iw];
                        } else {
                            col[row + col_idx * (cfg.C * cfg.R * cfg.S)] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

double run_cpu_conv_openblas_float(const ConvConfig& cfg, const ConvOutputDims& d,
                                   const std::vector<float>& x,
                                   const std::vector<float>& w,
                                   std::vector<float>& y) {
    const int M = cfg.K;
    const int Ncol = d.outH * d.outW;
    const int Kcol = cfg.C * cfg.R * cfg.S;
    std::vector<float> col(static_cast<size_t>(Kcol) * Ncol);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < cfg.iters; ++it) {
        for (int n = 0; n < cfg.N; ++n) {
            im2col_float_single_image(&x[static_cast<size_t>(n) * cfg.C * cfg.H * cfg.W], col.data(), cfg, d);
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        M, Ncol, Kcol,
                        alpha,
                        w.data(), M,
                        col.data(), Kcol,
                        beta,
                        &y[static_cast<size_t>(n) * cfg.K * d.outH * d.outW], M);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / cfg.iters;
}

double run_cpu_conv_openblas_double(const ConvConfig& cfg, const ConvOutputDims& d,
                                    const std::vector<double>& x,
                                    const std::vector<double>& w,
                                    std::vector<double>& y) {
    const int M = cfg.K;
    const int Ncol = d.outH * d.outW;
    const int Kcol = cfg.C * cfg.R * cfg.S;
    std::vector<double> col(static_cast<size_t>(Kcol) * Ncol);
    const double alpha = 1.0;
    const double beta = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < cfg.iters; ++it) {
        for (int n = 0; n < cfg.N; ++n) {
            im2col_double_single_image(&x[static_cast<size_t>(n) * cfg.C * cfg.H * cfg.W], col.data(), cfg, d);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        M, Ncol, Kcol,
                        alpha,
                        w.data(), M,
                        col.data(), Kcol,
                        beta,
                        &y[static_cast<size_t>(n) * cfg.K * d.outH * d.outW], M);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / cfg.iters;
}

double run_gpu_cudnn_float(const ConvConfig& cfg, const ConvOutputDims& d,
                           const std::vector<float>& x,
                           const std::vector<float>& w,
                           std::vector<float>& y) {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreate(&handle));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           cfg.N, cfg.C, cfg.H, cfg.W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           cfg.K, cfg.C, cfg.R, cfg.S));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                cfg.pad_h, cfg.pad_w,
                                                cfg.stride_h, cfg.stride_w,
                                                cfg.dilation_h, cfg.dilation_w,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           d.outN, d.outC, d.outH, d.outW));

    float* d_x = nullptr;
    float* d_w = nullptr;
    float* d_y = nullptr;
    const size_t xBytes = x.size() * sizeof(float);
    const size_t wBytes = w.size() * sizeof(float);
    const size_t yBytes = y.size() * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_x, xBytes));
    CHECK_CUDA(cudaMalloc(&d_w, wBytes));
    CHECK_CUDA(cudaMalloc(&d_y, yBytes));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), xBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w.data(), wBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y, 0, yBytes));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspaceBytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc,
                                                        algo, &workspaceBytes));
    void* d_workspace = nullptr;
    if (workspaceBytes > 0) CHECK_CUDA(cudaMalloc(&d_workspace, workspaceBytes));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w,
                                        convDesc, algo, d_workspace, workspaceBytes,
                                        &beta, yDesc, d_y));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));
    CHECK_CUDA(cudaEventRecord(startEvent));
    for (int it = 0; it < cfg.iters; ++it) {
        CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_workspace, workspaceBytes,
                                            &beta, yDesc, d_y));
    }
    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float totalMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(y.data(), d_y, yBytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    if (d_workspace) CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroy(handle));

    return static_cast<double>(totalMs) / cfg.iters;
}

double run_gpu_cudnn_double(const ConvConfig& cfg, const ConvOutputDims& d,
                            const std::vector<double>& x,
                            const std::vector<double>& w,
                            std::vector<double>& y) {
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreate(&handle));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                                           cfg.N, cfg.C, cfg.H, cfg.W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW,
                                           cfg.K, cfg.C, cfg.R, cfg.S));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                cfg.pad_h, cfg.pad_w,
                                                cfg.stride_h, cfg.stride_w,
                                                cfg.dilation_h, cfg.dilation_w,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_DOUBLE));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                                           d.outN, d.outC, d.outH, d.outW));

    double* d_x = nullptr;
    double* d_w = nullptr;
    double* d_y = nullptr;
    const size_t xBytes = x.size() * sizeof(double);
    const size_t wBytes = w.size() * sizeof(double);
    const size_t yBytes = y.size() * sizeof(double);
    CHECK_CUDA(cudaMalloc(&d_x, xBytes));
    CHECK_CUDA(cudaMalloc(&d_w, wBytes));
    CHECK_CUDA(cudaMalloc(&d_y, yBytes));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), xBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w.data(), wBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y, 0, yBytes));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspaceBytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc,
                                                        algo, &workspaceBytes));
    void* d_workspace = nullptr;
    if (workspaceBytes > 0) CHECK_CUDA(cudaMalloc(&d_workspace, workspaceBytes));

    const double alpha = 1.0;
    const double beta = 0.0;

    CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w,
                                        convDesc, algo, d_workspace, workspaceBytes,
                                        &beta, yDesc, d_y));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));
    CHECK_CUDA(cudaEventRecord(startEvent));
    for (int it = 0; it < cfg.iters; ++it) {
        CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_workspace, workspaceBytes,
                                            &beta, yDesc, d_y));
    }
    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float totalMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, startEvent, stopEvent));
    CHECK_CUDA(cudaMemcpy(y.data(), d_y, yBytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    if (d_workspace) CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroy(handle));

    return static_cast<double>(totalMs) / cfg.iters;
}

void compute_error_float(const std::vector<float>& ref, const std::vector<float>& test,
                         double& max_abs_error, double& rel_l2_error) {
    double max_err = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = static_cast<double>(ref[i]) - static_cast<double>(test[i]);
        double abs_diff = std::fabs(diff);
        if (abs_diff > max_err) max_err = abs_diff;
        sum_sq_diff += diff * diff;
        double r = static_cast<double>(ref[i]);
        sum_sq_ref += r * r;
    }
    max_abs_error = max_err;
    rel_l2_error = std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_ref) + 1e-30);
}

void compute_error_double(const std::vector<double>& ref, const std::vector<double>& test,
                          double& max_abs_error, double& rel_l2_error) {
    double max_err = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = ref[i] - test[i];
        double abs_diff = std::fabs(diff);
        if (abs_diff > max_err) max_err = abs_diff;
        sum_sq_diff += diff * diff;
        sum_sq_ref += ref[i] * ref[i];
    }
    max_abs_error = max_err;
    rel_l2_error = std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_ref) + 1e-300);
}

Metrics run_experiment_float(const ConvConfig& cfg, const ConvOutputDims& d) {
    const size_t xCount = static_cast<size_t>(cfg.N) * cfg.C * cfg.H * cfg.W;
    const size_t wCount = static_cast<size_t>(cfg.K) * cfg.C * cfg.R * cfg.S;
    const size_t yCount = static_cast<size_t>(d.outN) * d.outC * d.outH * d.outW;

    std::vector<float> x(xCount), w(wCount), y_cpu(yCount, 0.0f), y_gpu(yCount, 0.0f);
    initialize_vector_float(x, 42u);
    initialize_vector_float(w, 1337u);

    Metrics m;
    m.cpu_ms = run_cpu_conv_openblas_float(cfg, d, x, w, y_cpu);
    m.gpu_ms = run_gpu_cudnn_float(cfg, d, x, w, y_gpu);

    const double flops = conv_flop_count(cfg, d);
    m.cpu_gflops = flops / (m.cpu_ms * 1e6);
    m.gpu_gflops = flops / (m.gpu_ms * 1e6);
    m.speedup = m.cpu_ms / m.gpu_ms;
    compute_error_float(y_cpu, y_gpu, m.max_abs_error, m.rel_l2_error);
    return m;
}

Metrics run_experiment_double(const ConvConfig& cfg, const ConvOutputDims& d) {
    const size_t xCount = static_cast<size_t>(cfg.N) * cfg.C * cfg.H * cfg.W;
    const size_t wCount = static_cast<size_t>(cfg.K) * cfg.C * cfg.R * cfg.S;
    const size_t yCount = static_cast<size_t>(d.outN) * d.outC * d.outH * d.outW;

    std::vector<double> x(xCount), w(wCount), y_cpu(yCount, 0.0), y_gpu(yCount, 0.0);
    initialize_vector_double(x, 42u);
    initialize_vector_double(w, 1337u);

    Metrics m;
    m.cpu_ms = run_cpu_conv_openblas_double(cfg, d, x, w, y_cpu);
    m.gpu_ms = run_gpu_cudnn_double(cfg, d, x, w, y_gpu);

    const double flops = conv_flop_count(cfg, d);
    m.cpu_gflops = flops / (m.cpu_ms * 1e6);
    m.gpu_gflops = flops / (m.gpu_ms * 1e6);
    m.speedup = m.cpu_ms / m.gpu_ms;
    compute_error_double(y_cpu, y_gpu, m.max_abs_error, m.rel_l2_error);
    return m;
}

void print_config(const ConvConfig& cfg, const ConvOutputDims& d) {
    std::cout << "Configuracion del experimento\n";
    std::cout << "Precision                 : " << (cfg.use_double ? "FP64" : "FP32") << "\n";
    std::cout << "Entrada                   : N=" << cfg.N << ", C=" << cfg.C
              << ", H=" << cfg.H << ", W=" << cfg.W << "\n";
    std::cout << "Filtro                    : K=" << cfg.K << ", C=" << cfg.C
              << ", R=" << cfg.R << ", S=" << cfg.S << "\n";
    std::cout << "Salida                    : N=" << d.outN << ", C=" << d.outC
              << ", H=" << d.outH << ", W=" << d.outW << "\n";
}

void print_results(const Metrics& m) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "---------------- RESULTADOS ----------------\n";
    std::cout << "CPU OpenBLAS - tiempo medio : " << m.cpu_ms << " ms\n";
    std::cout << "CPU OpenBLAS - rendimiento  : " << m.cpu_gflops << " GFLOP/s ("
              << m.cpu_gflops / 1000.0 << " TFLOP/s)\n";
    std::cout << "GPU cuDNN - tiempo medio    : " << m.gpu_ms << " ms\n";
    std::cout << "GPU cuDNN - rendimiento     : " << m.gpu_gflops << " GFLOP/s ("
              << m.gpu_gflops / 1000.0 << " TFLOP/s)\n";
    std::cout << "Speedup GPU/CPU             : " << m.speedup << "x\n";
    std::cout << "Error max abs               : " << m.max_abs_error << "\n";
    std::cout << "Error relativo L2           : " << m.rel_l2_error << "\n";
    std::cout << "--------------------------------------------\n";
}

int main(int argc, char** argv) {
    ConvConfig cfg = parse_args(argc, argv);
    ConvOutputDims d = get_output_dims(cfg);

    print_gpu_info();
    print_config(cfg, d);

    Metrics metrics = cfg.use_double ? run_experiment_double(cfg, d)
                                     : run_experiment_float(cfg, d);
    print_results(metrics);
    return 0;
}
