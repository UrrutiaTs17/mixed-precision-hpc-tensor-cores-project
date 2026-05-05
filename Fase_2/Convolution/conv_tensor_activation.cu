// Compilar con:
// nvcc -std=c++17 conv_tensor_activation.cu -o conv_tc \
//      -I/usr/include/openblas \
//      -lcudnn -lopenblas \
//      -gencode arch=compute_86,code=sm_86
//
// Este programa compara tres rutas de convolucion 2D hacia adelante:
// 1. CPU con OpenBLAS (via transformacion im2col + SGEMM/DGEMM).
// 2. GPU con cuDNN FP32 clasico (sin Tensor Cores).
// 3. GPU con cuDNN y Tensor Cores (entradas FP16, acumulacion y salida FP32).
//
// Con --double: unicamente se ejecutan las rutas CPU FP64 y cuDNN FP64,
// ya que la ruta Tensor Core opera con FP16 de entrada.
//
// Las matrices de activacion y filtros se almacenan en formato NCHW,
// convencion usada tanto por cuDNN como por el im2col de referencia en CPU.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cblas.h>
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

// Valida llamadas a la API de CUDA y termina si ocurre un error.
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

// Valida llamadas a cuDNN y termina si la biblioteca reporta fallo.
#define CHECK_CUDNN(call) do { \
    cudnnStatus_t status = (call); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

namespace {

constexpr int kWarmupIters       = 3;
constexpr int kConversionThreads = 256;

// Dimensiones del fragmento WMMA para FP16 (unicas soportadas en sm >= 7.0).
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// 2x2 warps por bloque → 4 warps = 128 hilos; tile de salida 32×32.
constexpr int kBlockWarpsM = 2;
constexpr int kBlockWarpsN = 2;
constexpr int kBlockTileM  = kBlockWarpsM * kWmmaM;  // 32
constexpr int kBlockTileN  = kBlockWarpsN * kWmmaN;  // 32

// Padding en shared memory para evitar bank conflicts (16 bytes extra por fila).
constexpr int kWmmaShmemPad = 8;

// Parametros configurables desde linea de comandos.
// Convencion de nombres: N=batch, C=canales entrada, H/W=alto/ancho,
// K=filtros (canales salida), R/S=alto/ancho del filtro.
struct Options {
    int N          = 1;
    int C          = 32;
    int H          = 128;
    int W          = 128;
    int K          = 64;
    int R          = 3;
    int S          = 3;
    int pad_h      = 1;
    int pad_w      = 1;
    int stride_h   = 1;
    int stride_w   = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int iters      = 20;
    bool use_double = false;
};

// Dimensiones de la salida: derivadas de la formula estandar de convolucion.
struct OutputDims {
    int outN, outC, outH, outW;
};

// Metricas de rendimiento promedio para una ruta de convolucion.
struct Metrics {
    double ms     = 0.0;
    double gflops = 0.0;
    double tflops = 0.0;
};

// Metricas de error numerico frente a la referencia en CPU.
struct ErrorMetrics {
    double max_abs = 0.0;
    double rel_l2  = 0.0;
};

// Envueltura RAII para el handle de cuDNN.
// El handle encapsula el estado interno de la biblioteca (streams, cache, etc).
class CudnnHandle {
public:
    CudnnHandle()  { CHECK_CUDNN(cudnnCreate(&handle_)); }
    ~CudnnHandle() { if (handle_) CHECK_CUDNN(cudnnDestroy(handle_)); }

    CudnnHandle(const CudnnHandle&)            = delete;
    CudnnHandle& operator=(const CudnnHandle&) = delete;

    cudnnHandle_t get() const { return handle_; }

private:
    cudnnHandle_t handle_ = nullptr;
};

// Temporizador basado en eventos CUDA para medir tiempo en la GPU.
class CudaEventTimer {
public:
    CudaEventTimer() {
        CHECK_CUDA(cudaEventCreate(&start_));
        CHECK_CUDA(cudaEventCreate(&stop_));
    }

    ~CudaEventTimer() {
        CHECK_CUDA(cudaEventDestroy(start_));
        CHECK_CUDA(cudaEventDestroy(stop_));
    }

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start() { CHECK_CUDA(cudaEventRecord(start_)); }

    float stop_and_elapsed_ms() {
        CHECK_CUDA(cudaEventRecord(stop_));
        CHECK_CUDA(cudaEventSynchronize(stop_));
        float elapsed = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start_, stop_));
        return elapsed;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_  = nullptr;
};

// =========================================================================
// Utilidades de linea de comandos
// =========================================================================

static void print_usage(const char* prog) {
    std::cout
        << "Uso:\n"
        << "  " << prog << " [--N N] [--C C] [--H H] [--W W] [--K K] [--R R] [--S S]\n"
        << "             [--pad_h P] [--pad_w P] [--stride_h S] [--stride_w S]\n"
        << "             [--dilation_h D] [--dilation_w D] [--iters I] [--double]\n\n"
        << "Descripcion:\n"
        << "  Compara cuatro rutas de convolucion 2D hacia adelante:\n"
        << "    1. CPU im2col + OpenBLAS (FP32/FP64)\n"
        << "    2. GPU cuDNN clasico (FP32/FP64, sin Tensor Cores)\n"
        << "    3. GPU cuDNN con Tensor Cores (FP16 entrada, FP32 acumulacion)\n"
        << "    4. GPU im2col FP16 + kernel WMMA custom (Tensor Cores directos)\n"
        << "  La ruta WMMA (4) requiere K multiplo de 32, outH*outW multiplo de 32\n"
        << "  y C*R*S multiplo de 16. Si no se cumple se omite con aviso.\n"
        << "  Con --double solo se ejecutan las rutas 1 y 2.\n\n"
        << "Ejemplos:\n"
        << "  " << prog << "\n"
        << "  " << prog << " --N 1 --C 64 --H 224 --W 224 --K 64 --R 3 --S 3 --iters 10\n"
        << "  " << prog << " --double --N 1 --C 16 --H 64 --W 64 --K 32 --R 3 --S 3\n";
}

static int parse_int_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Falta valor para " << argv[i] << "\n";
        std::exit(EXIT_FAILURE);
    }
    return std::atoi(argv[++i]);
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--N")          == 0) opt.N          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--C")          == 0) opt.C          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--H")          == 0) opt.H          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--W")          == 0) opt.W          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--K")          == 0) opt.K          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--R")          == 0) opt.R          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--S")          == 0) opt.S          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--pad_h")      == 0) opt.pad_h      = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--pad_w")      == 0) opt.pad_w      = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--stride_h")   == 0) opt.stride_h   = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--stride_w")   == 0) opt.stride_w   = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--dilation_h") == 0) opt.dilation_h = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--dilation_w") == 0) opt.dilation_w = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--iters")      == 0) opt.iters      = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--double")     == 0) opt.use_double = true;
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Argumento no reconocido: " << argv[i] << "\n\n";
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    if (opt.N <= 0 || opt.C <= 0 || opt.H <= 0 || opt.W <= 0 ||
        opt.K <= 0 || opt.R <= 0 || opt.S <= 0 || opt.iters <= 0) {
        std::cerr << "Todos los parametros deben ser positivos.\n";
        std::exit(EXIT_FAILURE);
    }
    return opt;
}

// =========================================================================
// Utilidades de dimension y rendimiento
// =========================================================================

static OutputDims compute_output_dims(const Options& opt) {
    OutputDims d;
    d.outN = opt.N;
    d.outC = opt.K;
    d.outH = (opt.H + 2 * opt.pad_h - opt.dilation_h * (opt.R - 1) - 1) / opt.stride_h + 1;
    d.outW = (opt.W + 2 * opt.pad_w - opt.dilation_w * (opt.S - 1) - 1) / opt.stride_w + 1;
    if (d.outH <= 0 || d.outW <= 0) {
        std::cerr << "Dimensiones de salida invalidas. Revisa padding/stride/dilation/filtro.\n";
        std::exit(EXIT_FAILURE);
    }
    return d;
}

// Un MAC (multiply-accumulate) = 1 mul + 1 add = 2 operaciones de punto flotante.
// Cada posicion de salida (n, k, oh, ow) requiere C*R*S MACs.
static double conv_flops(const Options& opt, const OutputDims& d) {
    return 2.0
        * static_cast<double>(opt.N)
        * static_cast<double>(opt.K)
        * static_cast<double>(d.outH)
        * static_cast<double>(d.outW)
        * static_cast<double>(opt.C)
        * static_cast<double>(opt.R)
        * static_cast<double>(opt.S);
}

static Metrics build_metrics(const Options& opt, const OutputDims& d, double avg_ms) {
    Metrics m;
    m.ms     = avg_ms;
    m.gflops = conv_flops(opt, d) / (m.ms * 1e6);
    m.tflops = m.gflops / 1000.0;
    return m;
}

// =========================================================================
// Inicializacion de datos
// =========================================================================

// Valores deterministicos acotados: evita depender de semilla aleatoria
// y facilita reproducir los experimentos exactamente.
static void initialize_matrix_float(std::vector<float>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        const int c = static_cast<int>(i % 101) - 50;
        v[i] = static_cast<float>(c) / 25.0f;
    }
}

static void initialize_matrix_double(std::vector<double>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        const int c = static_cast<int>(i % 101) - 50;
        v[i] = static_cast<double>(c) / 25.0;
    }
}

// =========================================================================
// Metricas de error numerico
// =========================================================================

static ErrorMetrics compare_float_vectors(const std::vector<float>& ref,
                                          const std::vector<float>& test) {
    ErrorMetrics out;
    double sq_err = 0.0, sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double diff = static_cast<double>(ref[i]) - static_cast<double>(test[i]);
        out.max_abs = std::max(out.max_abs, std::abs(diff));
        sq_err += diff * diff;
        sq_ref += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    out.rel_l2 = sq_ref > 0.0 ? std::sqrt(sq_err / sq_ref) : 0.0;
    return out;
}

static ErrorMetrics compare_double_vectors(const std::vector<double>& ref,
                                           const std::vector<double>& test) {
    ErrorMetrics out;
    double sq_err = 0.0, sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double diff = ref[i] - test[i];
        out.max_abs = std::max(out.max_abs, std::abs(diff));
        sq_err += diff * diff;
        sq_ref += ref[i] * ref[i];
    }
    out.rel_l2 = sq_ref > 0.0 ? std::sqrt(sq_err / sq_ref) : 0.0;
    return out;
}

// =========================================================================
// Impresion de resultados
// =========================================================================

static void print_gpu_info() {
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    int gpu_clock_khz = 0, mem_clock_khz = 0, mem_bus_width = 0;
    cudaError_t e1 = cudaDeviceGetAttribute(&gpu_clock_khz, cudaDevAttrClockRate, dev);
    cudaError_t e2 = cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaError_t e3 = cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth, dev);

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
    std::cout << "Reloj GPU                  : "
              << (e1 == cudaSuccess ? gpu_clock_khz / 1000.0 : 0.0)
              << (e1 == cudaSuccess ? " MHz\n" : " no disponible\n");
    std::cout << "Reloj memoria              : "
              << (e2 == cudaSuccess ? mem_clock_khz / 1000.0 : 0.0)
              << (e2 == cudaSuccess ? " MHz\n" : " no disponible\n");
    std::cout << "Bus de memoria             : "
              << (e3 == cudaSuccess ? std::to_string(mem_bus_width) + " bits" : "no disponible")
              << "\n";
    std::cout << "Memoria compartida/bloque  : " << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
    std::cout << "===========================================================\n\n";
}

static void print_reference_comparison(const char* label,
                                       const Metrics& m,
                                       double ref_ms,
                                       const ErrorMetrics& e) {
    std::cout << label << " - tiempo         : " << m.ms << " ms\n";
    std::cout << label << " - rendimiento    : " << m.gflops << " GFLOP/s ("
              << m.tflops << " TFLOP/s)\n";
    std::cout << "Speedup vs CPU             : " << ref_ms / m.ms << "x\n";
    std::cout << "Error max abs vs CPU       : " << e.max_abs << "\n";
    std::cout << "Error relativo L2 vs CPU   : " << e.rel_l2 << "\n\n";
}

// =========================================================================
// Ruta 1 - CPU: im2col + OpenBLAS
//
// La convolucion se reescribe como una multiplicacion de matrices:
//   Y[K, outH*outW] = W[K, C*R*S] * col[C*R*S, outH*outW]
//
// "col" se construye con im2col: cada columna contiene los C*R*S elementos
// de la ventana receptiva correspondiente a una posicion de salida (oh, ow).
// Los ceros por fuera del borde (padding) se insertan explicitamente.
// =========================================================================

static void im2col_float(const float* x, float* col,
                         const Options& opt, const OutputDims& d) {
    const int stride_col = opt.C * opt.R * opt.S;
    for (int c = 0; c < opt.C; ++c) {
        for (int r = 0; r < opt.R; ++r) {
            for (int s = 0; s < opt.S; ++s) {
                const int row = (c * opt.R + r) * opt.S + s;
                for (int oh = 0; oh < d.outH; ++oh) {
                    for (int ow = 0; ow < d.outW; ++ow) {
                        const int ih = oh * opt.stride_h - opt.pad_h + r * opt.dilation_h;
                        const int iw = ow * opt.stride_w - opt.pad_w + s * opt.dilation_w;
                        const int col_idx = oh * d.outW + ow;
                        col[row + col_idx * stride_col] =
                            (ih >= 0 && ih < opt.H && iw >= 0 && iw < opt.W)
                            ? x[(c * opt.H + ih) * opt.W + iw]
                            : 0.0f;
                    }
                }
            }
        }
    }
}

static void im2col_double(const double* x, double* col,
                          const Options& opt, const OutputDims& d) {
    const int stride_col = opt.C * opt.R * opt.S;
    for (int c = 0; c < opt.C; ++c) {
        for (int r = 0; r < opt.R; ++r) {
            for (int s = 0; s < opt.S; ++s) {
                const int row = (c * opt.R + r) * opt.S + s;
                for (int oh = 0; oh < d.outH; ++oh) {
                    for (int ow = 0; ow < d.outW; ++ow) {
                        const int ih = oh * opt.stride_h - opt.pad_h + r * opt.dilation_h;
                        const int iw = ow * opt.stride_w - opt.pad_w + s * opt.dilation_w;
                        const int col_idx = oh * d.outW + ow;
                        col[row + col_idx * stride_col] =
                            (ih >= 0 && ih < opt.H && iw >= 0 && iw < opt.W)
                            ? x[(c * opt.H + ih) * opt.W + iw]
                            : 0.0;
                    }
                }
            }
        }
    }
}

static Metrics benchmark_cpu_float(const std::vector<float>& x,
                                   const std::vector<float>& w,
                                   std::vector<float>& y,
                                   const Options& opt,
                                   const OutputDims& d) {
    const int M    = opt.K;
    const int Ncol = d.outH * d.outW;
    const int Kcol = opt.C * opt.R * opt.S;
    std::vector<float> col(static_cast<size_t>(Kcol) * Ncol);
    const float alpha = 1.0f, beta = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < opt.iters; ++it) {
        for (int n = 0; n < opt.N; ++n) {
            const float* xn = x.data() + static_cast<size_t>(n) * opt.C * opt.H * opt.W;
            float*       yn = y.data() + static_cast<size_t>(n) * opt.K * d.outH * d.outW;
            im2col_float(xn, col.data(), opt, d);
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        M, Ncol, Kcol,
                        alpha, w.data(), M,
                        col.data(), Kcol,
                        beta, yn, M);
        }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / opt.iters;
    return build_metrics(opt, d, avg_ms);
}

static Metrics benchmark_cpu_double(const std::vector<double>& x,
                                    const std::vector<double>& w,
                                    std::vector<double>& y,
                                    const Options& opt,
                                    const OutputDims& d) {
    const int M    = opt.K;
    const int Ncol = d.outH * d.outW;
    const int Kcol = opt.C * opt.R * opt.S;
    std::vector<double> col(static_cast<size_t>(Kcol) * Ncol);
    const double alpha = 1.0, beta = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < opt.iters; ++it) {
        for (int n = 0; n < opt.N; ++n) {
            const double* xn = x.data() + static_cast<size_t>(n) * opt.C * opt.H * opt.W;
            double*       yn = y.data() + static_cast<size_t>(n) * opt.K * d.outH * d.outW;
            im2col_double(xn, col.data(), opt, d);
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        M, Ncol, Kcol,
                        alpha, w.data(), M,
                        col.data(), Kcol,
                        beta, yn, M);
        }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / opt.iters;
    return build_metrics(opt, d, avg_ms);
}

// =========================================================================
// Ruta 2 - GPU cuDNN FP32 clasico (sin Tensor Cores)
// =========================================================================

static Metrics benchmark_gpu_cudnn_float(const std::vector<float>& x,
                                         const std::vector<float>& w,
                                         std::vector<float>& y,
                                         const Options& opt,
                                         const OutputDims& d) {
    cudnnTensorDescriptor_t    xDesc, yDesc;
    cudnnFilterDescriptor_t    wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // Todos los tensores en FP32, formato NCHW.
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           opt.N, opt.C, opt.H, opt.W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           opt.K, opt.C, opt.R, opt.S));
    // El tipo de computo (computeType) determina la precision de la acumulacion interna.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                opt.pad_h, opt.pad_w,
                                                opt.stride_h, opt.stride_w,
                                                opt.dilation_h, opt.dilation_w,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           d.outN, d.outC, d.outH, d.outW));

    CudnnHandle handle;

    // Seleccion automatica del mejor algoritmo disponible con los descriptores dados.
    cudnnConvolutionFwdAlgoPerf_t perf_results[8];
    int algo_count = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle.get(), xDesc, wDesc, convDesc, yDesc, 8, &algo_count, perf_results));
    const cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;

    // El workspace es memoria temporal en GPU que algunos algoritmos necesitan.
    size_t ws_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle.get(), xDesc, wDesc, convDesc, yDesc, algo, &ws_bytes));

    float *d_x, *d_w, *d_y;
    void*  d_ws = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, w.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(float)));
    if (ws_bytes > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_bytes));

    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < kWarmupIters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < opt.iters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y));
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    CHECK_CUDA(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    if (d_ws) CHECK_CUDA(cudaFree(d_ws));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

    return build_metrics(opt, d, static_cast<double>(total_ms) / opt.iters);
}

// =========================================================================
// Ruta 2b - GPU cuDNN FP64 (solo para --double)
// =========================================================================

static Metrics benchmark_gpu_cudnn_double(const std::vector<double>& x,
                                          const std::vector<double>& w,
                                          std::vector<double>& y,
                                          const Options& opt,
                                          const OutputDims& d) {
    cudnnTensorDescriptor_t    xDesc, yDesc;
    cudnnFilterDescriptor_t    wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                                           opt.N, opt.C, opt.H, opt.W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW,
                                           opt.K, opt.C, opt.R, opt.S));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                opt.pad_h, opt.pad_w,
                                                opt.stride_h, opt.stride_w,
                                                opt.dilation_h, opt.dilation_w,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_DOUBLE));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                                           d.outN, d.outC, d.outH, d.outW));

    CudnnHandle handle;

    cudnnConvolutionFwdAlgoPerf_t perf_results[8];
    int algo_count = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle.get(), xDesc, wDesc, convDesc, yDesc, 8, &algo_count, perf_results));
    const cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;

    size_t ws_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle.get(), xDesc, wDesc, convDesc, yDesc, algo, &ws_bytes));

    double *d_x, *d_w, *d_y;
    void* d_ws = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_w, w.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(double)));
    if (ws_bytes > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_bytes));

    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w.data(), w.size() * sizeof(double), cudaMemcpyHostToDevice));

    const double alpha = 1.0, beta = 0.0;

    for (int i = 0; i < kWarmupIters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < opt.iters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y));
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    CHECK_CUDA(cudaMemcpy(y.data(), d_y, y.size() * sizeof(double), cudaMemcpyDeviceToHost));

    if (d_ws) CHECK_CUDA(cudaFree(d_ws));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

    return build_metrics(opt, d, static_cast<double>(total_ms) / opt.iters);
}

// =========================================================================
// Kernel de conversion FP32 -> FP16 en GPU (mismo patron que en GEMM)
// =========================================================================

__global__ static void convert_float_to_half_kernel(const float* src, __half* dst, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Convierte un buffer FP16 a FP32 en la GPU elemento a elemento.
__global__ static void convert_half_to_float_kernel(const __half* src, float* dst, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __half2float(src[idx]);
    }
}

// Sube un vector a GPU como FP32, lo convierte a FP16 en el device y devuelve
// el puntero FP16. El buffer FP32 intermedio se libera antes de retornar.
static __half* upload_and_convert_to_half(const std::vector<float>& host_data) {
    const size_t n = host_data.size();
    float*  d_fp32 = nullptr;
    __half* d_fp16 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_fp32, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fp16, n * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_fp32, host_data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    const int blocks = static_cast<int>((n + kConversionThreads - 1) / kConversionThreads);
    convert_float_to_half_kernel<<<blocks, kConversionThreads>>>(d_fp32, d_fp16, static_cast<int>(n));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_fp32));
    return d_fp16;
}

// =========================================================================
// Ruta 3 - GPU cuDNN con Tensor Cores: FP16 in / FP32 acumulacion y salida
//
// Tres pasos obligatorios para activar Tensor Cores en cuDNN:
//
//  1. Descriptores de entrada (x) y filtro (w) con CUDNN_DATA_HALF
//     -> los operandos de la multiplicacion son de 16 bits.
//
//  2. Tipo de computo CUDNN_DATA_FLOAT en cudnnSetConvolution2dDescriptor
//     -> la acumulacion de productos parciales se realiza en 32 bits,
//        lo que evita desbordamiento y mantiene la precision numerica util.
//
//  3. cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH)
//     -> le comunica a cuDNN que puede usar las unidades Tensor Core
//        para esta convolucion. Sin esta llamada cuDNN puede ignorarlas.
// =========================================================================

static Metrics benchmark_gpu_tensor_cores_conv(const std::vector<float>& x,
                                               const std::vector<float>& w,
                                               std::vector<float>& y,
                                               const Options& opt,
                                               const OutputDims& d) {
    // Paso 1: preparar entradas FP16 en GPU.
    __half* d_x_fp16 = upload_and_convert_to_half(x);
    __half* d_w_fp16 = upload_and_convert_to_half(w);

    // Paso 2: crear descriptores con tipos mixtos.
    cudnnTensorDescriptor_t    xDesc, yDesc;
    cudnnFilterDescriptor_t    wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // Entrada y filtro en FP16.
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           opt.N, opt.C, opt.H, opt.W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
                                           opt.K, opt.C, opt.R, opt.S));
    // computeType = FLOAT: acumula productos FP16 en un acumulador de 32 bits.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                opt.pad_h, opt.pad_w,
                                                opt.stride_h, opt.stride_w,
                                                opt.dilation_h, opt.dilation_w,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    // Paso 3: activar Tensor Cores explicitamente en el descriptor.
    CHECK_CUDNN(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

    // En cuDNN 9, FP16 entrada / FP32 salida no es compatible con NCHW.
    // Se usa FP16 para la salida y se convierte a FP32 despues de la convolucion.
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           d.outN, d.outC, d.outH, d.outW));

    CudnnHandle handle;

    // Con CUDNN_TENSOR_OP_MATH activo, cuDNN priorizara algoritmos compatibles con TC
    // al evaluar las opciones mediante cudnnGetConvolutionForwardAlgorithm_v7.
    cudnnConvolutionFwdAlgoPerf_t perf_results[8];
    int algo_count = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle.get(), xDesc, wDesc, convDesc, yDesc, 8, &algo_count, perf_results));

    // Algunos algoritmos devueltos pueden tener status != SUCCESS para FP16;
    // se elige el primero que cuDNN declara ejecutable.
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    for (int ai = 0; ai < algo_count; ++ai) {
        if (perf_results[ai].status == CUDNN_STATUS_SUCCESS) {
            algo = perf_results[ai].algo;
            break;
        }
    }

    size_t ws_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle.get(), xDesc, wDesc, convDesc, yDesc, algo, &ws_bytes));

    // Buffer de salida en FP16 (cuDNN 9 requiere FP16 out con FP16 in en NCHW).
    __half* d_y_fp16 = nullptr;
    float*  d_y_fp32 = nullptr;
    void*   d_ws     = nullptr;
    CHECK_CUDA(cudaMalloc(&d_y_fp16, y.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_y_fp32, y.size() * sizeof(float)));
    if (ws_bytes > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_bytes));

    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < kWarmupIters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x_fp16, wDesc, d_w_fp16,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y_fp16));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < opt.iters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x_fp16, wDesc, d_w_fp16,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y_fp16));
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    // Convertir salida FP16 → FP32 fuera del intervalo medido.
    const int y_count = static_cast<int>(y.size());
    const int conv_blocks = (y_count + kConversionThreads - 1) / kConversionThreads;
    convert_half_to_float_kernel<<<conv_blocks, kConversionThreads>>>(
        d_y_fp16, d_y_fp32, y_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y.data(), d_y_fp32, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    if (d_ws) CHECK_CUDA(cudaFree(d_ws));
    CHECK_CUDA(cudaFree(d_y_fp32));
    CHECK_CUDA(cudaFree(d_y_fp16));
    CHECK_CUDA(cudaFree(d_x_fp16));
    CHECK_CUDA(cudaFree(d_w_fp16));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

    return build_metrics(opt, d, static_cast<double>(total_ms) / opt.iters);
}

// =========================================================================
// Ruta 4 - im2col en GPU (FP16) + kernel WMMA
//
// La convolucion 2D se descompone en dos pasos:
//   1. im2col_fp16_kernel: transforma la entrada [C,H,W] en una matriz
//      col[C*R*S, outH*outW] FP16 row-major. Cada columna de col contiene
//      los C*R*S valores de la ventana receptiva de una posicion de salida.
//
//   2. wmma_gemm_kernel: calcula Y[K, outH*outW] = W[K, C*R*S] * col row-major
//      usando la API WMMA (Tensor Cores) de la misma forma que en GEMM.
//
// Layout:
//   W NCHW [K,C,R,S] visto como [K, C*R*S] es ya row-major → conversion
//   FP32→FP16 elemento a elemento sin transposicion.
//   Y[K, Ncol] row-major = NCHW [K, outH, outW] para N=1 → comparable
//   directamente con la salida de cuDNN.
// =========================================================================

// Construye la matriz col en FP16 row-major a partir de un batch element FP32.
// Thread id cubre todos los C*R*S × outH*outW elementos de col.
// Coalescencia: hilos consecutivos comparten el mismo crs y tienen pos
// consecutivos (mismo oh, ow consecutivo → iw consecutivo con stride_w=1),
// por lo que leen x a lo largo de la dimension W → acceso coalescente.
__global__ static void im2col_fp16_kernel(
        const float* __restrict__ x,
        __half*      __restrict__ col,
        int C, int H, int W,
        int R, int S,
        int pad_h,  int pad_w,
        int stride_h, int stride_w,
        int dilation_h, int dilation_w,
        int outH, int outW) {
    const int Ncol = outH * outW;
    const int id   = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= C * R * S * Ncol) return;

    const int crs = id / Ncol;
    const int pos = id % Ncol;
    const int c   = crs / (R * S);
    const int rs  = crs % (R * S);
    const int r   = rs  / S;
    const int s   = rs  % S;
    const int oh  = pos / outW;
    const int ow  = pos % outW;
    const int ih  = oh * stride_h - pad_h + r * dilation_h;
    const int iw  = ow * stride_w - pad_w + s * dilation_w;

    col[id] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
              ? __float2half(x[(c * H + ih) * W + iw])
              : __float2half(0.0f);
}

// Kernel GEMM con API WMMA: C(M,N) = A(M,K) * B(K,N), todos row-major FP16→FP32.
// Identico al de la ruta GEMM: bloque 128 hilos = 4 warps en grid 2x2 de fragmentos,
// shared memory con padding anti-bank-conflict, carga coalescente desde global memory.
__global__ static void wmma_gemm_kernel(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        float*        __restrict__ C,
        int M, int N, int K) {
    using namespace nvcuda;

    __shared__ __half sA[kBlockTileM][kWmmaK + kWmmaShmemPad];   // [32][24]
    __shared__ __half sB[kWmmaK][kBlockTileN + kWmmaShmemPad];   // [16][40]

    const int warp_id  = threadIdx.x / 32;
    const int warp_row = warp_id / kBlockWarpsN;
    const int warp_col = warp_id % kBlockWarpsN;

    const int block_row     = blockIdx.x * kBlockTileM;
    const int block_col     = blockIdx.y * kBlockTileN;
    const int warp_row_base = block_row + warp_row * kWmmaM;
    const int warp_col_base = block_col + warp_col * kWmmaN;

    wmma::fragment<wmma::matrix_a,    kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>                   c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_base = 0; k_base < K; k_base += kWmmaK) {
        // Carga coalescente del tile de A (32 x kWmmaK) a shared memory.
        for (int i = threadIdx.x; i < kBlockTileM * kWmmaK; i += blockDim.x) {
            const int row = i / kWmmaK;
            const int col = i % kWmmaK;
            sA[row][col] = (block_row + row < M && k_base + col < K)
                           ? A[(block_row + row) * K + (k_base + col)]
                           : __float2half(0.0f);
        }
        // Carga coalescente del tile de B (kWmmaK x 32) a shared memory.
        for (int i = threadIdx.x; i < kWmmaK * kBlockTileN; i += blockDim.x) {
            const int row = i / kBlockTileN;
            const int col = i % kBlockTileN;
            sB[row][col] = (k_base + row < K && block_col + col < N)
                           ? B[(k_base + row) * N + (block_col + col)]
                           : __float2half(0.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag,
            reinterpret_cast<const __half*>(&sA[warp_row * kWmmaM][0]),
            kWmmaK + kWmmaShmemPad);
        wmma::load_matrix_sync(b_frag,
            reinterpret_cast<const __half*>(&sB[0][warp_col * kWmmaN]),
            kBlockTileN + kWmmaShmemPad);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    if (warp_row_base < M && warp_col_base < N) {
        wmma::store_matrix_sync(
            C + warp_row_base * N + warp_col_base,
            c_frag, N,
            wmma::mem_row_major);
    }
}

// Benchmark de la ruta WMMA para convolucion.
// Convierte W a FP16 una sola vez; en cada iteracion lanza im2col_fp16_kernel
// seguido de wmma_gemm_kernel para cada elemento del batch.
// La salida Y queda en NCHW FP32 (mismo layout que cuDNN), comparable directamente.
static Metrics benchmark_gpu_wmma_conv(const std::vector<float>& x,
                                        const std::vector<float>& w,
                                        std::vector<float>& y,
                                        const Options& opt,
                                        const OutputDims& d) {
    const int M    = opt.K;
    const int Ncol = d.outH * d.outW;
    const int Kcol = opt.C * opt.R * opt.S;

    if (M % kBlockTileM != 0 || Ncol % kBlockTileN != 0 || Kcol % kWmmaK != 0) {
        std::cerr << "WMMA conv omitida: K=" << M << " (req. mult. de " << kBlockTileM
                  << "), outH*outW=" << Ncol << " (req. mult. de " << kBlockTileN
                  << "), C*R*S=" << Kcol << " (req. mult. de " << kWmmaK << ").\n";
        return Metrics{};
    }

    // Convertir filtros W: NCHW [K,C,R,S] = row-major [K, C*R*S] → FP16.
    const int w_count = static_cast<int>(w.size());
    float*  d_w_fp32 = nullptr;
    __half* d_w_fp16 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_w_fp32, static_cast<size_t>(w_count) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w_fp16, static_cast<size_t>(w_count) * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_w_fp32, w.data(),
                          static_cast<size_t>(w_count) * sizeof(float),
                          cudaMemcpyHostToDevice));
    const int bw = (w_count + kConversionThreads - 1) / kConversionThreads;
    convert_float_to_half_kernel<<<bw, kConversionThreads>>>(d_w_fp32, d_w_fp16, w_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_w_fp32));

    // Subir entrada X al GPU (usada por im2col cada iteracion).
    const int x_count = static_cast<int>(x.size());
    float* d_x = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, static_cast<size_t>(x_count) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(),
                          static_cast<size_t>(x_count) * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Buffer temporal col: [Kcol, Ncol] FP16 (reutilizado por batch element).
    __half* d_col = nullptr;
    CHECK_CUDA(cudaMalloc(&d_col,
        static_cast<size_t>(Kcol) * static_cast<size_t>(Ncol) * sizeof(__half)));

    // Buffer de salida Y: [N, K, outH, outW] FP32 NCHW.
    float* d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(float)));

    const int single_x = opt.C * opt.H * opt.W;
    const int single_y = opt.K * d.outH * d.outW;
    const int col_elems = Kcol * Ncol;
    const int col_blocks = (col_elems + kConversionThreads - 1) / kConversionThreads;

    const dim3 gemm_block(static_cast<unsigned int>(kBlockWarpsM * kBlockWarpsN * 32));
    const dim3 gemm_grid(
        static_cast<unsigned int>((M    + kBlockTileM - 1) / kBlockTileM),
        static_cast<unsigned int>((Ncol + kBlockTileN - 1) / kBlockTileN));

    auto run_conv_iter = [&]() {
        for (int n = 0; n < opt.N; ++n) {
            const float* x_n = d_x + static_cast<size_t>(n) * single_x;
            float*       y_n = d_y + static_cast<size_t>(n) * single_y;
            im2col_fp16_kernel<<<col_blocks, kConversionThreads>>>(
                x_n, d_col,
                opt.C, opt.H, opt.W,
                opt.R, opt.S,
                opt.pad_h, opt.pad_w,
                opt.stride_h, opt.stride_w,
                opt.dilation_h, opt.dilation_w,
                d.outH, d.outW);
            CHECK_CUDA(cudaGetLastError());
            wmma_gemm_kernel<<<gemm_grid, gemm_block>>>(
                d_w_fp16, d_col, y_n, M, Ncol, Kcol);
            CHECK_CUDA(cudaGetLastError());
        }
    };

    for (int i = 0; i < kWarmupIters; ++i) run_conv_iter();
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < opt.iters; ++i) run_conv_iter();
    const float total_ms = timer.stop_and_elapsed_ms();

    CHECK_CUDA(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_col));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w_fp16));

    return build_metrics(opt, d, static_cast<double>(total_ms) / opt.iters);
}

// =========================================================================
// Reportes finales
// =========================================================================

static void print_float_report(const Metrics& cpu,
                                const Metrics& gpu,
                                const Metrics& tc,
                                const Metrics& wmma,
                                const ErrorMetrics& gpu_err,
                                const ErrorMetrics& tc_err,
                                const ErrorMetrics& wmma_err,
                                const ErrorMetrics& wmma_vs_tc_err) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=========== RESULTADOS CONV 2D FP32 ===========\n";
    std::cout << "CPU im2col+BLAS - tiempo    : " << cpu.ms << " ms\n";
    std::cout << "CPU im2col+BLAS - rend.     : " << cpu.gflops << " GFLOP/s ("
              << cpu.tflops << " TFLOP/s)\n\n";

    print_reference_comparison("GPU cuDNN clasico  ", gpu, cpu.ms, gpu_err);

    std::cout << "GPU Tensor Core - tiempo    : " << tc.ms << " ms\n";
    std::cout << "GPU Tensor Core - rend.     : " << tc.gflops << " GFLOP/s ("
              << tc.tflops << " TFLOP/s)\n";
    std::cout << "Speedup TC vs CPU           : " << cpu.ms / tc.ms << "x\n";
    std::cout << "Speedup TC vs GPU clasico   : " << gpu.ms / tc.ms << "x\n";
    std::cout << "Error max abs vs CPU        : " << tc_err.max_abs << "\n";
    std::cout << "Error relativo L2 vs CPU    : " << tc_err.rel_l2 << "\n\n";

    if (wmma.ms > 0.0) {
        std::cout << "GPU WMMA custom - tiempo    : " << wmma.ms << " ms\n";
        std::cout << "GPU WMMA custom - rend.     : " << wmma.gflops << " GFLOP/s ("
                  << wmma.tflops << " TFLOP/s)\n";
        std::cout << "Speedup WMMA vs CPU         : " << cpu.ms / wmma.ms << "x\n";
        std::cout << "Speedup WMMA vs GPU clasico : " << gpu.ms / wmma.ms << "x\n";
        std::cout << "Speedup WMMA vs cuDNN TC    : " << tc.ms / wmma.ms << "x\n";
        std::cout << "Error max abs vs CPU        : " << wmma_err.max_abs << "\n";
        std::cout << "Error relativo L2 vs CPU    : " << wmma_err.rel_l2 << "\n";
        std::cout << "Error max abs vs cuDNN TC   : " << wmma_vs_tc_err.max_abs << "\n";
        std::cout << "Error relativo L2 vs TC     : " << wmma_vs_tc_err.rel_l2 << "\n";
    } else {
        std::cout << "GPU WMMA custom             : omitida (alineacion no cumplida)\n";
    }
    std::cout << "===============================================\n";
}

static void print_double_report(const Metrics& cpu,
                                 const Metrics& gpu,
                                 const ErrorMetrics& gpu_err) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=========== RESULTADOS CONV 2D FP64 ===========\n";
    std::cout << "CPU im2col+BLAS - tiempo    : " << cpu.ms << " ms\n";
    std::cout << "CPU im2col+BLAS - rend.     : " << cpu.gflops << " GFLOP/s ("
              << cpu.tflops << " TFLOP/s)\n\n";
    print_reference_comparison("GPU cuDNN clasico  ", gpu, cpu.ms, gpu_err);
    std::cout << "===============================================\n";
}

// =========================================================================
// Orquestacion de experimentos
// =========================================================================

static void run_experiment_float(const Options& opt) {
    const OutputDims d      = compute_output_dims(opt);
    const size_t x_count    = static_cast<size_t>(opt.N) * opt.C * opt.H * opt.W;
    const size_t w_count    = static_cast<size_t>(opt.K) * opt.C * opt.R * opt.S;
    const size_t y_count    = static_cast<size_t>(d.outN) * d.outC * d.outH * d.outW;

    std::vector<float> x(x_count), w(w_count);
    std::vector<float> y_cpu(y_count, 0.0f), y_gpu(y_count, 0.0f);
    std::vector<float> y_tc(y_count, 0.0f),  y_wmma(y_count, 0.0f);
    initialize_matrix_float(x);
    initialize_matrix_float(w);

    const Metrics cpu  = benchmark_cpu_float(x, w, y_cpu, opt, d);
    const Metrics gpu  = benchmark_gpu_cudnn_float(x, w, y_gpu, opt, d);
    const Metrics tc   = benchmark_gpu_tensor_cores_conv(x, w, y_tc, opt, d);
    const Metrics wmma = benchmark_gpu_wmma_conv(x, w, y_wmma, opt, d);

    const ErrorMetrics gpu_err       = compare_float_vectors(y_cpu, y_gpu);
    const ErrorMetrics tc_err        = compare_float_vectors(y_cpu, y_tc);
    const ErrorMetrics wmma_err      = compare_float_vectors(y_cpu, y_wmma);
    const ErrorMetrics wmma_vs_tc    = compare_float_vectors(y_tc,  y_wmma);

    print_float_report(cpu, gpu, tc, wmma, gpu_err, tc_err, wmma_err, wmma_vs_tc);
}

static void run_experiment_double(const Options& opt) {
    const OutputDims d      = compute_output_dims(opt);
    const size_t x_count    = static_cast<size_t>(opt.N) * opt.C * opt.H * opt.W;
    const size_t w_count    = static_cast<size_t>(opt.K) * opt.C * opt.R * opt.S;
    const size_t y_count    = static_cast<size_t>(d.outN) * d.outC * d.outH * d.outW;

    std::vector<double> x(x_count), w(w_count);
    std::vector<double> y_cpu(y_count, 0.0), y_gpu(y_count, 0.0);
    initialize_matrix_double(x);
    initialize_matrix_double(w);

    const Metrics cpu = benchmark_cpu_double(x, w, y_cpu, opt, d);
    const Metrics gpu = benchmark_gpu_cudnn_double(x, w, y_gpu, opt, d);

    const ErrorMetrics gpu_err = compare_double_vectors(y_cpu, y_gpu);

    print_double_report(cpu, gpu, gpu_err);
}

static void run_benchmark(const Options& opt) {
    const OutputDims d = compute_output_dims(opt);
    std::cout << "================== CONFIGURACION ==================\n";
    std::cout << "Precision                  : "
              << (opt.use_double ? "FP64 (double)" : "FP32 (float)") << "\n";
    std::cout << "Entrada (N,C,H,W)          : "
              << opt.N << ", " << opt.C << ", " << opt.H << ", " << opt.W << "\n";
    std::cout << "Filtro  (K,C,R,S)          : "
              << opt.K << ", " << opt.C << ", " << opt.R << ", " << opt.S << "\n";
    std::cout << "Salida  (N,K,outH,outW)    : "
              << d.outN << ", " << d.outC << ", " << d.outH << ", " << d.outW << "\n";
    std::cout << "Padding  (h,w)             : " << opt.pad_h << ", " << opt.pad_w << "\n";
    std::cout << "Stride   (h,w)             : " << opt.stride_h << ", " << opt.stride_w << "\n";
    std::cout << "Dilation (h,w)             : " << opt.dilation_h << ", " << opt.dilation_w << "\n";
    std::cout << "Iteraciones                : " << opt.iters << "\n";
    std::cout << "===================================================\n\n";

    if (opt.use_double) {
        run_experiment_double(opt);
    } else {
        run_experiment_float(opt);
    }
}

}  // namespace

// Punto de entrada del programa.
int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);
    print_gpu_info();
    run_benchmark(opt);
    return 0;
}
