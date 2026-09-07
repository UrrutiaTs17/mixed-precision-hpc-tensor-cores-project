// Fase_1/Convolution/conv_baseline.cu
//
// Baseline de Fase 1 para convolucion 2D hacia adelante: compara una
// referencia CPU (im2col + OpenBLAS SGEMM/DGEMM) contra cuDNN en GPU, sin
// Tensor Cores. Es la linea base "sin precision mixta" contra la que se mide
// el efecto de activarlos en Fase 2 (Fase_2/Convolution/conv_tensor_activation.cu).
//
// Rutas:
//   1. CPU: im2col + OpenBLAS (FP32 o FP64 segun --double).
//   2. GPU: cuDNN con Tensor Cores desactivados explicitamente (ver el
//      comentario en run_gpu_cudnn_float sobre CUDNN_FMA_MATH).
//
// Con --double, ambas rutas corren en FP64 (no hay ruta Tensor Core que
// comparar en ese caso: TF32/FP16/BF16 son temas de FP32 de entrada).
//
// Migrado de old/Fase_1/Convolution/cudnn_conv_balanced.cu. Sin cambios de
// logica numerica: misma inicializacion de datos, mismo im2col, mismo
// seteo de cuDNN, mismo criterio de seleccion de algoritmo, mismo esquema de
// medicion (1 corrida de calentamiento sin medir + cfg.iters corridas
// medidas con eventos CUDA). Ver Fase_1/Convolution/README.md para el
// detalle de que cambio en la migracion (macros/metricas movidas a
// common/, struct de resultados reorganizado) y que no.
//
// Compilar y correr: ver Fase_1/Convolution/run_conv_fase1.sbatch, o
// Fase_1/Convolution/README.md para invocar nvcc a mano.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cblas.h>
#include <cuda_runtime.h>
#include <cudnn.h>

namespace {

// CHECK_CUDA / CHECK_CUDNN (common/cuda_checks.cuh) y CudaEventTimer /
// Metrics / ErrorMetrics / compare_float_vectors / compare_double_vectors
// (common/metrics.cuh) reemplazan las macros y structs que este archivo
// tenia duplicadas antes de la migracion (ver old/Fase_1/Convolution/
// cudnn_conv_balanced.cu). Deben incluirse dentro de este namespace anonimo
// -- ver la nota de uso en la cabecera de cada header de common/.
#include "../../common/cuda_checks.cuh"
#include "../../common/metrics.cuh"

// =========================================================================
// Configuracion
// =========================================================================

// Parametros del problema, todos expuestos por linea de comandos (ver
// parse_args). Convencion de nombres: N=batch, C=canales entrada,
// H/W=alto/ancho de entrada, K=filtros (canales salida), R/S=alto/ancho del
// filtro.
struct ConvConfig {
    int N          = 1;
    int C          = 64;
    int H          = 64;
    int W          = 64;
    int K          = 64;
    int R          = 3;
    int S          = 3;
    int pad_h      = 1;
    int pad_w      = 1;
    int stride_h   = 1;
    int stride_w   = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int iters      = 10;
    bool use_double = false;
};

// Dimensiones de salida derivadas de ConvConfig por la formula estandar de
// convolucion (ver compute_output_dims).
struct ConvOutputDims {
    int outN;
    int outC;
    int outH;
    int outW;
};

// Resultado completo de un experimento: metricas de tiempo/rendimiento para
// cada ruta (CPU, GPU) y el error de la GPU respecto a la CPU en la misma
// precision. No es una comparacion contra una referencia FP64 externa (a
// diferencia de Fase 2): aqui CPU y GPU corren siempre en la misma precision
// (ambas FP32 o ambas FP64), asi que el error medido es una prueba de
// correccion GPU-vs-CPU, no de perdida de precision por formato.
struct ExperimentResult {
    Metrics cpu;
    Metrics gpu;
    ErrorMetrics err;
};

// =========================================================================
// Linea de comandos
// =========================================================================

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

// Lee el entero que sigue a argv[i] (el flag), avanza i, y termina el
// programa con un mensaje claro si falta el valor.
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

// =========================================================================
// Dimensiones y rendimiento
// =========================================================================

// Deriva las dimensiones de salida con la formula estandar de convolucion y
// aborta con un mensaje claro si el resultado es invalido (padding/stride/
// dilation/filtro incompatibles con la entrada).
ConvOutputDims compute_output_dims(const ConvConfig& cfg) {
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

// Un MAC (multiply-accumulate) = 1 mul + 1 add = 2 operaciones de punto
// flotante. Cada posicion de salida (n, k, oh, ow) requiere C*R*S MACs.
double conv_flop_count(const ConvConfig& cfg, const ConvOutputDims& d) {
    return 2.0 * static_cast<double>(cfg.N) * static_cast<double>(d.outH) *
           static_cast<double>(d.outW) * static_cast<double>(cfg.K) *
           static_cast<double>(cfg.C) * static_cast<double>(cfg.R) *
           static_cast<double>(cfg.S);
}

// Empaqueta un tiempo promedio medido (ms) en Metrics (GFLOP/s y TFLOP/s
// derivados de conv_flop_count). Punto unico de calculo de rendimiento para
// que CPU y GPU se reporten de forma consistente.
Metrics build_metrics(const ConvConfig& cfg, const ConvOutputDims& d, double avg_ms) {
    Metrics m;
    m.ms     = avg_ms;
    m.gflops = conv_flop_count(cfg, d) / (m.ms * 1e6);
    m.tflops = m.gflops / 1000.0;
    return m;
}

// =========================================================================
// Inicializacion de datos
// =========================================================================

// Ruido uniforme en [-1, 1] con semilla fija: reproducible entre corridas,
// sin depender de una fuente de entropia del sistema.
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

void im2col_float_single_image(const float* x, float* col, const ConvConfig& cfg, const ConvOutputDims& d) {
    const int stride_col = cfg.C * cfg.R * cfg.S;
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
                            col[row + col_idx * stride_col] = x[(c * cfg.H + ih) * cfg.W + iw];
                        } else {
                            col[row + col_idx * stride_col] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void im2col_double_single_image(const double* x, double* col, const ConvConfig& cfg, const ConvOutputDims& d) {
    const int stride_col = cfg.C * cfg.R * cfg.S;
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
                            col[row + col_idx * stride_col] = x[(c * cfg.H + ih) * cfg.W + iw];
                        } else {
                            col[row + col_idx * stride_col] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

// Corre cfg.iters convoluciones CPU (im2col + SGEMM) y devuelve el tiempo
// promedio en ms. El buffer "col" se reutiliza entre elementos del batch e
// iteraciones (una sola reserva).
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
            // El filtro NCHW [K,C,R,S] es row-major [M, Kcol] e im2col escribe
            // col-major [Kcol, Ncol] (= row-major [Ncol, Kcol], de ahi
            // CblasTrans). La llamada anterior declaraba ColMajor con lda=M para
            // ambos: BLAS releia el filtro con stride M en vez de Kcol y escribia
            // la salida con el canal como indice rapido (NHWC), de modo que
            // y_cpu no era comparable elemento a elemento con la salida NCHW de
            // cuDNN -> error L2 relativo ~sqrt(2) (vectores independientes).
            // Mismo layout que Fase_2/Convolution/conv_tensor_activation.cu.
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, Ncol, Kcol,
                        alpha,
                        w.data(), Kcol,
                        col.data(), Kcol,
                        beta,
                        &y[static_cast<size_t>(n) * cfg.K * d.outH * d.outW], Ncol);
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
            // Mismo layout que la version float: salida NCHW row-major.
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, Ncol, Kcol,
                        alpha,
                        w.data(), Kcol,
                        col.data(), Kcol,
                        beta,
                        &y[static_cast<size_t>(n) * cfg.K * d.outH * d.outW], Ncol);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / cfg.iters;
}

// =========================================================================
// Ruta 2 - GPU: cuDNN sin Tensor Cores
// =========================================================================

// Deja que cuDNN elija el algoritmo de convolucion directa en vez de forzar
// CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, que era el unico que no pide
// workspace y tambien el mas lento en estas formas: fijarlo hacia que Fase 1
// midiera ~9.8x mas que Fase 2 (153.96 ms vs 15.63 ms con C=K=1024) sobre la
// misma convolucion y el mismo dispositivo. Se descartan los candidatos cuyo
// workspace no cabe en la memoria libre; hay que llamarla despues de reservar
// los tensores para que free_bytes ya los descuente.
// Misma seleccion que Fase_2/Convolution/conv_tensor_activation.cu.
cudnnConvolutionFwdAlgo_t select_forward_algo(cudnnHandle_t handle,
                                              cudnnTensorDescriptor_t xDesc,
                                              cudnnFilterDescriptor_t wDesc,
                                              cudnnConvolutionDescriptor_t convDesc,
                                              cudnnTensorDescriptor_t yDesc,
                                              size_t& workspaceBytes) {
    constexpr int kMaxAlgos = 8;
    cudnnConvolutionFwdAlgoPerf_t perf[kMaxAlgos];
    int algo_count = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, xDesc, wDesc, convDesc, yDesc, kMaxAlgos, &algo_count, perf));

    size_t free_bytes = 0, total_bytes = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t ws_limit = free_bytes / 4 * 3;

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    for (int i = 0; i < algo_count; ++i) {
        if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= ws_limit) {
            algo = perf[i].algo;
            break;
        }
    }

    workspaceBytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc,
                                                        algo, &workspaceBytes));
    std::cout << "cuDNN algoritmo elegido   : " << static_cast<int>(algo)
              << " (workspace " << std::fixed << std::setprecision(1)
              << workspaceBytes / (1024.0 * 1024.0) << " MiB)\n";
    return algo;
}

// Corre cfg.iters convoluciones cuDNN FP32 (mas 1 corrida de calentamiento
// sin medir) y devuelve el tiempo promedio en ms.
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
    // Baseline "sin Tensor Cores": con CUDNN_DEFAULT_MATH cuDNN habilita TF32 en
    // Ampere por su cuenta y la linea base FP32 termina corriendo en Tensor
    // Cores. CUDNN_FMA_MATH obliga a la ruta FP32 escalar (pico 19.5 TFLOP/s en
    // A100, no los 156 TFLOP/s de TF32). Esta es la unica linea de este archivo
    // que hace que "conv_baseline" sea realmente un baseline sin Tensor Cores:
    // no moverla ni quitarla sin cambiar tambien el nombre/proposito del binario.
    CHECK_CUDNN(cudnnSetConvolutionMathType(convDesc, CUDNN_FMA_MATH));
    std::cout << "cuDNN math type           : CUDNN_FMA_MATH (TF32 desactivado)\n";
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

    size_t workspaceBytes = 0;
    const cudnnConvolutionFwdAlgo_t algo =
        select_forward_algo(handle, xDesc, wDesc, convDesc, yDesc, workspaceBytes);
    void* d_workspace = nullptr;
    if (workspaceBytes > 0) CHECK_CUDA(cudaMalloc(&d_workspace, workspaceBytes));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Corrida de calentamiento (no medida): fuerza la inicializacion perezosa
    // de cuDNN (JIT de kernels, cache de heuristica) fuera de la ventana
    // cronometrada.
    CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w,
                                        convDesc, algo, d_workspace, workspaceBytes,
                                        &beta, yDesc, d_y));
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int it = 0; it < cfg.iters; ++it) {
        CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_workspace, workspaceBytes,
                                            &beta, yDesc, d_y));
    }
    const float totalMs = timer.stop_and_elapsed_ms();

    CHECK_CUDA(cudaMemcpy(y.data(), d_y, yBytes, cudaMemcpyDeviceToHost));

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

// Analoga a run_gpu_cudnn_float en FP64. No hay math type que desactivar:
// TF32 solo existe como sustituto de FP32, FP64 siempre corre en la unidad
// escalar de doble precision.
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

    size_t workspaceBytes = 0;
    const cudnnConvolutionFwdAlgo_t algo =
        select_forward_algo(handle, xDesc, wDesc, convDesc, yDesc, workspaceBytes);
    void* d_workspace = nullptr;
    if (workspaceBytes > 0) CHECK_CUDA(cudaMalloc(&d_workspace, workspaceBytes));

    const double alpha = 1.0;
    const double beta = 0.0;

    CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w,
                                        convDesc, algo, d_workspace, workspaceBytes,
                                        &beta, yDesc, d_y));
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int it = 0; it < cfg.iters; ++it) {
        CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_workspace, workspaceBytes,
                                            &beta, yDesc, d_y));
    }
    const float totalMs = timer.stop_and_elapsed_ms();

    CHECK_CUDA(cudaMemcpy(y.data(), d_y, yBytes, cudaMemcpyDeviceToHost));

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

// =========================================================================
// Orquestacion de experimentos y reportes
// =========================================================================

ExperimentResult run_experiment_float(const ConvConfig& cfg, const ConvOutputDims& d) {
    const size_t xCount = static_cast<size_t>(cfg.N) * cfg.C * cfg.H * cfg.W;
    const size_t wCount = static_cast<size_t>(cfg.K) * cfg.C * cfg.R * cfg.S;
    const size_t yCount = static_cast<size_t>(d.outN) * d.outC * d.outH * d.outW;

    std::vector<float> x(xCount), w(wCount), y_cpu(yCount, 0.0f), y_gpu(yCount, 0.0f);
    initialize_vector_float(x, 42u);
    initialize_vector_float(w, 1337u);

    ExperimentResult r;
    r.cpu = build_metrics(cfg, d, run_cpu_conv_openblas_float(cfg, d, x, w, y_cpu));
    r.gpu = build_metrics(cfg, d, run_gpu_cudnn_float(cfg, d, x, w, y_gpu));
    r.err = compare_float_vectors(y_cpu, y_gpu);
    return r;
}

ExperimentResult run_experiment_double(const ConvConfig& cfg, const ConvOutputDims& d) {
    const size_t xCount = static_cast<size_t>(cfg.N) * cfg.C * cfg.H * cfg.W;
    const size_t wCount = static_cast<size_t>(cfg.K) * cfg.C * cfg.R * cfg.S;
    const size_t yCount = static_cast<size_t>(d.outN) * d.outC * d.outH * d.outW;

    std::vector<double> x(xCount), w(wCount), y_cpu(yCount, 0.0), y_gpu(yCount, 0.0);
    initialize_vector_double(x, 42u);
    initialize_vector_double(w, 1337u);

    ExperimentResult r;
    r.cpu = build_metrics(cfg, d, run_cpu_conv_openblas_double(cfg, d, x, w, y_cpu));
    r.gpu = build_metrics(cfg, d, run_gpu_cudnn_double(cfg, d, x, w, y_gpu));
    r.err = compare_double_vectors(y_cpu, y_gpu);
    return r;
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

void print_results(const ExperimentResult& r) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "---------------- RESULTADOS ----------------\n";
    std::cout << "CPU OpenBLAS - tiempo medio : " << r.cpu.ms << " ms\n";
    std::cout << "CPU OpenBLAS - rendimiento  : " << r.cpu.gflops << " GFLOP/s ("
              << r.cpu.tflops << " TFLOP/s)\n";
    std::cout << "GPU cuDNN - tiempo medio    : " << r.gpu.ms << " ms\n";
    std::cout << "GPU cuDNN - rendimiento     : " << r.gpu.gflops << " GFLOP/s ("
              << r.gpu.tflops << " TFLOP/s)\n";
    std::cout << "Speedup GPU/CPU             : " << r.cpu.ms / r.gpu.ms << "x\n";
    std::cout << "Error max abs               : " << r.err.max_abs << "\n";
    std::cout << "Error relativo L2           : " << r.err.rel_l2 << "\n";
    std::cout << "--------------------------------------------\n";
}

}  // namespace

// Punto de entrada: parsea argumentos, corre la ruta CPU vs GPU en la
// precision solicitada, e imprime el reporte.
int main(int argc, char** argv) {
    const ConvConfig cfg = parse_args(argc, argv);
    const ConvOutputDims d = compute_output_dims(cfg);

    print_gpu_info();
    print_config(cfg, d);

    const ExperimentResult r = cfg.use_double ? run_experiment_double(cfg, d)
                                              : run_experiment_float(cfg, d);
    print_results(r);
    return 0;
}
