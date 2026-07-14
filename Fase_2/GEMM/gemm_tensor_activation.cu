// nvcc -std=c++17 gemm_tensor_activation.cu -o gemm_tc -I/usr/include/openblas -lcublas -lopenblas -gencode arch=compute_80,code=sm_80 --allow-unsupported-compiler
// ./gemm_tc --m 2048 --n 2048 --k 2048 --iters 20
// En PACCA (A100, sm_80) la compilacion y el perfilado con Nsight Compute se
// lanzan via SLURM: sbatch run_gemm_tc.sbatch  (no ejecutar ncu con sudo).


#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>

namespace {

#include "../common.cuh"

constexpr int kWarmupIters = 3;
constexpr int kCpuWarmupIters = 3;
constexpr int kConversionThreads = 256;

// Dimensiones del fragmento WMMA para FP16: unicas dimensiones soportadas en sm_70+.
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// Numero de elementos K cargados a shared memory por iteracion del bucle externo.
// Debe ser multiplo de kWmmaK. A mayor kKStep, menos sincronizaciones y mayor
// reuso de datos en shared memory, a costa de mas shared memory usada.
constexpr int kKStep = 32;

// Numero de warps por dimension dentro de un bloque (4×4 = 16 warps = 512 hilos).
// Con 512 hilos/bloque y ~28.5 KiB shared por bloque, el SM puede alojar 3 bloques
// simultaneos -> 48/48 warps activos -> 100% ocupancia teorica en Ampere (CC 8.6).
constexpr int kBlockWarpsM = 4;
constexpr int kBlockWarpsN = 4;

// Tile de salida que maneja un bloque completo: 64×64 elementos FP32.
constexpr int kBlockTileM = kBlockWarpsM * kWmmaM;  // 64
constexpr int kBlockTileN = kBlockWarpsN * kWmmaN;  // 64

// Etapas del pipeline de triple buffer para cp.async.
// Mientras el warp computa el tile[i], la DMA ya esta cargando tile[i+2],
// eliminando la espera sincrona de global memory en cada iteracion K.
constexpr int kNumStages = 3;

// Padding en shared memory para evitar bank conflicts entre warps.
// Con FP16 (2 bytes) y 32 bancos de 4 bytes, añadir 8 elementos desplaza
// cada fila 16 bytes extra, eliminando el patron de conflicto ciclico.
constexpr int kWmmaShmemPad = 8;

// Formatos de datos soportados en las rutas Tensor Core (cuBLAS TC y WMMA custom).
enum class TensorCoreFormat {
    FP16,
    BF16,
    Both
};

// Parametros de entrada configurables desde linea de comandos.
struct Options {
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int iters = 20;
    bool use_double = false;
    TensorCoreFormat tc_format = TensorCoreFormat::FP16;
};

// Traduce codigos de cuBLAS a texto legible para diagnosticar fallos.
static const char* cublas_status_to_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

// Envueltura minima para crear y destruir el contexto de cuBLAS.
// El handle representa el estado interno que cuBLAS usa para ejecutar operaciones.
class CublasHandle {
public:
    CublasHandle() {
        CHECK_CUBLAS(cublasCreate(&handle_));
    }

    ~CublasHandle() noexcept {
        if (handle_ != nullptr) {
            const cublasStatus_t status = cublasDestroy(handle_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::cerr << "Warning: cublasDestroy fallo -> "
                          << cublas_status_to_string(status) << std::endl;
            }
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

// Buffer RAII para memoria en GPU: libera automaticamente incluso si una validacion falla.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ == 0) {
            return;
        }

        if (count_ > std::numeric_limits<size_t>::max() / sizeof(T)) {
            std::cerr << "La reserva solicitada desborda size_t." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * count_));
    }

    ~DeviceBuffer() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    size_t size() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// Muestra ayuda de uso y ejemplos basicos de ejecucion.
static void print_usage(const char* prog) {
    std::cout << "Uso:\n"
              << "  " << prog << " [--m M] [--n N] [--k K] [--iters I] [--double]"
              << " [--tc-format fp16|bf16|both]\n\n"
              << "Descripcion:\n"
              << "  Compara cuatro rutas de GEMM en precision mixta:\n"
              << "    1. CPU BLAS (OpenBLAS, FP32/FP64)\n"
              << "    2. GPU cuBLAS clasico (FP32/FP64, sin Tensor Cores)\n"
              << "    3. GPU cuBLAS con Tensor Cores (FP16/BF16->FP32, cublasGemmEx)\n"
              << "    4. GPU WMMA custom (FP16/BF16->FP32, kernel propio con shared memory)\n"
              << "  La ruta WMMA (4) requiere m y n multiplos de 32 y k multiplo de 16.\n"
              << "  Con --double solo se ejecutan las rutas 1 y 2 (WMMA y TC son FP16/BF16).\n"
              << "  --tc-format selecciona el formato de las rutas 3 y 4 (por defecto fp16).\n"
              << "  BF16 requiere GPU Ampere o superior (compute capability >= 8.0).\n\n"
              << "Ejemplos:\n"
              << "  " << prog << "\n"
              << "  " << prog << " --m 4096 --n 4096 --k 4096 --iters 10\n"
              << "  " << prog << " --double --m 2048 --n 2048 --k 2048 --iters 5\n"
              << "  " << prog << " --m 1024 --n 1024 --k 1024 --iters 5 --tc-format bf16\n";
}

static const char* require_arg_value(int& index, int argc, char** argv, const char* flag) {
    if (index + 1 >= argc) {
        std::cerr << "Falta valor para " << flag << ".\n\n";
        print_usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }
    return argv[++index];
}

static int parse_positive_int(const char* flag, const char* value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);

    if (errno != 0 || end == value || *end != '\0' ||
        parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        std::cerr << "Valor invalido para " << flag << ": " << value
                  << ". Debe ser un entero positivo." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return static_cast<int>(parsed);
}

// Interpreta el valor de --tc-format y termina el programa si no es reconocido.
static TensorCoreFormat parse_tc_format(const char* value) {
    if (std::strcmp(value, "fp16") == 0) return TensorCoreFormat::FP16;
    if (std::strcmp(value, "bf16") == 0) return TensorCoreFormat::BF16;
    if (std::strcmp(value, "both") == 0) return TensorCoreFormat::Both;

    std::cerr << "Formato Tensor Core no reconocido: " << value
              << ". Use fp16, bf16 o both." << std::endl;
    std::exit(EXIT_FAILURE);
}

// Interpreta los argumentos de la linea de comandos y valida sus valores.
static Options parse_args(int argc, char** argv) {
    Options opt;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--m") == 0) {
            opt.m = parse_positive_int("--m", require_arg_value(i, argc, argv, "--m"));
        } else if (std::strcmp(argv[i], "--n") == 0) {
            opt.n = parse_positive_int("--n", require_arg_value(i, argc, argv, "--n"));
        } else if (std::strcmp(argv[i], "--k") == 0) {
            opt.k = parse_positive_int("--k", require_arg_value(i, argc, argv, "--k"));
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opt.iters = parse_positive_int("--iters", require_arg_value(i, argc, argv, "--iters"));
        } else if (std::strcmp(argv[i], "--double") == 0) {
            opt.use_double = true;
        } else if (std::strcmp(argv[i], "--tc-format") == 0) {
            opt.tc_format = parse_tc_format(require_arg_value(i, argc, argv, "--tc-format"));
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Argumento no reconocido: " << argv[i] << "\n\n";
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    return opt;
}

// Imprime informacion del dispositivo activo para contextualizar los resultados.
static void print_gpu_info() {
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    int gpu_clock_khz = 0;
    int mem_clock_khz = 0;
    int mem_bus_width = 0;
    cudaError_t err_clock = cudaDeviceGetAttribute(&gpu_clock_khz, cudaDevAttrClockRate, dev);
    cudaError_t err_mem_clock = cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaError_t err_mem_bus = cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth, dev);

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
              << (err_clock == cudaSuccess ? gpu_clock_khz / 1000.0 : 0.0)
              << (err_clock == cudaSuccess ? " MHz\n" : " no disponible\n");
    std::cout << "Reloj memoria              : "
              << (err_mem_clock == cudaSuccess ? mem_clock_khz / 1000.0 : 0.0)
              << (err_mem_clock == cudaSuccess ? " MHz\n" : " no disponible\n");
    std::cout << "Bus de memoria             : "
              << (err_mem_bus == cudaSuccess ? std::to_string(mem_bus_width) + " bits" : "no disponible")
              << "\n";
    std::cout << "Memoria compartida/bloque  : " << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
    std::cout << "Tensor Cores FP16 (HMMA)   : " << (prop.major >= 7 ? "si" : "no") << "\n";
    std::cout << "Tensor Cores TF32          : " << (prop.major >= 8 ? "si" : "no") << "\n";
    std::cout << "===========================================================\n\n";
}

// FLOPs teoricos de una GEMM densa: C = A * B implica 2*m*n*k operaciones.
static double gemm_flops_standard(int m, int n, int k) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
}

static size_t checked_element_count(int rows, int cols, const char* label) {
    const size_t r = static_cast<size_t>(rows);
    const size_t c = static_cast<size_t>(cols);

    if (c != 0 && r > std::numeric_limits<size_t>::max() / c) {
        std::cerr << "Las dimensiones de " << label << " desbordan size_t." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return r * c;
}

static unsigned int blocks_for_elements(size_t count) {
    const size_t blocks = (count + kConversionThreads - 1) / kConversionThreads;
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        std::cerr << "La conversion FP32->FP16 requiere demasiados bloques CUDA." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return static_cast<unsigned int>(blocks);
}

static bool active_device_supports_fp16_tensor_cores() {
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    return prop.major >= 7;
}

// BF16 Tensor Core (HMMA con operandos __nv_bfloat16) requiere Ampere o superior.
static bool active_device_supports_bf16_tensor_cores() {
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    return prop.major >= 8;
}

// Construye las metricas de rendimiento a partir del tiempo medio por iteracion.
static Metrics build_metrics(int m, int n, int k, double avg_ms) {
    Metrics out;
    out.ms = avg_ms;
    out.gflops = gemm_flops_standard(m, n, k) / (out.ms * 1e6);
    out.tflops = out.gflops / 1000.0;
    return out;
}

// Llena una matriz FP32 con valores deterministas pequeños.
// Esto evita depender de datos aleatorios y facilita repetir experimentos.
static void initialize_matrix_float(std::vector<float>& mat) {
    for (size_t i = 0; i < mat.size(); ++i) {
        const int centered_value = static_cast<int>(i % 101) - 50;
        mat[i] = static_cast<float>(centered_value) / 25.0f;
    }
}

// Version FP64 de la inicializacion anterior.
static void initialize_matrix_double(std::vector<double>& mat) {
    for (size_t i = 0; i < mat.size(); ++i) {
        const int centered_value = static_cast<int>(i % 101) - 50;
        mat[i] = static_cast<double>(centered_value) / 25.0;
    }
}

// Copia un vector de floats desde CPU hacia GPU.
static void copy_float_vector_to_device(const std::vector<float>& src, float* dst) {
    CHECK_CUDA(cudaMemcpy(dst, src.data(), sizeof(float) * src.size(), cudaMemcpyHostToDevice));
}

// Copia un vector de doubles desde CPU hacia GPU.
static void copy_double_vector_to_device(const std::vector<double>& src, double* dst) {
    CHECK_CUDA(cudaMemcpy(dst, src.data(), sizeof(double) * src.size(), cudaMemcpyHostToDevice));
}

// Copia un vector de floats desde GPU hacia CPU.
static void copy_float_vector_to_host(const float* src, std::vector<float>& dst) {
    CHECK_CUDA(cudaMemcpy(dst.data(), src, sizeof(float) * dst.size(), cudaMemcpyDeviceToHost));
}

// Copia un vector de doubles desde GPU hacia CPU.
static void copy_double_vector_to_host(const double* src, std::vector<double>& dst) {
    CHECK_CUDA(cudaMemcpy(dst.data(), src, sizeof(double) * dst.size(), cudaMemcpyDeviceToHost));
}

// Compara una referencia FP64 col-major contra un resultado FP32 row-major (salida WMMA).
static ErrorMetrics compare_fp64_ref_colmaj_vs_fp32_rowmaj(
        const std::vector<double>& ref_fp64_colmaj,
        const std::vector<float>&  test_fp32_rowmaj,
        int m, int n) {
    ErrorMetrics out;
    double sq_err = 0.0, sq_ref = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const double r    = ref_fp64_colmaj[i + j * m];
            const double t    = static_cast<double>(test_fp32_rowmaj[i * n + j]);
            const double diff = r - t;
            out.max_abs = std::max(out.max_abs, std::abs(diff));
            sq_err += diff * diff;
            sq_ref += r * r;
        }
    }
    out.rel_l2 = sq_ref > 0.0 ? std::sqrt(sq_err / sq_ref) : 0.0;
    return out;
}

// Imprime el rendimiento y el error de una ruta respecto a la referencia en CPU.
static void print_reference_comparison(const char* label,
                                       const Metrics& metrics,
                                       double reference_ms,
                                       const ErrorMetrics& error) {
    std::cout << label << " - tiempo         : " << metrics.ms << " ms\n";
    std::cout << label << " - rendimiento    : " << metrics.gflops << " GFLOP/s ("
              << metrics.tflops << " TFLOP/s)\n";
    std::cout << "Speedup vs CPU             : " << reference_ms / metrics.ms << "x\n";
    std::cout << "Error max abs vs FP64      : " << error.max_abs << "\n";
    std::cout << "Error relativo L2 vs FP64  : " << error.rel_l2 << "\n\n";
}

// Ejecuta GEMM en CPU usando BLAS.
// `cblas_sgemm` realiza C = alpha * A * B + beta * C para datos float.
static Metrics benchmark_cpu_float(const std::vector<float>& A,
                                   const std::vector<float>& B,
                                   std::vector<float>& C,
                                   int m,
                                   int n,
                                   int k,
                                   int iters) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < kCpuWarmupIters; ++i) {
        cblas_sgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            alpha,
            A.data(), m,
            B.data(), k,
            beta,
            C.data(), m);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        // sgemm:
        // s -> single precision (float)
        // gemm -> General Matrix-Matrix Multiplication
        // CblasColMajor indica que las matrices se almacenan por columnas.
        cblas_sgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            alpha,
            A.data(), m,
            B.data(), k,
            beta,
            C.data(), m);
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    return build_metrics(m, n, k, avg_ms);
}

// Ejecuta GEMM en CPU usando BLAS en precision doble.
// `cblas_dgemm` es la version double precision de la misma operacion GEMM.
static Metrics benchmark_cpu_double(const std::vector<double>& A,
                                    const std::vector<double>& B,
                                    std::vector<double>& C,
                                    int m,
                                    int n,
                                    int k,
                                    int iters) {
    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < kCpuWarmupIters; ++i) {
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            alpha,
            A.data(), m,
            B.data(), k,
            beta,
            C.data(), m);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        // dgemm:
        // d -> double precision
        // gemm -> multiplicacion matricial general
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            alpha,
            A.data(), m,
            B.data(), k,
            beta,
            C.data(), m);
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    return build_metrics(m, n, k, avg_ms);
}

// Ejecuta GEMM en GPU con cuBLAS para datos float.
// El flujo general es: reservar memoria, copiar datos, hacer warmup, medir, copiar salida.
static Metrics benchmark_gpu_cublas_float(const std::vector<float>& A,
                                          const std::vector<float>& B,
                                          std::vector<float>& C,
                                          int m,
                                          int n,
                                          int k,
                                          int iters) {
    DeviceBuffer<float> dA(A.size());
    DeviceBuffer<float> dB(B.size());
    DeviceBuffer<float> dC(C.size());
    copy_float_vector_to_device(A, dA.get());
    copy_float_vector_to_device(B, dB.get());

    CublasHandle handle;
    // CUBLAS_PEDANTIC_MATH desactiva TF32 en Ampere: garantiza FP32 puro sin Tensor Cores.
    CHECK_CUBLAS(cublasSetMathMode(handle.get(), CUBLAS_PEDANTIC_MATH));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < kWarmupIters; ++i) {
        // cublasSgemm:
        // S -> single precision
        // GEMM -> C = alpha * A * B + beta * C
        // handle: contexto de cuBLAS
        // CUBLAS_OP_N: no transpone A ni B
        // m, n, k: dimensiones de la multiplicacion
        // dA, dB, dC: punteros en memoria GPU
        CHECK_CUBLAS(cublasSgemm(handle.get(),
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m, n, k,
                                 &alpha,
                                 dA.get(), m,
                                 dB.get(), k,
                                 &beta,
                                 dC.get(), m));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        // Se mide exactamente la llamada GEMM sobre la GPU.
        CHECK_CUBLAS(cublasSgemm(handle.get(),
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m, n, k,
                                 &alpha,
                                 dA.get(), m,
                                 dB.get(), k,
                                 &beta,
                                 dC.get(), m));
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    copy_float_vector_to_host(dC.get(), C);

    return build_metrics(m, n, k, total_ms / iters);
}

// Ejecuta GEMM en GPU con cuBLAS para datos double.
// Es equivalente a la version float, cambiando el tipo de dato y la rutina usada.
static Metrics benchmark_gpu_cublas_double(const std::vector<double>& A,
                                           const std::vector<double>& B,
                                           std::vector<double>& C,
                                           int m,
                                           int n,
                                           int k,
                                           int iters) {
    DeviceBuffer<double> dA(A.size());
    DeviceBuffer<double> dB(B.size());
    DeviceBuffer<double> dC(C.size());
    copy_double_vector_to_device(A, dA.get());
    copy_double_vector_to_device(B, dB.get());

    CublasHandle handle;
    // CUBLAS_PEDANTIC_MATH desactiva TF32 en Ampere: garantiza FP64 puro sin Tensor Cores.
    CHECK_CUBLAS(cublasSetMathMode(handle.get(), CUBLAS_PEDANTIC_MATH));
    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < kWarmupIters; ++i) {
        // cublasDgemm:
        // D -> double precision
        // El resto de parametros tienen el mismo significado que en cublasSgemm.
        CHECK_CUBLAS(cublasDgemm(handle.get(),
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m, n, k,
                                 &alpha,
                                 dA.get(), m,
                                 dB.get(), k,
                                 &beta,
                                 dC.get(), m));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasDgemm(handle.get(),
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 m, n, k,
                                 &alpha,
                                 dA.get(), m,
                                 dB.get(), k,
                                 &beta,
                                 dC.get(), m));
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    copy_double_vector_to_host(dC.get(), C);

    return build_metrics(m, n, k, total_ms / iters);
}

// Kernel CUDA sencillo para convertir cada elemento de float a half.
__global__ static void convert_float_to_half_kernel(const float* src, __half* dst, size_t size) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Kernel CUDA sencillo para convertir cada elemento de float a bfloat16.
__global__ static void convert_float_to_bfloat16_kernel(const float* src, __nv_bfloat16* dst, size_t size) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

// Convierte dos buffers FP32 a FP16 dentro de la GPU.
// Esto prepara las entradas para la ruta Tensor Core.
static void convert_fp32_buffers_to_fp16(const float* src_a,
                                         const float* src_b,
                                         __half* dst_a,
                                         __half* dst_b,
                                         size_t size_a,
                                         size_t size_b) {
    const unsigned int blocks_a = blocks_for_elements(size_a);
    const unsigned int blocks_b = blocks_for_elements(size_b);

    convert_float_to_half_kernel<<<blocks_a, kConversionThreads>>>(
        src_a, dst_a, size_a);
    CHECK_CUDA(cudaGetLastError());

    convert_float_to_half_kernel<<<blocks_b, kConversionThreads>>>(
        src_b, dst_b, size_b);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());
}

// Lanza la GEMM con Tensor Cores mediante cuBLAS.
// `cublasGemmEx` permite elegir tipos de entrada, salida y acumulacion por separado.
static void run_tensor_core_gemm(cublasHandle_t handle,
                                 const __half* A,
                                 const __half* B,
                                 float* C,
                                 int m,
                                 int n,
                                 int k,
                                 const float* alpha,
                                 const float* beta) {
    // cublasGemmEx:
    // - A y B entran como FP16 (CUDA_R_16F)
    // - C sale como FP32 (CUDA_R_32F)
    // - CUBLAS_COMPUTE_32F_FAST_16F: acumulacion FP32 acelerada con Tensor Cores FP16 (HMMA).
    // - CUBLAS_GEMM_DEFAULT: seleccion automatica del mejor algoritmo (no deprecado).
    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m, n, k,
                              alpha,
                              A, CUDA_R_16F, m,
                              B, CUDA_R_16F, k,
                              beta,
                              C, CUDA_R_32F, m,
                              CUBLAS_COMPUTE_32F_FAST_16F,
                              CUBLAS_GEMM_DEFAULT));
}

// Ejecuta la ruta de precision mixta con Tensor Cores.
// Primero copia A y B en FP32, luego las convierte a FP16 y finalmente ejecuta GEMM.
static Metrics benchmark_gpu_tensor_cores(const std::vector<float>& A,
                                          const std::vector<float>& B,
                                          std::vector<float>& C,
                                          int m,
                                          int n,
                                          int k,
                                          int iters) {
    DeviceBuffer<__half> dA_fp16(A.size());
    DeviceBuffer<__half> dB_fp16(B.size());
    DeviceBuffer<float> dC(C.size());

    // Los buffers FP32 de scratch solo se necesitan para la conversion; se liberan
    // al salir del bloque para no ocupar memoria durante el benchmark.
    {
        DeviceBuffer<float> dA_fp32(A.size());
        DeviceBuffer<float> dB_fp32(B.size());
        copy_float_vector_to_device(A, dA_fp32.get());
        copy_float_vector_to_device(B, dB_fp32.get());
        convert_fp32_buffers_to_fp16(
            dA_fp32.get(),
            dB_fp32.get(),
            dA_fp16.get(),
            dB_fp16.get(),
            A.size(),
            B.size());
    }

    CublasHandle handle;
    // CUBLAS_COMPUTE_32F_FAST_16F en cublasGemmEx selecciona la ruta Tensor Core FP16->FP32.
    // No se necesita cublasSetMathMode: el tipo de computo ya especifica la precision.

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Se hace warmup para estabilizar clocks y evitar medir costos de primer uso.
    for (int i = 0; i < kWarmupIters; ++i) {
        run_tensor_core_gemm(
            handle.get(), dA_fp16.get(), dB_fp16.get(), dC.get(), m, n, k, &alpha, &beta);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        run_tensor_core_gemm(
            handle.get(), dA_fp16.get(), dB_fp16.get(), dC.get(), m, n, k, &alpha, &beta);
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    copy_float_vector_to_host(dC.get(), C);

    return build_metrics(m, n, k, total_ms / iters);
}

// Convierte dos buffers FP32 a BF16 dentro de la GPU.
// Esto prepara las entradas para la ruta Tensor Core BF16.
static void convert_fp32_buffers_to_bf16(const float* src_a,
                                         const float* src_b,
                                         __nv_bfloat16* dst_a,
                                         __nv_bfloat16* dst_b,
                                         size_t size_a,
                                         size_t size_b) {
    const unsigned int blocks_a = blocks_for_elements(size_a);
    const unsigned int blocks_b = blocks_for_elements(size_b);

    convert_float_to_bfloat16_kernel<<<blocks_a, kConversionThreads>>>(
        src_a, dst_a, size_a);
    CHECK_CUDA(cudaGetLastError());

    convert_float_to_bfloat16_kernel<<<blocks_b, kConversionThreads>>>(
        src_b, dst_b, size_b);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());
}

// Lanza la GEMM con Tensor Cores mediante cuBLAS, para operandos BF16.
static void run_tensor_core_gemm_bf16(cublasHandle_t handle,
                                      const __nv_bfloat16* A,
                                      const __nv_bfloat16* B,
                                      float* C,
                                      int m,
                                      int n,
                                      int k,
                                      const float* alpha,
                                      const float* beta) {
    // cublasGemmEx:
    // - A y B entran como BF16 (CUDA_R_16BF)
    // - C sale como FP32 (CUDA_R_32F)
    // - CUBLAS_COMPUTE_32F: BF16 no tiene un modo "fast" dedicado como
    //   CUBLAS_COMPUTE_32F_FAST_16F; el modo estandar ya enruta a Tensor
    //   Cores (HMMA) para operandos BF16 en Ampere (sm_80+).
    // - CUBLAS_GEMM_DEFAULT: seleccion automatica del mejor algoritmo (no deprecado).
    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m, n, k,
                              alpha,
                              A, CUDA_R_16BF, m,
                              B, CUDA_R_16BF, k,
                              beta,
                              C, CUDA_R_32F, m,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

// Ejecuta la ruta de precision mixta con Tensor Cores en BF16.
// Analoga a benchmark_gpu_tensor_cores, cambiando FP16 por BF16.
static Metrics benchmark_gpu_tensor_cores_bf16(const std::vector<float>& A,
                                               const std::vector<float>& B,
                                               std::vector<float>& C,
                                               int m,
                                               int n,
                                               int k,
                                               int iters) {
    DeviceBuffer<__nv_bfloat16> dA_bf16(A.size());
    DeviceBuffer<__nv_bfloat16> dB_bf16(B.size());
    DeviceBuffer<float> dC(C.size());

    {
        DeviceBuffer<float> dA_fp32(A.size());
        DeviceBuffer<float> dB_fp32(B.size());
        copy_float_vector_to_device(A, dA_fp32.get());
        copy_float_vector_to_device(B, dB_fp32.get());
        convert_fp32_buffers_to_bf16(
            dA_fp32.get(),
            dB_fp32.get(),
            dA_bf16.get(),
            dB_bf16.get(),
            A.size(),
            B.size());
    }

    CublasHandle handle;
    // CUBLAS_COMPUTE_32F en cublasGemmEx selecciona la ruta Tensor Core BF16->FP32.

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < kWarmupIters; ++i) {
        run_tensor_core_gemm_bf16(
            handle.get(), dA_bf16.get(), dB_bf16.get(), dC.get(), m, n, k, &alpha, &beta);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        run_tensor_core_gemm_bf16(
            handle.get(), dA_bf16.get(), dB_bf16.get(), dC.get(), m, n, k, &alpha, &beta);
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    copy_float_vector_to_host(dC.get(), C);

    return build_metrics(m, n, k, total_ms / iters);
}

// =========================================================================
// Ruta 4 - Kernel WMMA personalizado con Tensor Cores
//
// La API WMMA (Warp Matrix Multiply-Accumulate) de CUDA expone las unidades
// Tensor Core a nivel de warp. Cada warp opera sobre fragmentos de 16x16x16
// elementos FP16 y acumula en FP32.
//
// Diferencia clave respecto a cuBLAS TC:
//   cuBLAS TC  -> biblioteca optimizada, inaccessible internamente.
//   WMMA custom -> el programador controla el tiling, la carga desde memoria
//                  compartida y el patron de coalescencia explicitamente.
//
// Convencion de datos usada en este kernel: row-major.
// Las matrices del benchmark principal estan en col-major; por eso existe
// float_colmaj_to_tc_rowmaj_kernel que transpone y convierte antes del
// benchmark. El tipo T (__half o __nv_bfloat16) selecciona el formato de
// Tensor Core; la logica de tiling/pipeline es identica para ambos.
// =========================================================================

// Conversion escalar float -> T. Cada tipo Tensor Core soportado provee su
// propia especializacion mediante el intrinseco de CUDA correspondiente.
template <typename T>
__device__ inline T float_to_tc_scalar(float x);

template <>
__device__ inline __half float_to_tc_scalar<__half>(float x) {
    return __float2half(x);
}

template <>
__device__ inline __nv_bfloat16 float_to_tc_scalar<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// Convierte una matriz FP32 col-major (rows x cols) a T row-major.
// Thread i escribe dst[i] = src[r + c*rows] donde r=i/cols, c=i%cols.
// Hilos consecutivos leen src con paso 1 (misma columna, filas contiguas),
// lo que produce accesos coalescentes en la lectura global.
template <typename T>
__global__ static void float_colmaj_to_tc_rowmaj_kernel(
        const float* __restrict__ src,
        T*           __restrict__ dst,
        int rows, int cols) {
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx < total) {
        const int r = idx / cols;
        const int c = idx % cols;
        dst[idx] = float_to_tc_scalar<T>(src[r + c * rows]);
    }
}

// Kernel GEMM con API WMMA + pipeline cp.async de 3 etapas (Ampere sm_80+).
// C(M,N) = A(M,K) * B(K,N), todos row-major T->FP32 (T = __half o __nv_bfloat16).
//
// Organizacion de hilos:
//   Bloque: 512 hilos = 16 warps en cuadricula 4x4 de fragmentos WMMA.
//   Cada warp calcula un fragmento de salida 16x16 en FP32.
//   Un bloque cubre un tile de salida 64x64.
//   Grid: (ceildiv(M,64), ceildiv(N,64)).
//
// Triple buffer con cp.async:
//   El SM tiene 3 copias de sA/sB (etapas 0,1,2). Mientras el warp ejecuta
//   instrucciones HMMA sobre la etapa[i], la DMA ya transfiere la etapa[i+2]
//   desde global memory sin pasar por registros (cp.async). La barrera de cada
//   iteracion se reemplaza por consumer_wait_prior<kNumStages-1>(), que solo
//   bloquea si el tile necesario aun no llego, en vez de vaciar todo el pipeline.
//
// Requisito: M multiplo de kBlockTileM(64), N de kBlockTileN(64), K de kKStep(32).
// Ocupancia esperada en CC 8.6: 3 bloques/SM x 16 warps = 48/48 warps = 100%.
template <typename T>
__launch_bounds__(kBlockWarpsM * kBlockWarpsN * 32, 3)
__global__ static void wmma_gemm_kernel(
        const T* __restrict__ A,
        const T* __restrict__ B,
        float*   __restrict__ C,
        int M, int N, int K) {
    using namespace nvcuda;

    // Triple buffer: sA[etapa][fila][col], sB[etapa][fila][col].
    // El padding por fila evita bank conflicts cuando warps distintos
    // acceden a columnas separadas por kKStep o kBlockTileN elementos.
    __shared__ T sA[kNumStages][kBlockTileM][kKStep      + kWmmaShmemPad];
    __shared__ T sB[kNumStages][kKStep]     [kBlockTileN + kWmmaShmemPad];

    const int warp_id       = threadIdx.x / 32;
    const int warp_row      = warp_id / kBlockWarpsN;
    const int warp_col      = warp_id % kBlockWarpsN;
    const int block_row     = blockIdx.x * kBlockTileM;
    const int block_col     = blockIdx.y * kBlockTileN;
    const int warp_row_base = block_row + warp_row * kWmmaM;
    const int warp_col_base = block_col + warp_col * kWmmaN;

    wmma::fragment<wmma::matrix_a,    kWmmaM, kWmmaN, kWmmaK, T, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    kWmmaM, kWmmaN, kWmmaK, T, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>              c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // API primitiva de pipeline (cuda_pipeline_primitives.h):
    //   __pipeline_memcpy_async(dst, src, size) -> emite cp.async (min 4 bytes en Ampere)
    //   __pipeline_commit()                     -> cierra el grupo de copias actual
    //   __pipeline_wait_prior(N)                -> espera hasta que queden <= N grupos pendientes
    // No requiere acquire/release: el estado del pipeline es implicito por hilo.

    // Numero de tiles K. K es multiplo de kKStep (validado en benchmark_gpu_wmma).
    const int num_tiles = K / kKStep;

    // Pares uint32_t (2 fp16) por tile: cp.async necesita minimo 4 bytes.
    // kBlockTileM*kKStep = 64*32 = 2048 fp16 = 1024 uint32_t
    // kKStep*kBlockTileN = 32*64 = 2048 fp16 = 1024 uint32_t
    const int n_pairs_A = kBlockTileM * kKStep / 2;
    const int n_pairs_B = kKStep      * kBlockTileN / 2;

    // -- Precarga de las primeras kNumStages etapas antes del bucle principal --
    // Emite kNumStages grupos de cp.async sin esperar ninguno todavia.
    for (int s = 0; s < kNumStages && s < num_tiles; ++s) {
        const int k_off = s * kKStep;

        for (int i = threadIdx.x; i < n_pairs_A; i += blockDim.x) {
            const int elem = i * 2;
            const int row  = elem / kKStep;
            const int col  = elem % kKStep;   // siempre par -> alineado a 4 bytes
            __pipeline_memcpy_async(
                reinterpret_cast<uint32_t*>(&sA[s][row][col]),
                reinterpret_cast<const uint32_t*>(&A[(block_row + row) * K + k_off + col]),
                sizeof(uint32_t));
        }
        for (int i = threadIdx.x; i < n_pairs_B; i += blockDim.x) {
            const int elem = i * 2;
            const int row  = elem / kBlockTileN;
            const int col  = elem % kBlockTileN;  // siempre par
            __pipeline_memcpy_async(
                reinterpret_cast<uint32_t*>(&sB[s][row][col]),
                reinterpret_cast<const uint32_t*>(&B[(k_off + row) * N + block_col + col]),
                sizeof(uint32_t));
        }
        __pipeline_commit();  // cierra el grupo s
    }

    // -- Bucle principal sobre tiles K --
    for (int tile = 0; tile < num_tiles; ++tile) {
        // prior decrece en los ultimos kNumStages-1 tiles porque ya no se emiten
        // commits nuevos: sin el ajuste, wait_prior(2) dejaria el tile actual pendiente.
        // Formula: min(kNumStages-1, tiles restantes despues del actual).
        const int prior = min(kNumStages - 1, num_tiles - tile - 1);
        __pipeline_wait_prior(prior);

        // Barrera de bloque: sincroniza los cp.async de todos los hilos antes de
        // que cualquier warp lea sA/sB con wmma::load_matrix_sync.
        __syncthreads();

        // -- Computo WMMA sobre la etapa actual --
        const int stage_c = tile % kNumStages;
        for (int k_inner = 0; k_inner < kKStep; k_inner += kWmmaK) {
            wmma::load_matrix_sync(a_frag,
                reinterpret_cast<const T*>(&sA[stage_c][warp_row * kWmmaM][k_inner]),
                kKStep + kWmmaShmemPad);
            wmma::load_matrix_sync(b_frag,
                reinterpret_cast<const T*>(&sB[stage_c][k_inner][warp_col * kWmmaN]),
                kBlockTileN + kWmmaShmemPad);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Barrera entre computo y carga futura: garantiza que TODOS los warps
        // terminaron wmma::load_matrix_sync sobre stage_c antes de que cualquier
        // warp empiece a sobreescribirlo con cp.async.
        // Sin esta barrera, un warp adelantado podria escribir stage_c mientras
        // otro warp aun lo esta leyendo -> race condition en shared memory.
        __syncthreads();

        // Emitir la carga futura DESPUES del computo: stage_c quedo libre ahora
        // y puede recibir el tile[tile+kNumStages] sin race condition.
        // El overlap con el computo de las proximas iteraciones se mantiene.
        const int future = tile + kNumStages;
        if (future < num_tiles) {
            const int k_off = future * kKStep;
            for (int i = threadIdx.x; i < n_pairs_A; i += blockDim.x) {
                const int elem = i * 2;
                const int row  = elem / kKStep;
                const int col  = elem % kKStep;
                __pipeline_memcpy_async(
                    reinterpret_cast<uint32_t*>(&sA[stage_c][row][col]),
                    reinterpret_cast<const uint32_t*>(&A[(block_row + row) * K + k_off + col]),
                    sizeof(uint32_t));
            }
            for (int i = threadIdx.x; i < n_pairs_B; i += blockDim.x) {
                const int elem = i * 2;
                const int row  = elem / kBlockTileN;
                const int col  = elem % kBlockTileN;
                __pipeline_memcpy_async(
                    reinterpret_cast<uint32_t*>(&sB[stage_c][row][col]),
                    reinterpret_cast<const uint32_t*>(&B[(k_off + row) * N + block_col + col]),
                    sizeof(uint32_t));
            }
            __pipeline_commit();
        }
    }

    // -- Escritura del fragmento acumulado a C (row-major) --
    if (warp_row_base < M && warp_col_base < N) {
        wmma::store_matrix_sync(
            C + warp_row_base * N + warp_col_base,
            c_frag, N,
            wmma::mem_row_major);
    }
}

// Benchmark de la ruta WMMA personalizada.
// Convierte A y B de FP32 col-major a T row-major en la GPU (T = __half o
// __nv_bfloat16), luego lanza wmma_gemm_kernel<T> y mide su tiempo con
// eventos CUDA.
// La salida C queda en FP32 row-major en el host para comparar con la
// referencia col-major usando compare_fp64_ref_colmaj_vs_fp32_rowmaj.
template <typename T>
static Metrics benchmark_gpu_wmma(const std::vector<float>& A,
                                   const std::vector<float>& B,
                                   std::vector<float>& C,
                                   int m, int n, int k,
                                   int iters) {
    if (m % kBlockTileM != 0 || n % kBlockTileN != 0 || k % kWmmaK != 0) {
        std::cerr << "WMMA requiere m y n multiplos de " << kBlockTileM
                  << " y k multiplo de " << kWmmaK << ".\n"
                  << "  m=" << m << " n=" << n << " k=" << k << "\n";
        std::exit(EXIT_FAILURE);
    }

    DeviceBuffer<float> dA_fp32(A.size());
    DeviceBuffer<float> dB_fp32(B.size());
    copy_float_vector_to_device(A, dA_fp32.get());
    copy_float_vector_to_device(B, dB_fp32.get());

    DeviceBuffer<T> dA_tc(A.size());
    DeviceBuffer<T> dB_tc(B.size());

    // Conversion: col-major FP32 → row-major T.
    // Se libera memoria FP32 scratch al salir del bloque.
    {
        const unsigned int bA = blocks_for_elements(A.size());
        const unsigned int bB = blocks_for_elements(B.size());
        float_colmaj_to_tc_rowmaj_kernel<T><<<bA, kConversionThreads>>>(
            dA_fp32.get(), dA_tc.get(), m, k);
        CHECK_CUDA(cudaGetLastError());
        float_colmaj_to_tc_rowmaj_kernel<T><<<bB, kConversionThreads>>>(
            dB_fp32.get(), dB_tc.get(), k, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    DeviceBuffer<float> dC(C.size());
    // 512 hilos/bloque (16 warps): permite 3 bloques simultaneos en el SM -> 100% ocupancia.
    const dim3 block(static_cast<unsigned int>(kBlockWarpsM * kBlockWarpsN * 32));
    // Grid 2D: un bloque por tile de salida 64x64.
    const dim3 grid(
        static_cast<unsigned int>((m + kBlockTileM - 1) / kBlockTileM),
        static_cast<unsigned int>((n + kBlockTileN - 1) / kBlockTileN));

    for (int i = 0; i < kWarmupIters; ++i) {
        wmma_gemm_kernel<T><<<grid, block>>>(
            dA_tc.get(), dB_tc.get(), dC.get(), m, n, k);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        wmma_gemm_kernel<T><<<grid, block>>>(
            dA_tc.get(), dB_tc.get(), dC.get(), m, n, k);
    }
    const float total_ms = timer.stop_and_elapsed_ms();
    CHECK_CUDA(cudaGetLastError());

    copy_float_vector_to_host(dC.get(), C);
    return build_metrics(m, n, k, static_cast<double>(total_ms) / iters);
}

// Presenta los resultados del experimento FP32.
// Los bloques TC/WMMA FP16 se imprimen si opt.tc_format es FP16 o Both;
// los bloques BF16 se imprimen si opt.tc_format es BF16 o Both.
static void print_float_report(const Options& opt,
                               const Metrics& cpu,
                               const Metrics& gpu,
                               const Metrics& tc,
                               const Metrics& wmma,
                               const Metrics& tc_bf16,
                               const Metrics& wmma_bf16,
                               const ErrorMetrics& cpu_error,
                               const ErrorMetrics& gpu_error,
                               const ErrorMetrics& tc_error,
                               const ErrorMetrics& wmma_error,
                               const ErrorMetrics& tc_bf16_error,
                               const ErrorMetrics& wmma_bf16_error) {
    const bool show_fp16 = (opt.tc_format == TensorCoreFormat::FP16 ||
                            opt.tc_format == TensorCoreFormat::Both);
    const bool show_bf16 = (opt.tc_format == TensorCoreFormat::BF16 ||
                            opt.tc_format == TensorCoreFormat::Both);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================ RESULTADOS GEMM FP32 =================\n";
    std::cout << "CPU BLAS - tiempo medio    : " << cpu.ms << " ms\n";
    std::cout << "CPU BLAS - rendimiento     : " << cpu.gflops << " GFLOP/s ("
              << cpu.tflops << " TFLOP/s)\n";
    std::cout << "Error max abs vs FP64      : " << cpu_error.max_abs << "\n";
    std::cout << "Error relativo L2 vs FP64  : " << cpu_error.rel_l2 << "\n\n";

    print_reference_comparison("GPU cuBLAS clasico", gpu, cpu.ms, gpu_error);

    if (show_fp16) {
        std::cout << "GPU cuBLAS TC - tiempo     : " << tc.ms << " ms\n";
        std::cout << "GPU cuBLAS TC - rend.      : " << tc.gflops << " GFLOP/s ("
                  << tc.tflops << " TFLOP/s)\n";
        std::cout << "Speedup TC vs CPU          : " << cpu.ms / tc.ms << "x\n";
        std::cout << "Speedup TC vs GPU clasico  : " << gpu.ms / tc.ms << "x\n";
        std::cout << "Error max abs vs FP64      : " << tc_error.max_abs << "\n";
        std::cout << "Error relativo L2 vs FP64  : " << tc_error.rel_l2 << "\n\n";

        std::cout << "GPU WMMA custom - tiempo   : " << wmma.ms << " ms\n";
        std::cout << "GPU WMMA custom - rend.    : " << wmma.gflops << " GFLOP/s ("
                  << wmma.tflops << " TFLOP/s)\n";
        std::cout << "Speedup WMMA vs CPU        : " << cpu.ms / wmma.ms << "x\n";
        std::cout << "Speedup WMMA vs GPU clasico: " << gpu.ms / wmma.ms << "x\n";
        std::cout << "Speedup WMMA vs cuBLAS TC  : " << tc.ms / wmma.ms << "x\n";
        std::cout << "Error max abs vs FP64      : " << wmma_error.max_abs << "\n";
        std::cout << "Error relativo L2 vs FP64  : " << wmma_error.rel_l2 << "\n\n";
    }

    if (show_bf16) {
        std::cout << "GPU cuBLAS TC BF16 - tiempo     : " << tc_bf16.ms << " ms\n";
        std::cout << "GPU cuBLAS TC BF16 - rend.      : " << tc_bf16.gflops << " GFLOP/s ("
                  << tc_bf16.tflops << " TFLOP/s)\n";
        std::cout << "Speedup TC BF16 vs CPU          : " << cpu.ms / tc_bf16.ms << "x\n";
        std::cout << "Speedup TC BF16 vs GPU clasico  : " << gpu.ms / tc_bf16.ms << "x\n";
        std::cout << "Error max abs vs FP64           : " << tc_bf16_error.max_abs << "\n";
        std::cout << "Error relativo L2 vs FP64       : " << tc_bf16_error.rel_l2 << "\n\n";

        std::cout << "GPU WMMA BF16 custom - tiempo   : " << wmma_bf16.ms << " ms\n";
        std::cout << "GPU WMMA BF16 custom - rend.    : " << wmma_bf16.gflops << " GFLOP/s ("
                  << wmma_bf16.tflops << " TFLOP/s)\n";
        std::cout << "Speedup WMMA BF16 vs CPU        : " << cpu.ms / wmma_bf16.ms << "x\n";
        std::cout << "Speedup WMMA BF16 vs GPU clasico: " << gpu.ms / wmma_bf16.ms << "x\n";
        std::cout << "Speedup WMMA BF16 vs cuBLAS TC  : " << tc_bf16.ms / wmma_bf16.ms << "x\n";
        std::cout << "Error max abs vs FP64           : " << wmma_bf16_error.max_abs << "\n";
        std::cout << "Error relativo L2 vs FP64       : " << wmma_bf16_error.rel_l2 << "\n";
    }
    std::cout << "=======================================================\n";
}

// Presenta los resultados del experimento FP64.
static void print_double_report(const Metrics& cpu,
                                const Metrics& gpu,
                                const ErrorMetrics& gpu_error) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================ RESULTADOS GEMM FP64 =================\n";
    std::cout << "CPU BLAS - tiempo medio    : " << cpu.ms << " ms\n";
    std::cout << "CPU BLAS - rendimiento     : " << cpu.gflops << " GFLOP/s ("
              << cpu.tflops << " TFLOP/s)\n\n";

    print_reference_comparison("GPU cuBLAS clasico", gpu, cpu.ms, gpu_error);

    std::cout << "=======================================================\n";
}

// Orquesta el experimento FP32 completo:
// inicializacion, benchmark en CPU, benchmark en GPU y comparacion numerica.
static void run_experiment_float(const Options& opt) {
    if (!active_device_supports_fp16_tensor_cores()) {
        std::cerr << "La ruta Tensor Core FP16 requiere compute capability 7.0 o superior."
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const bool want_fp16 = (opt.tc_format == TensorCoreFormat::FP16 ||
                            opt.tc_format == TensorCoreFormat::Both);
    const bool want_bf16 = (opt.tc_format == TensorCoreFormat::BF16 ||
                            opt.tc_format == TensorCoreFormat::Both);

    if (want_bf16 && !active_device_supports_bf16_tensor_cores()) {
        std::cerr << "La ruta Tensor Core BF16 requiere arquitectura Ampere o superior"
                     " (compute capability >= 8.0)." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const size_t size_a = checked_element_count(opt.m, opt.k, "A");
    const size_t size_b = checked_element_count(opt.k, opt.n, "B");
    const size_t size_c = checked_element_count(opt.m, opt.n, "C");

    std::vector<float> A(size_a);
    std::vector<float> B(size_b);
    std::vector<float> C_cpu(size_c, 0.0f);
    std::vector<float> C_gpu(size_c, 0.0f);
    std::vector<float> C_tc(size_c, 0.0f);
    std::vector<float> C_wmma(size_c, 0.0f);
    std::vector<float> C_tc_bf16(size_c, 0.0f);
    std::vector<float> C_wmma_bf16(size_c, 0.0f);

    initialize_matrix_float(A);
    initialize_matrix_float(B);

    // Referencia FP64: las mismas entradas casteadas a double para un ground truth preciso.
    std::vector<double> A_d(size_a), B_d(size_b), C_ref(size_c, 0.0);
    for (size_t i = 0; i < size_a; ++i) A_d[i] = static_cast<double>(A[i]);
    for (size_t i = 0; i < size_b; ++i) B_d[i] = static_cast<double>(B[i]);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                opt.m, opt.n, opt.k,
                1.0, A_d.data(), opt.m, B_d.data(), opt.k,
                0.0, C_ref.data(), opt.m);

    const Metrics cpu = benchmark_cpu_float(A, B, C_cpu, opt.m, opt.n, opt.k, opt.iters);
    const Metrics gpu = benchmark_gpu_cublas_float(A, B, C_gpu, opt.m, opt.n, opt.k, opt.iters);

    Metrics tc{}, wmma{}, tc_bf16{}, wmma_bf16{};
    ErrorMetrics tc_error{}, wmma_error{}, tc_bf16_error{}, wmma_bf16_error{};

    if (want_fp16) {
        tc   = benchmark_gpu_tensor_cores(A, B, C_tc, opt.m, opt.n, opt.k, opt.iters);
        wmma = benchmark_gpu_wmma<__half>(A, B, C_wmma, opt.m, opt.n, opt.k, opt.iters);
        tc_error   = compare_fp64_ref_vs_fp32(C_ref, C_tc);
        wmma_error = compare_fp64_ref_colmaj_vs_fp32_rowmaj(C_ref, C_wmma, opt.m, opt.n);
    }
    if (want_bf16) {
        tc_bf16   = benchmark_gpu_tensor_cores_bf16(A, B, C_tc_bf16, opt.m, opt.n, opt.k, opt.iters);
        wmma_bf16 = benchmark_gpu_wmma<__nv_bfloat16>(A, B, C_wmma_bf16, opt.m, opt.n, opt.k, opt.iters);
        tc_bf16_error   = compare_fp64_ref_vs_fp32(C_ref, C_tc_bf16);
        wmma_bf16_error = compare_fp64_ref_colmaj_vs_fp32_rowmaj(C_ref, C_wmma_bf16, opt.m, opt.n);
    }

    const ErrorMetrics cpu_error = compare_fp64_ref_vs_fp32(C_ref, C_cpu);
    const ErrorMetrics gpu_error = compare_fp64_ref_vs_fp32(C_ref, C_gpu);

    print_float_report(opt, cpu, gpu, tc, wmma, tc_bf16, wmma_bf16,
                       cpu_error, gpu_error, tc_error, wmma_error,
                       tc_bf16_error, wmma_bf16_error);
}

// Orquesta el experimento FP64 completo.
static void run_experiment_double(const Options& opt) {
    const size_t size_a = checked_element_count(opt.m, opt.k, "A");
    const size_t size_b = checked_element_count(opt.k, opt.n, "B");
    const size_t size_c = checked_element_count(opt.m, opt.n, "C");

    std::vector<double> A(size_a);
    std::vector<double> B(size_b);
    std::vector<double> C_cpu(size_c, 0.0);
    std::vector<double> C_gpu(size_c, 0.0);

    initialize_matrix_double(A);
    initialize_matrix_double(B);

    const Metrics cpu = benchmark_cpu_double(A, B, C_cpu, opt.m, opt.n, opt.k, opt.iters);
    const Metrics gpu = benchmark_gpu_cublas_double(A, B, C_gpu, opt.m, opt.n, opt.k, opt.iters);
    const ErrorMetrics gpu_error = compare_double_vectors(C_cpu, C_gpu);

    print_double_report(cpu, gpu, gpu_error);
}

// Imprime la configuracion y selecciona la ruta segun la precision pedida.
static void run_benchmark(const Options& opt) {
    std::cout << "================== CONFIGURACION ==================\n";
    std::cout << "Precision                  : " << (opt.use_double ? "FP64 (double)" : "FP32 (float)") << "\n";
    std::cout << "Dimensiones (M, N, K)      : " << opt.m << ", " << opt.n << ", " << opt.k << "\n";
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
