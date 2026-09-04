// Fase_2/GEMM/gemm_tensor_activation.cu
//
// Migrado desde old/Fase_2/GEMM/gemm_tensor_activation.cu (codigo ya
// auditado). Compara cinco rutas de GEMM en precision mixta: CPU OpenBLAS,
// cuBLAS clasico (sin Tensor Cores), cuBLAS con Tensor Cores (cublasGemmEx,
// FP16/BF16), un kernel WMMA propio (FP16/BF16) y CUTLASS
// (cutlass::gemm::device::Gemm, FP16/BF16), todas contra una referencia
// FP64. Ver Fase_2/GEMM/README.md para como interpretar los resultados sin
// combinar comparaciones que no son equivalentes.
//
// Unico cambio de fondo respecto al original: CHECK_CUDA/CHECK_CUBLAS,
// CudaEventTimer, Metrics, ErrorMetrics y las funciones compare_* ya no se
// definen en este archivo (venian de old/Fase_2/common.cuh, incluido como
// "../common.cuh") -- ahora se toman de common/cuda_checks.cuh y
// common/metrics.cuh, que exponen los mismos nombres y firmas. El resto del
// codigo, incluyendo la logica numerica de cada kernel, es identico.
//
// nvcc -std=c++17 gemm_tensor_activation.cu -o gemm_tc -I/usr/include/openblas -lcublas -lopenblas -gencode arch=compute_80,code=sm_80 --allow-unsupported-compiler
// ./gemm_tc --m 2048 --n 2048 --k 2048 --iters 20
// En PACCA (A100, sm_80) la compilacion y el perfilado con Nsight Compute se
// lanzan via SLURM: sbatch run_gemm_tc.sbatch  (no ejecutar ncu con sudo).
//
// Ruta 5 (CUTLASS, --cutlass): requiere ademas -I$CUTLASS_DIR/include
// apuntando a un checkout de github.com/NVIDIA/cutlass (serie 2.x, ver
// REQUIREMENTS.md) y --expt-relaxed-constexpr, que CUTLASS 2.x exige para
// su metaprogramacion constexpr host/device. Si CUTLASS no esta disponible
// en el include path, el binario compila igual (ver el guard __has_include
// mas abajo) pero --cutlass termina el proceso con un mensaje explicando
// como habilitarlo, en vez de fallar la compilacion para todos.
// nvcc -std=c++17 gemm_tensor_activation.cu -o gemm_tc -I/usr/include/openblas -I$CUTLASS_DIR/include -lcublas -lopenblas -gencode arch=compute_80,code=sm_80 --expt-relaxed-constexpr --allow-unsupported-compiler
// ./gemm_tc --m 2048 --n 2048 --k 2048 --iters 20 --cutlass


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

// CUTLASS (ruta 5) es header-only y opcional: si el include path no tiene un
// checkout de github.com/NVIDIA/cutlass (variable CUTLASS_DIR en
// run_gemm_tc.sbatch, ver REQUIREMENTS.md), el binario debe seguir
// compilando para las cuatro rutas restantes -- __has_include evita que la
// ausencia de CUTLASS rompa la compilacion de todo el archivo. Se incluye a
// nivel de archivo (NO dentro del namespace anonimo de mas abajo) porque,
// a diferencia de common/cuda_checks.cuh y common/metrics.cuh, CUTLASS es
// una libreria externa con su propio namespace (cutlass::), no un header
// interno del proyecto cuyos simbolos haya que aislar con enlace interno.
#if __has_include(<cutlass/gemm/device/gemm.h>)
#define HAVE_CUTLASS 1
#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#else
#define HAVE_CUTLASS 0
#endif

namespace {

// Ver la nota de uso al inicio de cada header: se incluyen DENTRO de este
// bloque de namespace anonimo para que sus simbolos conserven enlace
// interno, igual que cuando estaban duplicados dentro de old/Fase_2/common.cuh.
// metrics.cuh incluye cuda_checks.cuh internamente (CudaEventTimer usa
// CHECK_CUDA), pero se listan ambos explicitamente porque este archivo usa
// CHECK_CUDA/CHECK_CUBLAS directamente y no solo a traves de metrics.cuh.
#include "../../common/cuda_checks.cuh"
#include "../../common/metrics.cuh"
// wmma_gemm.cuh define kWmmaM/N/K, kKStep, kBlockWarps*, kBlockTile*,
// kNumStages, kSmemStride*, kVecElems/A/B, float_to_tc_scalar<T>,
// tc_scalar_to_float<T>, float_colmaj_to_tc_rowmaj_kernel<T>,
// issue_stage_copy<T> y wmma_gemm_kernel<T> -- todo lo que antes estaba
// definido localmente en este archivo, ahora compartido con Fase 3/4 (ver
// common/wmma_gemm.cuh). Mismos nombres, misma logica: el resto de este
// archivo no necesita mas cambios que este include.
#include "../../common/wmma_gemm.cuh"

constexpr int kWarmupIters = 3;
constexpr int kCpuWarmupIters = 3;
constexpr int kConversionThreads = 256;

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
    // Ruta 5. Reutiliza tc_format para elegir fp16/bf16/both, igual que las
    // rutas 3 y 4 -- ver print_usage y run_experiment_float.
    bool use_cutlass = false;
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
              << " [--tc-format fp16|bf16|both] [--cutlass]\n\n"
              << "Descripcion:\n"
              << "  Compara hasta cinco rutas de GEMM en precision mixta:\n"
              << "    1. CPU BLAS (OpenBLAS, FP32/FP64)\n"
              << "    2. GPU cuBLAS clasico (FP32/FP64, sin Tensor Cores)\n"
              << "    3. GPU cuBLAS con Tensor Cores (FP16/BF16->FP32, cublasGemmEx)\n"
              << "    4. GPU WMMA custom (FP16/BF16->FP32, kernel propio con shared memory)\n"
              << "    5. GPU CUTLASS (FP16/BF16->FP32, cutlass::gemm::device::Gemm), con --cutlass\n"
              << "  La ruta WMMA (4) requiere m y n multiplos de 64 y k multiplo de 32.\n"
              << "  Con --double solo se ejecutan las rutas 1 y 2 (3, 4 y 5 son FP16/BF16).\n"
              << "  --tc-format selecciona el formato de las rutas 3, 4 y 5 (por defecto fp16).\n"
              << "  BF16 requiere GPU Ampere o superior (compute capability >= 8.0).\n"
              << "  --cutlass activa la ruta 5. Requiere compute capability >= 8.0 (el\n"
              << "  template usa ArchTag Sm80) y que el binario se haya compilado con\n"
              << "  -I$CUTLASS_DIR/include (ver REQUIREMENTS.md); si no, termina con error.\n\n"
              << "Ejemplos:\n"
              << "  " << prog << "\n"
              << "  " << prog << " --m 4096 --n 4096 --k 4096 --iters 10\n"
              << "  " << prog << " --double --m 2048 --n 2048 --k 2048 --iters 5\n"
              << "  " << prog << " --m 1024 --n 1024 --k 1024 --iters 5 --tc-format bf16\n"
              << "  " << prog << " --m 2048 --n 2048 --k 2048 --iters 10 --tc-format both --cutlass\n";
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
        } else if (std::strcmp(argv[i], "--cutlass") == 0) {
            opt.use_cutlass = true;
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
    // El kernel recorre K en num_tiles = K / kKStep pasos, de modo que un k
    // multiplo de kWmmaK pero no de kKStep (p. ej. 48) descartaba en silencio
    // la cola de K. La condicion correcta es k % kKStep == 0, que ademas es la
    // que garantiza el alineamiento a 16 B de las cp.async sobre A.
    if (m % kBlockTileM != 0 || n % kBlockTileN != 0 || k % kKStep != 0) {
        std::cerr << "WMMA requiere m multiplo de " << kBlockTileM
                  << ", n multiplo de " << kBlockTileN
                  << " y k multiplo de " << kKStep << ".\n"
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
    // 512 hilos/bloque (16 warps); ver WMMA_MIN_BLOCKS_PER_SM para la ocupancia.
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

// =========================================================================
// Ruta 5 - CUTLASS (cutlass::gemm::device::Gemm, API 2.x)
//
// CUTLASS es la libreria de templates de NVIDIA para GEMM con Tensor Cores.
// A diferencia de cuBLAS TC (ruta 3, binario cerrado y afinado por NVIDIA) y
// del kernel WMMA propio (ruta 4, tiling/pipeline escritos a mano en este
// proyecto), CUTLASS da una implementacion de referencia OFICIAL pero
// instanciada por templates de C++ en tiempo de compilacion -- un punto
// intermedio en calidad de implementacion entre (3) y (4). Ver
// Fase_2/GEMM/README.md, seccion "Como interpretar los resultados", punto (c).
//
// Toda la combinacion de tipos/formas de abajo (ThreadblockShape, WarpShape,
// InstructionShape, EpilogueOp, numero de etapas, y el patron de
// Gemm::Arguments con listas {puntero, ld}) sigue, casi literal, el patron
// de examples/08_turing_tensorop_gemm.cu del repositorio NVIDIA/cutlass (API
// 2.x, cutlass::gemm::device::Gemm) mas el snippet canonico del README
// principal de NVIDIA/cutlass para construir Gemm::Arguments. El unico
// cambio deliberado es ArchTag: Sm75 (Turing, el ejemplo original) ->
// Sm80 (Ampere, A100 de PACCA); el resto de la forma (128x128x32 de
// threadblock, 64x64x32 de warp, 16x8x16 de instruccion HMMA, 3 etapas) es
// la combinacion que usan la mayoria de los ejemplos oficiales de CUTLASS
// para FP16/BF16 en Ampere, no un valor inventado para este proyecto.
//
// Verificado contra CUTLASS v2.11.0 (compila y corre en Ampere/Ada real):
//   1. GemmIdentityThreadblockSwizzle y su header resolvieron sin cambios.
//   2. La lista de campos de Gemm::Arguments (problem_size, ref_A, ref_B,
//      ref_C, ref_D, {alpha,beta}, split_k_slices), con split_k_slices=1
//      como ultimo campo posicional, coincide con el patron de los ejemplos
//      oficiales.
//   3. cutlass::Status no se decodifica con un switch propio (como
//      cublas_status_to_string); CHECK_CUTLASS solo imprime el valor
//      numerico. cutlass::cutlassGetStatusString() da un mensaje legible en
//      la mayoria de versiones 2.x si se prefiere ese detalle.
// Una version de CUTLASS distinta a 2.11.0 podria diferir en nombres de
// header o firma de estos simbolos -- revisar aqui primero si aparece un
// error de compilacion con otro checkout.
// =========================================================================
#if HAVE_CUTLASS

// Traduce (parcialmente) un cutlass::Status a texto. Solo distingue
// kSuccess de "cualquier otra cosa": un nombre de enumerador equivocado
// rompe la compilacion de este archivo entero, mientras que imprimir el
// codigo numerico crudo como fallback es seguro en cualquier version de
// CUTLASS 2.x.
static const char* cutlass_status_to_string(cutlass::Status status) {
    if (status == cutlass::Status::kSuccess) {
        return "kSuccess";
    }
    return "CUTLASS_STATUS_ERROR (ver codigo numerico impreso junto a este mensaje)";
}

#define CHECK_CUTLASS(call)                                                  \
    do {                                                                     \
        cutlass::Status cutlass_status = (call);                             \
        if (cutlass_status != cutlass::Status::kSuccess) {                   \
            std::cerr << "CUTLASS error at " << __FILE__ << ":" << __LINE__  \
                      << " -> " << cutlass_status_to_string(cutlass_status)  \
                      << " (status code "                                    \
                      << static_cast<int>(cutlass_status) << ")" << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// Configuracion de tipos/formas de CUTLASS para un ElementInput dado
// (cutlass::half_t o cutlass::bfloat16_t). Ver la nota grande de arriba
// sobre el origen de cada parametro de forma.
template <typename ElementInput>
struct CutlassGemmConfig {
    using ElementOutput = float;
    using ElementAccumulator = float;
    using LayoutInput = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::ColumnMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput, LayoutInput,
        ElementInput, LayoutInput,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,   // ThreadblockShape
        cutlass::gemm::GemmShape<64, 64, 32>,     // WarpShape
        cutlass::gemm::GemmShape<16, 8, 16>,      // InstructionShape (HMMA FP16 en Ampere)
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator,
            ElementAccumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3>;  // NumStages
};

// Conversion escalar float -> ElementInput usando el constructor explicito
// que cutlass::half_t/cutlass::bfloat16_t proveen desde float (cutlass/half.h,
// cutlass/bfloat16.h). Kernel separado de convert_float_to_half_kernel /
// convert_float_to_bfloat16_kernel (ruta 3) porque cutlass::half_t y
// __half (idem bfloat16_t / __nv_bfloat16) son tipos de C++ distintos,
// aunque bit-compatibles en memoria.
template <typename ElementInput>
__global__ static void convert_float_to_cutlass_kernel(
        const float* __restrict__ src, ElementInput* __restrict__ dst, size_t size) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = static_cast<ElementInput>(src[idx]);
    }
}

// Convierte dos buffers FP32 a ElementInput dentro de la GPU. Analoga a
// convert_fp32_buffers_to_fp16/convert_fp32_buffers_to_bf16 (ruta 3).
template <typename ElementInput>
static void convert_fp32_buffers_to_cutlass(const float* src_a,
                                             const float* src_b,
                                             ElementInput* dst_a,
                                             ElementInput* dst_b,
                                             size_t size_a,
                                             size_t size_b) {
    const unsigned int blocks_a = blocks_for_elements(size_a);
    const unsigned int blocks_b = blocks_for_elements(size_b);

    convert_float_to_cutlass_kernel<ElementInput><<<blocks_a, kConversionThreads>>>(
        src_a, dst_a, size_a);
    CHECK_CUDA(cudaGetLastError());

    convert_float_to_cutlass_kernel<ElementInput><<<blocks_b, kConversionThreads>>>(
        src_b, dst_b, size_b);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());
}

// Lanza una GEMM de CUTLASS ya inicializada (gemm_op.initialize ya corrio).
// Analoga a run_tensor_core_gemm: una llamada delgada con verificacion de
// error, pensada para reutilizarse igual dentro del bucle de warmup y del
// bucle cronometrado.
template <typename Gemm>
static void run_cutlass_gemm(Gemm& gemm_op) {
    CHECK_CUTLASS(gemm_op());
}

// Ejecuta la ruta de precision mixta con CUTLASS. Misma estructura que
// benchmark_gpu_tensor_cores/benchmark_gpu_wmma: conversion FP32->ElementInput,
// warmup, medicion con CudaEventTimer, copia de vuelta a host.
//
// Diferencia deliberada con las otras rutas: la construccion de argumentos
// y gemm_op.initialize() (que incluye can_implement y el workspace) se hacen
// UNA sola vez antes del warmup, no en cada iteracion -- igual que el
// handle de cuBLAS se crea una sola vez fuera del bucle en las rutas 2 y 3.
// Repetir initialize() en cada iteracion mediria tambien el costo de
// configurar la grilla, no solo el de ejecutar la GEMM, y la comparacion de
// tiempos con las demas rutas dejaria de ser justa.
template <typename ElementInput>
static Metrics benchmark_gpu_cutlass(const std::vector<float>& A,
                                     const std::vector<float>& B,
                                     std::vector<float>& C,
                                     int m, int n, int k,
                                     int iters) {
    using Config = CutlassGemmConfig<ElementInput>;
    using Gemm = typename Config::Gemm;

    DeviceBuffer<ElementInput> dA(A.size());
    DeviceBuffer<ElementInput> dB(B.size());
    DeviceBuffer<float> dC(C.size());

    {
        DeviceBuffer<float> dA_fp32(A.size());
        DeviceBuffer<float> dB_fp32(B.size());
        copy_float_vector_to_device(A, dA_fp32.get());
        copy_float_vector_to_device(B, dB_fp32.get());
        convert_fp32_buffers_to_cutlass<ElementInput>(
            dA_fp32.get(), dB_fp32.get(), dA.get(), dB.get(), A.size(), B.size());
    }

    // Leading dimensions: A, B y C col-major, igual que en las rutas 1-3
    // (m, k, m respectivamente) -- ver benchmark_gpu_cublas_float. A
    // diferencia de WMMA (ruta 4), CUTLASS aqui no necesita una conversion a
    // row-major: la salida se compara con compare_fp64_ref_vs_fp32 (misma
    // funcion que usan cpu/gpu/tc), no con la variante _colmaj_vs_fp32_rowmaj.
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Aggregate-init con listas {puntero, ld}: patron canonico de la seccion
    // "Instantiate CUTLASS GEMM..." del README de NVIDIA/cutlass, donde cada
    // {ptr, ld} construye implicitamente un cutlass::TensorRef via el
    // constructor no-explicito de cutlass::layout::ColumnMajor(ld).
    typename Gemm::Arguments arguments{
        {m, n, k},
        {dA.get(), lda},
        {dB.get(), ldb},
        {dC.get(), ldc},
        {dC.get(), ldc},
        {alpha, beta},
        1  // split_k_slices: sin split-K, un solo bloque de acumulacion en K.
    };

    Gemm gemm_op;
    CHECK_CUTLASS(gemm_op.can_implement(arguments));

    const size_t workspace_size = Gemm::get_workspace_size(arguments);
    // DeviceBuffer<T> con count==0 no reserva memoria y get() devuelve
    // nullptr (ver la clase mas arriba) -- exactamente lo que
    // gemm_op.initialize espera cuando no hace falta workspace.
    DeviceBuffer<uint8_t> workspace(workspace_size);
    CHECK_CUTLASS(gemm_op.initialize(arguments, workspace.get()));

    for (int i = 0; i < kWarmupIters; ++i) {
        run_cutlass_gemm(gemm_op);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        run_cutlass_gemm(gemm_op);
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    copy_float_vector_to_host(dC.get(), C);

    return build_metrics(m, n, k, total_ms / iters);
}

#endif  // HAVE_CUTLASS

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
                               const Metrics& cutlass_fp16,
                               const Metrics& cutlass_bf16,
                               const ErrorMetrics& cpu_error,
                               const ErrorMetrics& gpu_error,
                               const ErrorMetrics& tc_error,
                               const ErrorMetrics& wmma_error,
                               const ErrorMetrics& tc_bf16_error,
                               const ErrorMetrics& wmma_bf16_error,
                               const ErrorMetrics& cutlass_fp16_error,
                               const ErrorMetrics& cutlass_bf16_error) {
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

        if (opt.use_cutlass) {
            std::cout << "GPU CUTLASS - tiempo       : " << cutlass_fp16.ms << " ms\n";
            std::cout << "GPU CUTLASS - rend.        : " << cutlass_fp16.gflops << " GFLOP/s ("
                      << cutlass_fp16.tflops << " TFLOP/s)\n";
            std::cout << "Speedup CUTLASS vs CPU        : " << cpu.ms / cutlass_fp16.ms << "x\n";
            std::cout << "Speedup CUTLASS vs GPU clasico: " << gpu.ms / cutlass_fp16.ms << "x\n";
            std::cout << "Speedup CUTLASS vs cuBLAS TC  : " << tc.ms / cutlass_fp16.ms << "x\n";
            std::cout << "Speedup CUTLASS vs WMMA custom: " << wmma.ms / cutlass_fp16.ms << "x\n";
            std::cout << "Error max abs vs FP64      : " << cutlass_fp16_error.max_abs << "\n";
            std::cout << "Error relativo L2 vs FP64  : " << cutlass_fp16_error.rel_l2 << "\n\n";
        }
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

        if (opt.use_cutlass) {
            std::cout << "\n";
            std::cout << "GPU CUTLASS BF16 - tiempo       : " << cutlass_bf16.ms << " ms\n";
            std::cout << "GPU CUTLASS BF16 - rend.        : " << cutlass_bf16.gflops << " GFLOP/s ("
                      << cutlass_bf16.tflops << " TFLOP/s)\n";
            std::cout << "Speedup CUTLASS BF16 vs CPU        : " << cpu.ms / cutlass_bf16.ms << "x\n";
            std::cout << "Speedup CUTLASS BF16 vs GPU clasico: " << gpu.ms / cutlass_bf16.ms << "x\n";
            std::cout << "Speedup CUTLASS BF16 vs cuBLAS TC  : " << tc_bf16.ms / cutlass_bf16.ms << "x\n";
            std::cout << "Speedup CUTLASS BF16 vs WMMA custom: " << wmma_bf16.ms / cutlass_bf16.ms << "x\n";
            std::cout << "Error max abs vs FP64           : " << cutlass_bf16_error.max_abs << "\n";
            std::cout << "Error relativo L2 vs FP64       : " << cutlass_bf16_error.rel_l2 << "\n";
        }
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

    // La ruta 5 (CUTLASS) usa ArchTag Sm80 en su template (ver
    // CutlassGemmConfig): igual que BF16, requiere Ampere o superior.
    // Reutilizamos active_device_supports_bf16_tensor_cores() porque su
    // condicion (compute capability >= 8.0) es exactamente la que Sm80
    // exige, no porque tenga relacion logica con BF16 en si.
    if (opt.use_cutlass && !active_device_supports_bf16_tensor_cores()) {
        std::cerr << "La ruta CUTLASS (ArchTag Sm80) requiere arquitectura Ampere o superior"
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
    std::vector<float> C_cutlass_fp16(size_c, 0.0f);
    std::vector<float> C_cutlass_bf16(size_c, 0.0f);

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

    Metrics tc{}, wmma{}, tc_bf16{}, wmma_bf16{}, cutlass_fp16{}, cutlass_bf16{};
    ErrorMetrics tc_error{}, wmma_error{}, tc_bf16_error{}, wmma_bf16_error{};
    ErrorMetrics cutlass_fp16_error{}, cutlass_bf16_error{};

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

    if (opt.use_cutlass) {
#if HAVE_CUTLASS
        if (want_fp16) {
            cutlass_fp16 = benchmark_gpu_cutlass<cutlass::half_t>(
                A, B, C_cutlass_fp16, opt.m, opt.n, opt.k, opt.iters);
            cutlass_fp16_error = compare_fp64_ref_vs_fp32(C_ref, C_cutlass_fp16);
        }
        if (want_bf16) {
            cutlass_bf16 = benchmark_gpu_cutlass<cutlass::bfloat16_t>(
                A, B, C_cutlass_bf16, opt.m, opt.n, opt.k, opt.iters);
            cutlass_bf16_error = compare_fp64_ref_vs_fp32(C_ref, C_cutlass_bf16);
        }
#else
        std::cerr << "Se pidio --cutlass pero el binario se compilo sin CUTLASS disponible"
                     " en el include path.\n"
                  << "Recompila agregando -I$CUTLASS_DIR/include, apuntando a un checkout de"
                     " github.com/NVIDIA/cutlass (serie 2.x) -- ver REQUIREMENTS.md y"
                     " Fase_2/GEMM/README.md." << std::endl;
        std::exit(EXIT_FAILURE);
#endif
    }

    const ErrorMetrics cpu_error = compare_fp64_ref_vs_fp32(C_ref, C_cpu);
    const ErrorMetrics gpu_error = compare_fp64_ref_vs_fp32(C_ref, C_gpu);

    print_float_report(opt, cpu, gpu, tc, wmma, tc_bf16, wmma_bf16,
                       cutlass_fp16, cutlass_bf16,
                       cpu_error, gpu_error, tc_error, wmma_error,
                       tc_bf16_error, wmma_bf16_error,
                       cutlass_fp16_error, cutlass_bf16_error);
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
