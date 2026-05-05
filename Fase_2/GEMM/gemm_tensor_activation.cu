// nvcc -std=c++17 gemm_tensor_activation.cu -o gemm_tc -I/usr/include/openblas -lcublas -lopenblas -gencode arch=compute_86,code=sm_86
// ./gemm_tc --m 2048 --n 2048 --k 2048 --iters 20
// sudo /usr/local/cuda/bin/ncu --set full --export reporte_fase2_rtx3050 --force-overwrite ./gemm_tc


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

#include <openblas/cblas.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

// Valida llamadas a la API de CUDA y termina el programa si ocurre un error.
#define CHECK_CUDA(call) do { \
cudaError_t err = (call); \
if (err != cudaSuccess) { \
    std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
    << " -> " << cudaGetErrorString(err) << std::endl; \
    std::exit(EXIT_FAILURE); \
} \
} while(0)

// Valida llamadas a cuBLAS y termina el programa si la biblioteca reporta fallo.
#define CHECK_CUBLAS(call) do { \
cublasStatus_t status = (call); \
if (status != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
    << " -> " << cublas_status_to_string(status) \
    << " (status code " << status << ")" << std::endl; \
    std::exit(EXIT_FAILURE); \
} \
} while(0)

namespace {

constexpr int kWarmupIters = 3;
constexpr int kCpuWarmupIters = 1;
constexpr int kConversionThreads = 256;

// Dimensiones del fragmento WMMA para FP16: unicas dimensiones soportadas en sm_70+.
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// Numero de warps por dimension dentro de un bloque (2×2 = 4 warps = 128 hilos).
constexpr int kBlockWarpsM = 2;
constexpr int kBlockWarpsN = 2;

// Tile de salida que maneja un bloque completo: 32×32 elementos FP32.
constexpr int kBlockTileM = kBlockWarpsM * kWmmaM;  // 32
constexpr int kBlockTileN = kBlockWarpsN * kWmmaN;  // 32

// Padding en shared memory para evitar bank conflicts entre warps.
// Con FP16 (2 bytes) y 32 bancos de 4 bytes, añadir 8 elementos desplaza
// cada fila 16 bytes extra, eliminando el patron de conflicto ciclico.
constexpr int kWmmaShmemPad = 8;

// Parametros de entrada configurables desde linea de comandos.
struct Options {
    int m = 2048;
    int n = 2048;
    int k = 2048;
    int iters = 20;
    bool use_double = false;
};

// Metricas de rendimiento promedio para una ruta de GEMM.
struct Metrics {
    double ms = 0.0;
    double gflops = 0.0;
    double tflops = 0.0;
};

// Metricas para comparar la salida numerica frente a una referencia.
struct ErrorMetrics {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
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

// Temporizador basado en eventos CUDA.
// Mide tiempo en la GPU, evitando incluir latencias del lado del CPU.
class CudaEventTimer {
public:
    CudaEventTimer() {
        CHECK_CUDA(cudaEventCreate(&start_));
        CHECK_CUDA(cudaEventCreate(&stop_));
    }

    ~CudaEventTimer() noexcept {
        if (start_ != nullptr) {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr) {
            cudaEventDestroy(stop_);
        }
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start() {
        CHECK_CUDA(cudaEventRecord(start_));
    }

    float stop_and_elapsed_ms() {
        CHECK_CUDA(cudaEventRecord(stop_));
        CHECK_CUDA(cudaEventSynchronize(stop_));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
        return elapsed_ms;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
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
              << "  " << prog << " [--m M] [--n N] [--k K] [--iters I] [--double]\n\n"
              << "Descripcion:\n"
              << "  Compara cuatro rutas de GEMM en precision mixta:\n"
              << "    1. CPU BLAS (OpenBLAS, FP32/FP64)\n"
              << "    2. GPU cuBLAS clasico (FP32/FP64, sin Tensor Cores)\n"
              << "    3. GPU cuBLAS con Tensor Cores (FP16->FP32, cublasGemmEx)\n"
              << "    4. GPU WMMA custom (FP16->FP32, kernel propio con shared memory)\n"
              << "  La ruta WMMA (4) requiere m y n multiplos de 32 y k multiplo de 16.\n"
              << "  Con --double solo se ejecutan las rutas 1 y 2 (WMMA y TC son FP16).\n\n"
              << "Ejemplos:\n"
              << "  " << prog << "\n"
              << "  " << prog << " --m 4096 --n 4096 --k 4096 --iters 10\n"
              << "  " << prog << " --double --m 2048 --n 2048 --k 2048 --iters 5\n";
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

// Compara dos resultados FP32 usando error maximo absoluto y norma relativa L2.
static ErrorMetrics compare_float_vectors(const std::vector<float>& ref, const std::vector<float>& test) {
    ErrorMetrics out;
    double squared_error_sum = 0.0;
    double squared_ref_sum = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const double diff = static_cast<double>(ref[i]) - static_cast<double>(test[i]);
        out.max_abs = std::max(out.max_abs, std::abs(diff));
        squared_error_sum += diff * diff;
        squared_ref_sum += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }

    out.rel_l2 = squared_ref_sum > 0.0 ? std::sqrt(squared_error_sum / squared_ref_sum) : 0.0;
    return out;
}

// Compara dos matrices M×N con layouts diferentes:
// ref  en col-major (convencion cuBLAS/BLAS): elemento (i,j) en ref[i + j*m]
// test en row-major (salida del kernel WMMA): elemento (i,j) en test[i*n + j]
static ErrorMetrics compare_float_rowmaj_vs_colmaj(
        const std::vector<float>& ref_colmaj,
        const std::vector<float>& test_rowmaj,
        int m, int n) {
    ErrorMetrics out;
    double sq_err = 0.0, sq_ref = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const double r = static_cast<double>(ref_colmaj[i + j * m]);
            const double t = static_cast<double>(test_rowmaj[i * n + j]);
            const double diff = r - t;
            out.max_abs = std::max(out.max_abs, std::abs(diff));
            sq_err += diff * diff;
            sq_ref  += r * r;
        }
    }
    out.rel_l2 = sq_ref > 0.0 ? std::sqrt(sq_err / sq_ref) : 0.0;
    return out;
}

// Version FP64 de la comparacion anterior.
static ErrorMetrics compare_double_vectors(const std::vector<double>& ref, const std::vector<double>& test) {
    ErrorMetrics out;
    double squared_error_sum = 0.0;
    double squared_ref_sum = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const double diff = ref[i] - test[i];
        out.max_abs = std::max(out.max_abs, std::abs(diff));
        squared_error_sum += diff * diff;
        squared_ref_sum += ref[i] * ref[i];
    }

    out.rel_l2 = squared_ref_sum > 0.0 ? std::sqrt(squared_error_sum / squared_ref_sum) : 0.0;
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
    std::cout << "Error max abs vs CPU       : " << error.max_abs << "\n";
    std::cout << "Error relativo L2 vs CPU   : " << error.rel_l2 << "\n\n";
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
// float_colmaj_to_half_rowmaj_kernel que transpone y convierte antes del
// benchmark.
// =========================================================================

// Convierte una matriz FP32 col-major (rows x cols) a FP16 row-major.
// Thread i escribe dst[i] = src[r + c*rows] donde r=i/cols, c=i%cols.
// Hilos consecutivos leen src con paso 1 (misma columna, filas contiguas),
// lo que produce accesos coalescentes en la lectura global.
__global__ static void float_colmaj_to_half_rowmaj_kernel(
        const float* __restrict__ src,
        __half*      __restrict__ dst,
        int rows, int cols) {
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx < total) {
        const int r = idx / cols;
        const int c = idx % cols;
        dst[idx] = __float2half(src[r + c * rows]);
    }
}

// Kernel GEMM con API WMMA: C(M,N) = A(M,K) * B(K,N), todos row-major FP16→FP32.
//
// Organizacion de hilos:
//   Bloque: 128 hilos = 4 warps dispuestos en una cuadricula 2x2 de fragmentos.
//   Cada warp calcula un fragmento de salida de 16x16 en FP32.
//   Un bloque cubre un tile de salida de 32x32.
//   Grid: (ceildiv(M,32), ceildiv(N,32)).
//
// Flujo por bloque:
//   1. Carga coalescente del tile de A (32x16) a shared memory sA.
//   2. Carga coalescente del tile de B (16x32) a shared memory sB.
//   3. Cada warp llama wmma::load_matrix_sync sobre su sub-tile en sA/sB.
//   4. Cada warp acumula con wmma::mma_sync (operacion Tensor Core HMMA).
//   5. Al finalizar el bucle K, cada warp escribe su fragmento a C con
//      wmma::store_matrix_sync.
__global__ static void wmma_gemm_kernel(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        float*        __restrict__ C,
        int M, int N, int K) {
    using namespace nvcuda;

    // Shared memory: cada fila incluye padding para evitar bank conflicts.
    // sA almacena el tile actual de A (32 filas x 16 cols).
    // sB almacena el tile actual de B (16 filas x 32 cols).
    __shared__ __half sA[kBlockTileM][kWmmaK + kWmmaShmemPad];
    __shared__ __half sB[kWmmaK][kBlockTileN + kWmmaShmemPad];

    const int warp_id  = threadIdx.x / 32;
    const int warp_row = warp_id / kBlockWarpsN;
    const int warp_col = warp_id % kBlockWarpsN;

    const int block_row = blockIdx.x * kBlockTileM;
    const int block_col = blockIdx.y * kBlockTileN;

    // Posicion de inicio del fragmento de salida de este warp en C global.
    const int warp_row_base = block_row + warp_row * kWmmaM;
    const int warp_col_base = block_col + warp_col * kWmmaN;

    wmma::fragment<wmma::matrix_a,    kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>                   c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_base = 0; k_base < K; k_base += kWmmaK) {
        // -- Carga del tile de A (32 x kWmmaK) desde global a shared memory --
        // Cada hilo carga un elemento usando su indice lineal.
        // Para el hilo i: row = i/kWmmaK, col = i%kWmmaK.
        // Todos los hilos de un warp acceden a columnas consecutivas de la
        // misma fila → direcciones globales contiguas → acceso coalescente.
        for (int i = threadIdx.x; i < kBlockTileM * kWmmaK; i += blockDim.x) {
            const int row        = i / kWmmaK;
            const int col        = i % kWmmaK;
            const int global_row = block_row + row;
            const int global_col = k_base    + col;
            sA[row][col] = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col]
                           : __float2half(0.0f);
        }

        // -- Carga del tile de B (kWmmaK x 32) desde global a shared memory --
        // Patron simetrico: hilo i carga row = i/kBlockTileN, col = i%kBlockTileN.
        // Los kBlockTileN (32) hilos mas bajos de cada warp acceden a 32
        // columnas consecutivas de la misma fila → coalescente.
        for (int i = threadIdx.x; i < kWmmaK * kBlockTileN; i += blockDim.x) {
            const int row        = i / kBlockTileN;
            const int col        = i % kBlockTileN;
            const int global_row = k_base    + row;
            const int global_col = block_col + col;
            sB[row][col] = (global_row < K && global_col < N)
                           ? B[global_row * N + global_col]
                           : __float2half(0.0f);
        }

        // Barrera: todos los hilos del bloque deben haber completado las cargas
        // antes de que cualquier warp lea los fragmentos desde shared memory.
        __syncthreads();

        // -- Carga de fragmentos WMMA desde shared memory --
        // Cada warp lee su sub-tile de 16x16 de sA y sB.
        // El leading dimension (ldm) incluye el padding para que wmma acceda
        // correctamente a filas consecutivas sin conflictos de banco.
        wmma::load_matrix_sync(a_frag,
            reinterpret_cast<const __half*>(&sA[warp_row * kWmmaM][0]),
            kWmmaK + kWmmaShmemPad);

        wmma::load_matrix_sync(b_frag,
            reinterpret_cast<const __half*>(&sB[0][warp_col * kWmmaN]),
            kBlockTileN + kWmmaShmemPad);

        // -- Multiplicacion Tensor Core: acumula en FP32 --
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Barrera: protege sA/sB para la siguiente iteracion del bucle K.
        __syncthreads();
    }

    // -- Escritura del fragmento acumulado a la matriz de salida C (row-major) --
    if (warp_row_base < M && warp_col_base < N) {
        wmma::store_matrix_sync(
            C + warp_row_base * N + warp_col_base,
            c_frag, N,
            wmma::mem_row_major);
    }
}

// Benchmark de la ruta WMMA personalizada.
// Convierte A y B de FP32 col-major a FP16 row-major en la GPU,
// luego lanza wmma_gemm_kernel y mide su tiempo con eventos CUDA.
// La salida C queda en FP32 row-major en el host para comparar con la
// referencia col-major usando compare_float_rowmaj_vs_colmaj.
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

    DeviceBuffer<__half> dA_fp16(A.size());
    DeviceBuffer<__half> dB_fp16(B.size());

    // Conversion: col-major FP32 → row-major FP16.
    // Se libera memoria FP32 scratch al salir del bloque.
    {
        const unsigned int bA = blocks_for_elements(A.size());
        const unsigned int bB = blocks_for_elements(B.size());
        float_colmaj_to_half_rowmaj_kernel<<<bA, kConversionThreads>>>(
            dA_fp32.get(), dA_fp16.get(), m, k);
        CHECK_CUDA(cudaGetLastError());
        float_colmaj_to_half_rowmaj_kernel<<<bB, kConversionThreads>>>(
            dB_fp32.get(), dB_fp16.get(), k, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    DeviceBuffer<float> dC(C.size());

    const dim3 block(static_cast<unsigned int>(kBlockWarpsM * kBlockWarpsN * 32));
    const dim3 grid(
        static_cast<unsigned int>((m + kBlockTileM - 1) / kBlockTileM),
        static_cast<unsigned int>((n + kBlockTileN - 1) / kBlockTileN));

    for (int i = 0; i < kWarmupIters; ++i) {
        wmma_gemm_kernel<<<grid, block>>>(
            dA_fp16.get(), dB_fp16.get(), dC.get(), m, n, k);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        wmma_gemm_kernel<<<grid, block>>>(
            dA_fp16.get(), dB_fp16.get(), dC.get(), m, n, k);
    }
    const float total_ms = timer.stop_and_elapsed_ms();
    CHECK_CUDA(cudaGetLastError());

    copy_float_vector_to_host(dC.get(), C);
    return build_metrics(m, n, k, static_cast<double>(total_ms) / iters);
}

// Presenta los resultados del experimento FP32.
static void print_float_report(const Metrics& cpu,
                               const Metrics& gpu,
                               const Metrics& tc,
                               const Metrics& wmma,
                               const ErrorMetrics& gpu_error,
                               const ErrorMetrics& tc_error,
                               const ErrorMetrics& wmma_error) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================ RESULTADOS GEMM FP32 =================\n";
    std::cout << "CPU BLAS - tiempo medio    : " << cpu.ms << " ms\n";
    std::cout << "CPU BLAS - rendimiento     : " << cpu.gflops << " GFLOP/s ("
              << cpu.tflops << " TFLOP/s)\n\n";

    print_reference_comparison("GPU cuBLAS clasico", gpu, cpu.ms, gpu_error);

    std::cout << "GPU cuBLAS TC - tiempo     : " << tc.ms << " ms\n";
    std::cout << "GPU cuBLAS TC - rend.      : " << tc.gflops << " GFLOP/s ("
              << tc.tflops << " TFLOP/s)\n";
    std::cout << "Speedup TC vs CPU          : " << cpu.ms / tc.ms << "x\n";
    std::cout << "Speedup TC vs GPU clasico  : " << gpu.ms / tc.ms << "x\n";
    std::cout << "Error max abs vs CPU       : " << tc_error.max_abs << "\n";
    std::cout << "Error relativo L2 vs CPU   : " << tc_error.rel_l2 << "\n\n";

    std::cout << "GPU WMMA custom - tiempo   : " << wmma.ms << " ms\n";
    std::cout << "GPU WMMA custom - rend.    : " << wmma.gflops << " GFLOP/s ("
              << wmma.tflops << " TFLOP/s)\n";
    std::cout << "Speedup WMMA vs CPU        : " << cpu.ms / wmma.ms << "x\n";
    std::cout << "Speedup WMMA vs GPU clasico: " << gpu.ms / wmma.ms << "x\n";
    std::cout << "Speedup WMMA vs cuBLAS TC  : " << tc.ms / wmma.ms << "x\n";
    std::cout << "Error max abs vs CPU       : " << wmma_error.max_abs << "\n";
    std::cout << "Error relativo L2 vs CPU   : " << wmma_error.rel_l2 << "\n";
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

    const size_t size_a = checked_element_count(opt.m, opt.k, "A");
    const size_t size_b = checked_element_count(opt.k, opt.n, "B");
    const size_t size_c = checked_element_count(opt.m, opt.n, "C");

    std::vector<float> A(size_a);
    std::vector<float> B(size_b);
    std::vector<float> C_cpu(size_c, 0.0f);
    std::vector<float> C_gpu(size_c, 0.0f);
    std::vector<float> C_tc(size_c, 0.0f);
    std::vector<float> C_wmma(size_c, 0.0f);

    initialize_matrix_float(A);
    initialize_matrix_float(B);

    const Metrics cpu  = benchmark_cpu_float(A, B, C_cpu, opt.m, opt.n, opt.k, opt.iters);
    const Metrics gpu  = benchmark_gpu_cublas_float(A, B, C_gpu, opt.m, opt.n, opt.k, opt.iters);
    const Metrics tc   = benchmark_gpu_tensor_cores(A, B, C_tc, opt.m, opt.n, opt.k, opt.iters);
    const Metrics wmma = benchmark_gpu_wmma(A, B, C_wmma, opt.m, opt.n, opt.k, opt.iters);

    const ErrorMetrics gpu_error  = compare_float_vectors(C_cpu, C_gpu);
    const ErrorMetrics tc_error   = compare_float_vectors(C_cpu, C_tc);
    const ErrorMetrics wmma_error = compare_float_rowmaj_vs_colmaj(C_cpu, C_wmma, opt.m, opt.n);

    print_float_report(cpu, gpu, tc, wmma, gpu_error, tc_error, wmma_error);
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
