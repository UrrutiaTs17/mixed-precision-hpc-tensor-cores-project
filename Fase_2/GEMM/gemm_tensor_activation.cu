// nvcc -std=c++17 gemm_tensor_activation.cu -o gemm_tc -I/usr/include/openblas -lcublas -lopenblas -gencode arch=compute_86,code=sm_86



#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
    << " -> status code " << status << std::endl; \
    std::exit(EXIT_FAILURE); \
} \
} while(0)

namespace {

constexpr int kWarmupIters = 3;
constexpr int kConversionThreads = 256;

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

// Envueltura minima para crear y destruir el contexto de cuBLAS.
// El handle representa el estado interno que cuBLAS usa para ejecutar operaciones.
class CublasHandle {
public:
    CublasHandle() {
        CHECK_CUBLAS(cublasCreate(&handle_));
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            CHECK_CUBLAS(cublasDestroy(handle_));
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

    ~CudaEventTimer() {
        CHECK_CUDA(cudaEventDestroy(start_));
        CHECK_CUDA(cudaEventDestroy(stop_));
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

// Muestra ayuda de uso y ejemplos basicos de ejecucion.
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

// Interpreta los argumentos de la linea de comandos y valida sus valores.
static Options parse_args(int argc, char** argv) {
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
    std::cout << "===========================================================\n\n";
}

// FLOPs teoricos de una GEMM densa: C = A * B implica 2*m*n*k operaciones.
static double gemm_flops_standard(int m, int n, int k) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
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

// Reserva memoria en la GPU para un arreglo de floats.
static float* allocate_device_float_buffer(size_t count) {
    float* ptr = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(float) * count));
    return ptr;
}

// Reserva memoria en la GPU para un arreglo de doubles.
static double* allocate_device_double_buffer(size_t count) {
    double* ptr = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(double) * count));
    return ptr;
}

// Reserva memoria en la GPU para un arreglo de half precision.
static __half* allocate_device_half_buffer(size_t count) {
    __half* ptr = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(__half) * count));
    return ptr;
}

// Libera memoria previamente reservada en la GPU.
static void free_device_buffer(void* ptr) {
    if (ptr != nullptr) {
        CHECK_CUDA(cudaFree(ptr));
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
    float* dA = allocate_device_float_buffer(A.size());
    float* dB = allocate_device_float_buffer(B.size());
    float* dC = allocate_device_float_buffer(C.size());
    copy_float_vector_to_device(A, dA);
    copy_float_vector_to_device(B, dB);

    CublasHandle handle;
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
                                 dA, m,
                                 dB, k,
                                 &beta,
                                 dC, m));
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
                                 dA, m,
                                 dB, k,
                                 &beta,
                                 dC, m));
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    copy_float_vector_to_host(dC, C);

    free_device_buffer(dA);
    free_device_buffer(dB);
    free_device_buffer(dC);

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
    double* dA = allocate_device_double_buffer(A.size());
    double* dB = allocate_device_double_buffer(B.size());
    double* dC = allocate_device_double_buffer(C.size());
    copy_double_vector_to_device(A, dA);
    copy_double_vector_to_device(B, dB);

    CublasHandle handle;
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
                                 dA, m,
                                 dB, k,
                                 &beta,
                                 dC, m));
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
                                 dA, m,
                                 dB, k,
                                 &beta,
                                 dC, m));
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    copy_double_vector_to_host(dC, C);

    free_device_buffer(dA);
    free_device_buffer(dB);
    free_device_buffer(dC);

    return build_metrics(m, n, k, total_ms / iters);
}

// Kernel CUDA sencillo para convertir cada elemento de float a half.
__global__ static void convert_float_to_half_kernel(const float* src, __half* dst, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
    const int blocks_a = static_cast<int>((size_a + kConversionThreads - 1) / kConversionThreads);
    const int blocks_b = static_cast<int>((size_b + kConversionThreads - 1) / kConversionThreads);

    convert_float_to_half_kernel<<<blocks_a, kConversionThreads>>>(
        src_a, dst_a, static_cast<int>(size_a));
    convert_float_to_half_kernel<<<blocks_b, kConversionThreads>>>(
        src_b, dst_b, static_cast<int>(size_b));

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
    // - la acumulacion se pide en FP32 (CUBLAS_COMPUTE_32F_FAST_16F)
    // Esto aprovecha Tensor Cores para acelerar el calculo con una precision mixta.
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
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
    float* dA_fp32 = allocate_device_float_buffer(A.size());
    float* dB_fp32 = allocate_device_float_buffer(B.size());
    __half* dA_fp16 = allocate_device_half_buffer(A.size());
    __half* dB_fp16 = allocate_device_half_buffer(B.size());
    float* dC = allocate_device_float_buffer(C.size());

    copy_float_vector_to_device(A, dA_fp32);
    copy_float_vector_to_device(B, dB_fp32);
    convert_fp32_buffers_to_fp16(dA_fp32, dB_fp32, dA_fp16, dB_fp16, A.size(), B.size());

    CublasHandle handle;
    // Activar Tensor Cores explicitamente en este handle.
    // CUBLAS_TENSOR_OP_MATH le indica a cuBLAS que prefiera unidades Tensor Core
    // cuando la operacion y los tipos de dato lo permitan.
    CHECK_CUBLAS(cublasSetMathMode(handle.get(), CUBLAS_TENSOR_OP_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Se hace warmup para estabilizar clocks y evitar medir costos de primer uso.
    for (int i = 0; i < kWarmupIters; ++i) {
        run_tensor_core_gemm(handle.get(), dA_fp16, dB_fp16, dC, m, n, k, &alpha, &beta);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        run_tensor_core_gemm(handle.get(), dA_fp16, dB_fp16, dC, m, n, k, &alpha, &beta);
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    copy_float_vector_to_host(dC, C);

    free_device_buffer(dA_fp32);
    free_device_buffer(dB_fp32);
    free_device_buffer(dA_fp16);
    free_device_buffer(dB_fp16);
    free_device_buffer(dC);

    return build_metrics(m, n, k, total_ms / iters);
}

// Presenta los resultados del experimento FP32.
static void print_float_report(const Metrics& cpu,
                               const Metrics& gpu,
                               const Metrics& tc,
                               const ErrorMetrics& gpu_error,
                               const ErrorMetrics& tc_error) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================ RESULTADOS GEMM FP32 =================\n";
    std::cout << "CPU BLAS - tiempo medio    : " << cpu.ms << " ms\n";
    std::cout << "CPU BLAS - rendimiento     : " << cpu.gflops << " GFLOP/s ("
              << cpu.tflops << " TFLOP/s)\n\n";

    print_reference_comparison("GPU cuBLAS clasico", gpu, cpu.ms, gpu_error);

    std::cout << "GPU Tensor Core - tiempo   : " << tc.ms << " ms\n";
    std::cout << "GPU Tensor Core - rend.    : " << tc.gflops << " GFLOP/s ("
              << tc.tflops << " TFLOP/s)\n";
    std::cout << "Speedup vs CPU             : " << cpu.ms / tc.ms << "x\n";
    std::cout << "Speedup vs GPU clasico     : " << gpu.ms / tc.ms << "x\n";
    std::cout << "Error max abs vs CPU       : " << tc_error.max_abs << "\n";
    std::cout << "Error relativo L2 vs CPU   : " << tc_error.rel_l2 << "\n";
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
    const size_t size_a = static_cast<size_t>(opt.m) * opt.k;
    const size_t size_b = static_cast<size_t>(opt.k) * opt.n;
    const size_t size_c = static_cast<size_t>(opt.m) * opt.n;

    std::vector<float> A(size_a);
    std::vector<float> B(size_b);
    std::vector<float> C_cpu(size_c, 0.0f);
    std::vector<float> C_gpu(size_c, 0.0f);
    std::vector<float> C_tc(size_c, 0.0f);

    initialize_matrix_float(A);
    initialize_matrix_float(B);

    const Metrics cpu = benchmark_cpu_float(A, B, C_cpu, opt.m, opt.n, opt.k, opt.iters);
    const Metrics gpu = benchmark_gpu_cublas_float(A, B, C_gpu, opt.m, opt.n, opt.k, opt.iters);
    const Metrics tc = benchmark_gpu_tensor_cores(A, B, C_tc, opt.m, opt.n, opt.k, opt.iters);

    const ErrorMetrics gpu_error = compare_float_vectors(C_cpu, C_gpu);
    const ErrorMetrics tc_error = compare_float_vectors(C_cpu, C_tc);

    print_float_report(cpu, gpu, tc, gpu_error, tc_error);
}

// Orquesta el experimento FP64 completo.
static void run_experiment_double(const Options& opt) {
    const size_t size_a = static_cast<size_t>(opt.m) * opt.k;
    const size_t size_b = static_cast<size_t>(opt.k) * opt.n;
    const size_t size_c = static_cast<size_t>(opt.m) * opt.n;

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
