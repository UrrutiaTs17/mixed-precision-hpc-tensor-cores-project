// Fase_1/GEMM/gemm_baseline.cu
//
// Linea base de Fase 1: GEMM densa (C = A*B) en FP32 o FP64, comparando CPU
// (OpenBLAS, cblas_sgemm/cblas_dgemm) contra GPU (cuBLAS clasico,
// cublasSgemm/cublasDgemm), SIN Tensor Cores. Es el punto de referencia con
// el que se contrastan las rutas Tensor Core de Fase 2 (Fase_2/GEMM).
//
// Migrado desde old/Fase_1/GEMM/gemm_compare_balanced.cu (el binario vigente
// de esa carpeta -- gemm_benchmark.cu y gemm_compare.cu son iteraciones
// tempranas ya descartadas, no se migran). El comportamiento numerico de
// cada ruta es identico al original; lo que cambia es organizacion,
// documentacion, validacion de argumentos y eliminacion de duplicacion con
// common/.
//
// Compilacion (ver Fase_1/GEMM/README.md y run_gemm_fase1.sbatch para el
// patron completo en PACCA):
//   nvcc -std=c++17 -O3 gemm_baseline.cu -o gemm_baseline \
//        -I/usr/include/openblas -lcublas -lopenblas \
//        -gencode arch=compute_80,code=sm_80 --allow-unsupported-compiler
//
// Ejecucion:
//   ./gemm_baseline --m 4096 --n 4096 --k 4096 --iters 30
//   ./gemm_baseline --double --m 2048 --n 2048 --k 2048 --iters 10 --seed 7

#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {

// common/cuda_checks.cuh se incluye DENTRO del namespace anonimo (ver la
// nota de uso al inicio de ese header) y provee CHECK_CUDA/CHECK_CUBLAS,
// reemplazando las macros que este archivo definia localmente en la version
// anterior. CHECK_CUBLAS solo queda activa porque cublas_v2.h ya se incluyo
// arriba (CUBLAS_VER_MAJOR visible) y requiere que cublas_status_to_string
// este definida antes de que la macro se USE mas abajo (no antes de que se
// defina: al ser una macro, el cuerpo se resuelve en cada punto de uso).
#include "../../common/cuda_checks.cuh"

// Traduce codigos de cuBLAS a texto legible para CHECK_CUBLAS. El binario
// anterior imprimia solo el codigo numerico; este mensaje es mas util para
// diagnosticar fallos y es identico al que usa Fase_2/GEMM.
const char* cublas_status_to_string(cublasStatus_t status) {
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

// Parametros de entrada configurables desde linea de comandos. Nada de esto
// esta hardcodeado en el cuerpo del programa: dimensiones, iteraciones y la
// semilla del generador aleatorio son todos flags (ver print_usage).
struct Options {
    int m = 2048;
    int n = 2048;
    int k = 2048;
    int iters = 10;
    unsigned int seed = 42;
    bool use_double = false;
};

// Metricas de una ruta de benchmark: tiempo promedio por iteracion y
// rendimiento aritmetico derivado.
struct Metrics {
    double milliseconds = 0.0;
    double tflops = 0.0;
};

// TFLOP/s = FLOP / segundos / 1e12, con segundos = ms * 1e-3:
//   TFLOP/s = FLOP / (ms * 1e-3) / 1e12 = FLOP / (ms * 1e9)
// La version anterior dividia por (ms * 1e12), es decir calculaba GFLOP/s pero
// rotulaba el resultado como TFLOP/s: reportaba 1000x menos de lo real
// (0.0103552 en vez de 10.3552 TFLOP/s para M=N=K=12288 FP32). Ese fix ya
// esta aplicado aqui (viene de la version "balanced" ya auditada).
double tflops_from_ms(double flop, double ms) {
    return flop / (ms * 1e9);
}

void print_usage(const char* prog) {
    std::cout << "Uso:\n"
              << "  " << prog << " [--m M] [--n N] [--k K] [--iters I]"
              << " [--seed S] [--double]\n\n"
              << "Descripcion:\n"
              << "  Compara GEMM densa C(MxN) = A(MxK) * B(KxN) en CPU (OpenBLAS)\n"
              << "  contra GPU (cuBLAS clasico, sin Tensor Cores), en FP32 (por\n"
              << "  defecto) o FP64 (--double).\n\n"
              << "Flags:\n"
              << "  --m M       Filas de A y de C (default " << Options{}.m << ").\n"
              << "  --n N       Columnas de B y de C (default " << Options{}.n << ").\n"
              << "  --k K       Columnas de A / filas de B (default " << Options{}.k << ").\n"
              << "  --iters I   Iteraciones cronometradas, promediadas (default "
              << Options{}.iters << ").\n"
              << "  --seed S    Semilla del generador de numeros aleatorios usado\n"
              << "              para llenar A y B (default " << Options{}.seed << ").\n"
              << "  --double    Usa FP64 en vez de FP32 (default FP32).\n\n"
              << "Ejemplos:\n"
              << "  " << prog << " --m 4096 --n 4096 --k 4096 --iters 30\n"
              << "  " << prog << " --double --m 2048 --n 2048 --k 2048 --iters 10 --seed 7\n";
}

const char* require_arg_value(int& index, int argc, char** argv, const char* flag) {
    if (index + 1 >= argc) {
        std::cerr << "Falta valor para " << flag << ".\n\n";
        print_usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }
    return argv[++index];
}

// Valida que el valor sea un entero positivo (>0): usado para m/n/k/iters,
// donde 0 o negativo no describe un problema valido.
int parse_positive_int(const char* flag, const char* value) {
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

// Valida que el valor sea un entero no negativo (>=0): usado para --seed,
// donde 0 es una semilla valida para std::mt19937.
unsigned int parse_nonnegative_int(const char* flag, const char* value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);

    if (errno != 0 || end == value || *end != '\0' ||
        parsed > std::numeric_limits<unsigned int>::max()) {
        std::cerr << "Valor invalido para " << flag << ": " << value
                  << ". Debe ser un entero no negativo." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return static_cast<unsigned int>(parsed);
}

Options parse_args(int argc, char** argv) {
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
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            opt.seed = parse_nonnegative_int("--seed", require_arg_value(i, argc, argv, "--seed"));
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

// Imprime informacion del dispositivo activo para contextualizar los
// resultados (nombre, compute capability, memoria, SMs, etc.).
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

// FLOPs teoricos de una GEMM densa: C = A*B implica 2*m*n*k operaciones
// (una multiplicacion y una suma por producto parcial).
double gemm_operations(int m, int n, int k) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
}

// Llena una matriz FP32 con valores uniformes en [-1, 1). La semilla es un
// parametro (Options::seed), no una constante embebida, para poder repetir
// o variar el experimento sin recompilar.
void initialize_matrix_float(std::vector<float>& mat, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dist(gen);
    }
}

// Version FP64 de la inicializacion anterior.
void initialize_matrix_double(std::vector<double>& mat, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dist(gen);
    }
}

// Error absoluto maximo entre dos resultados FP32 (mismo layout lineal).
double max_abs_diff_float(const std::vector<float>& a, const std::vector<float>& b) {
    double max_err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (diff > max_err) {
            max_err = diff;
        }
    }
    return max_err;
}

// Version FP64 de la funcion anterior.
double max_abs_diff_double(const std::vector<double>& a, const std::vector<double>& b) {
    double max_err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::abs(a[i] - b[i]);
        if (diff > max_err) {
            max_err = diff;
        }
    }
    return max_err;
}

// Error relativo L2 (norma euclidiana del residuo, normalizada por la norma
// de la referencia). El termino 1e-30 en el denominador evita division por
// cero si la referencia fuera identicamente nula, sin afectar el resultado
// en ningun caso practico (matrices GEMM no triviales).
double rel_l2_error_float(const std::vector<float>& ref, const std::vector<float>& test) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = static_cast<double>(ref[i]);
        const double t = static_cast<double>(test[i]);
        const double d = r - t;
        num += d * d;
        den += r * r;
    }
    return std::sqrt(num) / (std::sqrt(den) + 1e-30);
}

// Version FP64 de la funcion anterior.
double rel_l2_error_double(const std::vector<double>& ref, const std::vector<double>& test) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = ref[i] - test[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num) / (std::sqrt(den) + 1e-30);
}

// Ejecuta GEMM en CPU con OpenBLAS (cblas_sgemm), cronometrando el promedio
// de `iters` llamadas tras una llamada de precalentamiento fuera del
// cronometro (estabiliza caches/frecuencia antes de medir).
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

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha,
                    A.data(), m,
                    B.data(), k,
                    beta,
                    C.data(), m);
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    return {ms, tflops_from_ms(gemm_operations(m, n, k), ms)};
}

// Version FP64 (cblas_dgemm) de la funcion anterior.
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

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha,
                    A.data(), m,
                    B.data(), k,
                    beta,
                    C.data(), m);
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    return {ms, tflops_from_ms(gemm_operations(m, n, k), ms)};
}

// Ejecuta GEMM en GPU con cuBLAS clasico (cublasSgemm, sin Tensor Cores):
// reserva memoria de dispositivo, copia entradas, hace una llamada de
// precalentamiento sincrona, y cronometra `iters` llamadas con eventos CUDA
// (que miden solo tiempo de GPU, sin la latencia de sincronizacion del host).
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

    const size_t sizeA = static_cast<size_t>(m) * k * sizeof(float);
    const size_t sizeB = static_cast<size_t>(k) * n * sizeof(float);
    const size_t sizeC = static_cast<size_t>(m) * n * sizeof(float);

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

    const double ms = static_cast<double>(total_ms) / iters;
    return {ms, tflops_from_ms(gemm_operations(m, n, k), ms)};
}

// Version FP64 (cublasDgemm) de la funcion anterior.
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

    const size_t sizeA = static_cast<size_t>(m) * k * sizeof(double);
    const size_t sizeB = static_cast<size_t>(k) * n * sizeof(double);
    const size_t sizeC = static_cast<size_t>(m) * n * sizeof(double);

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

    const double ms = static_cast<double>(total_ms) / iters;
    return {ms, tflops_from_ms(gemm_operations(m, n, k), ms)};
}

// Imprime la configuracion del experimento antes de correrlo.
void print_experiment_header(const Options& opt, const char* precision_name) {
    std::cout << "Configuracion del experimento\n";
    std::cout << "Precision                 : " << precision_name << "\n";
    std::cout << "Dimensiones GEMM          : C(" << opt.m << "x" << opt.n << ") = A("
              << opt.m << "x" << opt.k << ") * B(" << opt.k << "x" << opt.n << ")\n";
    std::cout << "Iteraciones promedio      : " << opt.iters << "\n";
    std::cout << "Semilla RNG               : " << opt.seed << "\n";
    std::cout << "Disposicion de memoria    : Column-major (igual para BLAS y cuBLAS)\n";
}

// Orquesta el experimento FP32 completo: inicializa A/B, corre CPU y GPU,
// y compara los resultados entre si (no hay referencia FP64 en esta fase --
// eso es una diferencia deliberada respecto a Fase_2, que si construye una
// referencia FP64 porque compara mas de dos rutas).
void run_experiment_float(const Options& opt) {
    const int m = opt.m;
    const int n = opt.n;
    const int k = opt.k;

    std::vector<float> A(static_cast<size_t>(m) * k);
    std::vector<float> B(static_cast<size_t>(k) * n);
    std::vector<float> C_cpu(static_cast<size_t>(m) * n, 0.0f);
    std::vector<float> C_gpu(static_cast<size_t>(m) * n, 0.0f);

    // NOTA MIGRACION: A y B se inicializan con la MISMA semilla (comportamiento
    // identico al original old/Fase_1/GEMM/gemm_compare_balanced.cu, que creaba
    // un std::mt19937 nuevo con semilla fija 42 dentro de cada llamada). Esto
    // hace que A y B compartan el mismo prefijo de la secuencia pseudoaleatoria
    // en vez de ser independientes entre si -- no se corrige aqui porque
    // cambiaria los datos de entrada y por tanto los resultados numericos
    // reportados; queda documentado para que el usuario decida si vale la pena
    // usar semillas distintas (p. ej. seed y seed+1) en una version futura.
    initialize_matrix_float(A, opt.seed);
    initialize_matrix_float(B, opt.seed);
    print_experiment_header(opt, "FP32");

    const Metrics cpu = run_cpu_blas_float(m, n, k, A, B, C_cpu, opt.iters);
    const Metrics gpu = run_gpu_cublas_float(m, n, k, A, B, C_gpu, opt.iters);

    const double max_err = max_abs_diff_float(C_cpu, C_gpu);
    const double rel_err = rel_l2_error_float(C_cpu, C_gpu);

    std::cout << std::fixed << std::setprecision(7);
    std::cout << "---------------- RESULTADOS ----------------\n";
    std::cout << "CPU BLAS  - tiempo medio  : " << cpu.milliseconds << " ms\n";
    std::cout << "CPU BLAS  - rendimiento   : " << cpu.tflops << " TFLOP/s\n";
    std::cout << "GPU cuBLAS- tiempo medio  : " << gpu.milliseconds << " ms\n";
    std::cout << "GPU cuBLAS- rendimiento   : " << gpu.tflops << " TFLOP/s\n";
    std::cout << "Speedup GPU/CPU           : " << (cpu.milliseconds / gpu.milliseconds) << "x\n";
    std::cout << "Error max abs             : " << max_err << "\n";
    std::cout << "Error relativo L2         : " << rel_err << "\n";
    std::cout << "--------------------------------------------\n\n";
}

// Version FP64 de la funcion anterior.
void run_experiment_double(const Options& opt) {
    const int m = opt.m;
    const int n = opt.n;
    const int k = opt.k;

    std::vector<double> A(static_cast<size_t>(m) * k);
    std::vector<double> B(static_cast<size_t>(k) * n);
    std::vector<double> C_cpu(static_cast<size_t>(m) * n, 0.0);
    std::vector<double> C_gpu(static_cast<size_t>(m) * n, 0.0);

    // NOTA MIGRACION: misma observacion que en run_experiment_float -- A y B
    // comparten semilla (comportamiento original preservado, ver ese comentario).
    initialize_matrix_double(A, opt.seed);
    initialize_matrix_double(B, opt.seed);
    print_experiment_header(opt, "FP64");

    const Metrics cpu = run_cpu_blas_double(m, n, k, A, B, C_cpu, opt.iters);
    const Metrics gpu = run_gpu_cublas_double(m, n, k, A, B, C_gpu, opt.iters);

    const double max_err = max_abs_diff_double(C_cpu, C_gpu);
    const double rel_err = rel_l2_error_double(C_cpu, C_gpu);

    std::cout << std::fixed << std::setprecision(7);
    std::cout << "---------------- RESULTADOS ----------------\n";
    std::cout << "CPU BLAS  - tiempo medio  : " << cpu.milliseconds << " ms\n";
    std::cout << "CPU BLAS  - rendimiento   : " << cpu.tflops << " TFLOP/s\n";
    std::cout << "GPU cuBLAS- tiempo medio  : " << gpu.milliseconds << " ms\n";
    std::cout << "GPU cuBLAS- rendimiento   : " << gpu.tflops << " TFLOP/s\n";
    std::cout << "Speedup GPU/CPU           : " << (cpu.milliseconds / gpu.milliseconds) << "x\n";
    std::cout << "Error max abs             : " << max_err << "\n";
    std::cout << "Error relativo L2         : " << rel_err << "\n";
    std::cout << "--------------------------------------------\n\n";
}

}  // namespace

// Punto de entrada: parsea argumentos, imprime info de la GPU activa, y
// corre la ruta FP32 o FP64 segun --double.
int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);
    print_gpu_info();

    if (opt.use_double) {
        run_experiment_double(opt);
    } else {
        run_experiment_float(opt);
    }

    return 0;
}
