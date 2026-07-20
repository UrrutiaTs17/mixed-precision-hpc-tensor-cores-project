// Utilidades compartidas entre los benchmarks de Fase_2 (GEMM, Convolution,
// Stencil): validacion de errores CUDA/cuBLAS/cuDNN, temporizador de eventos
// y estructuras/funciones de metricas de rendimiento y error numerico.
//
// Se incluye desde dentro del bloque `namespace { ... }` de cada .cu (no a
// nivel de archivo), para que los simbolos que no son macros conserven el
// enlace interno (anonymous namespace) que ya tenian cuando estaban
// duplicados en cada archivo.
//
// CHECK_CUBLAS y CHECK_CUDNN solo se definen si el .cu que incluye este
// header ya incluyo cublas_v2.h / cudnn.h respectivamente (Stencil no usa
// ninguna de las dos bibliotecas).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

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

// CUBLAS_VER_MAJOR lo define cublas_api.h (incluido por cublas_v2.h): solo
// esta visible si el .cu incluyo cublas_v2.h antes de este header.
#ifdef CUBLAS_VER_MAJOR
// Valida llamadas a cuBLAS y termina el programa si la biblioteca reporta fallo.
// Requiere que el .cu que incluye este header defina `cublas_status_to_string`
// antes del primer uso de CHECK_CUBLAS (la macro se expande en el punto de
// uso, no en el de definicion).
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cublas_status_to_string(status) \
                  << " (status code " << status << ")" << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)
#endif  // CUBLAS_VER_MAJOR

// CUDNN_MAJOR lo define cudnn_version.h (incluido por cudnn.h): solo esta
// visible si el .cu incluyo cudnn.h antes de este header.
#ifdef CUDNN_MAJOR
// Valida llamadas a cuDNN y termina el programa si la biblioteca reporta fallo.
#define CHECK_CUDNN(call) do { \
    cudnnStatus_t status = (call); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ \
                  << " -> " << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)
#endif  // CUDNN_MAJOR

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

// Metricas de rendimiento promedio para una ruta de benchmark.
struct Metrics {
    double ms     = 0.0;
    double gflops = 0.0;
    double tflops = 0.0;
};

// Metricas para comparar la salida numerica frente a una referencia.
struct ErrorMetrics {
    double max_abs = 0.0;
    double rel_l2  = 0.0;
    bool   reference_finite = true;  // false si algun valor de la referencia no es finito
    bool   solution_finite  = true;  // false si algun valor de la solucion no es finito
};

// Compara una referencia FP64 contra un resultado FP32, elemento a elemento
// sobre un buffer lineal (mismo indice logico en ambos vectores). Valida
// para cualquier layout siempre que ref y test compartan el mismo orden de
// almacenamiento (col-major en GEMM, NCHW en Convolucion, grilla en Stencil).
static ErrorMetrics compare_fp64_ref_vs_fp32(const std::vector<double>& ref_fp64,
                                             const std::vector<float>&  test_fp32) {
    ErrorMetrics out;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    for (size_t i = 0; i < ref_fp64.size(); ++i) {
        const double r = ref_fp64[i];
        const double t = static_cast<double>(test_fp32[i]);
        if (!std::isfinite(r)) { out.reference_finite = false; continue; }
        if (!std::isfinite(t)) { out.solution_finite = false; continue; }
        const double diff = r - t;
        out.max_abs = std::max(out.max_abs, std::abs(diff));
        sq_err += diff * diff;
        sq_ref += r * r;
    }
    out.rel_l2 = (out.reference_finite && std::isfinite(sq_ref) && sq_ref > 0.0)
                 ? std::sqrt(sq_err / sq_ref) : 0.0;
    return out;
}

// Compara dos vectores FP32 elemento a elemento (p. ej. GPU vs CPU, ambos
// FP32, mismo layout lineal).
static ErrorMetrics compare_float_vectors(const std::vector<float>& ref,
                                          const std::vector<float>& test) {
    ErrorMetrics out;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = static_cast<double>(ref[i]);
        const double t = static_cast<double>(test[i]);
        if (!std::isfinite(r)) { out.reference_finite = false; continue; }
        if (!std::isfinite(t)) { out.solution_finite = false; continue; }
        const double diff = r - t;
        out.max_abs = std::max(out.max_abs, std::abs(diff));
        sq_err += diff * diff;
        sq_ref += r * r;
    }
    out.rel_l2 = (out.reference_finite && std::isfinite(sq_ref) && sq_ref > 0.0)
                 ? std::sqrt(sq_err / sq_ref) : 0.0;
    return out;
}

// Compara dos vectores FP64 elemento a elemento (ruta --double).
static ErrorMetrics compare_double_vectors(const std::vector<double>& ref,
                                           const std::vector<double>& test) {
    ErrorMetrics out;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i];
        const double t = test[i];
        if (!std::isfinite(r)) { out.reference_finite = false; continue; }
        if (!std::isfinite(t)) { out.solution_finite = false; continue; }
        const double diff = r - t;
        out.max_abs = std::max(out.max_abs, std::abs(diff));
        sq_err += diff * diff;
        sq_ref += r * r;
    }
    out.rel_l2 = (out.reference_finite && std::isfinite(sq_ref) && sq_ref > 0.0)
                 ? std::sqrt(sq_err / sq_ref) : 0.0;
    return out;
}
