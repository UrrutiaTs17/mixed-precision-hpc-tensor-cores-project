// common/metrics.cuh
//
// Cronometro de eventos CUDA y funciones de metricas de rendimiento/error
// numerico, compartidas por GEMM, Convolucion y Stencil en las cuatro fases.
//
// Origen: Fase_2/common.cuh del codigo anterior (ver old/Fase_2/common.cuh),
// separado aqui de las macros de validacion (cuda_checks.cuh) siguiendo
// responsabilidad unica: este archivo es sobre medir tiempo y error, no
// sobre validar llamadas a la API.
//
// USO: incluir DENTRO del bloque `namespace { ... }` anonimo de cada .cu,
// igual que cuda_checks.cuh (ver la nota de uso en ese archivo).
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_checks.cuh"

// Cronometro basado en eventos CUDA: mide tiempo transcurrido en la GPU,
// sin incluir la latencia de sincronizacion del lado del CPU que tendria un
// std::chrono alrededor del lanzamiento del kernel.
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

  void start() { CHECK_CUDA(cudaEventRecord(start_)); }

  // Registra el evento de fin, sincroniza sobre el, y devuelve el tiempo
  // transcurrido en milisegundos. Bloquea el host hasta que el kernel
  // termina -- es la sincronizacion que hace que la medicion sea valida.
  float stop_and_elapsed_ms() {
    CHECK_CUDA(cudaEventRecord(stop_));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    float elapsed = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start_, stop_));
    return elapsed;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

// Metricas de rendimiento promedio de una ruta de benchmark (una precision,
// un tamano de problema, un numero de repeticiones).
struct Metrics {
  double ms = 0.0;
  double gflops = 0.0;
  double tflops = 0.0;
};

// Metricas de comparacion entre una solucion calculada y una referencia.
//
// max_abs / rel_l2 son las metricas base, usadas desde Fase 1. l2_abs,
// ref_l2_norm, ref_linf y rel_linf se agregaron en Fase 3 para el drift por
// checkpoint de Stencil (ver Fase_3/Stencil): l2_abs es ||test-ref||_2 sin
// normalizar (el numerador de rel_l2); ref_l2_norm es ||ref||_2; ref_linf es
// ||ref||_inf sobre todos los elementos FINITOS de la referencia (no solo
// donde la solucion tambien es finita -- ver el comentario dentro de
// compare_fp64_ref_vs_fp32 sobre por que la norma de la referencia no debe
// depender de que tan lejos llego la ruta evaluada antes de divergir);
// rel_linf = max_abs / ref_linf, el analogo en norma infinito de rel_l2.
struct ErrorMetrics {
  double max_abs = 0.0;
  double rel_l2 = 0.0;
  double l2_abs = 0.0;
  double ref_l2_norm = 0.0;
  double ref_linf = 0.0;
  double rel_linf = 0.0;
  bool reference_finite = true;  // false si algun valor de la referencia no es finito
  bool solution_finite = true;   // false si algun valor de la solucion no es finito
};

namespace metrics_detail {

// Implementacion comun de las tres funciones compare_* de abajo: acumula
// sq_err/sq_ref/ref_linf/max_abs sobre dos secuencias del mismo tamano,
// usando double para las sumas de reduccion sin importar en que precision
// esten almacenados ref/test (evita que el propio calculo del error
// introduzca mas redondeo del que se esta midiendo).
template <typename RefT, typename TestT>
inline ErrorMetrics compare_sequences(const RefT* ref, const TestT* test, size_t n) {
  ErrorMetrics out;
  double sq_err = 0.0;
  double sq_ref = 0.0;
  double ref_linf = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double r = static_cast<double>(ref[i]);
    const double t = static_cast<double>(test[i]);
    if (!std::isfinite(r)) {
      out.reference_finite = false;
      continue;
    }

    // La norma de la REFERENCIA se acumula antes de mirar la solucion: si se
    // saltara junto con el elemento divergente, ref_l2_norm/ref_linf
    // dependerian de cuantos puntos de la ruta evaluada siguen finitos, no
    // de la referencia en si (dos rutas con el mismo snapshot de referencia
    // reportarian normas distintas tras divergir).
    sq_ref += r * r;
    ref_linf = std::max(ref_linf, std::abs(r));

    if (!std::isfinite(t)) {
      out.solution_finite = false;
      continue;
    }

    const double diff = r - t;
    out.max_abs = std::max(out.max_abs, std::abs(diff));
    sq_err += diff * diff;
  }
  out.rel_l2 = (out.reference_finite && std::isfinite(sq_ref) && sq_ref > 0.0)
                   ? std::sqrt(sq_err / sq_ref)
                   : 0.0;
  out.l2_abs = (out.reference_finite && std::isfinite(sq_err)) ? std::sqrt(sq_err) : 0.0;
  out.ref_l2_norm = (out.reference_finite && std::isfinite(sq_ref)) ? std::sqrt(sq_ref) : 0.0;
  out.ref_linf = (out.reference_finite && std::isfinite(ref_linf)) ? ref_linf : 0.0;
  out.rel_linf = (out.reference_finite && out.ref_linf > 0.0) ? out.max_abs / out.ref_linf : 0.0;
  return out;
}

}  // namespace metrics_detail

// Compara una referencia FP64 (ground truth) contra un resultado FP32,
// elemento a elemento sobre un buffer lineal. Valida para cualquier layout
// siempre que ref y test compartan el mismo orden de almacenamiento
// (col-major en GEMM, NCHW en Convolucion, grilla en Stencil).
inline ErrorMetrics compare_fp64_ref_vs_fp32(const std::vector<double>& ref_fp64,
                                              const std::vector<float>& test_fp32) {
  return metrics_detail::compare_sequences(ref_fp64.data(), test_fp32.data(), ref_fp64.size());
}

// Compara dos vectores FP32 elemento a elemento (p. ej. GPU vs CPU, ambos
// FP32, mismo layout lineal).
inline ErrorMetrics compare_float_vectors(const std::vector<float>& ref,
                                           const std::vector<float>& test) {
  return metrics_detail::compare_sequences(ref.data(), test.data(), ref.size());
}

// Compara dos vectores FP64 elemento a elemento (ruta --double).
inline ErrorMetrics compare_double_vectors(const std::vector<double>& ref,
                                            const std::vector<double>& test) {
  return metrics_detail::compare_sequences(ref.data(), test.data(), ref.size());
}
