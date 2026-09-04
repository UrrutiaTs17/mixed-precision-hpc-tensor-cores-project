// common/cuda_checks.cuh
//
// Macros de validacion de errores para CUDA, cuBLAS y cuDNN. Usadas por los
// tres kernels (GEMM, Convolucion, Stencil) en las cuatro fases del proyecto.
//
// USO: incluir DENTRO del bloque `namespace { ... }` anonimo de cada .cu (no
// a nivel de archivo), para que los simbolos no-macro que pudiera agregar
// este header en el futuro conserven enlace interno. Las macros en si no se
// ven afectadas por el namespace (el preprocesador las expande antes de que
// el compilador vea el namespace), pero mantener la convencion evita que
// alguien copie codigo de un .cu a otro con enlace externo por accidente.
//
// CHECK_CUBLAS y CHECK_CUDNN solo quedan definidas si el .cu que incluye
// este header ya incluyo cublas_v2.h / cudnn.h ANTES de este #include (se
// detecta via CUBLAS_VER_MAJOR / CUDNN_MAJOR, que esos headers definen).
// Stencil no usa cuBLAS ni cuDNN, asi que en Stencil ninguna de las dos
// macros existe -- es intencional, no un descuido.
#pragma once

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

// Valida una llamada a la API de runtime de CUDA. Si falla, imprime el
// archivo/linea y el mensaje de error de CUDA, y termina el proceso.
//
// Por que abortar en vez de propagar el error: estos binarios son
// benchmarks de un solo proceso sin estado que valga la pena preservar tras
// un fallo de CUDA (el contexto de la GPU puede haber quedado invalido) --
// no hay una operacion de recuperacion razonable, asi que fallar rapido con
// un mensaje claro es mas util que propagar un codigo de error que el
// llamador tendria que volver a chequear en cada sitio.
#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> " \
                << cudaGetErrorString(err) << std::endl;                     \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#ifdef CUBLAS_VER_MAJOR
// Valida una llamada a cuBLAS. Requiere que el .cu que incluye este header
// defina `cublas_status_to_string(cublasStatus_t)` antes del primer uso de
// esta macro (se expande en el punto de uso, no en el de definicion del
// header, asi que el compilador solo ve esa dependencia donde CHECK_CUBLAS
// realmente se invoca).
#define CHECK_CUBLAS(call)                                                    \
  do {                                                                        \
    cublasStatus_t status = (call);                                           \
    if (status != CUBLAS_STATUS_SUCCESS) {                                    \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__          \
                << " -> " << cublas_status_to_string(status)                  \
                << " (status code " << status << ")" << std::endl;            \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)
#endif  // CUBLAS_VER_MAJOR

#ifdef CUDNN_MAJOR
// Valida una llamada a cuDNN.
#define CHECK_CUDNN(call)                                                    \
  do {                                                                       \
    cudnnStatus_t status = (call);                                           \
    if (status != CUDNN_STATUS_SUCCESS) {                                    \
      std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__          \
                << " -> " << cudnnGetErrorString(status) << std::endl;       \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)
#endif  // CUDNN_MAJOR
