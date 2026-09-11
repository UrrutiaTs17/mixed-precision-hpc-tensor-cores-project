// Fase_4/GEMM/gemm_chained.cu
//
// Encadenamiento genuino de GEMM: X(n+1) = X(n) * A, con A un operador FIJO
// (no cambia entre iteraciones), aplicado repetidamente -- la misma
// estructura que la iteracion de potencias para autovalores, y la que ya usa
// por dentro la reformulacion "Y = X*H + V*X" del Stencil de Fase 4 (ahi
// vive escondida dentro del kernel de Stencil; aqui se mide como GEMM por
// derecho propio). Especificado en el documento "Plan de Precision Mixta",
// secciones 02 (encadenamiento) y 01/"Ancla FP64" (esta extension) -- esa es
// la referencia normativa; si un comentario de aqui no coincide con lo que
// el plan describe, el plan tiene razon.
//
// ESTE ARCHIVO ES Fase_3/GEMM/gemm_chained.cu + EL ANCLA FP64:
// Punto de partida identico a Fase_3/GEMM/gemm_chained.cu (encadenamiento
// con compensacion por linealidad en FLOAT, sin ancla). La unica adicion es
// --anchor-every K: cada K iteraciones de la ruta CON compensacion
// (--comp on), en vez de dar el paso WMMA normal, el codigo (a) reconstruye
// el estado exacto T+comp en DOUBLE, (b) avanza UN paso con la referencia
// FP64 -- reutilizando gpu_fp64_step() TAL CUAL, la misma funcion que ya usa
// la ruta de referencia de este archivo, sin codigo cuBLAS nuevo -- y
// (c) re-siembra T (cuantizado) y el residuo de compensacion, ahora en
// DOUBLE (no en float), para que el proximo ancla no pierda precision por el
// camino. Busca el marcador "ANCLA FP64 (Fase 4)" en este archivo para ver
// cada pieza. El razonamiento completo (por que DOUBLE y no float para el
// residuo, por que --anchor-every requiere --comp on, las dos puertas de
// validacion K=0/K=1) esta en el comentario de cabecera de
// Fase_4/Stencil/stencil_tensor_activation.cu -- es el mismo patron,
// adaptado aqui a las funciones libres de este archivo (grids 1D sobre un
// buffer N*N, en vez de los grids 2D de Stencil) y simplificado porque GEMM,
// a diferencia de Stencil, no tiene ExecutionMode::Graph (nada que declarar
// incompatible) ni un kernel WMMA monolitico que evitar tocar: el paso
// encadenado ya estaba compuesto de kernels separados y reutilizables antes
// de esta extension.
//
// Compilar con:
// nvcc -std=c++17 gemm_chained.cu -o gemm_chained \
//      -lcublas -gencode arch=compute_80,code=sm_80 \
//      --allow-unsupported-compiler
// (agregar -DUSE_NVML_TELEMETRY -lnvidia-ml para telemetria de energia GPU)
//
// Ejecutar:
// ./gemm_chained --n 1024 --iters 20 --tc fp16 --comp on --anchor-every 5
//
// ELECCION DEL OPERADOR A -- LEER ANTES DE CAMBIAR N:
// A = c * H, con H la matriz de Hadamard de Sylvester (entradas EXACTAS +-1,
// sin aproximacion en ningun formato IEEE 754) y c una potencia de 2 exacta
// (por lo tanto A_T = A bit a bit al castear a FP16/BF16 -- ver la seccion
// "Hallazgo mas profundo" del documento de plan sobre por que esto importa:
// una matriz ortogonal generica de una QR aleatoria introduce un error de
// REDONDEO DEL OPERADOR que ninguna compensacion del estado puede corregir).
// La construccion de Sylvester exige N potencia de 2 -- los tamanos 512,
// 1024, 2048, 4096 del rango prometido por el plan lo son todos.
//
// La referencia FP64 usa cuBLAS (cublasDgemm), que es COLUMN-MAJOR por
// convencion; el kernel WMMA de common/wmma_gemm.cuh es ROW-MAJOR. Para que
// ambas rutas calculen EXACTAMENTE la misma operacion matematica X*A sobre
// el mismo buffer sin transponer nada explicitamente, la llamada a cuBLAS
// invierte el orden de los operandos -- ver el comentario junto a
// gpu_fp64_step() mas abajo para la derivacion completa. Es el punto de
// mayor riesgo de error silencioso de este archivo (si esta al reves, el
// binario compila y corre igual, pero compara peras con manzanas) --
// VERIFICAR EN PACCA con un N chico contra una referencia CPU antes de
// confiar en cualquier resultado.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

#include "../../common/cuda_checks.cuh"
#include "../../common/metrics.cuh"
#include "../../common/wmma_gemm.cuh"

}  // namespace

// power_sampling.h declara simbolos extern "C" (NVML/dirent) que deben
// quedar con enlace externo -- se incluye FUERA del namespace anonimo, igual
// que ya documenta su propio comentario de cabecera y que ya respeta
// Fase_3/Stencil/stencil_tensor_activation.cu.
#include "../../common/power_sampling.h"

namespace {

// Traduce codigos de cuBLAS a texto legible (necesario para CHECK_CUBLAS,
// definida en common/cuda_checks.cuh, expandida en el punto de uso).
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

constexpr int kConversionThreads = 256;
constexpr int kWarmupIters = 3;

enum class TcFormat { FP16, BF16, Both };

struct Options {
  int n = 1024;                 // tamano de la matriz cuadrada N x N
  int iters = 20;
  TcFormat tc_format = TcFormat::Both;
  bool comp = false;             // compensacion por linealidad (off por defecto)
  int checkpoint_every = 0;      // 0 = solo checkpoint en la ultima iteracion
  // ANCLA FP64 (Fase 4): 0 = deshabilitada (identico a Fase_3/GEMM). > 0
  // exige --comp on -- ver la validacion en parse_args() y el comentario de
  // cabecera del archivo.
  int anchor_every = 0;
  std::string csv_path;
  unsigned int seed = 42;
  // Factor de amplificacion objetivo (aprox.) para elegir c = 2^p en
  // A = c*H: lambda_aprox = c * sqrt(N). Ver build_hadamard_operator().
  double target_lambda = 1.1;
};

// =========================================================================
// Operador fijo A = c * H (Hadamard de Sylvester, entradas +-1, escalado por
// una potencia de 2 exacta).
// =========================================================================

// Construccion recursiva de Sylvester: H_1 = [1]; H_2k = [[H_k, H_k],
// [H_k, -H_k]]. N debe ser potencia de 2 (validado en parse_args). Entradas
// siempre +-1 -- exactas en cualquier formato de punto flotante.
static void build_sylvester_hadamard(std::vector<float>& H, int n) {
  H.assign(static_cast<size_t>(n) * n, 1.0f);
  for (int size = 1; size < n; size *= 2) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        const float v = H[static_cast<size_t>(i) * n + j];
        H[static_cast<size_t>(i + size) * n + j] = v;
        H[static_cast<size_t>(i) * n + (j + size)] = v;
        H[static_cast<size_t>(i + size) * n + (j + size)] = -v;
      }
    }
  }
}

// Elige c = 2^p (p entero, puede ser negativo) tal que c * sqrt(n) quede lo
// mas cerca posible de target_lambda -- crecimiento por iteracion analogo al
// lambda~2 de Stencil, pero aqui es EXACTO en aritmetica ideal (H es
// exactamente ortogonal salvo por el factor sqrt(n): ||H x|| = sqrt(n) ||x||
// para cualquier x, en aritmetica exacta) y c es representable sin
// redondeo. sqrt(n) en si mismo NO necesita ser una potencia de 2 -- eso
// solo importaria si se normalizara H a norma unitaria (H/sqrt(n)), lo cual
// esta funcion evita a proposito: escala por c aparte, dejando H con
// entradas +-1 intactas.
static double choose_amplification_c(int n, double target_lambda) {
  const double sqrt_n = std::sqrt(static_cast<double>(n));
  const double raw_c = target_lambda / sqrt_n;
  const double p = std::round(std::log2(raw_c));
  return std::pow(2.0, p);
}

// Construye A = c*H directamente en T (row-major) y en double (para la
// referencia FP64). Como c y las entradas de H son exactas, A_T y A_fp64
// representan EXACTAMENTE el mismo operador matematico -- no hay error de
// representacion del operador que compensar (a diferencia de una matriz
// ortogonal densa generica, ver el aviso de cabecera del archivo).
template <typename T>
static void build_hadamard_operator(int n, double target_lambda, std::vector<T>& A_tc,
                                     std::vector<double>& A_fp64, double& c_out,
                                     double& lambda_out) {
  std::vector<float> H;
  build_sylvester_hadamard(H, n);
  const double c = choose_amplification_c(n, target_lambda);
  c_out = c;
  lambda_out = c * std::sqrt(static_cast<double>(n));

  const size_t count = static_cast<size_t>(n) * n;
  A_tc.resize(count);
  A_fp64.resize(count);
  for (size_t i = 0; i < count; ++i) {
    const double val = static_cast<double>(H[i]) * c;  // +-c exactamente
    A_fp64[i] = val;
    // float_to_tc_scalar<T> es __device__-only y no puede llamarse desde
    // esta funcion host -- ver Fase_3/GEMM/gemm_chained.cu. T(float) es la
    // conversion correcta en codigo host.
    A_tc[i] = T(static_cast<float>(val));
  }
}

// =========================================================================
// Kernels elementales del encadenamiento (comp en FLOAT para el paso normal
// -- el residuo en DOUBLE que necesita el ancla vive en los kernels de la
// seccion "ANCLA FP64" mas abajo).
// =========================================================================

// Castea un buffer FLOAT (residuo de compensacion) a T, para poder pasarlo
// por wmma_gemm_kernel como si fuera un operando normal -- explota que el
// operador es LINEAL: (T+comp)*A = T*A + comp*A. comp es pequeno (residuo de
// redondeo de almacenamiento), asi que el redondeo adicional de truncarlo a
// T es un error de segundo orden.
template <typename T>
__global__ static void cast_float_to_tc_kernel(const float* __restrict__ src, T* __restrict__ dst,
                                                int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = float_to_tc_scalar<T>(src[i]);
}

// Combina el termino principal (T_raw, de wmma_gemm_kernel sobre T_in) y la
// correccion (comp_raw, de wmma_gemm_kernel sobre comp_in) en el resultado
// final, cuantiza a T, y calcula el nuevo residuo -- todo en un solo kernel.
// Con comp_on=false, comp_raw se ignora (puede ser nullptr).
template <typename T>
__global__ static void finalize_step_kernel(const float* __restrict__ t_raw,
                                             const float* __restrict__ comp_raw, bool comp_on,
                                             T* __restrict__ t_out, float* __restrict__ comp_out,
                                             int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float exact = comp_on ? (t_raw[i] + comp_raw[i]) : t_raw[i];
  const T q = float_to_tc_scalar<T>(exact);
  t_out[i] = q;
  // s_off (la ruta sin compensacion, que corre en todas las iteraciones sin
  // importar --comp) nunca reserva d_comp_out (chained_gemm_alloc con
  // comp_on=false) -- queda nullptr, de ahi la guardia.
  if (comp_on) comp_out[i] = exact - tc_scalar_to_float<T>(q);
}

// Siembra INICIAL del residuo de compensacion a partir del x0 real (en
// double, ya en el device como d_x64_in) y su cuantizacion a T -- NO desde
// cero. Mismo patron que Fase_4/Stencil (seed_comp_from_conversion_kernel/
// seed_comp64_from_conversion_kernel). Dejar comp/comp64 en 0 tras la
// siembra de X0 descartaria el redondeo x0->T de la primerisima conversion,
// que se propagaria amplificado por A en cada iteracion en vez de quedar
// capturado desde el principio, como exige la propiedad de reconstruccion
// Q(v)+comp=v que el resto del mecanismo asume ya valida desde t=0. El gate
// K=1 (ver README.md) depende de esta siembra para converger a la
// referencia FP64.
template <typename T>
__global__ static void seed_comp_from_double_kernel(const double* __restrict__ src64,
                                                      const T* __restrict__ src_tc,
                                                      float* __restrict__ comp, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) comp[i] = static_cast<float>(src64[i]) - tc_scalar_to_float<T>(src_tc[i]);
}

// Version DOBLE de la siembra anterior -- necesaria para d_comp64_in, para
// que el gate K=1 sea exacto desde la primera iteracion (sin pasar por
// float en el camino, igual que reseed_double_from_fp64_kernel mas abajo).
template <typename T>
__global__ static void seed_comp64_from_double_kernel(const double* __restrict__ src64,
                                                        const T* __restrict__ src_tc,
                                                        double* __restrict__ comp64, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    comp64[i] = src64[i] - static_cast<double>(tc_scalar_to_float<T>(src_tc[i]));
  }
}

inline int grid1d(int n, int block = 256) { return (n + block - 1) / block; }

// =========================================================================
// ANCLA FP64 (Fase 4): kernels elementales para la reconstruccion exacta y
// el reseed contra la referencia FP64. Mismo patron que ya usa
// Fase_4/Stencil/stencil_tensor_activation.cu, adaptado a grids 1D sobre un
// buffer plano N*N (aqui no hay estructura 2D que preservar).
// =========================================================================

// Ensancha el residuo de compensacion de FLOAT a DOUBLE (simple cast) --
// se usa despues de una iteracion NORMAL (no ancla), para mantener el estado
// double sincronizado por si la SIGUIENTE iteracion si es de ancla.
__global__ static void widen_comp_to_double_kernel(const float* __restrict__ comp_f,
                                                     double* __restrict__ comp_d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) comp_d[i] = static_cast<double>(comp_f[i]);
}

// Angosta el residuo de compensacion de DOUBLE a FLOAT (simple cast) -- se
// usa DESPUES de una iteracion de ancla, para sincronizar el residuo float
// que la siguiente iteracion NORMAL (via cast_float_to_tc_kernel) necesita.
__global__ static void narrow_double_to_float_kernel(const double* __restrict__ comp_d,
                                                       float* __restrict__ comp_f, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) comp_f[i] = static_cast<float>(comp_d[i]);
}

// Reconstruye el estado EXACTO (en double) a partir del par (T, comp64):
// exact = double(tc_to_float(T)) + comp64 -- la misma propiedad de
// reconstruccion que ya mantiene finalize_step_kernel en float
// (Q(v)+comp=v), aqui en double para no perder precision antes del paso
// FP64 de referencia.
template <typename T>
__global__ static void reconstruct_exact_double_kernel(const T* __restrict__ t_in,
                                                         const double* __restrict__ comp64_in,
                                                         double* __restrict__ exact_out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  exact_out[i] = static_cast<double>(tc_scalar_to_float<T>(t_in[i])) + comp64_in[i];
}

// Re-siembra el estado T y el residuo double a partir del resultado EXACTO
// del paso FP64 de referencia (out64). comp64_out se calcula como
// (out64 - double(tc_to_float(q))) enteramente en double, sin pasar por
// float en el camino -- una truncacion intermedia a float rompería la
// exactitud de --anchor-every 1 (K=1) frente a la referencia FP64 pura.
// Mismo patron que Fase_4/Stencil -- ver su comentario de cabecera.
template <typename T>
__global__ static void reseed_double_from_fp64_kernel(const double* __restrict__ out64,
                                                        T* __restrict__ t_out,
                                                        double* __restrict__ comp64_out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const T q = float_to_tc_scalar<T>(static_cast<float>(out64[i]));
  t_out[i] = q;
  comp64_out[i] = out64[i] - static_cast<double>(tc_scalar_to_float<T>(q));
}

// =========================================================================
// Ruta de referencia: GPU_FP64 encadenada (cuBLAS).
// =========================================================================

// Un paso X_out = X_in * A_fp64, en FP64, via cuBLAS. Se usa tanto para la
// trayectoria de referencia (siempre FP64, comparacion en cada checkpoint)
// como -- ANCLA FP64 (Fase 4) -- para el paso FP64 puntual que el ancla
// inyecta dentro de la trayectoria de baja precision. Es la MISMA funcion en
// ambos casos: el ancla no necesita ninguna llamada a cuBLAS nueva.
//
// DERIVACION DEL ORDEN DE OPERANDOS (leer antes de tocar esta funcion):
// cuBLAS es column-major. X_in y A_fp64 estan almacenados ROW-MAJOR (misma
// convencion que wmma_gemm_kernel). Un buffer row-major (R x C) leido como
// column-major tiene dimensiones (C x R) y representa la TRANSPUESTA de la
// matriz row-major original -- es la misma memoria, solo cambia como se
// interpreta.
//   X_in row-major (n x n) leido column-major = X_in^T (n x n)
//   A_fp64 row-major (n x n) leido column-major = A_fp64^T (n x n)
// Queremos C = X_in * A_fp64. Notar que C^T = A_fp64^T * X_in^T -- el
// PRODUCTO DE LAS TRANSPUESTAS, EN ORDEN INVERSO, es la transpuesta del
// producto. cuBLAS calcula, en su propia convencion column-major:
//   cublasDgemm(A=A_fp64_buf, B=X_in_buf) = A_fp64^T * X_in^T = C^T
// (col-major). Esa misma memoria, leida de vuelta como ROW-MAJOR, es
// (C^T)^T = C -- exactamente lo que queremos. Por eso el orden de los
// argumentos de cublasDgemm es (A_fp64_buf, X_in_buf), NO (X_in_buf,
// A_fp64_buf) -- invertido respecto al orden "natural" X*A.
static void gpu_fp64_step(cublasHandle_t handle, const double* d_x_in, const double* d_a,
                          double* d_x_out, int n) {
  const double alpha = 1.0;
  const double beta = 0.0;
  CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_x_in, n,
                           &beta, d_x_out, n));
}

// =========================================================================
// Ruta WMMA encadenada (con o sin compensacion por linealidad).
// =========================================================================

template <typename T>
struct ChainedGemmState {
  T* d_x_in = nullptr;
  T* d_x_out = nullptr;
  float* d_comp_in = nullptr;   // nullptr si comp_on == false
  float* d_comp_out = nullptr;
  float* d_t_raw = nullptr;     // scratch: salida cruda de wmma_gemm_kernel sobre x
  float* d_comp_tc_raw = nullptr;  // scratch: salida cruda de wmma_gemm_kernel sobre comp
  T* d_comp_as_tc = nullptr;    // scratch: comp castea a T antes del segundo wmma
};

template <typename T>
static void chained_gemm_alloc(ChainedGemmState<T>& s, int n, bool comp_on) {
  const size_t count = static_cast<size_t>(n) * n;
  CHECK_CUDA(cudaMalloc(&s.d_x_in, count * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&s.d_x_out, count * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&s.d_t_raw, count * sizeof(float)));
  if (comp_on) {
    CHECK_CUDA(cudaMalloc(&s.d_comp_in, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_comp_out, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_comp_tc_raw, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_comp_as_tc, count * sizeof(T)));
    CHECK_CUDA(cudaMemset(s.d_comp_in, 0, count * sizeof(float)));
  }
}

template <typename T>
static void chained_gemm_free(ChainedGemmState<T>& s) {
  cudaFree(s.d_x_in);
  cudaFree(s.d_x_out);
  cudaFree(s.d_t_raw);
  cudaFree(s.d_comp_in);
  cudaFree(s.d_comp_out);
  cudaFree(s.d_comp_tc_raw);
  cudaFree(s.d_comp_as_tc);
}

// Un paso encadenado: X_out = X_in * A (+ correccion si comp_on). No sabe
// nada del ancla -- ANCLA FP64 (Fase 4) es pura orquestacion en
// run_chained_route(), que decide, iteracion por iteracion, si llamar a
// esta funcion o al camino de reconstruccion+FP64+reseed. Se reutiliza tal
// cual desde Fase_3/GEMM/gemm_chained.cu, sin ningun cambio.
template <typename T>
static void chained_gemm_step(ChainedGemmState<T>& s, const T* d_a_tc, int n, bool comp_on) {
  const dim3 block(kBlockWarpsM * kBlockWarpsN * 32);
  const dim3 grid(static_cast<unsigned int>((n + kBlockTileM - 1) / kBlockTileM),
                  static_cast<unsigned int>((n + kBlockTileN - 1) / kBlockTileN));

  // Termino principal: T_raw = X_in * A (Tensor Cores).
  wmma_gemm_kernel<T><<<grid, block>>>(s.d_x_in, d_a_tc, s.d_t_raw, n, n, n);
  CHECK_CUDA(cudaGetLastError());

  if (comp_on) {
    // Correccion: comp_raw = comp_in * A (tambien Tensor Cores, sobre comp
    // truncado a T -- ver el comentario de cast_float_to_tc_kernel).
    cast_float_to_tc_kernel<T><<<grid1d(n * n), kConversionThreads>>>(s.d_comp_in,
                                                                       s.d_comp_as_tc, n * n);
    CHECK_CUDA(cudaGetLastError());
    wmma_gemm_kernel<T><<<grid, block>>>(s.d_comp_as_tc, d_a_tc, s.d_comp_tc_raw, n, n, n);
    CHECK_CUDA(cudaGetLastError());
  }

  finalize_step_kernel<T><<<grid1d(n * n), kConversionThreads>>>(
      s.d_t_raw, comp_on ? s.d_comp_tc_raw : nullptr, comp_on, s.d_x_out, s.d_comp_out, n * n);
  CHECK_CUDA(cudaGetLastError());
}

// =========================================================================
// CLI
// =========================================================================

static void print_usage(const char* prog) {
  std::cout << "Uso: " << prog
            << " [--n N] [--iters K] [--tc fp16|bf16|both] [--comp off|on]\n"
            << "         [--checkpoint-every K] [--anchor-every K] [--csv RUTA]"
            << " [--seed S] [--target-lambda L]\n\n"
            << "  --n N (potencia de 2, default 1024): tamano de la matriz cuadrada.\n"
            << "  --comp off|on (default off): compensacion por linealidad del\n"
            << "  redondeo de almacenamiento -- ver la seccion 02 del documento de plan.\n"
            << "  --checkpoint-every K (default 0 = solo la ultima iteracion): cada\n"
            << "  cuantas iteraciones se compara contra la referencia FP64.\n"
            << "  --anchor-every K (default 0 = deshabilitada, Fase 4): cada K\n"
            << "  iteraciones de la ruta con compensacion, en vez del paso WMMA normal,\n"
            << "  se reconstruye el estado exacto en double, se da un paso con la\n"
            << "  referencia FP64 (cublasDgemm) y se re-siembra T y el residuo -- ver el\n"
            << "  comentario de cabecera del archivo. Requiere --comp on. K=1 debe dar\n"
            << "  drift bit-identico a la referencia FP64; K=0 es identico a Fase 3.\n"
            << "\nEjemplos:\n"
            << "  " << prog << " --n 1024 --iters 20 --tc fp16 --comp on\n"
            << "  " << prog << " --n 1024 --iters 40 --tc fp16 --comp on --anchor-every 5\n"
            << "  " << prog << " --n 4096 --iters 10 --tc both --checkpoint-every 5\n";
}

static int parse_int_arg(int& i, int argc, char** argv) {
  if (i + 1 >= argc) {
    std::cerr << "Falta valor para " << argv[i] << "\n";
    std::exit(EXIT_FAILURE);
  }
  return std::atoi(argv[++i]);
}

static bool parse_on_off(const char* flag, const char* value) {
  if (std::strcmp(value, "off") == 0) return false;
  if (std::strcmp(value, "on") == 0) return true;
  std::cerr << "Valor invalido para " << flag << ": '" << value << "' (use off|on)\n";
  std::exit(EXIT_FAILURE);
}

static bool is_power_of_two(int n) { return n > 0 && (n & (n - 1)) == 0; }

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (std::strcmp(argv[i], "--n") == 0) {
      opt.n = parse_int_arg(i, argc, argv);
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      opt.iters = parse_int_arg(i, argc, argv);
    } else if (std::strcmp(argv[i], "--checkpoint-every") == 0) {
      opt.checkpoint_every = parse_int_arg(i, argc, argv);
    } else if (std::strcmp(argv[i], "--anchor-every") == 0) {
      opt.anchor_every = parse_int_arg(i, argc, argv);
    } else if (std::strcmp(argv[i], "--seed") == 0) {
      opt.seed = static_cast<unsigned int>(parse_int_arg(i, argc, argv));
    } else if (std::strcmp(argv[i], "--tc") == 0) {
      if (i + 1 >= argc) { std::cerr << "Falta valor para --tc\n"; std::exit(EXIT_FAILURE); }
      const std::string v = argv[++i];
      if (v == "fp16") opt.tc_format = TcFormat::FP16;
      else if (v == "bf16") opt.tc_format = TcFormat::BF16;
      else if (v == "both") opt.tc_format = TcFormat::Both;
      else { std::cerr << "Valor invalido para --tc: '" << v << "'\n"; std::exit(EXIT_FAILURE); }
    } else if (std::strcmp(argv[i], "--comp") == 0) {
      if (i + 1 >= argc) { std::cerr << "Falta valor para --comp\n"; std::exit(EXIT_FAILURE); }
      opt.comp = parse_on_off("--comp", argv[++i]);
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      if (i + 1 >= argc) { std::cerr << "Falta valor para --csv\n"; std::exit(EXIT_FAILURE); }
      opt.csv_path = argv[++i];
    } else if (std::strcmp(argv[i], "--target-lambda") == 0) {
      if (i + 1 >= argc) { std::cerr << "Falta valor para --target-lambda\n"; std::exit(EXIT_FAILURE); }
      opt.target_lambda = std::atof(argv[++i]);
    } else {
      std::cerr << "Argumento desconocido: " << argv[i] << "\n";
      print_usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  if (!is_power_of_two(opt.n)) {
    std::cerr << "--n debe ser potencia de 2 (recibido " << opt.n << "): la construccion de"
                 " Sylvester-Hadamard lo exige. Ver el aviso de cabecera del archivo.\n";
    std::exit(EXIT_FAILURE);
  }
  if (opt.n % kBlockTileM != 0 || opt.n % kKStep != 0) {
    std::cerr << "--n=" << opt.n << " no cumple los requisitos de wmma_gemm_kernel (multiplo de "
              << kBlockTileM << " y de " << kKStep << "). No debería pasar para una potencia de 2"
              << " >= 64 -- revisar common/wmma_gemm.cuh si esto dispara.\n";
    std::exit(EXIT_FAILURE);
  }
  if (opt.iters <= 0) {
    std::cerr << "--iters debe ser positivo.\n";
    std::exit(EXIT_FAILURE);
  }
  if (opt.checkpoint_every < 0) {
    std::cerr << "--checkpoint-every debe ser >= 0.\n";
    std::exit(EXIT_FAILURE);
  }
  if (opt.anchor_every < 0) {
    std::cerr << "--anchor-every debe ser >= 0.\n";
    std::exit(EXIT_FAILURE);
  }
  if (opt.anchor_every > 0 && !opt.comp) {
    std::cerr << "--anchor-every > 0 requiere --comp on: el ancla re-siembra el residuo de"
                 " compensacion, que no existe con --comp off. Ver el comentario de cabecera"
                 " del archivo.\n";
    std::exit(EXIT_FAILURE);
  }
  return opt;
}

// =========================================================================
// MEDICION POR RUTA -- leer antes de tocar la estructura de esta seccion.
//
// Las trayectorias (referencia FP64, WMMA sin compensacion, WMMA con
// compensacion + ancla) se miden en FASES SEPARADAS, cada una con su propio
// cronometro y su propia ventana de PowerBuffer.
//
// Antes corrian entrelazadas dentro de un mismo bucle, envueltas por UN solo
// cronometro y por dos PowerBuffer abiertos y cerrados en los MISMOS
// instantes. Resultado: t_iter_ms, t_total_ms, gflops y energy_gpu_j salian
// IDENTICOS en las dos filas CSV_SUMMARY, y ademas incluian el costo de la
// referencia FP64 -- que en A100 domina. Dos de los tres ejes del Frente de
// Pareto 3D quedaban inutilizables para este kernel (el de ERROR siempre
// estuvo bien: rel_l2/rel_linf nunca dependieron del cronometro). En Fase 4
// el dano era mayor todavia: el COSTO DEL ANCLA -- que es justo lo que el
// barrido de K existe para medir -- tampoco era visible, porque el tiempo de
// la ruta compensada no era suyo. Stencil nunca tuvo el problema, ya media
// por ruta; esto alinea GEMM con esa forma.
//
// POR QUE FASES SEPARADAS Y NO TRES CRONOMETROS EN UN BUCLE UNICO. El tiempo
// si se podia separar con eventos CUDA dentro del bucle. La ENERGIA no:
// nvmlDeviceGetTotalEnergyConsumption es un contador de TODO el dispositivo y
// su cuantizacion (~20-25 ms de GPU cargada, ver "REGIMEN DE VALIDEZ" en
// common/power_sampling.h) es mayor que el tramo de una trayectoria en una
// iteracion. Sumar cientos de tramos cuantizados no da una energia
// utilizable. Atribuir energia a una ruta exige darle una ventana CONTIGUA
// propia, y eso exige separar los bucles.
//
// DONDE QUEDA CADA COSTO FP64, que en Fase 4 son dos cosas distintas:
//   - La trayectoria de REFERENCIA (con la que se mide el error) queda FUERA
//     de las rutas de baja precision, en su propia fase, publicada como ruta
//     GPU_FP64.
//   - Los pasos FP64 que el ANCLA inyecta quedan DENTRO de la ruta compensada,
//     que es donde corresponde: son parte de su costo, y medirlos es el
//     objetivo del barrido de K.
// Antes las dos cosas estaban mezcladas en el mismo numero.
//
// LO QUE CUESTA: la referencia se corre UNA vez (no una por formato) y sus
// estados en los checkpoints se guardan en RAM del host. Mismo patron que ya
// usa Stencil (ckpt.fp64_checkpoints). El costo se imprime al arrancar.
//
// LO QUE NO ARREGLA: las fases corren una detras de otra, asi que la ultima
// ve una GPU mas caliente que la primera. Es la misma limitacion de
// aislamiento termico que el plan ya documenta (Etapa 5).
// =========================================================================

static bool es_checkpoint(const Options& opt, int iter) {
  return (opt.checkpoint_every > 0 && iter % opt.checkpoint_every == 0) || iter == opt.iters;
}

static int contar_checkpoints(const Options& opt) {
  int n = 0;
  for (int iter = 1; iter <= opt.iters; ++iter) {
    if (es_checkpoint(opt, iter)) ++n;
  }
  return n;
}

// Emite la fila CSV_SUMMARY de una fase ya medida.
//
// gflops usa SIEMPRE los FLOPs UTILES (un X*A por iteracion), tambien en la
// ruta compensada, que hace un segundo producto para la correccion y, con el
// ancla activa, un paso FP64 cada K iteraciones: esa ruta entrega el mismo
// resultado util por iteracion a mayor costo, asi que su gflops mas bajo es
// exactamente lo que hay que reportar, no un artefacto de contabilidad.
static void emit_csv_summary(const char* route, int n, int iters, double total_s,
                             int gpu_segments, PowerBuffer* pb, int anchor_every_col) {
  const double flops_per_iter = 2.0 * static_cast<double>(n) * n * n;  // N^3, 2 FLOPs/MAC
  const double gflops = (flops_per_iter * iters / 1e9) / total_s;
  // Mismo criterio de confiabilidad que Stencil (power_sampling.h,
  // kEnergyWindowReliableSeconds): con muchos tramos cortos el error de
  // cuantizacion del contador NVML por tramo domina y la energia deja de ser
  // comparable entre rutas. window_reliable=0 no invalida t_iter_ms/gflops
  // (vienen del reloj de pared, no de NVML), solo la columna de energia.
  const bool reliable = total_s >= kEnergyWindowReliableSeconds * gpu_segments;
  std::cout << "CSV_SUMMARY," << route << "," << n << "," << iters << ","
            << (total_s * 1000.0 / iters) << "," << (total_s * 1000.0) << "," << gflops
            << "," << energy_field(power_buffer_capture_valid(pb),
                                   power_buffer_energy_joules(pb))
            << "," << (reliable ? 1 : 0) << "," << gpu_segments << ","
            << anchor_every_col << "\n";
}

// =========================================================================
// FASE 1 -- trayectoria de referencia FP64, en su propia ventana.
//
// Se publica como ruta GPU_FP64: es el punto de Pareto "todo en FP64" que a
// este kernel le faltaba (Stencil ya lo tenia como ruta propia) y, a la vez,
// deja constancia auditable de que su costo NO esta dentro del de las rutas
// de baja precision.
//
// Corre UNA sola vez para toda la invocacion, no una por formato: la
// trayectoria FP64 no depende de T. Devuelve un snapshot del estado por cada
// iteracion de checkpoint, contra el que se compararan despues las rutas WMMA
// sin recalcular nada dentro de sus ventanas medidas.
//
// OJO: esta fase NO tiene nada que ver con el ancla. El ancla usa la misma
// funcion gpu_fp64_step(), pero desde dentro de la ruta compensada y sobre su
// propio estado reconstruido -- ver run_chained_route().
// =========================================================================
static std::vector<std::vector<double>> run_fp64_reference(const Options& opt,
                                                            cublasHandle_t handle,
                                                            const double* d_a_fp64,
                                                            const double* d_x0_64) {
  const int n = opt.n;
  const size_t count = static_cast<size_t>(n) * n;
  const int num_ckpt = contar_checkpoints(opt);

  const double snap_mib = static_cast<double>(count) * sizeof(double) / (1024.0 * 1024.0);
  std::cout << "Snapshots FP64 de referencia: " << num_ckpt << " x " << snap_mib
            << " MiB = " << (num_ckpt * snap_mib / 1024.0) << " GiB de RAM del host\n";

  double* d_in = nullptr;
  double* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, count * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_out, count * sizeof(double)));
  auto reset = [&]() {
    CHECK_CUDA(cudaMemcpy(d_in, d_x0_64, count * sizeof(double), cudaMemcpyDeviceToDevice));
  };
  reset();

  for (int w = 0; w < kWarmupIters; ++w) {
    gpu_fp64_step(handle, d_in, d_a_fp64, d_out, n);
    std::swap(d_in, d_out);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  reset();

  std::vector<std::vector<double>> snapshots;
  snapshots.reserve(num_ckpt);

  PowerBuffer* pb = power_buffer_create(0);
  power_buffer_start_sampling(pb);
  const auto t0 = std::chrono::steady_clock::now();
  // Tiempo perdido en pausas de checkpoint (D2H + guardado), a restar del
  // total para que gflops/energia reflejen solo computo GPU -- mismo problema
  // y mismo arreglo que ya documenta Fase_3/Stencil (buscar "pause_t0" alli).
  double pause_s = 0.0;
  int gpu_segments = 1;

  for (int iter = 1; iter <= opt.iters; ++iter) {
    gpu_fp64_step(handle, d_in, d_a_fp64, d_out, n);
    std::swap(d_in, d_out);
    if (es_checkpoint(opt, iter)) {
      // Sincroniza ANTES de abrir la pausa -- ver la nota extensa en
      // medir_ruta(): sin esto el D2H de abajo espera a la cola asincrona
      // dentro de la pausa y ese tiempo de GPU se resta del medido.
      CHECK_CUDA(cudaDeviceSynchronize());
      const auto pause_t0 = std::chrono::steady_clock::now();
      power_buffer_stop_sampling(pb);
      snapshots.emplace_back(count);
      CHECK_CUDA(cudaMemcpy(snapshots.back().data(), d_in, count * sizeof(double),
                            cudaMemcpyDeviceToHost));
      power_buffer_start_sampling(pb);
      pause_s +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - pause_t0).count();
      ++gpu_segments;
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  const double total_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - pause_s;
  power_buffer_stop_sampling(pb);
  emit_csv_summary("GPU_FP64", n, opt.iters, total_s, gpu_segments, pb,
                   /*anchor_every_col=*/0);
  power_buffer_destroy(pb);

  cudaFree(d_in);
  cudaFree(d_out);
  return snapshots;
}

// =========================================================================
// FASES 2 y 3 -- las rutas WMMA de un formato T, cada una en su ventana.
// =========================================================================

template <typename T>
static void run_chained_route(const Options& opt, cublasHandle_t cublas_handle,
                               const T* d_a_tc, const double* d_a_fp64, const double* d_x0_64,
                               const std::vector<float>& x0,
                               const std::vector<std::vector<double>>& ref_snapshots,
                               const char* format_label) {
  const int n = opt.n;
  const size_t count = static_cast<size_t>(n) * n;
  const bool anchor_enabled = opt.anchor_every > 0;

  ChainedGemmState<T> s_off, s_on;
  chained_gemm_alloc(s_off, n, /*comp_on=*/false);
  if (opt.comp) chained_gemm_alloc(s_on, n, /*comp_on=*/true);

  // ANCLA FP64 (Fase 4): residuo de compensacion en DOUBLE, sombra del residuo
  // en float de s_on, mas dos buffers scratch (estado exacto reconstruido,
  // salida del paso FP64). Solo existen si --anchor-every > 0 (que ya exige
  // --comp on en parse_args).
  double* d_comp64_in = nullptr;
  double* d_comp64_out = nullptr;
  double* d_exact64 = nullptr;
  double* d_out64 = nullptr;
  if (anchor_enabled) {
    CHECK_CUDA(cudaMalloc(&d_comp64_in, count * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_comp64_out, count * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_exact64, count * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out64, count * sizeof(double)));
  }

  // float_to_tc_scalar<T> es __device__-only: la siembra en host necesita su
  // propio casteo. __half/__nv_bfloat16 tienen constructores desde float
  // utilizables en host -- si tu toolkit los marca __device__-only, reemplaza
  // por __float2half/__float2bfloat16 via un pequeno kernel de siembra, igual
  // que hace Fase_2/GEMM con convert_float_to_half_kernel.
  std::vector<T> x0_t(count);
  for (size_t i = 0; i < count; ++i) x0_t[i] = T(x0[i]);

  // Deja una ruta lista para arrancar desde x0. Se llama antes del warm-up y
  // otra vez despues, para que el bucle medido arranque del mismo estado que
  // arrancaria sin warm-up (incluido el residuo en double del ancla, asi que
  // el bucle medido arranca identico con o sin ancla).
  auto reset_ruta = [&](ChainedGemmState<T>& s, bool comp_on) {
    CHECK_CUDA(cudaMemcpy(s.d_x_in, x0_t.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    if (comp_on) {
      // Siembra el residuo desde el redondeo REAL de x0->T (no desde 0) -- ver
      // el comentario de seed_comp_from_double_kernel. d_x0_64 guarda x0 en
      // double y nunca se avanza, asi que da igual en que fase estemos.
      seed_comp_from_double_kernel<T><<<grid1d(n * n), kConversionThreads>>>(
          d_x0_64, s.d_x_in, s.d_comp_in, n * n);
      CHECK_CUDA(cudaGetLastError());
      if (anchor_enabled) {
        seed_comp64_from_double_kernel<T><<<grid1d(n * n), kConversionThreads>>>(
            d_x0_64, s.d_x_in, d_comp64_in, n * n);
        CHECK_CUDA(cudaGetLastError());
      }
    }
  };

  std::vector<float> host_route(count);
  std::vector<T> tmp_tc(count);

  auto medir_ruta = [&](ChainedGemmState<T>& s, bool comp_on, const char* sufijo,
                        int anchor_col) {
    reset_ruta(s, comp_on);
    // El warm-up NO ejecuta el camino de ancla (usa el paso WMMA normal para
    // las tres iteraciones descartables); el reset de abajo re-sincroniza
    // tambien el residuo en double.
    for (int w = 0; w < kWarmupIters; ++w) {
      chained_gemm_step(s, d_a_tc, n, comp_on);
      std::swap(s.d_x_in, s.d_x_out);
      if (comp_on) std::swap(s.d_comp_in, s.d_comp_out);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    reset_ruta(s, comp_on);

    // El ancla solo actua sobre la ruta compensada.
    const bool anclar = comp_on && anchor_enabled;

    PowerBuffer* pb = power_buffer_create(0);
    power_buffer_start_sampling(pb);
    const auto t0 = std::chrono::steady_clock::now();
    double pause_s = 0.0;
    int gpu_segments = 1;
    size_t ckpt_idx = 0;

    for (int iter = 1; iter <= opt.iters; ++iter) {
      const bool is_anchor_iter = anclar && (iter % opt.anchor_every == 0);
      if (is_anchor_iter) {
        // ANCLA FP64 (Fase 4): reconstruye el estado exacto T+comp en double,
        // avanza UN paso con gpu_fp64_step (la MISMA funcion que usa la
        // trayectoria de referencia, sin codigo cuBLAS nuevo) y re-siembra T +
        // el residuo double. narrow_double_to_float_kernel sincroniza el
        // residuo float que la siguiente iteracion NORMAL necesita (via
        // cast_float_to_tc_kernel dentro de chained_gemm_step).
        //
        // Este paso FP64 SI cuenta dentro del tiempo y la energia de esta
        // ruta: es su costo, y medirlo es el objetivo del barrido de K.
        reconstruct_exact_double_kernel<T><<<grid1d(n * n), kConversionThreads>>>(
            s.d_x_in, d_comp64_in, d_exact64, n * n);
        CHECK_CUDA(cudaGetLastError());
        gpu_fp64_step(cublas_handle, d_exact64, d_a_fp64, d_out64, n);
        reseed_double_from_fp64_kernel<T><<<grid1d(n * n), kConversionThreads>>>(
            d_out64, s.d_x_out, d_comp64_out, n * n);
        CHECK_CUDA(cudaGetLastError());
        narrow_double_to_float_kernel<<<grid1d(n * n), kConversionThreads>>>(
            d_comp64_out, s.d_comp_out, n * n);
        CHECK_CUDA(cudaGetLastError());
      } else {
        chained_gemm_step(s, d_a_tc, n, comp_on);
        if (anclar) {
          // Mantiene el residuo double sincronizado por si la SIGUIENTE
          // iteracion es de ancla.
          widen_comp_to_double_kernel<<<grid1d(n * n), kConversionThreads>>>(
              s.d_comp_out, d_comp64_out, n * n);
          CHECK_CUDA(cudaGetLastError());
        }
      }
      std::swap(s.d_x_in, s.d_x_out);
      if (comp_on) std::swap(s.d_comp_in, s.d_comp_out);
      if (anclar) std::swap(d_comp64_in, d_comp64_out);

      if (es_checkpoint(opt, iter)) {
        // Sincroniza ANTES de abrir la pausa. Los lanzamientos de kernel son
        // ASINCRONOS: sin esto, el cudaMemcpy D2H de mas abajo bloquea hasta
        // que la GPU termina toda la cola pendiente, esa espera cae DENTRO de
        // la pausa, y se resta de total_s -- es decir, el computo que se
        // quiere medir se descuenta del tiempo medido. Da t_iter_ms
        // imposiblemente bajos (se detecto a n=2048: 600 TFLOPS FP64 en una
        // tarjeta de 0.2 TFLOPS). Stencil ya lo hacia bien via
        // CudaEventTimer::stop_and_elapsed_ms(), que sincroniza.
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto pause_t0 = std::chrono::steady_clock::now();
        power_buffer_stop_sampling(pb);

        CHECK_CUDA(cudaMemcpy(tmp_tc.data(), s.d_x_in, count * sizeof(T),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) host_route[i] = static_cast<float>(tmp_tc[i]);
        const ErrorMetrics err =
            compare_fp64_ref_vs_fp32(ref_snapshots[ckpt_idx], host_route);
        // anchor_every va al FINAL de la fila. Vale 0 en la ruta "_none"
        // SIEMPRE, incluso si esta corrida se lanzo con --anchor-every K>0: el
        // ancla solo se aplica a la ruta con compensacion. Es una columna POR
        // FILA, no de la corrida -- ver Fase_4/tools/README.md.
        std::cout << "CSV_DRIFT," << format_label << sufijo << "," << n << "," << iter << ","
                  << err.rel_l2 << "," << err.rel_linf << "," << (err.solution_finite ? 1 : 0)
                  << "," << anchor_col << "\n";
        ++ckpt_idx;

        power_buffer_start_sampling(pb);
        pause_s +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - pause_t0).count();
        ++gpu_segments;
      }
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    const double total_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - pause_s;
    power_buffer_stop_sampling(pb);

    const std::string route = std::string(format_label) + sufijo;
    emit_csv_summary(route.c_str(), n, opt.iters, total_s, gpu_segments, pb, anchor_col);
    power_buffer_destroy(pb);
  };

  medir_ruta(s_off, /*comp_on=*/false, "_none", /*anchor_col=*/0);
  if (opt.comp) medir_ruta(s_on, /*comp_on=*/true, "_comp", opt.anchor_every);

  chained_gemm_free(s_off);
  if (opt.comp) chained_gemm_free(s_on);
  if (anchor_enabled) {
    cudaFree(d_comp64_in);
    cudaFree(d_comp64_out);
    cudaFree(d_exact64);
    cudaFree(d_out64);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);
  // Sin esto, power_buffer_create() nunca ve telemetry_nvml_enabled()==true
  // (esa funcion solo LEE la bandera, no la inicializa) y energy_gpu_j sale
  // NaN en TODAS las filas pase lo que pase el .sbatch compile con
  // -DUSE_NVML_TELEMETRY -- Stencil ya hace esta llamada en su main(), este
  // archivo nunca la tuvo.
  telemetry_nvml_initialize(0);

  std::cout << "N=" << opt.n << " iters=" << opt.iters << " comp=" << (opt.comp ? "on" : "off")
            << " checkpoint_every=" << opt.checkpoint_every
            << " anchor_every=" << opt.anchor_every
            << (opt.anchor_every > 0 ? " (activa)" : " (deshabilitada)") << "\n";

  double c = 0.0, lambda = 0.0;
  std::vector<__half> A_fp16;
  std::vector<__nv_bfloat16> A_bf16;
  std::vector<double> A_fp64;
  const bool need_fp16 = (opt.tc_format == TcFormat::FP16 || opt.tc_format == TcFormat::Both);
  const bool need_bf16 = (opt.tc_format == TcFormat::BF16 || opt.tc_format == TcFormat::Both);
  if (need_fp16) build_hadamard_operator<__half>(opt.n, opt.target_lambda, A_fp16, A_fp64, c, lambda);
  if (need_bf16) build_hadamard_operator<__nv_bfloat16>(opt.n, opt.target_lambda, A_bf16, A_fp64, c, lambda);
  std::cout << "Operador A = c*H (Hadamard de Sylvester): c=" << c << " lambda_por_iter=" << lambda
            << "\n";

  std::vector<float> x0(static_cast<size_t>(opt.n) * opt.n);
  {
    std::mt19937 gen(opt.seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x0) v = dist(gen);
  }

  __half* d_a_fp16 = nullptr;
  __nv_bfloat16* d_a_bf16 = nullptr;
  double* d_a_fp64 = nullptr;
  const size_t count = static_cast<size_t>(opt.n) * opt.n;
  CHECK_CUDA(cudaMalloc(&d_a_fp64, count * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_a_fp64, A_fp64.data(), count * sizeof(double), cudaMemcpyHostToDevice));

  // x0 en double, INMUTABLE durante toda la corrida: lo usan la referencia
  // FP64 (como estado inicial) y la siembra del residuo de compensacion de
  // cada ruta. Vive aqui, y no dentro de una fase, precisamente para que
  // ninguna fase pueda avanzarlo y dejar a la siguiente sembrando desde un
  // estado que ya no es x0.
  double* d_x0_64 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x0_64, count * sizeof(double)));
  {
    std::vector<double> x0_d(count);
    for (size_t i = 0; i < count; ++i) x0_d[i] = static_cast<double>(x0[i]);
    CHECK_CUDA(cudaMemcpy(d_x0_64, x0_d.data(), count * sizeof(double), cudaMemcpyHostToDevice));
  }

  cublasHandle_t cublas_handle;
  CHECK_CUBLAS(cublasCreate(&cublas_handle));

  // FASE 1, una sola vez para toda la invocacion (la trayectoria FP64 de
  // referencia no depende del formato T). Sus buffers se liberan al volver,
  // antes de que las rutas WMMA reserven los suyos.
  const std::vector<std::vector<double>> ref_snapshots =
      run_fp64_reference(opt, cublas_handle, d_a_fp64, d_x0_64);

  if (need_fp16) {
    CHECK_CUDA(cudaMalloc(&d_a_fp16, count * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_a_fp16, A_fp16.data(), count * sizeof(__half), cudaMemcpyHostToDevice));
    run_chained_route<__half>(opt, cublas_handle, d_a_fp16, d_a_fp64, d_x0_64, x0,
                              ref_snapshots, "FP16");
  }
  if (need_bf16) {
    CHECK_CUDA(cudaMalloc(&d_a_bf16, count * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(d_a_bf16, A_bf16.data(), count * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    run_chained_route<__nv_bfloat16>(opt, cublas_handle, d_a_bf16, d_a_fp64, d_x0_64, x0,
                                     ref_snapshots, "BF16");
  }

  cublasDestroy(cublas_handle);
  cudaFree(d_a_fp16);
  cudaFree(d_a_bf16);
  cudaFree(d_a_fp64);
  cudaFree(d_x0_64);
  return 0;
}
