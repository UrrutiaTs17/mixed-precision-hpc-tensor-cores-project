// Fase_4/Convolution/conv_chained.cu
//
// Encadenamiento genuino de Convolucion 2D: X(n+1) = conv(X(n), W), con W un
// filtro FIJO (no cambia entre iteraciones) aplicado repetidamente -- un
// proceso de suavizado/difusion iterativo. Especificado en el documento
// "Plan de Precision Mixta", secciones 02 (encadenamiento) y 01/"Ancla FP64"
// (esta extension) -- esa es la referencia normativa.
//
// ESTE ARCHIVO ES Fase_3/Convolution/conv_chained.cu + EL ANCLA FP64:
// Punto de partida identico a Fase_3/Convolution/conv_chained.cu
// (encadenamiento con compensacion por linealidad en FLOAT, sin ancla). La
// unica adicion es --anchor-every K: cada K iteraciones de la ruta CON
// compensacion (--comp on), en vez del paso WMMA normal, el codigo
// (a) reconstruye el estado exacto T+comp en DOUBLE, (b) avanza UN paso con
// gpu_fp64_conv_step() -- la MISMA funcion que usa la trayectoria de
// referencia de este archivo, sin codigo cuBLAS nuevo -- y (c) re-siembra T
// (cuantizado) y el residuo de compensacion, ahora en DOUBLE.
//
// (El ancla usaba antes el MISMO buffer scratch de im2col que la referencia,
// porque las dos llamadas ocurrian dentro de la misma iteracion del mismo
// bucle. Desde que la medicion se separo por fases -- ver "MEDICION POR RUTA"
// mas abajo -- la referencia corre en una fase propia y libera su scratch
// antes de que empiecen las rutas WMMA, asi que el ancla reserva el suyo.
// Con eso desaparece la unica diferencia estructural del mecanismo de ancla
// frente a GEMM, y con ella la pregunta sobre carreras de datos que obligaba
// a plantearse cada vez que alguien tocaba el bucle.)
//
// Mismo patron exacto que Fase_4/GEMM/gemm_chained.cu (que a su vez
// sigue a Fase_4/Stencil/stencil_tensor_activation.cu) -- ver el comentario
// de cabecera de ese archivo para el razonamiento completo (por que DOUBLE y
// no float para el residuo del ancla, por que --anchor-every requiere
// --comp on, las dos puertas de validacion K=0/K=1). Busca el marcador
// "ANCLA FP64 (Fase 4)" en este archivo para cada pieza.
//
// POR QUE ENCADENAR: ver Fase_3/GEMM/README.md, misma correccion #13 del
// plan, aplicada aqui a Convolucion.
//
// EL FILTRO W: el mismo Laplaciano de 5 puntos que usa Stencil (coeficientes
// EXACTOS en cualquier formato: 0.25 vecino, -1.0 centro, 0 en las esquinas),
// expresado como filtro de convolucion 3x3 -- no un filtro generico. Da una
// eleccion reproducible, ya justificada por el propio planteamiento del
// problema del plan (equivalencia estructural stencil/convolucion), y evita
// el "error de representacion del operador" que SI tendria un filtro con
// coeficientes arbitrarios no representables exactamente en FP16/BF16 (ver
// el mismo hallazgo aplicado a GEMM en Fase_3/GEMM/README.md).
//
// POR QUE C=K=64 CANALES (no C=K=1 como el campo escalar de Stencil): la
// convolucion se expresa como GEMM via im2col (Y[K,outH*outW] =
// W[K,C*R*S]*col[C*R*S,outH*outW], igual que la ruta 4 de Fase_2/Convolution
// -- ver el comentario junto a build_block_diagonal_filter() mas abajo). Con
// C=K=1, M=1: el kernel WMMA de common/wmma_gemm.cuh tiene tiles de salida
// 64x64, asi que M=1 desperdiciaria 63/64 del tile y daria numeros de
// rendimiento enganosos (mide ocupacion pesima del kernel, no el efecto de
// precision que el proyecto quiere caracterizar). En vez de eso, W se
// construye BLOQUE-DIAGONAL: C=K=64 canales, cada uno con el MISMO filtro de
// 5 puntos aplicado de forma INDEPENDIENTE (sin mezclar canales, W[k,c,*,*]
// es cero salvo c==k) -- matematicamente identico a 64 simulaciones de
// Stencil corriendo en paralelo, con mejor aprovechamiento del tile de 64.
//
// Compilar con:
// nvcc -std=c++17 conv_chained.cu -o conv_chained \
//      -lcublas -gencode arch=compute_80,code=sm_80 \
//      --allow-unsupported-compiler
//
// Ejecutar:
// ./conv_chained --hw 64 --iters 20 --tc fp16 --comp on --anchor-every 5
//
// El mismo aviso de GEMM aplica aqui: la referencia FP64 usa cuBLAS
// (column-major); el kernel WMMA es row-major. VERIFICAR EN PACCA con un
// --hw chico contra una referencia independiente antes de confiar en
// cualquier resultado -- ver el comentario junto a gpu_fp64_conv_step().

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

// power_sampling.h declara simbolos extern "C" (NVML/dirent): FUERA del
// namespace anonimo -- ver la nota de cabecera de ese header.
#include "../../common/power_sampling.h"

namespace {

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
constexpr int kChannels = 64;  // C = K, ver aviso de cabecera del archivo.
constexpr int kFilterR = 3;
constexpr int kFilterS = 3;
constexpr int kCRS = kChannels * kFilterR * kFilterS;  // 576 = 18 * kKStep(32)

// wmma_gemm_kernel (common/wmma_gemm.cuh) exige M multiplo de kBlockTileM,
// K multiplo de kKStep -- aqui M=kChannels (fijo) y K=kCRS (fijo, derivado
// de kChannels/kFilterR/kFilterS). Si alguien cambia esas constantes sin
// verificar esto, mejor un error de compilacion que un --hw que compila
// pero silenciosamente trunca la cola de K o M (ver el mismo aviso en
// benchmark_gpu_wmma de Fase_2/GEMM sobre por que la condicion correcta
// importa). N=hw*hw se valida en tiempo de ejecucion en parse_args, porque
// depende de --hw.
static_assert(kChannels % kBlockTileM == 0, "kChannels debe ser multiplo de kBlockTileM");
static_assert(kCRS % kKStep == 0, "kCRS (kChannels*kFilterR*kFilterS) debe ser multiplo de kKStep");

enum class TcFormat { FP16, BF16, Both };

struct Options {
  int hw = 64;  // alto y ancho del campo espacial (H = W = hw, "SAME" padding)
  int iters = 20;
  TcFormat tc_format = TcFormat::Both;
  bool comp = false;
  int checkpoint_every = 0;
  // ANCLA FP64 (Fase 4): 0 = deshabilitada (identico a Fase_3/Convolution).
  // > 0 exige --comp on -- ver la validacion en parse_args() y el
  // comentario de cabecera del archivo.
  int anchor_every = 0;
  unsigned int seed = 42;
};

// =========================================================================
// Filtro W bloque-diagonal: mismo Laplaciano de 5 puntos de Stencil (stress:
// c_neigh=0.25, c_center=-1.0, coeficientes EXACTOS), replicado en la
// diagonal de canales. W[k, c*R*S + r*S + s] = (c==k) ? filtro[r][s] : 0.
// =========================================================================

static void build_block_diagonal_filter(std::vector<double>& W) {
  constexpr double c_neigh = 0.25;
  constexpr double c_center = -1.0;
  // filtro[r][s], r,s en [0,3): 5 puntos de Von Neumann (sin esquinas).
  double filt[kFilterR][kFilterS] = {
      {0.0, c_neigh, 0.0},
      {c_neigh, c_center, c_neigh},
      {0.0, c_neigh, 0.0},
  };
  W.assign(static_cast<size_t>(kChannels) * kCRS, 0.0);
  for (int k = 0; k < kChannels; ++k) {
    for (int r = 0; r < kFilterR; ++r) {
      for (int s = 0; s < kFilterS; ++s) {
        const size_t idx =
            static_cast<size_t>(k) * kCRS + (static_cast<size_t>(k) * kFilterR + r) * kFilterS + s;
        W[idx] = filt[r][s];
      }
    }
  }
}

// =========================================================================
// im2col: construye col[C*R*S, outH*outW] a partir del campo X[C,H,W].
// "SAME" padding (pad=1, stride=1, dilation=1, R=S=3): outH=H, outW=W --
// requisito para que la salida tenga la MISMA forma que la entrada y pueda
// alimentar la siguiente iteracion.
// =========================================================================

// Version T->T (estado principal, ya en baja precision -- no hay conversion,
// solo reordenamiento de memoria).
template <typename T>
__global__ static void im2col_tc_kernel(const T* __restrict__ x, T* __restrict__ col, int hw) {
  const int Ncol = hw * hw;
  const int id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= kCRS * Ncol) return;
  const int crs = id / Ncol;
  const int pos = id % Ncol;
  const int c = crs / (kFilterR * kFilterS);
  const int rs = crs % (kFilterR * kFilterS);
  const int r = rs / kFilterS;
  const int s = rs % kFilterS;
  const int oh = pos / hw;
  const int ow = pos % hw;
  const int ih = oh - 1 + r;  // pad=1, stride=1, dilation=1
  const int iw = ow - 1 + s;
  T zero{};
  col[id] = (ih >= 0 && ih < hw && iw >= 0 && iw < hw) ? x[(c * hw + ih) * hw + iw] : zero;
}

// Version float->T (residuo de compensacion, con casteo -- comp se guarda en
// float, igual que GEMM Fase 3, y se trunca a T solo para este paso,
// explotando linealidad: conv(T+comp, W) = conv(T,W) + conv(comp,W)).
template <typename T>
__global__ static void im2col_float_to_tc_kernel(const float* __restrict__ x, T* __restrict__ col,
                                                   int hw) {
  const int Ncol = hw * hw;
  const int id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= kCRS * Ncol) return;
  const int crs = id / Ncol;
  const int pos = id % Ncol;
  const int c = crs / (kFilterR * kFilterS);
  const int rs = crs % (kFilterR * kFilterS);
  const int r = rs / kFilterS;
  const int s = rs % kFilterS;
  const int oh = pos / hw;
  const int ow = pos % hw;
  const int ih = oh - 1 + r;
  const int iw = ow - 1 + s;
  col[id] = (ih >= 0 && ih < hw && iw >= 0 && iw < hw)
                ? float_to_tc_scalar<T>(x[(c * hw + ih) * hw + iw])
                : float_to_tc_scalar<T>(0.0f);
}

// Version double->double (referencia FP64).
__global__ static void im2col_double_kernel(const double* __restrict__ x,
                                             double* __restrict__ col, int hw) {
  const int Ncol = hw * hw;
  const int id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= kCRS * Ncol) return;
  const int crs = id / Ncol;
  const int pos = id % Ncol;
  const int c = crs / (kFilterR * kFilterS);
  const int rs = crs % (kFilterR * kFilterS);
  const int r = rs / kFilterS;
  const int s = rs % kFilterS;
  const int oh = pos / hw;
  const int ow = pos % hw;
  const int ih = oh - 1 + r;
  const int iw = ow - 1 + s;
  col[id] = (ih >= 0 && ih < hw && iw >= 0 && iw < hw) ? x[(c * hw + ih) * hw + iw] : 0.0;
}

template <typename T>
__global__ static void cast_float_to_tc_kernel(const float* __restrict__ src, T* __restrict__ dst,
                                                int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = float_to_tc_scalar<T>(src[i]);
}

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
  // importar --comp) nunca reserva d_comp_out -- queda nullptr, de ahi la
  // guardia (ver la misma logica en Fase_3/GEMM/gemm_chained.cu).
  if (comp_on) comp_out[i] = exact - tc_scalar_to_float<T>(q);
}

// Siembra INICIAL del residuo de compensacion a partir del x0 real (en
// double, ya en el device como d_x64_in) y su cuantizacion a T -- NO desde
// cero. Mismo patron que Fase_4/Stencil (seed_comp_from_conversion_kernel/
// seed_comp64_from_conversion_kernel). Dejar comp/comp64 en 0 tras la
// siembra de X0 descartaria el redondeo x0->T de la primerisima conversion
// en vez de quedar capturado desde el principio, como exige la propiedad de
// reconstruccion Q(v)+comp=v que el resto del mecanismo asume ya valida
// desde t=0. El gate K=1 (ver README.md) depende de esta siembra para
// converger a la referencia FP64 -- ver Fase_4/GEMM/gemm_chained.cu para el
// razonamiento completo.
template <typename T>
__global__ static void seed_comp_from_double_kernel(const double* __restrict__ src64,
                                                      const T* __restrict__ src_tc,
                                                      float* __restrict__ comp, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) comp[i] = static_cast<float>(src64[i]) - tc_scalar_to_float<T>(src_tc[i]);
}

// Version DOBLE de la siembra anterior -- necesaria para d_comp64_in, para
// que el gate K=1 sea exacto desde la primera iteracion.
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
// el reseed contra la referencia FP64 -- operan sobre el estado "field"
// (kChannels*hw*hw), no sobre los buffers im2col. Mismo patron que
// Fase_4/GEMM/gemm_chained.cu (identico salvo el nombre del tamano: alli es
// n*n, aqui es field).
// =========================================================================

__global__ static void widen_comp_to_double_kernel(const float* __restrict__ comp_f,
                                                     double* __restrict__ comp_d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) comp_d[i] = static_cast<double>(comp_f[i]);
}

__global__ static void narrow_double_to_float_kernel(const double* __restrict__ comp_d,
                                                       float* __restrict__ comp_f, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) comp_f[i] = static_cast<float>(comp_d[i]);
}

// exact = double(tc_to_float(T)) + comp64 -- ver la misma funcion en
// Fase_4/GEMM/gemm_chained.cu para el razonamiento completo.
template <typename T>
__global__ static void reconstruct_exact_double_kernel(const T* __restrict__ t_in,
                                                         const double* __restrict__ comp64_in,
                                                         double* __restrict__ exact_out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  exact_out[i] = static_cast<double>(tc_scalar_to_float<T>(t_in[i])) + comp64_in[i];
}

// comp64_out = out64 - double(tc_to_float(q)) -- SIN pasar por float en el
// camino (critico para que --anchor-every 1 sea bit-identico a la
// referencia FP64; ver Fase_4/GEMM/gemm_chained.cu y el bug original en
// Fase_4/Stencil).
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
// Referencia FP64: un paso conv(X,W) via im2col + cublasDgemm.
//
// Mismo aviso que Fase_3/GEMM/gemm_chained.cu (gpu_fp64_step): cuBLAS es
// column-major, el resto de este archivo es row-major. col_fp64 (C*R*S x
// outH*outW) y W_fp64 (K x C*R*S) estan en row-major; queremos
// Y[K,outH*outW] = W * col. Con la misma derivacion que GEMM (una matriz
// row-major leida column-major es su transpuesta; (W*col)^T = col^T * W^T),
// el orden de argumentos de cublasDgemm es (col_fp64_buf, W_fp64_buf) --
// col PRIMERO, W SEGUNDO -- invertido respecto al orden "natural" W*col.
// VERIFICAR EN PACCA con --hw chico antes de confiar en cualquier resultado.
//
// ANCLA FP64 (Fase 4): esta funcion se usa TANTO para la trayectoria de
// referencia (fase 1, comparacion en cada checkpoint) COMO para el paso
// puntual que el ancla inyecta en la ruta de baja precision (fase 3) --
// misma funcion, sin cuBLAS nuevo. Cada uno pasa SU PROPIO d_col_scratch:
// desde que la medicion se separo por fases, la referencia libera el suyo
// antes de que empiecen las rutas WMMA, asi que no hay buffer compartido ni
// pregunta sobre carreras de datos que responder.
static void gpu_fp64_conv_step(cublasHandle_t handle, const double* d_x_in, const double* d_w,
                               double* d_col_scratch, double* d_x_out, int hw) {
  const int Ncol = hw * hw;
  im2col_double_kernel<<<grid1d(kCRS * Ncol), kConversionThreads>>>(d_x_in, d_col_scratch, hw);
  CHECK_CUDA(cudaGetLastError());
  const double alpha = 1.0;
  const double beta = 0.0;
  // Y[K, Ncol] = W[K, CRS] * col[CRS, Ncol], todos row-major -> cuBLAS con
  // argumentos invertidos (ver derivacion arriba): m=Ncol, n=K, k=CRS.
  CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Ncol, kChannels, kCRS, &alpha,
                           d_col_scratch, Ncol, d_w, kCRS, &beta, d_x_out, Ncol));
}

// =========================================================================
// Ruta WMMA encadenada.
// =========================================================================

template <typename T>
struct ChainedConvState {
  T* d_x_in = nullptr;
  T* d_x_out = nullptr;
  float* d_comp_in = nullptr;
  float* d_comp_out = nullptr;
  T* d_col = nullptr;          // scratch: im2col(x_in), [CRS, Ncol]
  float* d_t_raw = nullptr;    // scratch: W * col(x_in), [K, Ncol]
  T* d_comp_col = nullptr;     // scratch: im2col(comp_in as T), [CRS, Ncol]
  float* d_comp_raw = nullptr; // scratch: W * comp_col, [K, Ncol]
};

template <typename T>
static void chained_conv_alloc(ChainedConvState<T>& s, int hw, bool comp_on) {
  const size_t field = static_cast<size_t>(kChannels) * hw * hw;
  const size_t col_sz = static_cast<size_t>(kCRS) * hw * hw;
  CHECK_CUDA(cudaMalloc(&s.d_x_in, field * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&s.d_x_out, field * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&s.d_col, col_sz * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&s.d_t_raw, field * sizeof(float)));
  if (comp_on) {
    CHECK_CUDA(cudaMalloc(&s.d_comp_in, field * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_comp_out, field * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&s.d_comp_col, col_sz * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&s.d_comp_raw, field * sizeof(float)));
    CHECK_CUDA(cudaMemset(s.d_comp_in, 0, field * sizeof(float)));
  }
}

template <typename T>
static void chained_conv_free(ChainedConvState<T>& s) {
  cudaFree(s.d_x_in);
  cudaFree(s.d_x_out);
  cudaFree(s.d_col);
  cudaFree(s.d_t_raw);
  cudaFree(s.d_comp_in);
  cudaFree(s.d_comp_out);
  cudaFree(s.d_comp_col);
  cudaFree(s.d_comp_raw);
}

// Un paso encadenado: X_out = conv(X_in, W) (+ correccion si comp_on). No
// sabe nada del ancla -- ANCLA FP64 (Fase 4) es pura orquestacion en
// run_chained_route(), igual que en Fase_4/GEMM/gemm_chained.cu. Se
// reutiliza tal cual desde Fase_3/Convolution/conv_chained.cu, sin ningun
// cambio.
template <typename T>
static void chained_conv_step(ChainedConvState<T>& s, const T* d_w_tc, int hw, bool comp_on) {
  const int Ncol = hw * hw;
  const dim3 gemm_block(kBlockWarpsM * kBlockWarpsN * 32);
  const dim3 gemm_grid(static_cast<unsigned int>((kChannels + kBlockTileM - 1) / kBlockTileM),
                       static_cast<unsigned int>((Ncol + kBlockTileN - 1) / kBlockTileN));

  im2col_tc_kernel<T><<<grid1d(kCRS * Ncol), kConversionThreads>>>(s.d_x_in, s.d_col, hw);
  CHECK_CUDA(cudaGetLastError());
  wmma_gemm_kernel<T><<<gemm_grid, gemm_block>>>(d_w_tc, s.d_col, s.d_t_raw, kChannels, Ncol,
                                                  kCRS);
  CHECK_CUDA(cudaGetLastError());

  if (comp_on) {
    im2col_float_to_tc_kernel<T><<<grid1d(kCRS * Ncol), kConversionThreads>>>(
        s.d_comp_in, s.d_comp_col, hw);
    CHECK_CUDA(cudaGetLastError());
    wmma_gemm_kernel<T><<<gemm_grid, gemm_block>>>(d_w_tc, s.d_comp_col, s.d_comp_raw, kChannels,
                                                    Ncol, kCRS);
    CHECK_CUDA(cudaGetLastError());
  }

  const int field = kChannels * Ncol;
  finalize_step_kernel<T><<<grid1d(field), kConversionThreads>>>(
      s.d_t_raw, comp_on ? s.d_comp_raw : nullptr, comp_on, s.d_x_out, s.d_comp_out, field);
  CHECK_CUDA(cudaGetLastError());
}

// =========================================================================
// CLI
// =========================================================================

static void print_usage(const char* prog) {
  std::cout << "Uso: " << prog
            << " [--hw N] [--iters K] [--tc fp16|bf16|both] [--comp off|on]\n"
            << "         [--checkpoint-every K] [--anchor-every K] [--seed S]\n\n"
            << "  --hw N (multiplo de 64, default 64): alto y ancho del campo espacial\n"
            << "  (H=W=N, padding SAME). " << kChannels << " canales fijos, ver cabecera\n"
            << "  del archivo sobre por que.\n"
            << "  --comp off|on (default off): compensacion por linealidad.\n"
            << "  --anchor-every K (default 0 = deshabilitada, Fase 4): cada K\n"
            << "  iteraciones de la ruta con compensacion, en vez del paso WMMA normal,\n"
            << "  se reconstruye el estado exacto en double, se da un paso con la\n"
            << "  referencia FP64 (cublasDgemm) y se re-siembra T y el residuo -- ver el\n"
            << "  comentario de cabecera del archivo. Requiere --comp on. K=1 debe dar\n"
            << "  drift bit-identico a la referencia FP64; K=0 es identico a Fase 3.\n"
            << "\nEjemplos:\n"
            << "  " << prog << " --hw 64 --iters 20 --tc fp16 --comp on\n"
            << "  " << prog << " --hw 64 --iters 40 --tc fp16 --comp on --anchor-every 5\n"
            << "  " << prog << " --hw 128 --iters 10 --tc both --checkpoint-every 5\n";
}

static int parse_int_arg(int& i, int argc, char** argv) {
  if (i + 1 >= argc) { std::cerr << "Falta valor para " << argv[i] << "\n"; std::exit(EXIT_FAILURE); }
  return std::atoi(argv[++i]);
}

static bool parse_on_off(const char* flag, const char* value) {
  if (std::strcmp(value, "off") == 0) return false;
  if (std::strcmp(value, "on") == 0) return true;
  std::cerr << "Valor invalido para " << flag << ": '" << value << "' (use off|on)\n";
  std::exit(EXIT_FAILURE);
}

static Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (std::strcmp(argv[i], "--hw") == 0) {
      opt.hw = parse_int_arg(i, argc, argv);
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
    } else {
      std::cerr << "Argumento desconocido: " << argv[i] << "\n";
      print_usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  const int ncol = opt.hw * opt.hw;
  if (opt.hw <= 0 || ncol % kBlockTileN != 0) {
    std::cerr << "--hw=" << opt.hw << " invalido: hw*hw debe ser multiplo de " << kBlockTileN
              << " (dimension N del GEMM subyacente). hw=64,128,192,256... funcionan.\n";
    std::exit(EXIT_FAILURE);
  }
  if (opt.iters <= 0) { std::cerr << "--iters debe ser positivo.\n"; std::exit(EXIT_FAILURE); }
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
// Las trayectorias se miden en FASES SEPARADAS, cada una con su propio
// cronometro y su propia ventana de PowerBuffer.
//
// Antes corrian entrelazadas dentro de un mismo bucle, envueltas por UN solo
// cronometro y por dos PowerBuffer abiertos y cerrados en los MISMOS
// instantes. Resultado: t_iter_ms, t_total_ms, gflops y energy_gpu_j salian
// IDENTICOS en las dos filas CSV_SUMMARY, y ademas incluian el costo de la
// referencia FP64. Dos de los tres ejes del Frente de Pareto 3D quedaban
// inutilizables para este kernel (el de ERROR siempre estuvo bien: rel_l2 y
// rel_linf nunca dependieron del cronometro). Stencil nunca tuvo el problema,
// ya media por ruta; esto alinea Convolucion con esa forma.
//
// POR QUE FASES SEPARADAS Y NO VARIOS CRONOMETROS EN UN BUCLE UNICO. El
// tiempo si se podia separar con eventos CUDA dentro del bucle. La ENERGIA
// no: nvmlDeviceGetTotalEnergyConsumption es un contador de TODO el
// dispositivo y su cuantizacion (~20-25 ms de GPU cargada, ver "REGIMEN DE
// VALIDEZ" en common/power_sampling.h) es mayor que el tramo de una
// trayectoria en una iteracion. Sumar cientos de tramos cuantizados no da una
// energia utilizable. Atribuir energia a una ruta exige darle una ventana
// CONTIGUA propia, y eso exige separar los bucles.
//
// SEGUNDO DEFECTO, CORREGIDO A LA VEZ: la pausa de checkpoint se abria ANTES
// de sincronizar. Los lanzamientos de kernel son asincronos, asi que el
// cudaMemcpy D2H del checkpoint bloqueaba esperando toda la cola pendiente,
// esa espera caia dentro de la pausa, y se restaba de total_s -- el computo
// que se queria medir se descontaba del tiempo medido. Ahora se sincroniza
// antes de abrir la pausa (Stencil ya lo hacia bien via
// CudaEventTimer::stop_and_elapsed_ms(), que sincroniza).
//
// LO QUE CUESTA: la referencia FP64 se corre UNA vez (no una por formato) y
// sus estados en los checkpoints se guardan en RAM del host. Mismo patron que
// ya usa Stencil (ckpt.fp64_checkpoints). El costo se imprime al arrancar.
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
// gflops usa SIEMPRE los FLOPs UTILES (una conv(X,W) por iteracion, contada
// sobre el GEMM subyacente del im2col), tambien en la ruta compensada, que
// hace un segundo producto para la correccion: esa ruta entrega el mismo
// resultado util por iteracion a mayor costo, asi que su gflops mas bajo es
// exactamente lo que hay que reportar, no un artefacto de contabilidad.
static void emit_csv_summary(const char* route, int hw, int iters, double total_s,
                             int gpu_segments, PowerBuffer* pb, int anchor_every_col) {
  const double flops_per_iter = 2.0 * kChannels * static_cast<double>(hw) * hw * kCRS;
  const double gflops = (flops_per_iter * iters / 1e9) / total_s;
  // Mismo criterio de confiabilidad que Stencil (power_sampling.h,
  // kEnergyWindowReliableSeconds): con muchos tramos cortos el error de
  // cuantizacion del contador NVML por tramo domina y la energia deja de ser
  // comparable entre rutas. window_reliable=0 no invalida t_iter_ms/gflops
  // (vienen del reloj de pared, no de NVML), solo la columna de energia.
  const bool reliable = total_s >= kEnergyWindowReliableSeconds * gpu_segments;
  std::cout << "CSV_SUMMARY," << route << "," << hw << "," << iters << ","
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
// =========================================================================
static std::vector<std::vector<double>> run_fp64_reference(const Options& opt,
                                                            cublasHandle_t handle,
                                                            const double* d_w_fp64,
                                                            const double* d_x0_64) {
  const int hw = opt.hw;
  const size_t field = static_cast<size_t>(kChannels) * hw * hw;
  const size_t col_sz = static_cast<size_t>(kCRS) * hw * hw;
  const int num_ckpt = contar_checkpoints(opt);

  const double snap_mib = static_cast<double>(field) * sizeof(double) / (1024.0 * 1024.0);
  std::cout << "Snapshots FP64 de referencia: " << num_ckpt << " x " << snap_mib
            << " MiB = " << (num_ckpt * snap_mib / 1024.0) << " GiB de RAM del host\n";

  double* d_in = nullptr;
  double* d_out = nullptr;
  double* d_col64 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, field * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_out, field * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_col64, col_sz * sizeof(double)));
  auto reset = [&]() {
    CHECK_CUDA(cudaMemcpy(d_in, d_x0_64, field * sizeof(double), cudaMemcpyDeviceToDevice));
  };
  reset();

  for (int w = 0; w < kWarmupIters; ++w) {
    gpu_fp64_conv_step(handle, d_in, d_w_fp64, d_col64, d_out, hw);
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
  // total para que gflops/energia reflejen solo computo GPU.
  double pause_s = 0.0;
  int gpu_segments = 1;

  for (int iter = 1; iter <= opt.iters; ++iter) {
    gpu_fp64_conv_step(handle, d_in, d_w_fp64, d_col64, d_out, hw);
    std::swap(d_in, d_out);
    if (es_checkpoint(opt, iter)) {
      // Sincroniza ANTES de abrir la pausa -- ver la nota extensa de cabecera
      // de esta seccion: sin esto el D2H de abajo espera a la cola asincrona
      // dentro de la pausa y ese tiempo de GPU se resta del medido.
      CHECK_CUDA(cudaDeviceSynchronize());
      const auto pause_t0 = std::chrono::steady_clock::now();
      power_buffer_stop_sampling(pb);
      snapshots.emplace_back(field);
      CHECK_CUDA(cudaMemcpy(snapshots.back().data(), d_in, field * sizeof(double),
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
  emit_csv_summary("GPU_FP64", hw, opt.iters, total_s, gpu_segments, pb,
                   /*anchor_every_col=*/0);
  power_buffer_destroy(pb);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_col64);
  return snapshots;
}

// =========================================================================
// FASES 2 y 3 -- las rutas WMMA de un formato T, cada una en su ventana.
//
// EL SCRATCH DE im2col YA NO SE COMPARTE ENTRE LA REFERENCIA Y EL ANCLA.
// Antes las dos llamadas a gpu_fp64_conv_step() ocurrian dentro de la misma
// iteracion del mismo bucle y reusaban el mismo d_col64; era seguro (mismo
// stream, secuenciales) pero era la UNICA diferencia estructural del
// mecanismo de ancla frente a GEMM, y obligaba a razonar sobre carreras de
// datos cada vez que alguien tocaba el bucle. Con las fases separadas, la
// referencia libera su scratch al terminar y el ancla reserva el suyo: la
// pregunta desaparece. El pico de memoria no sube (baja un poco), porque los
// dos buffers ya no coexisten -- ver el presupuesto en el .sbatch.
// =========================================================================

template <typename T>
static void run_chained_route(const Options& opt, cublasHandle_t cublas_handle,
                               const T* d_w_tc, const double* d_w_fp64, const double* d_x0_64,
                               const std::vector<float>& x0,
                               const std::vector<std::vector<double>>& ref_snapshots,
                               const char* format_label) {
  const int hw = opt.hw;
  const size_t field = static_cast<size_t>(kChannels) * hw * hw;
  const size_t col_sz = static_cast<size_t>(kCRS) * hw * hw;
  const bool anchor_enabled = opt.anchor_every > 0;

  ChainedConvState<T> s_off, s_on;
  chained_conv_alloc(s_off, hw, false);
  if (opt.comp) chained_conv_alloc(s_on, hw, true);

  // ANCLA FP64 (Fase 4): residuo de compensacion en DOUBLE (sombra del
  // residuo en float de s_on), dos buffers scratch de tamano "field" (estado
  // exacto reconstruido, salida del paso FP64) y el scratch de im2col en
  // double que necesita gpu_fp64_conv_step(). Solo existen si
  // --anchor-every > 0 (que ya exige --comp on en parse_args).
  double* d_comp64_in = nullptr;
  double* d_comp64_out = nullptr;
  double* d_exact64 = nullptr;
  double* d_out64 = nullptr;
  double* d_col64_anchor = nullptr;
  if (anchor_enabled) {
    CHECK_CUDA(cudaMalloc(&d_comp64_in, field * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_comp64_out, field * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_exact64, field * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out64, field * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_col64_anchor, col_sz * sizeof(double)));
  }

  std::vector<T> x0_t(field);
  for (size_t i = 0; i < field; ++i) x0_t[i] = T(x0[i]);  // ver nota en gemm_chained.cu

  // Deja una ruta lista para arrancar desde x0. Se llama antes del warm-up y
  // otra vez despues, para que el bucle medido arranque del mismo estado que
  // arrancaria sin warm-up (incluido el residuo en double del ancla, asi que
  // el bucle medido arranca identico con o sin ancla).
  auto reset_ruta = [&](ChainedConvState<T>& s, bool comp_on) {
    CHECK_CUDA(cudaMemcpy(s.d_x_in, x0_t.data(), field * sizeof(T), cudaMemcpyHostToDevice));
    if (comp_on) {
      // Siembra el residuo desde el redondeo REAL de x0->T (no desde 0) -- ver
      // el comentario de seed_comp_from_double_kernel. d_x0_64 guarda x0 en
      // double y nunca se avanza, asi que da igual en que fase estemos.
      seed_comp_from_double_kernel<T><<<grid1d(static_cast<int>(field)), kConversionThreads>>>(
          d_x0_64, s.d_x_in, s.d_comp_in, static_cast<int>(field));
      CHECK_CUDA(cudaGetLastError());
      if (anchor_enabled) {
        seed_comp64_from_double_kernel<T><<<grid1d(static_cast<int>(field)),
                                            kConversionThreads>>>(
            d_x0_64, s.d_x_in, d_comp64_in, static_cast<int>(field));
        CHECK_CUDA(cudaGetLastError());
      }
    }
  };

  std::vector<float> host_route(field);
  std::vector<T> tmp_tc(field);

  auto medir_ruta = [&](ChainedConvState<T>& s, bool comp_on, const char* sufijo,
                        int anchor_col) {
    reset_ruta(s, comp_on);
    // El warm-up NO ejecuta el camino de ancla (usa el paso WMMA normal);
    // el reset de abajo re-sincroniza tambien el residuo en double.
    for (int w = 0; w < kWarmupIters; ++w) {
      chained_conv_step(s, d_w_tc, hw, comp_on);
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
        // avanza UN paso con gpu_fp64_conv_step (la MISMA funcion que usa la
        // trayectoria de referencia, sobre su propio scratch) y re-siembra T +
        // el residuo double.
        //
        // Este paso FP64 SI cuenta dentro del tiempo y la energia de esta
        // ruta: es su costo, y medirlo es el objetivo del barrido de K.
        reconstruct_exact_double_kernel<T><<<grid1d(static_cast<int>(field)),
                                             kConversionThreads>>>(
            s.d_x_in, d_comp64_in, d_exact64, static_cast<int>(field));
        CHECK_CUDA(cudaGetLastError());
        gpu_fp64_conv_step(cublas_handle, d_exact64, d_w_fp64, d_col64_anchor, d_out64, hw);
        reseed_double_from_fp64_kernel<T><<<grid1d(static_cast<int>(field)),
                                            kConversionThreads>>>(
            d_out64, s.d_x_out, d_comp64_out, static_cast<int>(field));
        CHECK_CUDA(cudaGetLastError());
        narrow_double_to_float_kernel<<<grid1d(static_cast<int>(field)), kConversionThreads>>>(
            d_comp64_out, s.d_comp_out, static_cast<int>(field));
        CHECK_CUDA(cudaGetLastError());
      } else {
        chained_conv_step(s, d_w_tc, hw, comp_on);
        if (anclar) {
          // Mantiene el residuo double sincronizado por si la SIGUIENTE
          // iteracion es de ancla.
          widen_comp_to_double_kernel<<<grid1d(static_cast<int>(field)), kConversionThreads>>>(
              s.d_comp_out, d_comp64_out, static_cast<int>(field));
          CHECK_CUDA(cudaGetLastError());
        }
      }
      std::swap(s.d_x_in, s.d_x_out);
      if (comp_on) std::swap(s.d_comp_in, s.d_comp_out);
      if (anclar) std::swap(d_comp64_in, d_comp64_out);

      if (es_checkpoint(opt, iter)) {
        // Sincroniza ANTES de abrir la pausa -- ver la nota de cabecera de
        // esta seccion sobre por que, sin esto, el computo se descuenta del
        // tiempo medido.
        CHECK_CUDA(cudaDeviceSynchronize());
        const auto pause_t0 = std::chrono::steady_clock::now();
        power_buffer_stop_sampling(pb);

        CHECK_CUDA(cudaMemcpy(tmp_tc.data(), s.d_x_in, field * sizeof(T),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < field; ++i) host_route[i] = static_cast<float>(tmp_tc[i]);
        const ErrorMetrics err =
            compare_fp64_ref_vs_fp32(ref_snapshots[ckpt_idx], host_route);
        // anchor_every va al FINAL de la fila. Vale 0 en la ruta "_none"
        // SIEMPRE, incluso si esta corrida se lanzo con --anchor-every K>0: el
        // ancla solo se aplica a la ruta con compensacion. Es una columna POR
        // FILA, no de la corrida -- ver Fase_4/tools/README.md.
        std::cout << "CSV_DRIFT," << format_label << sufijo << "," << hw << "," << iter << ","
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
    emit_csv_summary(route.c_str(), hw, opt.iters, total_s, gpu_segments, pb, anchor_col);
    power_buffer_destroy(pb);
  };

  medir_ruta(s_off, /*comp_on=*/false, "_none", /*anchor_col=*/0);
  if (opt.comp) medir_ruta(s_on, /*comp_on=*/true, "_comp", opt.anchor_every);

  chained_conv_free(s_off);
  if (opt.comp) chained_conv_free(s_on);
  if (anchor_enabled) {
    cudaFree(d_comp64_in);
    cudaFree(d_comp64_out);
    cudaFree(d_exact64);
    cudaFree(d_out64);
    cudaFree(d_col64_anchor);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);

  std::cout << "HW=" << opt.hw << " C=K=" << kChannels << " iters=" << opt.iters
            << " comp=" << (opt.comp ? "on" : "off")
            << " checkpoint_every=" << opt.checkpoint_every
            << " anchor_every=" << opt.anchor_every
            << (opt.anchor_every > 0 ? " (activa)" : " (deshabilitada)") << "\n";

  std::vector<double> W_fp64;
  build_block_diagonal_filter(W_fp64);

  const bool need_fp16 = (opt.tc_format == TcFormat::FP16 || opt.tc_format == TcFormat::Both);
  const bool need_bf16 = (opt.tc_format == TcFormat::BF16 || opt.tc_format == TcFormat::Both);

  const size_t field = static_cast<size_t>(kChannels) * opt.hw * opt.hw;
  std::vector<float> x0(field);
  {
    std::mt19937 gen(opt.seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x0) v = dist(gen);
  }

  double* d_w_fp64 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_w_fp64, W_fp64.size() * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_w_fp64, W_fp64.data(), W_fp64.size() * sizeof(double),
                        cudaMemcpyHostToDevice));

  // x0 en double, INMUTABLE durante toda la corrida: lo usan la referencia
  // FP64 (como estado inicial) y la siembra del residuo de compensacion de
  // cada ruta. Vive aqui, y no dentro de una fase, precisamente para que
  // ninguna fase pueda avanzarlo y dejar a la siguiente sembrando desde un
  // estado que ya no es x0.
  double* d_x0_64 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x0_64, field * sizeof(double)));
  {
    std::vector<double> x0_d(field);
    for (size_t i = 0; i < field; ++i) x0_d[i] = static_cast<double>(x0[i]);
    CHECK_CUDA(cudaMemcpy(d_x0_64, x0_d.data(), field * sizeof(double), cudaMemcpyHostToDevice));
  }

  cublasHandle_t cublas_handle;
  CHECK_CUBLAS(cublasCreate(&cublas_handle));

  // FASE 1, una sola vez para toda la invocacion (la trayectoria FP64 de
  // referencia no depende del formato T). Sus buffers -- incluido el scratch
  // de im2col en double -- se liberan al volver, antes de que las rutas WMMA
  // reserven los suyos.
  const std::vector<std::vector<double>> ref_snapshots =
      run_fp64_reference(opt, cublas_handle, d_w_fp64, d_x0_64);

  if (need_fp16) {
    std::vector<__half> W_tc(W_fp64.size());
    for (size_t i = 0; i < W_fp64.size(); ++i) W_tc[i] = __half(static_cast<float>(W_fp64[i]));
    __half* d_w_tc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_w_tc, W_tc.size() * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_w_tc, W_tc.data(), W_tc.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    run_chained_route<__half>(opt, cublas_handle, d_w_tc, d_w_fp64, d_x0_64, x0,
                              ref_snapshots, "FP16");
    cudaFree(d_w_tc);
  }
  if (need_bf16) {
    std::vector<__nv_bfloat16> W_tc(W_fp64.size());
    for (size_t i = 0; i < W_fp64.size(); ++i) {
      W_tc[i] = __nv_bfloat16(static_cast<float>(W_fp64[i]));
    }
    __nv_bfloat16* d_w_tc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_w_tc, W_tc.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(d_w_tc, W_tc.data(), W_tc.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    run_chained_route<__nv_bfloat16>(opt, cublas_handle, d_w_tc, d_w_fp64, d_x0_64, x0,
                                     ref_snapshots, "BF16");
    cudaFree(d_w_tc);
  }

  cublasDestroy(cublas_handle);
  cudaFree(d_w_fp64);
  cudaFree(d_x0_64);
  return 0;
}
