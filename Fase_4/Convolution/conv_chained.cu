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
// la referencia FP64 -- reutilizando gpu_fp64_conv_step() TAL CUAL, la misma
// funcion que ya usa la ruta de referencia de este archivo, incluso
// reciclando su mismo buffer scratch de im2col en double (d_col64: la
// llamada de referencia y la del ancla ocurren una despues de la otra dentro
// de la MISMA iteracion, en el mismo stream por defecto, asi que reusar el
// scratch es seguro -- no hace falta un segundo buffer de tamano col_sz) --
// y (c) re-siembra T (cuantizado) y el residuo de compensacion, ahora en
// DOUBLE. Mismo patron exacto que Fase_4/GEMM/gemm_chained.cu (que a su vez
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
// referencia (siempre FP64, comparacion en cada checkpoint) COMO para el
// paso puntual que el ancla inyecta en la ruta de baja precision -- misma
// funcion, sin cuBLAS nuevo. d_col_scratch puede ser el mismo buffer
// (d_col64) en ambos usos dentro de la misma iteracion: las dos llamadas son
// secuenciales en el stream por defecto, asi que no hay una carrera de
// datos por reusar el scratch.
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
// Bucle principal de una ruta (un formato T).
// =========================================================================

template <typename T>
static void run_chained_route(const Options& opt, const T* d_w_tc, const double* d_w_fp64,
                               const std::vector<float>& x0, const char* format_label) {
  const int hw = opt.hw;
  const size_t field = static_cast<size_t>(kChannels) * hw * hw;
  const size_t col_sz = static_cast<size_t>(kCRS) * hw * hw;

  double* d_x64_in = nullptr;
  double* d_x64_out = nullptr;
  double* d_col64 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x64_in, field * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_x64_out, field * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_col64, col_sz * sizeof(double)));
  {
    std::vector<double> x0_d(field);
    for (size_t i = 0; i < field; ++i) x0_d[i] = static_cast<double>(x0[i]);
    CHECK_CUDA(cudaMemcpy(d_x64_in, x0_d.data(), field * sizeof(double), cudaMemcpyHostToDevice));
  }
  cublasHandle_t cublas_handle;
  CHECK_CUBLAS(cublasCreate(&cublas_handle));

  ChainedConvState<T> s_off, s_on;
  chained_conv_alloc(s_off, hw, false);
  if (opt.comp) chained_conv_alloc(s_on, hw, true);

  // ANCLA FP64 (Fase 4): residuo de compensacion en DOUBLE (sombra del
  // residuo en float de s_on) mas dos buffers scratch de tamano "field"
  // (estado exacto reconstruido, salida del paso FP64 de referencia). El
  // scratch de im2col en double se reutiliza de d_col64 (declarado arriba)
  // -- ver el comentario de gpu_fp64_conv_step(). Solo existen si
  // --anchor-every > 0 (que ya exige --comp on en parse_args).
  double* d_comp64_in = nullptr;
  double* d_comp64_out = nullptr;
  double* d_exact64 = nullptr;
  double* d_out64 = nullptr;
  const bool anchor_enabled = opt.anchor_every > 0;
  if (anchor_enabled) {
    CHECK_CUDA(cudaMalloc(&d_comp64_in, field * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_comp64_out, field * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_exact64, field * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out64, field * sizeof(double)));
  }

  // seed_state() se llama dos veces (siembra inicial y reset post-warm-up) y
  // en AMBAS d_x64_in todavia guarda x0 en double intacto (el warm-up de
  // arriba solo avanza s_off/s_on, nunca d_x64_in) -- por eso la siembra del
  // residuo puede vivir aqui adentro una sola vez.
  auto seed_state = [&]() {
    std::vector<T> x0_t(field);
    for (size_t i = 0; i < field; ++i) x0_t[i] = T(x0[i]);  // ver nota en gemm_chained.cu
    CHECK_CUDA(cudaMemcpy(s_off.d_x_in, x0_t.data(), field * sizeof(T), cudaMemcpyHostToDevice));
    if (opt.comp) {
      CHECK_CUDA(cudaMemcpy(s_on.d_x_in, x0_t.data(), field * sizeof(T), cudaMemcpyHostToDevice));
      // Siembra el residuo desde el redondeo REAL de x0->T (no desde 0) --
      // ver el comentario de seed_comp_from_double_kernel.
      seed_comp_from_double_kernel<T><<<grid1d(static_cast<int>(field)), kConversionThreads>>>(
          d_x64_in, s_on.d_x_in, s_on.d_comp_in, static_cast<int>(field));
      CHECK_CUDA(cudaGetLastError());
      if (anchor_enabled) {
        seed_comp64_from_double_kernel<T><<<grid1d(static_cast<int>(field)), kConversionThreads>>>(
            d_x64_in, s_on.d_x_in, d_comp64_in, static_cast<int>(field));
        CHECK_CUDA(cudaGetLastError());
      }
    }
  };
  seed_state();

  PowerBuffer* power_buffer_off = power_buffer_create(0);
  PowerBuffer* power_buffer_on = opt.comp ? power_buffer_create(0) : nullptr;

  // Warm-up (descartable) -- NO ejecuta el camino de ancla (usa
  // chained_conv_step normal); seed_state() de abajo re-sincroniza tambien
  // d_comp64_in, asi que el bucle medido arranca identico con o sin ancla.
  for (int w = 0; w < kWarmupIters; ++w) {
    chained_conv_step(s_off, d_w_tc, hw, false);
    std::swap(s_off.d_x_in, s_off.d_x_out);
    if (opt.comp) {
      chained_conv_step(s_on, d_w_tc, hw, true);
      std::swap(s_on.d_x_in, s_on.d_x_out);
      std::swap(s_on.d_comp_in, s_on.d_comp_out);
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  seed_state();  // reinicia tras el warm-up, igual que GEMM/Stencil

  power_buffer_start_sampling(power_buffer_off);
  if (opt.comp) power_buffer_start_sampling(power_buffer_on);
  const auto t0 = std::chrono::steady_clock::now();
  double checkpoint_pause_s = 0.0;
  int gpu_segments = 1;

  std::vector<double> ref_host(field);
  std::vector<float> off_host(field), on_host(field);

  for (int iter = 1; iter <= opt.iters; ++iter) {
    gpu_fp64_conv_step(cublas_handle, d_x64_in, d_w_fp64, d_col64, d_x64_out, hw);
    std::swap(d_x64_in, d_x64_out);

    chained_conv_step(s_off, d_w_tc, hw, false);
    std::swap(s_off.d_x_in, s_off.d_x_out);

    if (opt.comp) {
      const bool is_anchor_iter = anchor_enabled && (iter % opt.anchor_every == 0);
      if (is_anchor_iter) {
        // ANCLA FP64 (Fase 4): reconstruye el estado exacto T+comp en
        // double, avanza UN paso con la referencia FP64
        // (gpu_fp64_conv_step, la MISMA funcion de arriba, reusando
        // d_col64 como scratch) y re-siembra T + el residuo double.
        reconstruct_exact_double_kernel<T><<<grid1d(static_cast<int>(field)), kConversionThreads>>>(
            s_on.d_x_in, d_comp64_in, d_exact64, static_cast<int>(field));
        CHECK_CUDA(cudaGetLastError());
        gpu_fp64_conv_step(cublas_handle, d_exact64, d_w_fp64, d_col64, d_out64, hw);
        reseed_double_from_fp64_kernel<T><<<grid1d(static_cast<int>(field)), kConversionThreads>>>(
            d_out64, s_on.d_x_out, d_comp64_out, static_cast<int>(field));
        CHECK_CUDA(cudaGetLastError());
        narrow_double_to_float_kernel<<<grid1d(static_cast<int>(field)), kConversionThreads>>>(
            d_comp64_out, s_on.d_comp_out, static_cast<int>(field));
        CHECK_CUDA(cudaGetLastError());
      } else {
        chained_conv_step(s_on, d_w_tc, hw, true);
        if (anchor_enabled) {
          widen_comp_to_double_kernel<<<grid1d(static_cast<int>(field)), kConversionThreads>>>(
              s_on.d_comp_out, d_comp64_out, static_cast<int>(field));
          CHECK_CUDA(cudaGetLastError());
        }
      }
      std::swap(s_on.d_x_in, s_on.d_x_out);
      std::swap(s_on.d_comp_in, s_on.d_comp_out);
      if (anchor_enabled) std::swap(d_comp64_in, d_comp64_out);
    }

    const bool is_checkpoint =
        (opt.checkpoint_every > 0 && iter % opt.checkpoint_every == 0) || iter == opt.iters;
    if (is_checkpoint) {
      const auto pause_t0 = std::chrono::steady_clock::now();
      power_buffer_stop_sampling(power_buffer_off);
      if (opt.comp) power_buffer_stop_sampling(power_buffer_on);

      CHECK_CUDA(cudaMemcpy(ref_host.data(), d_x64_in, field * sizeof(double),
                            cudaMemcpyDeviceToHost));
      {
        std::vector<T> tmp(field);
        CHECK_CUDA(cudaMemcpy(tmp.data(), s_off.d_x_in, field * sizeof(T),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < field; ++i) off_host[i] = static_cast<float>(tmp[i]);
        const ErrorMetrics err = compare_fp64_ref_vs_fp32(ref_host, off_host);
        std::cout << "CSV_DRIFT," << format_label << "_none," << hw << "," << iter << ","
                  << err.rel_l2 << "," << err.rel_linf << "," << (err.solution_finite ? 1 : 0)
                  << "\n";
      }
      if (opt.comp) {
        std::vector<T> tmp(field);
        CHECK_CUDA(cudaMemcpy(tmp.data(), s_on.d_x_in, field * sizeof(T),
                              cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < field; ++i) on_host[i] = static_cast<float>(tmp[i]);
        const ErrorMetrics err = compare_fp64_ref_vs_fp32(ref_host, on_host);
        std::cout << "CSV_DRIFT," << format_label << "_comp," << hw << "," << iter << ","
                  << err.rel_l2 << "," << err.rel_linf << "," << (err.solution_finite ? 1 : 0)
                  << "\n";
      }

      power_buffer_start_sampling(power_buffer_off);
      if (opt.comp) power_buffer_start_sampling(power_buffer_on);
      checkpoint_pause_s +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - pause_t0).count();
      ++gpu_segments;
    }
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  const auto t1 = std::chrono::steady_clock::now();
  const double total_s = std::chrono::duration<double>(t1 - t0).count() - checkpoint_pause_s;
  power_buffer_stop_sampling(power_buffer_off);
  if (opt.comp) power_buffer_stop_sampling(power_buffer_on);

  // FLOPs por iteracion: K*Ncol*CRS*2 (el GEMM subyacente del im2col).
  const double flops_per_iter =
      2.0 * kChannels * static_cast<double>(hw) * hw * kCRS;
  const double gflops = (flops_per_iter * opt.iters / 1e9) / total_s;

  const bool window_reliable = total_s >= kEnergyWindowReliableSeconds * gpu_segments;
  const double energy_off_j = power_buffer_energy_joules(power_buffer_off);
  std::cout << "CSV_SUMMARY," << format_label << "_none," << hw << "," << opt.iters << ","
            << (total_s * 1000.0 / opt.iters) << "," << (total_s * 1000.0) << "," << gflops << ","
            << energy_field(power_buffer_capture_valid(power_buffer_off), energy_off_j) << ","
            << (window_reliable ? 1 : 0) << "," << gpu_segments << "\n";
  if (opt.comp) {
    const double energy_on_j = power_buffer_energy_joules(power_buffer_on);
    std::cout << "CSV_SUMMARY," << format_label << "_comp," << hw << "," << opt.iters << ","
              << (total_s * 1000.0 / opt.iters) << "," << (total_s * 1000.0) << "," << gflops
              << "," << energy_field(power_buffer_capture_valid(power_buffer_on), energy_on_j)
              << "," << (window_reliable ? 1 : 0) << "," << gpu_segments << "\n";
  }

  power_buffer_destroy(power_buffer_off);
  if (power_buffer_on) power_buffer_destroy(power_buffer_on);
  chained_conv_free(s_off);
  if (opt.comp) chained_conv_free(s_on);
  if (anchor_enabled) {
    cudaFree(d_comp64_in);
    cudaFree(d_comp64_out);
    cudaFree(d_exact64);
    cudaFree(d_out64);
  }
  cudaFree(d_x64_in);
  cudaFree(d_x64_out);
  cudaFree(d_col64);
  cublasDestroy(cublas_handle);
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

  std::vector<float> x0(static_cast<size_t>(kChannels) * opt.hw * opt.hw);
  {
    std::mt19937 gen(opt.seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x0) v = dist(gen);
  }

  double* d_w_fp64 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_w_fp64, W_fp64.size() * sizeof(double)));
  CHECK_CUDA(cudaMemcpy(d_w_fp64, W_fp64.data(), W_fp64.size() * sizeof(double),
                        cudaMemcpyHostToDevice));

  if (need_fp16) {
    std::vector<__half> W_tc(W_fp64.size());
    for (size_t i = 0; i < W_fp64.size(); ++i) W_tc[i] = __half(static_cast<float>(W_fp64[i]));
    __half* d_w_tc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_w_tc, W_tc.size() * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_w_tc, W_tc.data(), W_tc.size() * sizeof(__half),
                          cudaMemcpyHostToDevice));
    run_chained_route<__half>(opt, d_w_tc, d_w_fp64, x0, "FP16");
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
    run_chained_route<__nv_bfloat16>(opt, d_w_tc, d_w_fp64, x0, "BF16");
    cudaFree(d_w_tc);
  }

  cudaFree(d_w_fp64);
  return 0;
}
