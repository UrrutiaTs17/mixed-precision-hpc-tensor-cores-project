// common/wmma_gemm.cuh
//
// Kernel GEMM con Tensor Cores (API WMMA + pipeline cp.async de 3 etapas,
// Ampere sm_80+), compartido por GEMM en Fase 2 (una sola llamada) y Fase 3/4
// (encadenado: la salida de una llamada es la entrada de la siguiente, y el
// mismo kernel calcula tambien el termino de corrección por linealidad de la
// compensación — ver Fase_3/GEMM/README.md).
//
// Origen: extraído de Fase_2/GEMM/gemm_tensor_activation.cu (código ya
// migrado y verificado) sin cambios de lógica — mismos nombres, mismas
// constantes, mismo algoritmo. Fase_2/GEMM ahora incluye este header en vez
// de definir el kernel localmente.
//
// USO: incluir DENTRO del bloque `namespace { ... }` anónimo de cada .cu,
// igual que cuda_checks.cuh y metrics.cuh (ver la nota de uso en esos
// headers). Requiere <mma.h> y <cuda_pipeline_primitives.h> incluidos por el
// .cu ANTES de este header (no se incluyen aquí para no forzar un orden de
// includes distinto al que cada .cu ya usa).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Dimensiones del fragmento WMMA: únicas soportadas en sm_70+.
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// Elementos K cargados a shared memory por iteración del bucle externo.
// Debe ser múltiplo de kWmmaK. A mayor kKStep, menos sincronizaciones y mayor
// reuso de datos en shared memory, a costa de más shared memory usada.
constexpr int kKStep = 32;

// Warps por dimensión dentro de un bloque (4×4 = 16 warps = 512 hilos).
constexpr int kBlockWarpsM = 4;
constexpr int kBlockWarpsN = 4;

// Bloques residentes por SM que se le piden al compilador (segundo argumento
// de __launch_bounds__). Ajustado para sm_80 (A100): 3 bloques x 512 hilos =
// 1536 hilos = 48/64 warps = 75% de ocupancia, con presupuesto de
// 65536/(3*512) = 42 registros por hilo. Sobrescribible al compilar para
// barrer el parámetro en PACCA: nvcc -DWMMA_MIN_BLOCKS_PER_SM=2 ...
#ifndef WMMA_MIN_BLOCKS_PER_SM
#define WMMA_MIN_BLOCKS_PER_SM 3
#endif

// Tile de salida que maneja un bloque completo: 64×64 elementos FP32.
constexpr int kBlockTileM = kBlockWarpsM * kWmmaM;  // 64
constexpr int kBlockTileN = kBlockWarpsN * kWmmaN;  // 64

// Etapas del pipeline de triple buffer para cp.async. Mientras el warp
// computa el tile[i], la DMA ya está cargando tile[i+2], eliminando la
// espera síncrona de global memory en cada iteración K.
constexpr int kNumStages = 3;

// Padding en shared memory para evitar bank conflicts entre warps. Con FP16
// (2 bytes) y 32 bancos de 4 bytes, añadir 8 elementos desplaza cada fila 16
// bytes extra, eliminando el patrón de conflicto cíclico.
constexpr int kWmmaShmemPad = 8;

// Filas de shared memory, en elementos de 2 bytes (FP16 o BF16).
constexpr int kSmemStrideA = kKStep      + kWmmaShmemPad;  // 40 elem = 80 B
constexpr int kSmemStrideB = kBlockTileN + kWmmaShmemPad;  // 72 elem = 144 B

// Ancho de cada cp.async: Ampere admite 4, 8 o 16 bytes por LDGSTS. Con 16
// bytes (8 elementos de 2 bytes) el bloque copia un tile K completo en una
// sola pasada de sus 512 hilos.
constexpr int kVecElems = 16 / 2;                                 // 8
constexpr int kVecsA    = kBlockTileM * kKStep      / kVecElems;  // 256
constexpr int kVecsB    = kKStep      * kBlockTileN / kVecElems;  // 256

// cp.async exige que origen y destino estén alineados al tamaño copiado
// (16 B). Destino: la base de sA/sB lleva __align__(16) y cada fila/etapa
// debe medir un múltiplo de 16 B para que el alineamiento se propague.
static_assert(kKStep      % kVecElems == 0, "kKStep debe ser multiplo de kVecElems");
static_assert(kBlockTileN % kVecElems == 0, "kBlockTileN debe ser multiplo de kVecElems");
static_assert(kSmemStrideA * 2 % 16 == 0, "fila de sA no alineada a 16 B");
static_assert(kSmemStrideB * 2 % 16 == 0, "fila de sB no alineada a 16 B");
static_assert(kBlockTileM * kSmemStrideA * 2 % 16 == 0, "etapa de sA no alineada a 16 B");
static_assert(kKStep      * kSmemStrideB * 2 % 16 == 0, "etapa de sB no alineada a 16 B");
// Origen: los desplazamientos en global son (fila)*ld + k_off + col. Con col
// múltiplo de kVecElems y k_off múltiplo de kKStep, basta que los leading
// dimensions (K y N) sean múltiplos de kVecElems; se garantiza en tiempo de
// ejecución en el llamador exigiendo K % kKStep == 0 y N % kBlockTileN == 0.

// Conversión escalar float -> T. Cada tipo Tensor Core soportado provee su
// propia especialización mediante el intrínseco de CUDA correspondiente.
template <typename T>
__device__ inline T float_to_tc_scalar(float x);

template <>
__device__ inline __half float_to_tc_scalar<__half>(float x) {
  return __float2half(x);
}

template <>
__device__ inline __nv_bfloat16 float_to_tc_scalar<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

// Conversión escalar T -> float. NO existía en Fase_2/GEMM (nunca se
// necesitaba: el flujo de una sola llamada siempre convierte float->T de
// entrada, y el acumulador WMMA ya sale en float). Fase 3/4 sí la necesitan
// para reconstruir el valor exacto (Q(v) + comp) al encadenar iteraciones.
template <typename T>
__device__ inline float tc_scalar_to_float(T x);

template <>
__device__ inline float tc_scalar_to_float<__half>(__half x) {
  return __half2float(x);
}

template <>
__device__ inline float tc_scalar_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

// Convierte una matriz FP32 col-major (rows x cols) a T row-major. Thread i
// escribe dst[i] = src[r + c*rows] donde r=i/cols, c=i%cols. Hilos
// consecutivos leen src con paso 1 (misma columna, filas contiguas), lo que
// produce accesos coalescentes en la lectura global.
template <typename T>
__global__ static void float_colmaj_to_tc_rowmaj_kernel(const float* __restrict__ src,
                                                          T* __restrict__ dst, int rows,
                                                          int cols) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx < total) {
    const int r = idx / cols;
    const int c = idx % cols;
    dst[idx] = float_to_tc_scalar<T>(src[r + c * rows]);
  }
}

// Emite las cp.async de un tile K completo (sA + sB) hacia una etapa del
// triple buffer. Los kVecsA + kVecsB vectores de 16 bytes se reparten entre
// TODOS los hilos del bloque en un único bucle: con la configuración por
// defecto son 256 + 256 = 512 vectores para 512 hilos, es decir un
// LDGSTS.128 por hilo. El bucle mantiene la forma grid-stride para seguir
// siendo correcto si se cambian kBlockTile*/kKStep.
//
// Los índices globales se calculan en size_t: con dimensiones grandes
// (p. ej. 32768) el producto fila*ld ronda 1.07e9 y queda al borde del rango
// de int.
template <typename T>
__device__ __forceinline__ void issue_stage_copy(const T* __restrict__ A,
                                                   const T* __restrict__ B,
                                                   T* __restrict__ sA_stage,
                                                   T* __restrict__ sB_stage, int block_row,
                                                   int block_col, int k_off, int N, int K) {
  for (int i = threadIdx.x; i < kVecsA + kVecsB; i += blockDim.x) {
    if (i < kVecsA) {
      const int elem = i * kVecElems;
      const int row = elem / kKStep;
      const int col = elem % kKStep;
      __pipeline_memcpy_async(&sA_stage[row * kSmemStrideA + col],
                               &A[static_cast<size_t>(block_row + row) * K + k_off + col],
                               sizeof(uint4));
    } else {
      const int elem = (i - kVecsA) * kVecElems;
      const int row = elem / kBlockTileN;
      const int col = elem % kBlockTileN;
      __pipeline_memcpy_async(&sB_stage[row * kSmemStrideB + col],
                               &B[static_cast<size_t>(k_off + row) * N + block_col + col],
                               sizeof(uint4));
    }
  }
}

// Kernel GEMM con API WMMA + pipeline cp.async de 3 etapas (Ampere sm_80+).
// C(M,N) = A(M,K) * B(K,N), todos row-major T->FP32 (T = __half o
// __nv_bfloat16).
//
// Organización de hilos:
//   Bloque: 512 hilos = 16 warps en cuadrícula 4x4 de fragmentos WMMA.
//   Cada warp calcula un fragmento de salida 16x16 en FP32.
//   Un bloque cubre un tile de salida 64x64.
//   Grid: (ceildiv(M,64), ceildiv(N,64)).
//
// Triple buffer con cp.async: el SM tiene 3 copias de sA/sB (etapas 0,1,2).
// Mientras el warp ejecuta instrucciones HMMA sobre la etapa[i], la DMA ya
// transfiere la etapa[i+2] desde global memory sin pasar por registros
// (cp.async). La barrera de cada iteración se reemplaza por
// consumer_wait_prior<kNumStages-1>(), que solo bloquea si el tile
// necesario aún no llegó, en vez de vaciar todo el pipeline.
//
// Requisito: M múltiplo de kBlockTileM(64), N de kBlockTileN(64), K de
// kKStep(32) -- el llamador debe validarlo antes de lanzar (ver
// benchmark_gpu_wmma en Fase_2/GEMM y el equivalente encadenado en Fase 3/4).
// Ocupancia esperada en sm_80 (A100): WMMA_MIN_BLOCKS_PER_SM bloques/SM x 16
// warps; con el valor por defecto 3 son 48 de los 64 warps del SM (75%).
template <typename T>
__launch_bounds__(kBlockWarpsM* kBlockWarpsN * 32,
                   WMMA_MIN_BLOCKS_PER_SM) __global__ static void wmma_gemm_kernel(
    const T* __restrict__ A, const T* __restrict__ B, float* __restrict__ C, int M, int N,
    int K) {
  using namespace nvcuda;

  // Triple buffer: sA[etapa][fila][col], sB[etapa][fila][col]. El padding
  // por fila evita bank conflicts cuando warps distintos acceden a columnas
  // separadas por kKStep o kBlockTileN elementos. __align__(16) es
  // obligatorio para el destino de las cp.async de 16 bytes: nvcc solo
  // garantizaría el alineamiento natural del tipo (2 bytes).
  __shared__ __align__(16) T sA[kNumStages][kBlockTileM][kSmemStrideA];
  __shared__ __align__(16) T sB[kNumStages][kKStep][kSmemStrideB];

  const int warp_id = threadIdx.x / 32;
  const int warp_row = warp_id / kBlockWarpsN;
  const int warp_col = warp_id % kBlockWarpsN;
  const int block_row = blockIdx.x * kBlockTileM;
  const int block_col = blockIdx.y * kBlockTileN;
  const int warp_row_base = block_row + warp_row * kWmmaM;
  const int warp_col_base = block_col + warp_col * kWmmaN;

  wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, T, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, T, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  // API primitiva de pipeline (cuda_pipeline_primitives.h):
  //   __pipeline_memcpy_async(dst, src, size) -> emite cp.async (min 4 bytes en Ampere)
  //   __pipeline_commit()                     -> cierra el grupo de copias actual
  //   __pipeline_wait_prior(N)                -> espera hasta que queden <= N grupos pendientes
  // No requiere acquire/release: el estado del pipeline es implícito por hilo.

  // Número de tiles K. K es múltiplo de kKStep (validado por el llamador).
  const int num_tiles = K / kKStep;

  // -- Precarga de las primeras kNumStages etapas antes del bucle principal --
  // Emite kNumStages grupos de cp.async sin esperar ninguno todavía.
  for (int s = 0; s < kNumStages && s < num_tiles; ++s) {
    issue_stage_copy<T>(A, B, &sA[s][0][0], &sB[s][0][0], block_row, block_col, s * kKStep, N,
                         K);
    __pipeline_commit();  // cierra el grupo s
  }

  // -- Bucle principal sobre tiles K --
  for (int tile = 0; tile < num_tiles; ++tile) {
    // prior decrece en los últimos kNumStages-1 tiles porque ya no se emiten
    // commits nuevos: sin el ajuste, wait_prior(2) dejaría el tile actual
    // pendiente. Fórmula: min(kNumStages-1, tiles restantes después del actual).
    const int prior = min(kNumStages - 1, num_tiles - tile - 1);
    __pipeline_wait_prior(prior);

    // Barrera de bloque: sincroniza los cp.async de todos los hilos antes de
    // que cualquier warp lea sA/sB con wmma::load_matrix_sync.
    __syncthreads();

    // -- Cómputo WMMA sobre la etapa actual --
    const int stage_c = tile % kNumStages;
    for (int k_inner = 0; k_inner < kKStep; k_inner += kWmmaK) {
      wmma::load_matrix_sync(
          a_frag, reinterpret_cast<const T*>(&sA[stage_c][warp_row * kWmmaM][k_inner]),
          kSmemStrideA);
      wmma::load_matrix_sync(
          b_frag, reinterpret_cast<const T*>(&sB[stage_c][k_inner][warp_col * kWmmaN]),
          kSmemStrideB);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Barrera entre cómputo y carga futura: garantiza que TODOS los warps
    // terminaron wmma::load_matrix_sync sobre stage_c antes de que cualquier
    // warp empiece a sobreescribirlo con cp.async. Sin esta barrera, un warp
    // adelantado podría escribir stage_c mientras otro warp aún lo está
    // leyendo -> race condition en shared memory.
    __syncthreads();

    // Emitir la carga futura DESPUÉS del cómputo: stage_c quedó libre ahora
    // y puede recibir el tile[tile+kNumStages] sin race condition. El
    // overlap con el cómputo de las próximas iteraciones se mantiene.
    const int future = tile + kNumStages;
    if (future < num_tiles) {
      issue_stage_copy<T>(A, B, &sA[stage_c][0][0], &sB[stage_c][0][0], block_row, block_col,
                           future * kKStep, N, K);
      __pipeline_commit();
    }
  }

  // -- Escritura del fragmento acumulado a C (row-major) --
  if (warp_row_base < M && warp_col_base < N) {
    wmma::store_matrix_sync(C + warp_row_base * N + warp_col_base, c_frag, N,
                             wmma::mem_row_major);
  }
}
