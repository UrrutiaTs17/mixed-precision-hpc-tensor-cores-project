// Fase_3/GEMM/gemm_chained.cu
//
// Encadenamiento genuino de GEMM: X(n+1) = X(n) * A, con A un operador FIJO
// (no cambia entre iteraciones), aplicado repetidamente -- la misma
// estructura que la iteracion de potencias para autovalores, y la que ya usa
// por dentro la reformulacion "Y = X*H + V*X" del Stencil de Fase 4 (ahi
// vive escondida dentro del kernel de Stencil; aqui se mide como GEMM por
// derecho propio). Especificado en el documento "Plan de Precision Mixta",
// seccion 02 -- esa es la referencia normativa; si un comentario de aqui no
// coincide con lo que el plan describe, el plan tiene razon.
//
// POR QUE ENCADENAR (corregido tras revision, ver correccion #13 del plan):
// GEMM y Convolucion se median hasta Fase 2 como una llamada suelta -- no
// representa como se usan en HPC clasico (solucionadores iterativos,
// iteracion de potencias, capas apiladas), donde el error de una operacion
// se propaga a la siguiente. Este archivo le da a GEMM el mismo tratamiento
// que Fase 3 ya le dio a Stencil.
//
// Compilar con:
// nvcc -std=c++17 gemm_chained.cu -o gemm_chained \
//      -lcublas -gencode arch=compute_80,code=sm_80 \
//      --allow-unsupported-compiler
// (agregar -DUSE_NVML_TELEMETRY -lnvidia-ml para telemetria de energia GPU)
//
// Ejecutar:
// ./gemm_chained --n 1024 --iters 20 --tc fp16 --comp on
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
    // esta funcion host (nvcc no lo rechaza en compilacion, pero el stub de
    // host que genera aborta en tiempo de ejecucion). T(float) -- el
    // constructor de __half/__nv_bfloat16, valido en host -- es la
    // conversion correcta aqui; mismo patron que la siembra de x0 mas abajo.
    A_tc[i] = T(static_cast<float>(val));
  }
}

// =========================================================================
// Kernels elementales del encadenamiento (comp en FLOAT). Fase 3 no incluye
// el ancla FP64 -- eso es una extension de Fase 4, ver
// Fase_4/GEMM/gemm_chained.cu y su README.md. NOTA: esa extension NO usa
// common/chained_precision.cuh (motor generico escrito temprano en el
// proyecto, nunca conectado a ningun archivo real) -- termino escribiendo
// kernels locales concretos (T fijo por instanciacion, sin funtor generico
// como parametro de template) para evitar la complejidad de pasar lambdas
// __device__ como argumento de template entre traducciones separadas. Ver
// el comentario de cabecera de common/README.md, seccion chained_precision.cuh.
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
  // La escritura a comp_out va guardada: s_off (la ruta sin compensacion,
  // que corre en todas las iteraciones sin importar --comp) se asigna con
  // chained_gemm_alloc(..., comp_on=false), que nunca reserva d_comp_out --
  // queda nullptr. Escribir sin esta guardia es un acceso ilegal a memoria.
  if (comp_on) comp_out[i] = exact - tc_scalar_to_float<T>(q);
}

// Siembra INICIAL del residuo de compensacion a partir del x0 real (en
// double, ya en el device como d_x64_in) y su cuantizacion a T -- NO desde
// cero. Dejar comp en 0 tras la siembra de X0 descartaria el redondeo
// x0->T de la primerisima conversion; ese error se propagaria amplificado
// por A en cada iteracion, igual que cualquier otro termino del estado, en
// vez de quedar capturado en comp desde el principio, como exige la
// propiedad de reconstruccion Q(v)+comp=v que el resto del mecanismo asume
// ya valida desde t=0. El gate K=1 de Fase_4/GEMM/gemm_chained.cu depende
// de esta siembra para converger a la referencia FP64.
template <typename T>
__global__ static void seed_comp_from_double_kernel(const double* __restrict__ src64,
                                                      const T* __restrict__ src_tc,
                                                      float* __restrict__ comp, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) comp[i] = static_cast<float>(src64[i]) - tc_scalar_to_float<T>(src_tc[i]);
}

inline int grid1d(int n, int block = 256) { return (n + block - 1) / block; }

// =========================================================================
// Ruta de referencia: GPU_FP64 encadenada (cuBLAS).
// =========================================================================

// Un paso X_out = X_in * A_fp64, en FP64, via cuBLAS.
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

// Un paso encadenado: X_out = X_in * A (+ correccion si comp_on). No hace
// ancla FP64 -- eso es Fase 4 (ver Fase_4/GEMM/gemm_chained.cu, que
// reutiliza esta misma funcion sin cambios, decidiendo por fuera -- en
// run_chained_route() -- si una iteracion es de ancla o normal).
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
            << "         [--checkpoint-every K] [--csv RUTA] [--seed S]"
            << " [--target-lambda L]\n\n"
            << "  --n N (potencia de 2, default 1024): tamano de la matriz cuadrada.\n"
            << "  --comp off|on (default off): compensacion por linealidad del\n"
            << "  redondeo de almacenamiento -- ver la seccion 02 del documento de plan.\n"
            << "  --checkpoint-every K (default 0 = solo la ultima iteracion): cada\n"
            << "  cuantas iteraciones se compara contra la referencia FP64.\n"
            << "\nEjemplos:\n"
            << "  " << prog << " --n 1024 --iters 20 --tc fp16 --comp on\n"
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
  return opt;
}

// =========================================================================
// MEDICION POR RUTA -- leer antes de tocar la estructura de esta seccion.
//
// Las tres trayectorias (referencia FP64, WMMA sin compensacion, WMMA con
// compensacion) se miden en FASES SEPARADAS, cada una con su propio
// cronometro y su propia ventana de PowerBuffer.
//
// Antes corrian entrelazadas dentro de un mismo bucle, envueltas por UN solo
// cronometro y por dos PowerBuffer abiertos y cerrados en los MISMOS
// instantes. Resultado: t_iter_ms, t_total_ms, gflops y energy_gpu_j salian
// IDENTICOS en las dos filas CSV_SUMMARY, y ademas incluian el costo de la
// referencia FP64 -- que en A100 domina. Dos de los tres ejes del Frente de
// Pareto 3D quedaban inutilizables para este kernel (el de ERROR siempre
// estuvo bien: rel_l2/rel_linf nunca dependieron del cronometro). Stencil
// nunca tuvo el problema, ya media por ruta; esto alinea GEMM con esa forma.
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
// LO QUE CUESTA: la referencia FP64 se corre UNA vez (no una por formato) y
// sus estados en los checkpoints se guardan en RAM del host, para que las
// rutas de baja precision se comparen contra ella sin recalcularla dentro de
// sus ventanas. Mismo patron que ya usa Stencil (ckpt.fp64_checkpoints). El
// costo de memoria se imprime al arrancar.
//
// LO QUE NO ARREGLA: las fases corren una detras de otra, asi que la ultima
// ve una GPU mas caliente que la primera. Es la misma limitacion de
// aislamiento termico que el plan ya documenta (Etapa 5) para todo el
// proyecto, no algo que introduzca esta estructura.
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
// ruta compensada, que hace un segundo producto para la correccion: esa ruta
// entrega el mismo resultado util por iteracion a mayor costo, asi que su
// gflops mas bajo es exactamente lo que hay que reportar, no un artefacto de
// contabilidad.
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
// iteracion de checkpoint, que es contra lo que se compararan despues las
// rutas WMMA sin tener que recalcular nada dentro de sus ventanas medidas.
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
static void run_chained_route(const Options& opt, const T* d_a_tc, const double* d_x0_64,
                               const std::vector<float>& x0,
                               const std::vector<std::vector<double>>& ref_snapshots,
                               const char* format_label) {
  const int n = opt.n;
  const size_t count = static_cast<size_t>(n) * n;

  ChainedGemmState<T> s_off, s_on;
  chained_gemm_alloc(s_off, n, /*comp_on=*/false);
  if (opt.comp) chained_gemm_alloc(s_on, n, /*comp_on=*/true);

  // float_to_tc_scalar<T> es __device__-only: la siembra en host necesita su
  // propio casteo. __half/__nv_bfloat16 tienen constructores desde float
  // utilizables en host -- si tu toolkit los marca __device__-only, reemplaza
  // por __float2half/__float2bfloat16 via un pequeno kernel de siembra, igual
  // que hace Fase_2/GEMM con convert_float_to_half_kernel.
  std::vector<T> x0_t(count);
  for (size_t i = 0; i < count; ++i) x0_t[i] = T(x0[i]);

  // Deja una ruta lista para arrancar desde x0. Se llama antes del warm-up y
  // otra vez despues, para que el bucle medido arranque del mismo estado que
  // arrancaria sin warm-up.
  auto reset_ruta = [&](ChainedGemmState<T>& s, bool comp_on) {
    CHECK_CUDA(cudaMemcpy(s.d_x_in, x0_t.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    if (comp_on) {
      // Siembra el residuo desde el redondeo REAL de x0->T (no desde 0) -- ver
      // el comentario de seed_comp_from_double_kernel. d_x0_64 guarda x0 en
      // double y nunca se avanza, asi que da igual en que fase estemos.
      seed_comp_from_double_kernel<T><<<grid1d(n * n), kConversionThreads>>>(
          d_x0_64, s.d_x_in, s.d_comp_in, n * n);
      CHECK_CUDA(cudaGetLastError());
    }
  };

  std::vector<float> host_route(count);
  std::vector<T> tmp_tc(count);

  auto medir_ruta = [&](ChainedGemmState<T>& s, bool comp_on, const char* sufijo) {
    reset_ruta(s, comp_on);
    for (int w = 0; w < kWarmupIters; ++w) {
      chained_gemm_step(s, d_a_tc, n, comp_on);
      std::swap(s.d_x_in, s.d_x_out);
      if (comp_on) std::swap(s.d_comp_in, s.d_comp_out);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    reset_ruta(s, comp_on);

    PowerBuffer* pb = power_buffer_create(0);
    power_buffer_start_sampling(pb);
    const auto t0 = std::chrono::steady_clock::now();
    double pause_s = 0.0;
    int gpu_segments = 1;
    size_t ckpt_idx = 0;

    for (int iter = 1; iter <= opt.iters; ++iter) {
      chained_gemm_step(s, d_a_tc, n, comp_on);
      std::swap(s.d_x_in, s.d_x_out);
      if (comp_on) std::swap(s.d_comp_in, s.d_comp_out);

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
        // anchor_every va al FINAL de la fila y vale SIEMPRE 0 en Fase 3: este
        // binario no tiene ancla (es una extension de Fase 4). La columna
        // existe igual para que el esquema de CSV_DRIFT/CSV_SUMMARY sea
        // IDENTICO entre Fase 3 y Fase 4 -- run_full_pipeline.sh concatena los
        // results/ de ambas fases en el mismo analisis (ver
        // Fase_4/tools/common_analysis.py), y dos esquemas distintos obligarian
        // a ese modulo a ramificar por fase.
        std::cout << "CSV_DRIFT," << format_label << sufijo << "," << n << "," << iter << ","
                  << err.rel_l2 << "," << err.rel_linf << "," << (err.solution_finite ? 1 : 0)
                  << ",0\n";
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
    emit_csv_summary(route.c_str(), n, opt.iters, total_s, gpu_segments, pb,
                     /*anchor_every_col=*/0);
    power_buffer_destroy(pb);
  };

  medir_ruta(s_off, /*comp_on=*/false, "_none");
  if (opt.comp) medir_ruta(s_on, /*comp_on=*/true, "_comp");

  chained_gemm_free(s_off);
  if (opt.comp) chained_gemm_free(s_on);
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_args(argc, argv);

  std::cout << "N=" << opt.n << " iters=" << opt.iters << " comp=" << (opt.comp ? "on" : "off")
            << " checkpoint_every=" << opt.checkpoint_every << "\n";

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

  // FASE 1, una sola vez para toda la invocacion (la trayectoria FP64 no
  // depende del formato T). Sus buffers se liberan al volver, antes de que
  // las rutas WMMA reserven los suyos.
  const std::vector<std::vector<double>> ref_snapshots =
      run_fp64_reference(opt, cublas_handle, d_a_fp64, d_x0_64);

  if (need_fp16) {
    CHECK_CUDA(cudaMalloc(&d_a_fp16, count * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_a_fp16, A_fp16.data(), count * sizeof(__half), cudaMemcpyHostToDevice));
    run_chained_route<__half>(opt, d_a_fp16, d_x0_64, x0, ref_snapshots, "FP16");
  }
  if (need_bf16) {
    CHECK_CUDA(cudaMalloc(&d_a_bf16, count * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(d_a_bf16, A_bf16.data(), count * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    run_chained_route<__nv_bfloat16>(opt, d_a_bf16, d_x0_64, x0, ref_snapshots, "BF16");
  }

  cublasDestroy(cublas_handle);
  cudaFree(d_a_fp16);
  cudaFree(d_a_bf16);
  cudaFree(d_a_fp64);
  cudaFree(d_x0_64);
  return 0;
}
