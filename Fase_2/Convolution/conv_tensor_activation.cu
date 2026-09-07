// Fase_2/Convolution/conv_tensor_activation.cu
//
// Compilar con:
// nvcc -std=c++17 conv_tensor_activation.cu -o conv_tc \
//      -I/usr/include/openblas \
//      -lcudnn -lopenblas \
//      -gencode arch=compute_80,code=sm_80 \
//      --allow-unsupported-compiler
//
// En PACCA (A100, sm_80) la compilacion y el perfilado con Nsight Compute se
// lanzan via SLURM: sbatch run_conv_tc.sbatch  (no ejecutar ncu con sudo).
//
// Este programa compara CUATRO rutas de convolucion 2D hacia adelante:
// 1. CPU con OpenBLAS (via transformacion im2col + SGEMM/DGEMM).
// 2. GPU con cuDNN FP32 clasico (sin Tensor Cores).
// 3. GPU con cuDNN y Tensor Cores (entradas FP16 y/o BF16, acumulacion y
//    salida FP32) -- formato seleccionable con --tc-format.
// 4. GPU con im2col propio en GPU (FP16) + kernel WMMA propio (Tensor Cores
//    manejados directamente, sin pasar por cuDNN).
//
// Con --double: unicamente se ejecutan las rutas 1 y 2 (CPU FP64 y cuDNN
// FP64), ya que las rutas 3 y 4 operan con entrada de 16 bits.
//
// Las matrices de activacion y filtros se almacenan en formato NCHW,
// convencion usada tanto por cuDNN como por el im2col de referencia en CPU.
//
// Ver Fase_2/Convolution/README.md, seccion "Como interpretar los
// resultados", antes de resumir cualquier corrida en un solo numero de
// "speedup de Tensor Cores": las rutas 2 vs 3 y 2 vs 4 miden cosas distintas
// y no son intercambiables.
//
// Migrado de old/Fase_2/Convolution/conv_tensor_activation.cu sin cambios de
// logica numerica. El unico cambio de este archivo respecto al original es
// de donde vienen CHECK_CUDA/CHECK_CUDNN/CudaEventTimer/Metrics/ErrorMetrics/
// compare_* -- ver el comentario junto al #include de mas abajo.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cblas.h>
#include <cudnn.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>

// CUTLASS es header-only (repositorio github.com/NVIDIA/cutlass, serie 2.x,
// ver REQUIREMENTS.md) -- no agrega dependencias de enlazado, solo requiere
// -I<CUTLASS_DIR>/include al compilar (ver run_conv_tc.sbatch, CUTLASS_DIR).
// Kernel de convolucion implicita (Ruta 5, ver mas abajo) via la API 2.x
// clasica (cutlass::conv::kernel::DefaultConv2dFprop +
// cutlass::conv::device::ImplicitGemmConvolution), siguiendo la estructura
// de examples/16_ampere_tensorop_conv2dfprop del repo oficial de CUTLASS.
// Verificado contra CUTLASS v2.11.0. Con otra version, confirmar estos
// nombres de header contra examples/16_ampere_tensorop_conv2dfprop/
// ampere_tensorop_conv2dfprop.cu del arbol real antes de compilar.
//
// Los headers de CUTLASS quedan detras de __has_include (igual que
// Fase_2/GEMM/gemm_tensor_activation.cu) para que la ausencia de
// -I$CUTLASS_DIR/include solo desactive la ruta 5 en vez de romper la
// compilacion de las cuatro rutas 1-4, que no dependen de CUTLASS -- ver el
// uso de HAVE_CUTLASS junto a run_cutlass_conv_impl y en el punto de llamada
// de run_benchmark, mas abajo.
#if __has_include(<cutlass/conv/device/implicit_gemm_convolution.h>)
#define HAVE_CUTLASS 1
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>
#include <cutlass/tensor_coord.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/conv/convolution.h>
#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
// Lista de headers necesaria para los simbolos que usa directamente la
// Ruta 5 mas abajo (Tensor4DCoord, MatrixCoord, TensorRef, Conv2dProblemSize,
// Mode, IteratorAlgorithm, LinearCombination, DefaultConv2dFprop,
// ImplicitGemmConvolution), verificada contra CUTLASS v2.11.0. Con otra
// version, un simbolo incompleto casi siempre es un header transitivo que
// aqui se asume incluido y en esa version no lo esta (o esta en otra ruta)
// -- revisar examples/16_ampere_tensorop_conv2dfprop/CMakeLists.txt o el
// propio .cu del ejemplo para la lista exacta de includes que usa.
#else
#define HAVE_CUTLASS 0
#endif

namespace {

// CHECK_CUDA / CHECK_CUDNN (common/cuda_checks.cuh) y CudaEventTimer /
// Metrics / ErrorMetrics / compare_fp64_ref_vs_fp32 / compare_float_vectors /
// compare_double_vectors (common/metrics.cuh) reemplazan lo que este archivo
// obtenia de old/Fase_2/common.cuh -- un header que GEMM, Convolucion y
// Stencil de la Fase 2 anterior incluian cada uno por su lado. common/
// consolida esas mismas definiciones (mismos structs, mismos campos, misma
// formula de error) para las cuatro fases y los tres kernels; la firma de
// cada macro/clase/funcion usada abajo no cambio, asi que el resto de este
// archivo es identico al original. Deben incluirse dentro de este namespace
// anonimo -- ver la nota de uso en la cabecera de cada header de common/.
#include "../../common/cuda_checks.cuh"
#include "../../common/metrics.cuh"

// Valida una llamada a CUTLASS (cutlass::Status). Mismo patron y motivo que
// CHECK_CUDA/CHECK_CUDNN de common/cuda_checks.cuh; se define aqui (no en
// common/) porque CUTLASS solo lo usa la Ruta 5 de este archivo -- ningun
// otro .cu del proyecto lo necesita todavia.
// cutlass::cutlassGetStatusString (declarada en cutlass/cutlass.h) expone
// un string legible para cutlass::Status -- verificado contra CUTLASS
// v2.11.0.
#define CHECK_CUTLASS(call)                                                   \
  do {                                                                        \
    cutlass::Status status_cutlass_ = (call);                                 \
    if (status_cutlass_ != cutlass::Status::kSuccess) {                       \
      std::cerr << "CUTLASS error at " << __FILE__ << ":" << __LINE__         \
                << " -> " << cutlass::cutlassGetStatusString(status_cutlass_) \
                << std::endl;                                                 \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

constexpr int kWarmupIters       = 3;
constexpr int kConversionThreads = 256;

// Dimensiones del fragmento WMMA para FP16 (unicas soportadas en sm >= 7.0).
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

// 4×4 warps = 16 warps = 512 hilos; tile de salida 64×64.
constexpr int kBlockWarpsM = 4;
constexpr int kBlockWarpsN = 4;

// Bloques residentes por SM que se le piden al compilador (segundo argumento de
// __launch_bounds__). El valor se fijo originalmente pensando en CC 8.6
// (48 warps/SM), no en el A100 de PACCA. En sm_80 el techo es 2048 hilos/SM y
// 64 warps/SM:
//   3 bloques x 512 hilos = 1536 hilos = 48/64 warps = 75 % de ocupancia,
//   con presupuesto de 65536/(3*512) = 42 registros por hilo.
// Subir a 4 daria 100 % pero recorta el presupuesto a 32 registros por hilo y
// el kernel (3 fragmentos WMMA + indices) desbordaria a memoria local, que
// cuesta mas que los warps ganados. La memoria compartida no es el limite:
// 3 x 28.5 KiB = 85.5 KiB de los 164 KiB por SM del A100.
// Sobrescribible al compilar para barrer el parametro en PACCA:
//   nvcc -DWMMA_MIN_BLOCKS_PER_SM=2 ...
#ifndef WMMA_MIN_BLOCKS_PER_SM
#define WMMA_MIN_BLOCKS_PER_SM 3
#endif
constexpr int kBlockTileM  = kBlockWarpsM * kWmmaM;  // 64
constexpr int kBlockTileN  = kBlockWarpsN * kWmmaN;  // 64

// Elementos K cargados por iteracion externa (multiplo de kWmmaK).
constexpr int kKStep = 32;

// Etapas del pipeline cp.async: mientras se computa tile[i], la DMA
// ya transfiere tile[i+2] sin pasar por registros (Ampere sm_80+).
constexpr int kNumStages = 3;

// Padding en shared memory para evitar bank conflicts (16 bytes extra por fila).
constexpr int kWmmaShmemPad = 8;

// Filas de shared memory, en elementos de 2 bytes (FP16).
constexpr int kSmemStrideA = kKStep      + kWmmaShmemPad;  // 40 elem = 80 B
constexpr int kSmemStrideB = kBlockTileN + kWmmaShmemPad;  // 72 elem = 144 B

// Ancho de cada cp.async. Ampere admite 4, 8 o 16 bytes por LDGSTS; la version
// original usaba 4 (un uint32_t = 2 elementos), lo que emite 4x mas
// instrucciones de las necesarias para mover el mismo tile. Con 16 bytes
// (8 elementos de 2 bytes) el bloque copia un tile K completo en una sola
// pasada de sus 512 hilos.
constexpr int kVecElems = 16 / 2;                                 // 8
constexpr int kVecsA    = kBlockTileM * kKStep      / kVecElems;  // 256
constexpr int kVecsB    = kKStep      * kBlockTileN / kVecElems;  // 256

// cp.async exige que origen y destino esten alineados al tamaño copiado (16 B).
// Destino: la base de sA/sB lleva __align__(16) y cada fila/etapa debe medir un
// multiplo de 16 B para que el alineamiento se propague.
static_assert(kKStep      % kVecElems == 0, "kKStep debe ser multiplo de kVecElems");
static_assert(kBlockTileN % kVecElems == 0, "kBlockTileN debe ser multiplo de kVecElems");
static_assert(kSmemStrideA * 2 % 16 == 0, "fila de sA no alineada a 16 B");
static_assert(kSmemStrideB * 2 % 16 == 0, "fila de sB no alineada a 16 B");
static_assert(kBlockTileM * kSmemStrideA * 2 % 16 == 0, "etapa de sA no alineada a 16 B");
static_assert(kKStep      * kSmemStrideB * 2 % 16 == 0, "etapa de sB no alineada a 16 B");
// Origen: los desplazamientos en global son (fila)*ld + k_off + col. Con col
// multiplo de kVecElems y k_off multiplo de kKStep, basta que los leading
// dimensions (Kcol = C*R*S y Ncol = outH*outW) sean multiplos de kVecElems; lo
// garantiza la validacion de benchmark_gpu_wmma_conv (Kcol % kKStep == 0 y
// Ncol % kBlockTileN == 0).

// Formatos de datos soportados en la ruta cuDNN con Tensor Cores.
enum class TensorCoreFormat {
    FP16,
    BF16,
    Both
};

// Parametros configurables desde linea de comandos.
// Convencion de nombres: N=batch, C=canales entrada, H/W=alto/ancho,
// K=filtros (canales salida), R/S=alto/ancho del filtro.
struct Options {
    int N          = 1;
    int C          = 32;
    int H          = 128;
    int W          = 128;
    int K          = 64;
    int R          = 3;
    int S          = 3;
    int pad_h      = 1;
    int pad_w      = 1;
    int stride_h   = 1;
    int stride_w   = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int iters      = 20;
    bool use_double = false;
    TensorCoreFormat tc_format = TensorCoreFormat::FP16;
    // Ruta 5 (CUTLASS ImplicitGemm), opt-in via --cutlass. Reutiliza
    // tc_format (fp16/bf16/both) para elegir que formato(s) de la Ruta 5
    // correr, igual que ya hace la Ruta 3 (cuDNN Tensor Core) -- no se
    // agrega un flag de formato separado para no duplicar --tc-format.
    bool run_cutlass = false;
};

// Dimensiones de la salida: derivadas de la formula estandar de convolucion.
struct OutputDims {
    int outN, outC, outH, outW;
};

// Envueltura RAII para el handle de cuDNN.
// El handle encapsula el estado interno de la biblioteca (streams, cache, etc).
class CudnnHandle {
public:
    CudnnHandle()  { CHECK_CUDNN(cudnnCreate(&handle_)); }
    ~CudnnHandle() { if (handle_) CHECK_CUDNN(cudnnDestroy(handle_)); }

    CudnnHandle(const CudnnHandle&)            = delete;
    CudnnHandle& operator=(const CudnnHandle&) = delete;

    cudnnHandle_t get() const { return handle_; }

private:
    cudnnHandle_t handle_ = nullptr;
};

// =========================================================================
// Utilidades de linea de comandos
// =========================================================================

static void print_usage(const char* prog) {
    std::cout
        << "Uso:\n"
        << "  " << prog << " [--N N] [--C C] [--H H] [--W W] [--K K] [--R R] [--S S]\n"
        << "             [--pad_h P] [--pad_w P] [--stride_h S] [--stride_w S]\n"
        << "             [--dilation_h D] [--dilation_w D] [--iters I] [--double]\n"
        << "             [--tc-format fp16|bf16|both] [--cutlass]\n\n"
        << "Descripcion:\n"
        << "  Compara hasta cinco rutas de convolucion 2D hacia adelante:\n"
        << "    1. CPU im2col + OpenBLAS (FP32/FP64)\n"
        << "    2. GPU cuDNN clasico (FP32/FP64, sin Tensor Cores)\n"
        << "    3. GPU cuDNN con Tensor Cores (FP16/BF16 entrada, FP32 acumulacion)\n"
        << "    4. GPU im2col FP16 + kernel WMMA custom (Tensor Cores directos)\n"
        << "    5. GPU CUTLASS ImplicitGemmConvolution (FP16/BF16, opt-in con --cutlass)\n"
        << "  La ruta WMMA (4) requiere K multiplo de 64, outH*outW multiplo de 64\n"
        << "  y C*R*S multiplo de 32. Si no se cumple se omite con aviso.\n"
        << "  Con --double solo se ejecutan las rutas 1 y 2.\n"
        << "  --tc-format selecciona el formato de las rutas 3 y 5 (por defecto fp16).\n"
        << "  BF16 requiere GPU Ampere o superior (compute capability >= 8.0).\n"
        << "  --cutlass activa la ruta 5 (desactivada por defecto); usa layout NHWC\n"
        << "  internamente (conversion NCHW<->NHWC documentada junto al codigo de la ruta).\n"
        << "  CUTLASS puede rechazar en tiempo de ejecucion formas no soportadas por su\n"
        << "  iterador optimizado; en ese caso la ruta se omite con aviso, igual que la\n"
        << "  ruta WMMA (4) con formas no divisibles.\n\n"
        << "Ejemplos:\n"
        << "  " << prog << "\n"
        << "  " << prog << " --N 1 --C 64 --H 224 --W 224 --K 64 --R 3 --S 3 --iters 10\n"
        << "  " << prog << " --double --N 1 --C 16 --H 64 --W 64 --K 32 --R 3 --S 3\n"
        << "  " << prog << " --N 1 --C 64 --H 64 --W 64 --K 64 --R 3 --S 3 --iters 2"
        << " --tc-format bf16\n"
        << "  " << prog << " --N 1 --C 64 --H 64 --W 64 --K 64 --R 3 --S 3 --iters 2"
        << " --tc-format both --cutlass\n";
}

static int parse_int_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Falta valor para " << argv[i] << "\n";
        std::exit(EXIT_FAILURE);
    }
    return std::atoi(argv[++i]);
}

static TensorCoreFormat parse_tc_format(const char* value) {
    if (std::strcmp(value, "fp16") == 0) return TensorCoreFormat::FP16;
    if (std::strcmp(value, "bf16") == 0) return TensorCoreFormat::BF16;
    if (std::strcmp(value, "both") == 0) return TensorCoreFormat::Both;

    std::cerr << "Formato Tensor Core no reconocido: " << value
              << ". Use fp16, bf16 o both." << std::endl;
    std::exit(EXIT_FAILURE);
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--N")          == 0) opt.N          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--C")          == 0) opt.C          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--H")          == 0) opt.H          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--W")          == 0) opt.W          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--K")          == 0) opt.K          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--R")          == 0) opt.R          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--S")          == 0) opt.S          = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--pad_h")      == 0) opt.pad_h      = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--pad_w")      == 0) opt.pad_w      = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--stride_h")   == 0) opt.stride_h   = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--stride_w")   == 0) opt.stride_w   = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--dilation_h") == 0) opt.dilation_h = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--dilation_w") == 0) opt.dilation_w = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--iters")      == 0) opt.iters      = parse_int_arg(i, argc, argv);
        else if (std::strcmp(argv[i], "--double")     == 0) opt.use_double = true;
        else if (std::strcmp(argv[i], "--cutlass")    == 0) opt.run_cutlass = true;
        else if (std::strcmp(argv[i], "--tc-format")  == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --tc-format\n";
                std::exit(EXIT_FAILURE);
            }
            opt.tc_format = parse_tc_format(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Argumento no reconocido: " << argv[i] << "\n\n";
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    if (opt.N <= 0 || opt.C <= 0 || opt.H <= 0 || opt.W <= 0 ||
        opt.K <= 0 || opt.R <= 0 || opt.S <= 0 || opt.iters <= 0) {
        std::cerr << "Todos los parametros deben ser positivos.\n";
        std::exit(EXIT_FAILURE);
    }
    return opt;
}

// =========================================================================
// Utilidades de dimension y rendimiento
// =========================================================================

static OutputDims compute_output_dims(const Options& opt) {
    OutputDims d;
    d.outN = opt.N;
    d.outC = opt.K;
    d.outH = (opt.H + 2 * opt.pad_h - opt.dilation_h * (opt.R - 1) - 1) / opt.stride_h + 1;
    d.outW = (opt.W + 2 * opt.pad_w - opt.dilation_w * (opt.S - 1) - 1) / opt.stride_w + 1;
    if (d.outH <= 0 || d.outW <= 0) {
        std::cerr << "Dimensiones de salida invalidas. Revisa padding/stride/dilation/filtro.\n";
        std::exit(EXIT_FAILURE);
    }
    return d;
}

// Un MAC (multiply-accumulate) = 1 mul + 1 add = 2 operaciones de punto flotante.
// Cada posicion de salida (n, k, oh, ow) requiere C*R*S MACs.
static double conv_flops(const Options& opt, const OutputDims& d) {
    return 2.0
        * static_cast<double>(opt.N)
        * static_cast<double>(opt.K)
        * static_cast<double>(d.outH)
        * static_cast<double>(d.outW)
        * static_cast<double>(opt.C)
        * static_cast<double>(opt.R)
        * static_cast<double>(opt.S);
}

static Metrics build_metrics(const Options& opt, const OutputDims& d, double avg_ms) {
    Metrics m;
    m.ms     = avg_ms;
    m.gflops = conv_flops(opt, d) / (m.ms * 1e6);
    m.tflops = m.gflops / 1000.0;
    return m;
}

// =========================================================================
// Inicializacion de datos
// =========================================================================

// Valores deterministicos acotados: evita depender de semilla aleatoria
// y facilita reproducir los experimentos exactamente.
static void initialize_matrix_float(std::vector<float>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        const int c = static_cast<int>(i % 101) - 50;
        v[i] = static_cast<float>(c) / 25.0f;
    }
}

static void initialize_matrix_double(std::vector<double>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        const int c = static_cast<int>(i % 101) - 50;
        v[i] = static_cast<double>(c) / 25.0;
    }
}

// =========================================================================
// Metricas de error numerico
// =========================================================================

// =========================================================================
// Impresion de resultados
// =========================================================================

static void print_gpu_info() {
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    int gpu_clock_khz = 0, mem_clock_khz = 0, mem_bus_width = 0;
    cudaError_t e1 = cudaDeviceGetAttribute(&gpu_clock_khz, cudaDevAttrClockRate, dev);
    cudaError_t e2 = cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaError_t e3 = cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth, dev);

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
              << (e1 == cudaSuccess ? gpu_clock_khz / 1000.0 : 0.0)
              << (e1 == cudaSuccess ? " MHz\n" : " no disponible\n");
    std::cout << "Reloj memoria              : "
              << (e2 == cudaSuccess ? mem_clock_khz / 1000.0 : 0.0)
              << (e2 == cudaSuccess ? " MHz\n" : " no disponible\n");
    std::cout << "Bus de memoria             : "
              << (e3 == cudaSuccess ? std::to_string(mem_bus_width) + " bits" : "no disponible")
              << "\n";
    std::cout << "Memoria compartida/bloque  : " << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
    std::cout << "===========================================================\n\n";
}

// BF16 Tensor Core (HMMA con operandos __nv_bfloat16) requiere Ampere o superior.
static bool active_device_supports_bf16_tensor_cores() {
    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    return prop.major >= 8;
}

static void print_reference_comparison(const char* label,
                                       const Metrics& m,
                                       double ref_ms,
                                       const ErrorMetrics& e) {
    std::cout << label << " - tiempo         : " << m.ms << " ms\n";
    std::cout << label << " - rendimiento    : " << m.gflops << " GFLOP/s ("
              << m.tflops << " TFLOP/s)\n";
    std::cout << "Speedup vs CPU             : " << ref_ms / m.ms << "x\n";
    std::cout << "Error max abs vs CPU       : " << e.max_abs << "\n";
    std::cout << "Error relativo L2 vs CPU   : " << e.rel_l2 << "\n\n";
}

// =========================================================================
// Ruta 1 - CPU: im2col + OpenBLAS
//
// La convolucion se reescribe como una multiplicacion de matrices:
//   Y[K, outH*outW] = W[K, C*R*S] * col[C*R*S, outH*outW]
//
// "col" se construye con im2col: cada columna contiene los C*R*S elementos
// de la ventana receptiva correspondiente a una posicion de salida (oh, ow).
// Los ceros por fuera del borde (padding) se insertan explicitamente.
// =========================================================================

static void im2col_float(const float* x, float* col,
                         const Options& opt, const OutputDims& d) {
    const int stride_col = opt.C * opt.R * opt.S;
    for (int c = 0; c < opt.C; ++c) {
        for (int r = 0; r < opt.R; ++r) {
            for (int s = 0; s < opt.S; ++s) {
                const int row = (c * opt.R + r) * opt.S + s;
                for (int oh = 0; oh < d.outH; ++oh) {
                    for (int ow = 0; ow < d.outW; ++ow) {
                        const int ih = oh * opt.stride_h - opt.pad_h + r * opt.dilation_h;
                        const int iw = ow * opt.stride_w - opt.pad_w + s * opt.dilation_w;
                        const int col_idx = oh * d.outW + ow;
                        col[row + col_idx * stride_col] =
                            (ih >= 0 && ih < opt.H && iw >= 0 && iw < opt.W)
                            ? x[(c * opt.H + ih) * opt.W + iw]
                            : 0.0f;
                    }
                }
            }
        }
    }
}

static void im2col_double(const double* x, double* col,
                          const Options& opt, const OutputDims& d) {
    const int stride_col = opt.C * opt.R * opt.S;
    for (int c = 0; c < opt.C; ++c) {
        for (int r = 0; r < opt.R; ++r) {
            for (int s = 0; s < opt.S; ++s) {
                const int row = (c * opt.R + r) * opt.S + s;
                for (int oh = 0; oh < d.outH; ++oh) {
                    for (int ow = 0; ow < d.outW; ++ow) {
                        const int ih = oh * opt.stride_h - opt.pad_h + r * opt.dilation_h;
                        const int iw = ow * opt.stride_w - opt.pad_w + s * opt.dilation_w;
                        const int col_idx = oh * d.outW + ow;
                        col[row + col_idx * stride_col] =
                            (ih >= 0 && ih < opt.H && iw >= 0 && iw < opt.W)
                            ? x[(c * opt.H + ih) * opt.W + iw]
                            : 0.0;
                    }
                }
            }
        }
    }
}

static Metrics benchmark_cpu_float(const std::vector<float>& x,
                                   const std::vector<float>& w,
                                   std::vector<float>& y,
                                   const Options& opt,
                                   const OutputDims& d) {
    const int M    = opt.K;
    const int Ncol = d.outH * d.outW;
    const int Kcol = opt.C * opt.R * opt.S;
    std::vector<float> col(static_cast<size_t>(Kcol) * Ncol);
    const float alpha = 1.0f, beta = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < opt.iters; ++it) {
        for (int n = 0; n < opt.N; ++n) {
            const float* xn = x.data() + static_cast<size_t>(n) * opt.C * opt.H * opt.W;
            float*       yn = y.data() + static_cast<size_t>(n) * opt.K * d.outH * d.outW;
            im2col_float(xn, col.data(), opt, d);
            // w NCHW [K,C,R,S] es row-major [M, Kcol]; col se construye
            // col-major [Kcol, Ncol] (= row-major [Ncol, Kcol], de ahi el
            // CblasTrans) y la salida NCHW [K, outH*outW] exige row-major
            // [M, Ncol] con ldc = Ncol, elementwise-comparable con cuDNN.
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, Ncol, Kcol,
                        alpha, w.data(), Kcol,
                        col.data(), Kcol,
                        beta, yn, Ncol);
        }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / opt.iters;
    return build_metrics(opt, d, avg_ms);
}

static Metrics benchmark_cpu_double(const std::vector<double>& x,
                                    const std::vector<double>& w,
                                    std::vector<double>& y,
                                    const Options& opt,
                                    const OutputDims& d) {
    const int M    = opt.K;
    const int Ncol = d.outH * d.outW;
    const int Kcol = opt.C * opt.R * opt.S;
    std::vector<double> col(static_cast<size_t>(Kcol) * Ncol);
    const double alpha = 1.0, beta = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < opt.iters; ++it) {
        for (int n = 0; n < opt.N; ++n) {
            const double* xn = x.data() + static_cast<size_t>(n) * opt.C * opt.H * opt.W;
            double*       yn = y.data() + static_cast<size_t>(n) * opt.K * d.outH * d.outW;
            im2col_double(xn, col.data(), opt, d);
            // Mismo layout que la version float: salida NCHW row-major.
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, Ncol, Kcol,
                        alpha, w.data(), Kcol,
                        col.data(), Kcol,
                        beta, yn, Ncol);
        }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / opt.iters;
    return build_metrics(opt, d, avg_ms);
}

// =========================================================================
// Ruta 2 - GPU cuDNN FP32 clasico (sin Tensor Cores)
// =========================================================================

static Metrics benchmark_gpu_cudnn_float(const std::vector<float>& x,
                                         const std::vector<float>& w,
                                         std::vector<float>& y,
                                         const Options& opt,
                                         const OutputDims& d) {
    cudnnTensorDescriptor_t    xDesc, yDesc;
    cudnnFilterDescriptor_t    wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // Todos los tensores en FP32, formato NCHW.
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           opt.N, opt.C, opt.H, opt.W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           opt.K, opt.C, opt.R, opt.S));
    // El tipo de computo (computeType) determina la precision de la acumulacion interna.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                opt.pad_h, opt.pad_w,
                                                opt.stride_h, opt.stride_w,
                                                opt.dilation_h, opt.dilation_w,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    // Esta ruta es la linea base "sin Tensor Cores" contra la que se mide el
    // speedup de las rutas 3 y 4, asi que tiene que ser FP32 escalar de verdad.
    // Con CUDNN_DEFAULT_MATH cuDNN habilita TF32 por su cuenta en Ampere y la
    // supuesta linea base corria en Tensor Cores: 79.1 TFLOP/s medidos son el
    // 50.7 % del pico TF32 (156 TFLOP/s) y 4x por encima del pico FP32 escalar
    // del A100 (19.5 TFLOP/s), con lo que el speedup TC/FP32 salia deflactado
    // (1.9x-3.0x). CUDNN_FMA_MATH fuerza la ruta FP32 escalar.
    CHECK_CUDNN(cudnnSetConvolutionMathType(convDesc, CUDNN_FMA_MATH));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           d.outN, d.outC, d.outH, d.outW));

    CudnnHandle handle;

    // Seleccion automatica del mejor algoritmo disponible con los descriptores dados.
    cudnnConvolutionFwdAlgoPerf_t perf_results[8];
    int algo_count = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle.get(), xDesc, wDesc, convDesc, yDesc, 8, &algo_count, perf_results));

    // Con CUDNN_FMA_MATH quedan descartados los algoritmos que solo existen en
    // variante Tensor Core, y esos vuelven con status != SUCCESS: hay que tomar
    // el primero ejecutable, no el primero de la lista.
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    for (int ai = 0; ai < algo_count; ++ai) {
        if (perf_results[ai].status == CUDNN_STATUS_SUCCESS) {
            algo = perf_results[ai].algo;
            break;
        }
    }

    // El workspace es memoria temporal en GPU que algunos algoritmos necesitan.
    size_t ws_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle.get(), xDesc, wDesc, convDesc, yDesc, algo, &ws_bytes));

    float *d_x, *d_w, *d_y;
    void*  d_ws = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, w.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(float)));
    if (ws_bytes > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_bytes));

    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < kWarmupIters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < opt.iters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y));
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    CHECK_CUDA(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    if (d_ws) CHECK_CUDA(cudaFree(d_ws));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

    return build_metrics(opt, d, static_cast<double>(total_ms) / opt.iters);
}

// =========================================================================
// Ruta 2b - GPU cuDNN FP64 (solo para --double)
// =========================================================================

static Metrics benchmark_gpu_cudnn_double(const std::vector<double>& x,
                                          const std::vector<double>& w,
                                          std::vector<double>& y,
                                          const Options& opt,
                                          const OutputDims& d) {
    cudnnTensorDescriptor_t    xDesc, yDesc;
    cudnnFilterDescriptor_t    wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                                           opt.N, opt.C, opt.H, opt.W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW,
                                           opt.K, opt.C, opt.R, opt.S));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                opt.pad_h, opt.pad_w,
                                                opt.stride_h, opt.stride_w,
                                                opt.dilation_h, opt.dilation_w,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_DOUBLE));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                                           d.outN, d.outC, d.outH, d.outW));

    CudnnHandle handle;

    cudnnConvolutionFwdAlgoPerf_t perf_results[8];
    int algo_count = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle.get(), xDesc, wDesc, convDesc, yDesc, 8, &algo_count, perf_results));
    const cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;

    size_t ws_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle.get(), xDesc, wDesc, convDesc, yDesc, algo, &ws_bytes));

    double *d_x, *d_w, *d_y;
    void* d_ws = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_w, w.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(double)));
    if (ws_bytes > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_bytes));

    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, w.data(), w.size() * sizeof(double), cudaMemcpyHostToDevice));

    const double alpha = 1.0, beta = 0.0;

    for (int i = 0; i < kWarmupIters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < opt.iters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x, wDesc, d_w,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y));
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    CHECK_CUDA(cudaMemcpy(y.data(), d_y, y.size() * sizeof(double), cudaMemcpyDeviceToHost));

    if (d_ws) CHECK_CUDA(cudaFree(d_ws));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

    return build_metrics(opt, d, static_cast<double>(total_ms) / opt.iters);
}

// =========================================================================
// Kernel de conversion FP32 -> FP16 en GPU (mismo patron que en GEMM)
// =========================================================================

__global__ static void convert_float_to_half_kernel(const float* src, __half* dst, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Convierte un buffer FP16 a FP32 en la GPU elemento a elemento.
__global__ static void convert_half_to_float_kernel(const __half* src, float* dst, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __half2float(src[idx]);
    }
}

// Sube un vector a GPU como FP32, lo convierte a FP16 en el device y devuelve
// el puntero FP16. El buffer FP32 intermedio se libera antes de retornar.
static __half* upload_and_convert_to_half(const std::vector<float>& host_data) {
    const size_t n = host_data.size();
    float*  d_fp32 = nullptr;
    __half* d_fp16 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_fp32, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fp16, n * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_fp32, host_data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    const int blocks = static_cast<int>((n + kConversionThreads - 1) / kConversionThreads);
    convert_float_to_half_kernel<<<blocks, kConversionThreads>>>(d_fp32, d_fp16, static_cast<int>(n));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_fp32));
    return d_fp16;
}

// =========================================================================
// Kernels de conversion FP32 <-> BF16 en GPU (mismo patron que FP16 arriba)
// =========================================================================

__global__ static void convert_float_to_bfloat16_kernel(const float* src, __nv_bfloat16* dst, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

__global__ static void convert_bfloat16_to_float_kernel(const __nv_bfloat16* src, float* dst, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

// Sube un vector a GPU como FP32, lo convierte a BF16 en el device y devuelve
// el puntero BF16. El buffer FP32 intermedio se libera antes de retornar.
static __nv_bfloat16* upload_and_convert_to_bfloat16(const std::vector<float>& host_data) {
    const size_t n = host_data.size();
    float*         d_fp32 = nullptr;
    __nv_bfloat16* d_bf16 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_fp32, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bf16, n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMemcpy(d_fp32, host_data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    const int blocks = static_cast<int>((n + kConversionThreads - 1) / kConversionThreads);
    convert_float_to_bfloat16_kernel<<<blocks, kConversionThreads>>>(d_fp32, d_bf16, static_cast<int>(n));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_fp32));
    return d_bf16;
}

// =========================================================================
// Ruta 3 - GPU cuDNN con Tensor Cores: FP16 in / FP32 acumulacion y salida
//
// Tres pasos obligatorios para activar Tensor Cores en cuDNN:
//
//  1. Descriptores de entrada (x) y filtro (w) con CUDNN_DATA_HALF
//     -> los operandos de la multiplicacion son de 16 bits.
//
//  2. Tipo de computo CUDNN_DATA_FLOAT en cudnnSetConvolution2dDescriptor
//     -> la acumulacion de productos parciales se realiza en 32 bits,
//        lo que evita desbordamiento y mantiene la precision numerica util.
//
//  3. cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH)
//     -> le comunica a cuDNN que puede usar las unidades Tensor Core
//        para esta convolucion. Sin esta llamada cuDNN puede ignorarlas.
// =========================================================================

static Metrics benchmark_gpu_tensor_cores_conv(const std::vector<float>& x,
                                               const std::vector<float>& w,
                                               std::vector<float>& y,
                                               const Options& opt,
                                               const OutputDims& d) {
    // Paso 1: preparar entradas FP16 en GPU.
    __half* d_x_fp16 = upload_and_convert_to_half(x);
    __half* d_w_fp16 = upload_and_convert_to_half(w);

    // Paso 2: crear descriptores con tipos mixtos.
    cudnnTensorDescriptor_t    xDesc, yDesc;
    cudnnFilterDescriptor_t    wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // Entrada y filtro en FP16.
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           opt.N, opt.C, opt.H, opt.W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
                                           opt.K, opt.C, opt.R, opt.S));
    // computeType = FLOAT: acumula productos FP16 en un acumulador de 32 bits.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                opt.pad_h, opt.pad_w,
                                                opt.stride_h, opt.stride_w,
                                                opt.dilation_h, opt.dilation_w,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    // Paso 3: activar Tensor Cores explicitamente en el descriptor.
    CHECK_CUDNN(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

    // En cuDNN 9, FP16 entrada / FP32 salida no es compatible con NCHW.
    // Se usa FP16 para la salida y se convierte a FP32 despues de la convolucion.
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
                                           d.outN, d.outC, d.outH, d.outW));

    CudnnHandle handle;

    // Con CUDNN_TENSOR_OP_MATH activo, cuDNN priorizara algoritmos compatibles con TC
    // al evaluar las opciones mediante cudnnGetConvolutionForwardAlgorithm_v7.
    cudnnConvolutionFwdAlgoPerf_t perf_results[8];
    int algo_count = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle.get(), xDesc, wDesc, convDesc, yDesc, 8, &algo_count, perf_results));

    // Algunos algoritmos devueltos pueden tener status != SUCCESS para FP16;
    // se elige el primero que cuDNN declara ejecutable.
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    for (int ai = 0; ai < algo_count; ++ai) {
        if (perf_results[ai].status == CUDNN_STATUS_SUCCESS) {
            algo = perf_results[ai].algo;
            break;
        }
    }

    size_t ws_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle.get(), xDesc, wDesc, convDesc, yDesc, algo, &ws_bytes));

    // Buffer de salida en FP16 (cuDNN 9 requiere FP16 out con FP16 in en NCHW).
    __half* d_y_fp16 = nullptr;
    float*  d_y_fp32 = nullptr;
    void*   d_ws     = nullptr;
    CHECK_CUDA(cudaMalloc(&d_y_fp16, y.size() * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_y_fp32, y.size() * sizeof(float)));
    if (ws_bytes > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_bytes));

    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < kWarmupIters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x_fp16, wDesc, d_w_fp16,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y_fp16));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < opt.iters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x_fp16, wDesc, d_w_fp16,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y_fp16));
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    // Convertir salida FP16 → FP32 fuera del intervalo medido.
    const int y_count = static_cast<int>(y.size());
    const int conv_blocks = (y_count + kConversionThreads - 1) / kConversionThreads;
    convert_half_to_float_kernel<<<conv_blocks, kConversionThreads>>>(
        d_y_fp16, d_y_fp32, y_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y.data(), d_y_fp32, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    if (d_ws) CHECK_CUDA(cudaFree(d_ws));
    CHECK_CUDA(cudaFree(d_y_fp32));
    CHECK_CUDA(cudaFree(d_y_fp16));
    CHECK_CUDA(cudaFree(d_x_fp16));
    CHECK_CUDA(cudaFree(d_w_fp16));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

    return build_metrics(opt, d, static_cast<double>(total_ms) / opt.iters);
}

// =========================================================================
// Ruta 3b - GPU cuDNN con Tensor Cores: BF16 in / FP32 acumulacion y salida
//
// Analoga a benchmark_gpu_tensor_cores_conv, cambiando CUDNN_DATA_HALF por
// CUDNN_DATA_BFLOAT16. Misma limitacion de layout que se encontro para
// FP16: el "Type Configurations" de cudnnConvolutionForward() documenta
// la combinacion 16-bit-in/FLOAT-out (compute FLOAT) como soportada
// unicamente en NHWC; en NCHW solo esta garantizada la combinacion con
// entrada Y salida en el mismo tipo de 16 bits (16-bit-in/16-bit-out,
// compute FLOAT). Esa es la restriccion que ya obligo a usar yDesc=HALF
// en la ruta FP16 de este archivo (ver comentario original mas arriba).
//
// No se asume que BF16 se comporte igual sin verificarlo: por eso se usa
// aqui exactamente el mismo patron ya validado para FP16 (yDesc=BFLOAT16,
// conversion a FLOAT despues de medir). Si al ejecutar en PACCA se
// confirma que cuDNN sí acepta BF16-in/FLOAT-out en NCHW, este paso de
// conversion se podria eliminar como simplificacion futura; hasta
// entonces, esta es la ruta segura y coherente con la ruta FP16 existente.
// =========================================================================

static Metrics benchmark_gpu_tensor_cores_conv_bf16(const std::vector<float>& x,
                                                    const std::vector<float>& w,
                                                    std::vector<float>& y,
                                                    const Options& opt,
                                                    const OutputDims& d) {
    // Paso 1: preparar entradas BF16 en GPU.
    __nv_bfloat16* d_x_bf16 = upload_and_convert_to_bfloat16(x);
    __nv_bfloat16* d_w_bf16 = upload_and_convert_to_bfloat16(w);

    // Paso 2: crear descriptores con tipos mixtos.
    cudnnTensorDescriptor_t    xDesc, yDesc;
    cudnnFilterDescriptor_t    wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // Entrada y filtro en BF16.
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16,
                                           opt.N, opt.C, opt.H, opt.W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_BFLOAT16, CUDNN_TENSOR_NCHW,
                                           opt.K, opt.C, opt.R, opt.S));
    // computeType = FLOAT: acumula productos BF16 en un acumulador de 32 bits.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                opt.pad_h, opt.pad_w,
                                                opt.stride_h, opt.stride_w,
                                                opt.dilation_h, opt.dilation_w,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    // Paso 3: activar Tensor Cores explicitamente en el descriptor.
    CHECK_CUDNN(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

    // Salida BF16 (mismo motivo que la ruta FP16: ver nota de layout arriba).
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16,
                                           d.outN, d.outC, d.outH, d.outW));

    CudnnHandle handle;

    cudnnConvolutionFwdAlgoPerf_t perf_results[8];
    int algo_count = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle.get(), xDesc, wDesc, convDesc, yDesc, 8, &algo_count, perf_results));

    // Algunos algoritmos devueltos pueden tener status != SUCCESS para BF16;
    // se elige el primero que cuDNN declara ejecutable.
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    for (int ai = 0; ai < algo_count; ++ai) {
        if (perf_results[ai].status == CUDNN_STATUS_SUCCESS) {
            algo = perf_results[ai].algo;
            break;
        }
    }

    size_t ws_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle.get(), xDesc, wDesc, convDesc, yDesc, algo, &ws_bytes));

    // Buffer de salida en BF16 (ver nota de layout arriba).
    __nv_bfloat16* d_y_bf16 = nullptr;
    float*         d_y_fp32 = nullptr;
    void*          d_ws     = nullptr;
    CHECK_CUDA(cudaMalloc(&d_y_bf16, y.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_y_fp32, y.size() * sizeof(float)));
    if (ws_bytes > 0) CHECK_CUDA(cudaMalloc(&d_ws, ws_bytes));

    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < kWarmupIters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x_bf16, wDesc, d_w_bf16,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y_bf16));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < opt.iters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle.get(), &alpha,
                                            xDesc, d_x_bf16, wDesc, d_w_bf16,
                                            convDesc, algo, d_ws, ws_bytes,
                                            &beta, yDesc, d_y_bf16));
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    // Convertir salida BF16 → FP32 fuera del intervalo medido.
    const int y_count = static_cast<int>(y.size());
    const int conv_blocks = (y_count + kConversionThreads - 1) / kConversionThreads;
    convert_bfloat16_to_float_kernel<<<conv_blocks, kConversionThreads>>>(
        d_y_bf16, d_y_fp32, y_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y.data(), d_y_fp32, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    if (d_ws) CHECK_CUDA(cudaFree(d_ws));
    CHECK_CUDA(cudaFree(d_y_fp32));
    CHECK_CUDA(cudaFree(d_y_bf16));
    CHECK_CUDA(cudaFree(d_x_bf16));
    CHECK_CUDA(cudaFree(d_w_bf16));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

    return build_metrics(opt, d, static_cast<double>(total_ms) / opt.iters);
}

// =========================================================================
// Ruta 4 - im2col en GPU (FP16) + kernel WMMA
//
// La convolucion 2D se descompone en dos pasos:
//   1. im2col_fp16_kernel: transforma la entrada [C,H,W] en una matriz
//      col[C*R*S, outH*outW] FP16 row-major. Cada columna de col contiene
//      los C*R*S valores de la ventana receptiva de una posicion de salida.
//
//   2. wmma_gemm_kernel: calcula Y[K, outH*outW] = W[K, C*R*S] * col row-major
//      usando la API WMMA (Tensor Cores) de la misma forma que en GEMM.
//
// Layout:
//   W NCHW [K,C,R,S] visto como [K, C*R*S] es ya row-major → conversion
//   FP32→FP16 elemento a elemento sin transposicion.
//   Y[K, Ncol] row-major = NCHW [K, outH, outW] para N=1 → comparable
//   directamente con la salida de cuDNN.
// =========================================================================

// Construye la matriz col en FP16 row-major a partir de un batch element FP32.
// Thread id cubre todos los C*R*S × outH*outW elementos de col.
// Coalescencia: hilos consecutivos comparten el mismo crs y tienen pos
// consecutivos (mismo oh, ow consecutivo → iw consecutivo con stride_w=1),
// por lo que leen x a lo largo de la dimension W → acceso coalescente.
__global__ static void im2col_fp16_kernel(
        const float* __restrict__ x,
        __half*      __restrict__ col,
        int C, int H, int W,
        int R, int S,
        int pad_h,  int pad_w,
        int stride_h, int stride_w,
        int dilation_h, int dilation_w,
        int outH, int outW) {
    const int Ncol = outH * outW;
    const int id   = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= C * R * S * Ncol) return;

    const int crs = id / Ncol;
    const int pos = id % Ncol;
    const int c   = crs / (R * S);
    const int rs  = crs % (R * S);
    const int r   = rs  / S;
    const int s   = rs  % S;
    const int oh  = pos / outW;
    const int ow  = pos % outW;
    const int ih  = oh * stride_h - pad_h + r * dilation_h;
    const int iw  = ow * stride_w - pad_w + s * dilation_w;

    col[id] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
              ? __float2half(x[(c * H + ih) * W + iw])
              : __float2half(0.0f);
}

// Emite las cp.async de un tile K completo (sA + sB) hacia una etapa del
// triple buffer. Los kVecsA + kVecsB vectores de 16 bytes se reparten entre
// TODOS los hilos del bloque en un unico bucle: con la configuracion por
// defecto son 256 + 256 = 512 vectores para 512 hilos, es decir un LDGSTS.128
// por hilo. El bucle mantiene la forma grid-stride para seguir siendo correcto
// si se cambian kBlockTile*/kKStep.
//
// Los indices globales se calculan en size_t: con Ncol = 65536 y Kcol grande el
// producto fila*ld se acerca al rango de int.
__device__ __forceinline__ void issue_stage_copy(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        __half* __restrict__ sA_stage,
        __half* __restrict__ sB_stage,
        int block_row, int block_col, int k_off, int N, int K) {
    for (int i = threadIdx.x; i < kVecsA + kVecsB; i += blockDim.x) {
        if (i < kVecsA) {
            const int elem = i * kVecElems;
            const int row  = elem / kKStep;
            const int col  = elem % kKStep;
            __pipeline_memcpy_async(
                &sA_stage[row * kSmemStrideA + col],
                &A[static_cast<size_t>(block_row + row) * K + k_off + col],
                sizeof(uint4));
        } else {
            const int elem = (i - kVecsA) * kVecElems;
            const int row  = elem / kBlockTileN;
            const int col  = elem % kBlockTileN;
            __pipeline_memcpy_async(
                &sB_stage[row * kSmemStrideB + col],
                &B[static_cast<size_t>(k_off + row) * N + block_col + col],
                sizeof(uint4));
        }
    }
}

// Kernel GEMM con API WMMA + pipeline cp.async de 3 etapas (Ampere sm_80+).
// C(M,N) = A(M,K) * B(K,N), todos row-major FP16→FP32.
// En el contexto de convolucion: A = filtros W [K, C*R*S], B = im2col [C*R*S, outH*outW].
//
// 4×4 warps = 512 hilos, tile 64×64, triple buffer cp.async.
// Requisito: M mult. de 64, N mult. de 64, K mult. de 32 (validado en benchmark).
// Ocupancia esperada en sm_80 (A100): WMMA_MIN_BLOCKS_PER_SM bloques/SM × 16
// warps; con el valor por defecto 3 son 48 de los 64 warps del SM (75 %).
__launch_bounds__(kBlockWarpsM * kBlockWarpsN * 32, WMMA_MIN_BLOCKS_PER_SM)
__global__ static void wmma_gemm_kernel(
        const __half* __restrict__ A,
        const __half* __restrict__ B,
        float*        __restrict__ C,
        int M, int N, int K) {
    using namespace nvcuda;

    // Triple buffer: sA[etapa][fila][col], sB[etapa][fila][col].
    // sA[3][64][40]: 15 360 bytes; sB[3][32][72]: 13 824 bytes → 28.5 KiB total.
    // __align__(16) es obligatorio para el destino de las cp.async de 16 bytes:
    // nvcc solo garantizaria el alineamiento natural del tipo (2 bytes).
    __shared__ __align__(16) __half sA[kNumStages][kBlockTileM][kSmemStrideA];
    __shared__ __align__(16) __half sB[kNumStages][kKStep]     [kSmemStrideB];

    const int warp_id       = threadIdx.x / 32;
    const int warp_row      = warp_id / kBlockWarpsN;
    const int warp_col      = warp_id % kBlockWarpsN;
    const int block_row     = blockIdx.x * kBlockTileM;
    const int block_col     = blockIdx.y * kBlockTileN;
    const int warp_row_base = block_row + warp_row * kWmmaM;
    const int warp_col_base = block_col + warp_col * kWmmaN;

    wmma::fragment<wmma::matrix_a,    kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,    kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>                   c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int num_tiles = K / kKStep;

    // -- Precarga de las primeras kNumStages etapas --
    for (int s = 0; s < kNumStages && s < num_tiles; ++s) {
        issue_stage_copy(A, B, &sA[s][0][0], &sB[s][0][0],
                         block_row, block_col, s * kKStep, N, K);
        __pipeline_commit();
    }

    // -- Bucle principal --
    for (int tile = 0; tile < num_tiles; ++tile) {
        // Esperar que el tile actual este listo en shared memory.
        const int prior = min(kNumStages - 1, num_tiles - tile - 1);
        __pipeline_wait_prior(prior);
        __syncthreads();

        // Computo WMMA sobre la etapa actual.
        const int stage_c = tile % kNumStages;
        for (int k_inner = 0; k_inner < kKStep; k_inner += kWmmaK) {
            wmma::load_matrix_sync(a_frag,
                reinterpret_cast<const __half*>(&sA[stage_c][warp_row * kWmmaM][k_inner]),
                kSmemStrideA);
            wmma::load_matrix_sync(b_frag,
                reinterpret_cast<const __half*>(&sB[stage_c][k_inner][warp_col * kWmmaN]),
                kSmemStrideB);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Barrera obligatoria: garantiza que todos los warps terminaron de leer
        // stage_c antes de que cualquier warp empiece a sobreescribirlo con cp.async.
        __syncthreads();

        // Lanzar carga futura en stage_c (ahora libre).
        const int future = tile + kNumStages;
        if (future < num_tiles) {
            issue_stage_copy(A, B, &sA[stage_c][0][0], &sB[stage_c][0][0],
                             block_row, block_col, future * kKStep, N, K);
            __pipeline_commit();
        }
    }

    if (warp_row_base < M && warp_col_base < N) {
        wmma::store_matrix_sync(
            C + warp_row_base * N + warp_col_base,
            c_frag, N,
            wmma::mem_row_major);
    }
}

// Benchmark de la ruta WMMA para convolucion.
// Convierte W a FP16 una sola vez; en cada iteracion lanza im2col_fp16_kernel
// seguido de wmma_gemm_kernel para cada elemento del batch.
// La salida Y queda en NCHW FP32 (mismo layout que cuDNN), comparable directamente.
static Metrics benchmark_gpu_wmma_conv(const std::vector<float>& x,
                                        const std::vector<float>& w,
                                        std::vector<float>& y,
                                        const Options& opt,
                                        const OutputDims& d) {
    const int M    = opt.K;
    const int Ncol = d.outH * d.outW;
    const int Kcol = opt.C * opt.R * opt.S;

    if (M % kBlockTileM != 0 || Ncol % kBlockTileN != 0 || Kcol % kKStep != 0) {
        std::cerr << "WMMA conv omitida: K=" << M << " (req. mult. de " << kBlockTileM
                  << "), outH*outW=" << Ncol << " (req. mult. de " << kBlockTileN
                  << "), C*R*S=" << Kcol << " (req. mult. de " << kKStep << ").\n";
        return Metrics{};
    }

    // Convertir filtros W: NCHW [K,C,R,S] = row-major [K, C*R*S] → FP16.
    const int w_count = static_cast<int>(w.size());
    float*  d_w_fp32 = nullptr;
    __half* d_w_fp16 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_w_fp32, static_cast<size_t>(w_count) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w_fp16, static_cast<size_t>(w_count) * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_w_fp32, w.data(),
                          static_cast<size_t>(w_count) * sizeof(float),
                          cudaMemcpyHostToDevice));
    const int bw = (w_count + kConversionThreads - 1) / kConversionThreads;
    convert_float_to_half_kernel<<<bw, kConversionThreads>>>(d_w_fp32, d_w_fp16, w_count);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_w_fp32));

    // Subir entrada X al GPU (usada por im2col cada iteracion).
    const int x_count = static_cast<int>(x.size());
    float* d_x = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, static_cast<size_t>(x_count) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(),
                          static_cast<size_t>(x_count) * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Buffer temporal col: [Kcol, Ncol] FP16 (reutilizado por batch element).
    __half* d_col = nullptr;
    CHECK_CUDA(cudaMalloc(&d_col,
        static_cast<size_t>(Kcol) * static_cast<size_t>(Ncol) * sizeof(__half)));

    // Buffer de salida Y: [N, K, outH, outW] FP32 NCHW.
    float* d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_y, y.size() * sizeof(float)));

    const int single_x = opt.C * opt.H * opt.W;
    const int single_y = opt.K * d.outH * d.outW;
    const int col_elems = Kcol * Ncol;
    const int col_blocks = (col_elems + kConversionThreads - 1) / kConversionThreads;

    // 512 hilos/bloque (16 warps); ver WMMA_MIN_BLOCKS_PER_SM para la ocupancia.
    const dim3 gemm_block(static_cast<unsigned int>(kBlockWarpsM * kBlockWarpsN * 32));
    const dim3 gemm_grid(
        static_cast<unsigned int>((M    + kBlockTileM - 1) / kBlockTileM),
        static_cast<unsigned int>((Ncol + kBlockTileN - 1) / kBlockTileN));

    auto run_conv_iter = [&]() {
        for (int n = 0; n < opt.N; ++n) {
            const float* x_n = d_x + static_cast<size_t>(n) * single_x;
            float*       y_n = d_y + static_cast<size_t>(n) * single_y;
            im2col_fp16_kernel<<<col_blocks, kConversionThreads>>>(
                x_n, d_col,
                opt.C, opt.H, opt.W,
                opt.R, opt.S,
                opt.pad_h, opt.pad_w,
                opt.stride_h, opt.stride_w,
                opt.dilation_h, opt.dilation_w,
                d.outH, d.outW);
            CHECK_CUDA(cudaGetLastError());
            wmma_gemm_kernel<<<gemm_grid, gemm_block>>>(
                d_w_fp16, d_col, y_n, M, Ncol, Kcol);
            CHECK_CUDA(cudaGetLastError());
        }
    };

    for (int i = 0; i < kWarmupIters; ++i) run_conv_iter();
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < opt.iters; ++i) run_conv_iter();
    const float total_ms = timer.stop_and_elapsed_ms();

    CHECK_CUDA(cudaMemcpy(y.data(), d_y, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_col));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w_fp16));

    return build_metrics(opt, d, static_cast<double>(total_ms) / opt.iters);
}

// =========================================================================
// Ruta 5 - GPU CUTLASS ImplicitGemmConvolution (Tensor Cores, FP16/BF16)
//
// A diferencia de las rutas 3 (cuDNN) y 4 (WMMA propio), CUTLASS instancia
// la convolucion como una plantilla C++ resuelta en tiempo de compilacion
// (cutlass::conv::kernel::DefaultConv2dFprop) en vez de elegir un algoritmo
// en tiempo de ejecucion. La estructura de esta ruta sigue casi literal el
// ejemplo oficial examples/16_ampere_tensorop_conv2dfprop del repositorio
// github.com/NVIDIA/cutlass (API 2.x, no CuTe/3.x) -- ver el comentario
// junto al #include de CUTLASS mas arriba. Compilado y verificado en GPU
// Ampere+ real contra CUTLASS v2.11.0 -- ver README.md.
//
// *** LAYOUT: NHWC, no NCHW ***
// Toda esta ruta 5 usa cutlass::layout::TensorNHWC, tanto para la activacion
// (N,H,W,C) como para el filtro (que CUTLASS trata con la misma clase de
// layout, interpretando las dimensiones como K,R,S,C -- "KRSC") y para la
// salida (N,outH,outW,K). El resto del archivo (CPU, cuDNN, WMMA) trabaja
// en NCHW/KCRS. Por eso esta ruta:
//   1. Sube x (NCHW) y w (KCRS) a GPU en FP32.
//   2. Los convierte a NHWC/KRSC + FP16 o BF16 con convert_nchw_to_nhwc_kernel
//      (kernel generico de transposicion+cast, parametrizado por las 4
//      dimensiones -- funciona igual para activacion y filtro porque ambos
//      son un arreglo 4D con el eje de canales en la posicion 1).
//   3. Corre CUTLASS sobre esos buffers NHWC/KRSC, con salida NHWC FP32.
//   4. Convierte la salida NHWC -> NCHW con convert_nhwc_to_nchw_float_kernel
//      antes de copiarla a host, para que sea comparable elemento a elemento
//      con y_ref/y_cpu/y_gpu/y_tc/y_wmma (todos NCHW).
// Este paso de conversion es el punto real de riesgo de esta ruta: un error
// de indices en cualquiera de los dos kernels de abajo produciria una salida
// con error alto pero sin fallar la compilacion ni la ejecucion -- por eso
// se comparan explicitamente contra la referencia FP64 y contra la CPU FP32,
// igual que las otras rutas, en vez de asumir que "corrio sin abortar" basta.
// =========================================================================

// Transpone un tensor 4D FP32 de layout "canal en la posicion 1"
// (NCHW para activaciones [N,C,H,W], KCRS para filtros [K,C,R,S]) a layout
// "canal en la posicion 3" (NHWC / KRSC), casteando a la vez a ElementDst
// (cutlass::half_t o cutlass::bfloat16_t). Sirve para ambos casos porque
// solo depende de las 4 dimensiones (D0,D1,D2,D3), no de su significado.
template <typename ElementDst>
__global__ static void convert_nchw_to_nhwc_kernel(
        const float* __restrict__ src, ElementDst* __restrict__ dst,
        int D0, int D1, int D2, int D3) {
    const long long idx   = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(D0) * D1 * D2 * D3;
    if (idx >= total) return;

    // Descomponer idx segun el orden de almacenamiento NCHW/KCRS de src
    // (D1 = canales, la dimension mas "externa" tras D0).
    long long t = idx;
    const int d3 = static_cast<int>(t % D3); t /= D3;
    const int d2 = static_cast<int>(t % D2); t /= D2;
    const int d1 = static_cast<int>(t % D1); t /= D1;
    const int d0 = static_cast<int>(t);

    // Recomponer el indice destino en NHWC/KRSC (D1 = canales pasa a ser la
    // dimension mas "interna").
    const long long dst_idx =
        ((static_cast<long long>(d0) * D2 + d2) * D3 + d3) * D1 + d1;
    dst[dst_idx] = static_cast<ElementDst>(src[idx]);
}

// Inverso de convert_nchw_to_nhwc_kernel, especializado para el caso de la
// salida: NHWC FP32 (ElementOutput de CUTLASS en esta ruta) -> NCHW FP32,
// para que y quede en el mismo layout que las demas rutas.
__global__ static void convert_nhwc_to_nchw_float_kernel(
        const float* __restrict__ src, float* __restrict__ dst,
        int N, int H, int W, int C) {
    const long long idx   = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(N) * H * W * C;
    if (idx >= total) return;

    // src esta en NHWC: descomponer con C como dimension mas interna.
    long long t = idx;
    const int c = static_cast<int>(t % C); t /= C;
    const int w = static_cast<int>(t % W); t /= W;
    const int h = static_cast<int>(t % H); t /= H;
    const int n = static_cast<int>(t);

    // dst en NCHW.
    const long long dst_idx =
        ((static_cast<long long>(n) * C + c) * H + h) * W + w;
    dst[dst_idx] = src[idx];
}

// Implementacion generica de la Ruta 5, parametrizada por el tipo de dato de
// 16 bits de CUTLASS (cutlass::half_t o cutlass::bfloat16_t). Comparten toda
// la logica salvo el tipo de operando; separarla en un template evita tener
// dos copias que se puedan desincronizar (como pasaria si se duplicara a
// mano, igual que benchmark_gpu_tensor_cores_conv/_bf16 en la ruta 3, que si
// estan duplicadas porque alli cada una necesita su propio
// cudnnConvolutionDescriptor_t -- aqui no hay ese impedimento).
//
// La seccion de configuracion de tipos de CUTLASS
// (ThreadblockShape/WarpShape/InstructionShape/NumStages/EpilogueOp) y la
// construccion de Conv2dProblemSize/Arguments de abajo siguen la estructura
// de examples/16_ampere_tensorop_conv2dfprop, verificadas por compilacion y
// ejecucion contra CUTLASS v2.11.0. Con otra version de CUTLASS, los puntos
// con mas probabilidad de no coincidir exactamente son la lista de
// parametros de template de DefaultConv2dFprop, el orden de argumentos del
// constructor de Conv2dProblemSize, y los campos de ImplicitGemm::Arguments
// -- comparar contra el ejemplo real de esa version si algo no coincide.
#if HAVE_CUTLASS
template <typename ElementIO>
static Metrics run_cutlass_conv_impl(const std::vector<float>& x,
                                      const std::vector<float>& w,
                                      std::vector<float>& y,
                                      const Options& opt,
                                      const OutputDims& d,
                                      const char* format_label) {
    using ElementInputA        = ElementIO;
    using ElementInputB        = ElementIO;
    using ElementOutput        = float;
    using ElementAccumulator   = float;
    using ElementComputeEpilogue = float;

    using LayoutInputA = cutlass::layout::TensorNHWC;
    using LayoutInputB = cutlass::layout::TensorNHWC;  // "KRSC", ver nota de layout arriba.
    using LayoutOutput = cutlass::layout::TensorNHWC;

    // Forma de tile / instruccion de ejemplo 16 (Ampere, sm_80). 128x128x32
    // por bloque, 64x64x32 por warp, mma.sync 16x8x16 -- valida para FP16 y
    // BF16 por igual en Ampere (misma instruccion HMMA de 16 bits).
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue>;

    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

    // NumStages=3: pipeline multietapa de Ampere (analogo al cp.async de 3
    // etapas de la ruta 4 de este mismo archivo); Turing (sm_75) usaria 2.
    constexpr int NumStages = 3;

    using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        SwizzleThreadBlock,
        NumStages,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized
    >::Kernel;

    using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    // --- Forma del problema, en las coordenadas NHWC/KRSC de CUTLASS ---
    const cutlass::Tensor4DCoord input_size {opt.N, opt.H, opt.W, opt.C};
    const cutlass::Tensor4DCoord filter_size{opt.K, opt.R, opt.S, opt.C};
    const cutlass::Tensor4DCoord padding    {opt.pad_h, opt.pad_h, opt.pad_w, opt.pad_w};
    const cutlass::MatrixCoord   conv_stride{opt.stride_h, opt.stride_w};
    const cutlass::MatrixCoord   dilation   {opt.dilation_h, opt.dilation_w};

    // El constructor de Conv2dProblemSize usado aqui
    // (input/filter/padding/stride/dilation/mode/split_k_slices, sin
    // output_size explicito) calcula P/Q (alto/ancho de salida)
    // internamente con la misma formula estandar que compute_output_dims()
    // de este archivo. Se valida ese supuesto explicitamente mas abajo
    // (comparando N/P/Q/K contra d) en vez de confiar en que ambas formulas
    // coincidan sin chequeo.
    const cutlass::conv::Conv2dProblemSize problem_size(
        input_size, filter_size, padding, conv_stride, dilation,
        cutlass::conv::Mode::kCrossCorrelation,
        /*split_k_slices=*/1);

    // problem_size.output_size() devuelve int64_t (el conteo total de
    // elementos N*P*Q*K), no un Tensor4DCoord -- la forma 4D se construye
    // directamente desde los campos N/P/Q/K de Conv2dProblemSize.
    const cutlass::Tensor4DCoord cutlass_output_size(
        problem_size.N, problem_size.P, problem_size.Q, problem_size.K);
    if (cutlass_output_size.n() != d.outN || cutlass_output_size.h() != d.outH ||
        cutlass_output_size.w() != d.outW || cutlass_output_size.c() != d.outC) {
        std::cerr << "CUTLASS conv " << format_label << " omitida: "
                  << "output_size() de CUTLASS (" << cutlass_output_size.n() << ","
                  << cutlass_output_size.h() << "," << cutlass_output_size.w() << ","
                  << cutlass_output_size.c() << ") no coincide con la formula de "
                  << "compute_output_dims() de este archivo (" << d.outN << "," << d.outH
                  << "," << d.outW << "," << d.outC << "). Revisar la formula de padding/"
                  << "stride/dilation de Conv2dProblemSize contra el ejemplo oficial.\n";
        return Metrics{};
    }

    // --- Subir x (NCHW) y w (KCRS) en FP32, convertir a NHWC/KRSC ElementIO ---
    float* d_x_fp32 = nullptr;
    float* d_w_fp32 = nullptr;
    ElementInputA* d_x_nhwc = nullptr;
    ElementInputB* d_w_nhwc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x_fp32, x.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w_fp32, w.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x_nhwc, x.size() * sizeof(ElementInputA)));
    CHECK_CUDA(cudaMalloc(&d_w_nhwc, w.size() * sizeof(ElementInputB)));
    CHECK_CUDA(cudaMemcpy(d_x_fp32, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w_fp32, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));

    {
        const int total_x  = static_cast<int>(x.size());
        const int blocks_x = (total_x + kConversionThreads - 1) / kConversionThreads;
        convert_nchw_to_nhwc_kernel<ElementInputA><<<blocks_x, kConversionThreads>>>(
            d_x_fp32, d_x_nhwc, opt.N, opt.C, opt.H, opt.W);
        CHECK_CUDA(cudaGetLastError());

        const int total_w  = static_cast<int>(w.size());
        const int blocks_w = (total_w + kConversionThreads - 1) / kConversionThreads;
        convert_nchw_to_nhwc_kernel<ElementInputB><<<blocks_w, kConversionThreads>>>(
            d_w_fp32, d_w_nhwc, opt.K, opt.C, opt.R, opt.S);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaFree(d_x_fp32));
    CHECK_CUDA(cudaFree(d_w_fp32));

    // Salida NHWC FP32. beta=0 => el tensor "C" del epilogo no se lee: se
    // reutiliza el mismo buffer de salida como tensor_c y tensor_d (patron
    // habitual en los ejemplos de CUTLASS para evitar una allocacion extra).
    ElementOutput* d_y_nhwc = nullptr;
    float*         d_y_nchw = nullptr;
    CHECK_CUDA(cudaMalloc(&d_y_nhwc, y.size() * sizeof(ElementOutput)));
    CHECK_CUDA(cudaMalloc(&d_y_nchw, y.size() * sizeof(float)));

    const cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a(
        d_x_nhwc, LayoutInputA::packed(input_size));
    const cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b(
        d_w_nhwc, LayoutInputB::packed(filter_size));
    const cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c(
        d_y_nhwc, LayoutOutput::packed(cutlass_output_size));
    const cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_d(
        d_y_nhwc, LayoutOutput::packed(cutlass_output_size));

    const ElementComputeEpilogue alpha(1);
    const ElementComputeEpilogue beta(0);

    // VERIFICAR EN PACCA: orden y numero de campos de ImplicitGemm::Arguments
    // (problem_size, tensor_a, tensor_b, tensor_c, tensor_d, {alpha, beta}).
    // Es el mismo orden que usa examples/16_ampere_tensorop_conv2dfprop de
    // memoria, pero es exactamente el tipo de detalle que cambia entre
    // versiones de CUTLASS sin avisar en tiempo de compilacion (los campos
    // son todos convertibles entre si via inicializacion agregada).
    typename ImplicitGemm::Arguments arguments{
        problem_size,
        tensor_a,
        tensor_b,
        tensor_c,
        tensor_d,
        {alpha, beta}
    };

    ImplicitGemm implicit_gemm_op;

    // can_implement() es la forma de CUTLASS de decir "esta combinacion de
    // forma/tipo/algoritmo no esta soportada" en tiempo de ejecucion -- a
    // diferencia de la ruta 4 (WMMA propio), que valida divisibilidad a mano
    // antes de lanzar el kernel, aqui se deja que la propia libreria decida
    // y simplemente se omite la ruta con aviso si dice que no puede, en vez
    // de abortar todo el binario (CHECK_CUTLASS aborta; aqui no se usa esa
    // macro a proposito para este chequeo puntual).
    cutlass::Status status = implicit_gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS conv " << format_label << " omitida: can_implement()"
                     " devolvio " << cutlass::cutlassGetStatusString(status)
                  << " para N=" << opt.N << " C=" << opt.C << " H=" << opt.H
                  << " W=" << opt.W << " K=" << opt.K << " R=" << opt.R
                  << " S=" << opt.S << ".\n";
        CHECK_CUDA(cudaFree(d_x_nhwc));
        CHECK_CUDA(cudaFree(d_w_nhwc));
        CHECK_CUDA(cudaFree(d_y_nhwc));
        CHECK_CUDA(cudaFree(d_y_nchw));
        return Metrics{};
    }

    const size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    void* d_workspace = nullptr;
    if (workspace_size > 0) CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));

    CHECK_CUTLASS(implicit_gemm_op.initialize(arguments, d_workspace));

    for (int i = 0; i < kWarmupIters; ++i) {
        CHECK_CUTLASS(implicit_gemm_op());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < opt.iters; ++i) {
        CHECK_CUTLASS(implicit_gemm_op());
    }
    const float total_ms = timer.stop_and_elapsed_ms();

    // Convertir salida NHWC -> NCHW fuera del intervalo medido (ver nota de
    // layout al inicio de esta ruta).
    const int y_count     = static_cast<int>(y.size());
    const int conv_blocks = (y_count + kConversionThreads - 1) / kConversionThreads;
    convert_nhwc_to_nchw_float_kernel<<<conv_blocks, kConversionThreads>>>(
        d_y_nhwc, d_y_nchw, d.outN, d.outH, d.outW, d.outC);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(y.data(), d_y_nchw, y.size() * sizeof(float), cudaMemcpyDeviceToHost));

    if (d_workspace) CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(d_y_nchw));
    CHECK_CUDA(cudaFree(d_y_nhwc));
    CHECK_CUDA(cudaFree(d_x_nhwc));
    CHECK_CUDA(cudaFree(d_w_nhwc));

    return build_metrics(opt, d, static_cast<double>(total_ms) / opt.iters);
}

// Instancia FP16 de la Ruta 5. Nombre pedido explicitamente para esta ruta
// (en vez de benchmark_gpu_cutlass_conv, que seguiria mas de cerca el
// prefijo benchmark_gpu_* de las rutas 2-4).
static Metrics run_cutlass_conv(const std::vector<float>& x,
                                 const std::vector<float>& w,
                                 std::vector<float>& y,
                                 const Options& opt,
                                 const OutputDims& d) {
    return run_cutlass_conv_impl<cutlass::half_t>(x, w, y, opt, d, "FP16");
}

// Instancia BF16 de la Ruta 5, analoga a benchmark_gpu_tensor_cores_conv_bf16
// en la ruta 3 (mismo patron: una funcion por formato). Requiere Ampere o
// superior -- ya validado en run_experiment_float antes de llamar aqui,
// igual que para la ruta 3 BF16.
static Metrics run_cutlass_conv_bf16(const std::vector<float>& x,
                                      const std::vector<float>& w,
                                      std::vector<float>& y,
                                      const Options& opt,
                                      const OutputDims& d) {
    return run_cutlass_conv_impl<cutlass::bfloat16_t>(x, w, y, opt, d, "BF16");
}
#endif  // HAVE_CUTLASS

// =========================================================================
// Reportes finales
// =========================================================================

static void print_float_report(const Options& opt,
                                const Metrics& cpu,
                                const Metrics& gpu,
                                const Metrics& tc,
                                const Metrics& wmma,
                                const Metrics& tc_bf16,
                                const ErrorMetrics& cpu_err,
                                const ErrorMetrics& gpu_err,
                                const ErrorMetrics& tc_err,
                                const ErrorMetrics& wmma_err,
                                const ErrorMetrics& tc_bf16_err,
                                const ErrorMetrics& gpu_vs_cpu_err,
                                const ErrorMetrics& tc_vs_cpu_err,
                                const ErrorMetrics& wmma_vs_cpu_err,
                                const ErrorMetrics& wmma_vs_tc_err,
                                const ErrorMetrics& tc_bf16_vs_cpu_err,
                                // Ruta 5 (CUTLASS) -- parametros agregados al final de la
                                // lista para no reordenar/tocar los de las rutas 1-4.
                                const Metrics& cutlass_fp16,
                                const Metrics& cutlass_bf16,
                                const ErrorMetrics& cutlass_fp16_err,
                                const ErrorMetrics& cutlass_fp16_vs_cpu_err,
                                const ErrorMetrics& cutlass_bf16_err,
                                const ErrorMetrics& cutlass_bf16_vs_cpu_err) {
    const bool show_fp16 = (opt.tc_format == TensorCoreFormat::FP16 ||
                            opt.tc_format == TensorCoreFormat::Both);
    const bool show_bf16 = (opt.tc_format == TensorCoreFormat::BF16 ||
                            opt.tc_format == TensorCoreFormat::Both);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=========== RESULTADOS CONV 2D FP32 ===========\n";
    std::cout << "CPU im2col+BLAS - tiempo    : " << cpu.ms << " ms\n";
    std::cout << "CPU im2col+BLAS - rend.     : " << cpu.gflops << " GFLOP/s ("
              << cpu.tflops << " TFLOP/s)\n";
    std::cout << "Error max abs vs FP64       : " << cpu_err.max_abs << "\n";
    std::cout << "Error relativo L2 vs FP64   : " << cpu_err.rel_l2 << "\n\n";

    std::cout << "GPU cuDNN FP32 escalar      : math type CUDNN_FMA_MATH (TF32 desactivado)\n";
    std::cout << "GPU cuDNN FP32 - tiempo     : " << gpu.ms << " ms\n";
    std::cout << "GPU cuDNN FP32 - rend.      : " << gpu.gflops << " GFLOP/s ("
              << gpu.tflops << " TFLOP/s)\n";
    std::cout << "Speedup vs CPU              : " << cpu.ms / gpu.ms << "x\n";
    std::cout << "Error max abs vs FP64       : " << gpu_err.max_abs << "\n";
    std::cout << "Error relativo L2 vs FP64   : " << gpu_err.rel_l2 << "\n";
    std::cout << "Error max abs vs CPU FP32   : " << gpu_vs_cpu_err.max_abs << "\n";
    std::cout << "Error rel L2 vs CPU FP32    : " << gpu_vs_cpu_err.rel_l2 << "\n\n";

    if (show_fp16) {
        std::cout << "GPU Tensor Core - tiempo    : " << tc.ms << " ms\n";
        std::cout << "GPU Tensor Core - rend.     : " << tc.gflops << " GFLOP/s ("
                  << tc.tflops << " TFLOP/s)\n";
        std::cout << "Speedup TC vs CPU           : " << cpu.ms / tc.ms << "x\n";
        std::cout << "Speedup TC vs FP32 escalar  : " << gpu.ms / tc.ms << "x\n";
        std::cout << "Error max abs vs FP64       : " << tc_err.max_abs << "\n";
        std::cout << "Error relativo L2 vs FP64   : " << tc_err.rel_l2 << "\n";
        std::cout << "Error max abs vs CPU FP32   : " << tc_vs_cpu_err.max_abs << "\n";
        std::cout << "Error rel L2 vs CPU FP32    : " << tc_vs_cpu_err.rel_l2 << "\n\n";
    }

    if (show_bf16) {
        std::cout << "GPU Tensor Core BF16 - tiempo    : " << tc_bf16.ms << " ms\n";
        std::cout << "GPU Tensor Core BF16 - rend.     : " << tc_bf16.gflops << " GFLOP/s ("
                  << tc_bf16.tflops << " TFLOP/s)\n";
        std::cout << "Speedup TC BF16 vs CPU           : " << cpu.ms / tc_bf16.ms << "x\n";
        std::cout << "Speedup TC BF16 vs FP32 escalar  : " << gpu.ms / tc_bf16.ms << "x\n";
        std::cout << "Error max abs vs FP64            : " << tc_bf16_err.max_abs << "\n";
        std::cout << "Error relativo L2 vs FP64        : " << tc_bf16_err.rel_l2 << "\n";
        std::cout << "Error max abs vs CPU FP32        : " << tc_bf16_vs_cpu_err.max_abs << "\n";
        std::cout << "Error rel L2 vs CPU FP32         : " << tc_bf16_vs_cpu_err.rel_l2 << "\n\n";
    }

    if (wmma.ms > 0.0) {
        std::cout << "GPU WMMA custom - tiempo    : " << wmma.ms << " ms\n";
        std::cout << "GPU WMMA custom - rend.     : " << wmma.gflops << " GFLOP/s ("
                  << wmma.tflops << " TFLOP/s)\n";
        std::cout << "Speedup WMMA vs CPU         : " << cpu.ms / wmma.ms << "x\n";
        std::cout << "Speedup WMMA vs FP32 escalar: " << gpu.ms / wmma.ms << "x\n";
        std::cout << "Error max abs vs FP64       : " << wmma_err.max_abs << "\n";
        std::cout << "Error relativo L2 vs FP64   : " << wmma_err.rel_l2 << "\n";
        std::cout << "Error max abs vs CPU FP32   : " << wmma_vs_cpu_err.max_abs << "\n";
        std::cout << "Error rel L2 vs CPU FP32    : " << wmma_vs_cpu_err.rel_l2 << "\n";
        // WMMA custom (Ruta 4) es intrinsecamente FP16; la comparacion contra
        // cuDNN TC solo tiene referencia valida cuando esa ruta se ejecuto.
        if (show_fp16) {
            std::cout << "Speedup WMMA vs cuDNN TC    : " << tc.ms / wmma.ms << "x\n";
            std::cout << "Error max abs vs cuDNN TC   : " << wmma_vs_tc_err.max_abs << "\n";
            std::cout << "Error relativo L2 vs TC     : " << wmma_vs_tc_err.rel_l2 << "\n";
        }
    } else {
        std::cout << "GPU WMMA custom             : omitida (alineacion no cumplida)\n";
    }

    // Ruta 5 (CUTLASS ImplicitGemmConvolution). Solo se corrio si --cutlass
    // estaba activo; cutlass_fp16.ms/cutlass_bf16.ms quedan en 0 si la ruta
    // no corrio (flag desactivado, formato no pedido via --tc-format, o
    // can_implement() la rechazo en tiempo de ejecucion -- ver run_cutlass_conv_impl).
    if (opt.run_cutlass && show_fp16) {
        if (cutlass_fp16.ms > 0.0) {
            std::cout << "\nGPU CUTLASS ImplicitGemm FP16 - tiempo   : " << cutlass_fp16.ms << " ms\n";
            std::cout << "GPU CUTLASS ImplicitGemm FP16 - rend.    : " << cutlass_fp16.gflops
                      << " GFLOP/s (" << cutlass_fp16.tflops << " TFLOP/s)\n";
            std::cout << "Speedup CUTLASS FP16 vs CPU              : " << cpu.ms / cutlass_fp16.ms << "x\n";
            // Comparacion (c) del README: libreria FP32 (ruta 2) vs. CUTLASS (ruta 5).
            std::cout << "Speedup CUTLASS FP16 vs FP32 escalar     : " << gpu.ms / cutlass_fp16.ms << "x\n";
            std::cout << "Error max abs vs FP64                    : " << cutlass_fp16_err.max_abs << "\n";
            std::cout << "Error relativo L2 vs FP64                : " << cutlass_fp16_err.rel_l2 << "\n";
            std::cout << "Error max abs vs CPU FP32                : " << cutlass_fp16_vs_cpu_err.max_abs << "\n";
            std::cout << "Error rel L2 vs CPU FP32                 : " << cutlass_fp16_vs_cpu_err.rel_l2 << "\n";
            std::cout << "Speedup CUTLASS FP16 vs cuDNN TC         : " << tc.ms / cutlass_fp16.ms << "x\n";
        } else {
            std::cout << "\nGPU CUTLASS ImplicitGemm FP16            : omitida (ver aviso mas arriba)\n";
        }
    }
    if (opt.run_cutlass && show_bf16) {
        if (cutlass_bf16.ms > 0.0) {
            std::cout << "\nGPU CUTLASS ImplicitGemm BF16 - tiempo   : " << cutlass_bf16.ms << " ms\n";
            std::cout << "GPU CUTLASS ImplicitGemm BF16 - rend.    : " << cutlass_bf16.gflops
                      << " GFLOP/s (" << cutlass_bf16.tflops << " TFLOP/s)\n";
            std::cout << "Speedup CUTLASS BF16 vs CPU              : " << cpu.ms / cutlass_bf16.ms << "x\n";
            std::cout << "Speedup CUTLASS BF16 vs FP32 escalar     : " << gpu.ms / cutlass_bf16.ms << "x\n";
            std::cout << "Error max abs vs FP64                    : " << cutlass_bf16_err.max_abs << "\n";
            std::cout << "Error relativo L2 vs FP64                : " << cutlass_bf16_err.rel_l2 << "\n";
            std::cout << "Error max abs vs CPU FP32                : " << cutlass_bf16_vs_cpu_err.max_abs << "\n";
            std::cout << "Error rel L2 vs CPU FP32                 : " << cutlass_bf16_vs_cpu_err.rel_l2 << "\n";
            if (show_bf16 && tc_bf16.ms > 0.0) {
                std::cout << "Speedup CUTLASS BF16 vs cuDNN TC BF16    : " << tc_bf16.ms / cutlass_bf16.ms << "x\n";
            }
        } else {
            std::cout << "\nGPU CUTLASS ImplicitGemm BF16            : omitida (ver aviso mas arriba)\n";
        }
    }

    std::cout << "===============================================\n";
}

static void print_double_report(const Metrics& cpu,
                                 const Metrics& gpu,
                                 const ErrorMetrics& gpu_err) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=========== RESULTADOS CONV 2D FP64 ===========\n";
    std::cout << "CPU im2col+BLAS - tiempo    : " << cpu.ms << " ms\n";
    std::cout << "CPU im2col+BLAS - rend.     : " << cpu.gflops << " GFLOP/s ("
              << cpu.tflops << " TFLOP/s)\n\n";
    print_reference_comparison("GPU cuDNN clasico  ", gpu, cpu.ms, gpu_err);
    std::cout << "===============================================\n";
}

// =========================================================================
// Orquestacion de experimentos
// =========================================================================

static void run_experiment_float(const Options& opt) {
    const bool want_fp16 = (opt.tc_format == TensorCoreFormat::FP16 ||
                            opt.tc_format == TensorCoreFormat::Both);
    const bool want_bf16 = (opt.tc_format == TensorCoreFormat::BF16 ||
                            opt.tc_format == TensorCoreFormat::Both);

    if (want_bf16 && !active_device_supports_bf16_tensor_cores()) {
        std::cerr << "La ruta Tensor Core BF16 requiere arquitectura Ampere o superior"
                     " (compute capability >= 8.0)." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const OutputDims d      = compute_output_dims(opt);
    const size_t x_count    = static_cast<size_t>(opt.N) * opt.C * opt.H * opt.W;
    const size_t w_count    = static_cast<size_t>(opt.K) * opt.C * opt.R * opt.S;
    const size_t y_count    = static_cast<size_t>(d.outN) * d.outC * d.outH * d.outW;

    std::vector<float> x(x_count), w(w_count);
    std::vector<float> y_cpu(y_count, 0.0f), y_gpu(y_count, 0.0f);
    std::vector<float> y_tc(y_count, 0.0f),  y_wmma(y_count, 0.0f);
    std::vector<float> y_tc_bf16(y_count, 0.0f);
    // Ruta 5 (CUTLASS): buffers de salida separados, mismo patron que y_tc/y_tc_bf16.
    std::vector<float> y_cutlass_fp16(y_count, 0.0f), y_cutlass_bf16(y_count, 0.0f);
    initialize_matrix_float(x);
    initialize_matrix_float(w);

    // Referencia FP64 (ground truth): las mismas entradas casteadas a double y
    // la ruta im2col + cblas_dgemm en una sola pasada, mismo patron que GEMM.
    std::vector<double> y_ref(y_count, 0.0);
    {
        std::vector<double> x_d(x_count), w_d(w_count);
        for (size_t i = 0; i < x_count; ++i) x_d[i] = static_cast<double>(x[i]);
        for (size_t i = 0; i < w_count; ++i) w_d[i] = static_cast<double>(w[i]);

        const int M    = opt.K;
        const int Ncol = d.outH * d.outW;
        const int Kcol = opt.C * opt.R * opt.S;
        std::vector<double> col(static_cast<size_t>(Kcol) * Ncol);
        for (int n = 0; n < opt.N; ++n) {
            const double* xn = x_d.data() + static_cast<size_t>(n) * opt.C * opt.H * opt.W;
            double*       yn = y_ref.data() + static_cast<size_t>(n) * opt.K * d.outH * d.outW;
            im2col_double(xn, col.data(), opt, d);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, Ncol, Kcol,
                        1.0, w_d.data(), Kcol,
                        col.data(), Kcol,
                        0.0, yn, Ncol);
        }
    }

    const Metrics cpu  = benchmark_cpu_float(x, w, y_cpu, opt, d);
    const Metrics gpu  = benchmark_gpu_cudnn_float(x, w, y_gpu, opt, d);
    const Metrics wmma = benchmark_gpu_wmma_conv(x, w, y_wmma, opt, d);

    Metrics tc{}, tc_bf16{};
    ErrorMetrics tc_err{}, tc_vs_cpu{}, tc_bf16_err{}, tc_bf16_vs_cpu{};

    if (want_fp16) {
        tc = benchmark_gpu_tensor_cores_conv(x, w, y_tc, opt, d);
        tc_err     = compare_fp64_ref_vs_fp32(y_ref, y_tc);
        tc_vs_cpu  = compare_float_vectors(y_cpu, y_tc);
    }
    if (want_bf16) {
        tc_bf16 = benchmark_gpu_tensor_cores_conv_bf16(x, w, y_tc_bf16, opt, d);
        tc_bf16_err    = compare_fp64_ref_vs_fp32(y_ref, y_tc_bf16);
        tc_bf16_vs_cpu = compare_float_vectors(y_cpu, y_tc_bf16);
    }

    // Ruta 5 (CUTLASS), opt-in con --cutlass; reutiliza want_fp16/want_bf16
    // (derivados de --tc-format) para decidir que formato(s) correr, igual
    // que la ruta 3.
    Metrics cutlass_fp16{}, cutlass_bf16{};
    ErrorMetrics cutlass_fp16_err{}, cutlass_fp16_vs_cpu{};
    ErrorMetrics cutlass_bf16_err{}, cutlass_bf16_vs_cpu{};
    if (opt.run_cutlass) {
#if HAVE_CUTLASS
        if (want_fp16) {
            cutlass_fp16 = run_cutlass_conv(x, w, y_cutlass_fp16, opt, d);
            if (cutlass_fp16.ms > 0.0) {
                cutlass_fp16_err    = compare_fp64_ref_vs_fp32(y_ref, y_cutlass_fp16);
                cutlass_fp16_vs_cpu = compare_float_vectors(y_cpu, y_cutlass_fp16);
            }
        }
        if (want_bf16) {
            cutlass_bf16 = run_cutlass_conv_bf16(x, w, y_cutlass_bf16, opt, d);
            if (cutlass_bf16.ms > 0.0) {
                cutlass_bf16_err    = compare_fp64_ref_vs_fp32(y_ref, y_cutlass_bf16);
                cutlass_bf16_vs_cpu = compare_float_vectors(y_cpu, y_cutlass_bf16);
            }
        }
#else
        std::cerr << "Se pidio --cutlass pero el binario se compilo sin CUTLASS disponible"
                     " en el include path.\n"
                  << "Recompila agregando -I$CUTLASS_DIR/include, apuntando a un checkout de"
                     " github.com/NVIDIA/cutlass (serie 2.x) -- ver REQUIREMENTS.md y"
                     " Fase_2/Convolution/README.md." << std::endl;
        std::exit(EXIT_FAILURE);
#endif
    }

    // Metricas primarias: contra el ground truth FP64 (objetivo especifico #3).
    const ErrorMetrics cpu_err       = compare_fp64_ref_vs_fp32(y_ref, y_cpu);
    const ErrorMetrics gpu_err       = compare_fp64_ref_vs_fp32(y_ref, y_gpu);
    const ErrorMetrics wmma_err      = compare_fp64_ref_vs_fp32(y_ref, y_wmma);

    // Metricas secundarias: contra la CPU FP32 (trazabilidad con corridas previas).
    const ErrorMetrics gpu_vs_cpu    = compare_float_vectors(y_cpu, y_gpu);
    const ErrorMetrics wmma_vs_cpu   = compare_float_vectors(y_cpu, y_wmma);
    const ErrorMetrics wmma_vs_tc    = compare_float_vectors(y_tc,  y_wmma);

    print_float_report(opt, cpu, gpu, tc, wmma, tc_bf16,
                       cpu_err, gpu_err, tc_err, wmma_err, tc_bf16_err,
                       gpu_vs_cpu, tc_vs_cpu, wmma_vs_cpu, wmma_vs_tc, tc_bf16_vs_cpu,
                       cutlass_fp16, cutlass_bf16,
                       cutlass_fp16_err, cutlass_fp16_vs_cpu,
                       cutlass_bf16_err, cutlass_bf16_vs_cpu);
}

static void run_experiment_double(const Options& opt) {
    const OutputDims d      = compute_output_dims(opt);
    const size_t x_count    = static_cast<size_t>(opt.N) * opt.C * opt.H * opt.W;
    const size_t w_count    = static_cast<size_t>(opt.K) * opt.C * opt.R * opt.S;
    const size_t y_count    = static_cast<size_t>(d.outN) * d.outC * d.outH * d.outW;

    std::vector<double> x(x_count), w(w_count);
    std::vector<double> y_cpu(y_count, 0.0), y_gpu(y_count, 0.0);
    initialize_matrix_double(x);
    initialize_matrix_double(w);

    const Metrics cpu = benchmark_cpu_double(x, w, y_cpu, opt, d);
    const Metrics gpu = benchmark_gpu_cudnn_double(x, w, y_gpu, opt, d);

    const ErrorMetrics gpu_err = compare_double_vectors(y_cpu, y_gpu);

    print_double_report(cpu, gpu, gpu_err);
}

static void run_benchmark(const Options& opt) {
    const OutputDims d = compute_output_dims(opt);
    std::cout << "================== CONFIGURACION ==================\n";
    std::cout << "Precision                  : "
              << (opt.use_double ? "FP64 (double)" : "FP32 (float)") << "\n";
    std::cout << "Entrada (N,C,H,W)          : "
              << opt.N << ", " << opt.C << ", " << opt.H << ", " << opt.W << "\n";
    std::cout << "Filtro  (K,C,R,S)          : "
              << opt.K << ", " << opt.C << ", " << opt.R << ", " << opt.S << "\n";
    std::cout << "Salida  (N,K,outH,outW)    : "
              << d.outN << ", " << d.outC << ", " << d.outH << ", " << d.outW << "\n";
    std::cout << "Padding  (h,w)             : " << opt.pad_h << ", " << opt.pad_w << "\n";
    std::cout << "Stride   (h,w)             : " << opt.stride_h << ", " << opt.stride_w << "\n";
    std::cout << "Dilation (h,w)             : " << opt.dilation_h << ", " << opt.dilation_w << "\n";
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
