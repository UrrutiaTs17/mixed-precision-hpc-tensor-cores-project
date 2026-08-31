// Compilar con:
// nvcc -std=c++17 stencil_tensor_activation.cu -o stencil_tc \
//      -gencode arch=compute_80,code=sm_80
//
// Ejecutar:
// ./stencil_tc --nx 4096 --ny 4096 --iters 20 --tc both
//
// Validar Tensor Cores con Nsight Compute:
// ncu --kernel-name regex:.*stencil2d_wmma_kernel.* \
//     --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,\
//sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed \
//     ./stencil_tc --nx 4096 --ny 4096 --iters 20 --tc fp16
//
// Este programa compara tres rutas para un stencil 2D de 5 puntos:
// 1. CPU FP32 serial como referencia numerica.
// 2. GPU CUDA FP32 clasico, sin Tensor Cores.
// 3. GPU Tensor Core con WMMA: entradas FP16/BF16 y salida/acumulacion FP32.
//
// La ruta WMMA expresa cada tile interior 16x16 como DOS operaciones MMA:
//
//     Y = X H + V X
//
// donde X es el tile del estado, H es tridiagonal con el coeficiente de centro
// en la diagonal y el de vecino en las dos subdiagonales (desplazamiento en x,
// multiplicacion por la derecha), y V lleva el coeficiente de vecino en las dos
// subdiagonales y cero en la diagonal (desplazamiento en y, multiplicacion por
// la izquierda). cn/cc son los coeficientes del operador activo (ver OpMode:
// stress da cn=0.25, cc=-1.0; diffusive da cn=alpha, cc=1-4*alpha).
// H y V solo alcanzan a las vecinas internas al tile; las cuatro bandas
// exteriores (16 valores por lado, sin esquinas diagonales, que el stencil de 5
// puntos no usa) se cargan aparte y se suman en FP32 tras el mma.
// Carga por tile completo: 256 + 4*16 = 320 valores de 16 bits.
//
// Fase 3 (este archivo): a diferencia de Fase_2/Stencil/stencil_tensor_activation.cu
// -que congela la validacion de activacion de Tensor Cores y relanza --iters veces
// la misma operacion sobre el MISMO buffer de entrada, valido solo para medir
// throughput-, aqui las tres rutas encadenan genuinamente salida(i) -> entrada(i+1)
// para poder cuantificar drift numerico acumulado a traves de iteraciones reales.
// Reutiliza Fase_2/common.cuh por ruta relativa (no lo duplica). La suma
// compensada Kahan queda para una entrega posterior de Fase 3.

#include <mma.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

// El header de telemetria queda a nivel global porque incluye <nvml.h> cuando
// el sbatch habilita NVML; las declaraciones C de NVML no deben caer dentro
// del namespace anonimo de este archivo.
#include "tools/power_sampling.h"

namespace {

#include "../../Fase_2/common.cuh"

using namespace nvcuda;

constexpr int kTile = 16;
constexpr int kWarpThreads = 32;
constexpr int kWarmupIters = 3;
constexpr int kConversionThreads = 256;
// occupancy: con 1 warp/bloque el techo de 32 CTAs/SM del A100 fija 32
// warps/SM (50%) pese a que registros (64 warps/SM) y shared lo permitirian.
// Con 4 warps/bloque los CTAs bajan de 1.05M a 262144. Los cuatro warps ya no
// toman un tile 16x16 cada uno: cooperan sobre UN tile 16x(4*16) -- ver kTileW
// y el camino ancho del kernel.
constexpr int kWarpsPerBlock = 4;

// Desplazamiento de alineacion de los buffers del dominio 2D completo.
//
// El interior arranca en x = 1, asi que la primera celda interior de la fila y
// es el elemento idx2d(1, y, nx) = y*nx + 1: IMPAR. Con el puntero de
// cudaMalloc (alineado a 256 B) la direccion de esa celda es
//
//     A(y) = base + sizeof(T) * (y*nx + 1)
//
// y ningun acceso vectorizado es representable ahi: para T = __half hace falta
// A(y) % 16 == 0, es decir y*nx + 1 == 0 (mod 8), imposible con nx multiplo
// de 8. Desplazando el puntero LOGICO en kAlignOffsetElems elementos,
//
//     A(y) = base + sizeof(T) * (kAlignOffsetElems + y*nx + 1)
//
// y la condicion pasa a ser kAlignOffsetElems + 1 == 0 (mod V), con V el
// numero de elementos que caben en 16 B:
//
//     T = __half / __nv_bfloat16 (2 B) -> V = 8 -> 15 + 1 = 16 == 0 (mod 8) OK
//     T = float                  (4 B) -> V = 4 -> 15 + 1 = 16 == 0 (mod 4) OK
//
// El mismo 15 sirve para los dos tamanos porque el desplazamiento se cuenta en
// ELEMENTOS del tipo de cada buffer (30 B en los de 16 bits, 60 B en los FP32),
// no en bytes. Requiere nx multiplo de 8 (2048/4096/16384 lo son); con otro nx
// el desplazamiento sigue siendo CORRECTO -- es solo un puntero dentro de una
// asignacion con padding suficiente -- pero deja de alinear, y la etapa de
// vectorizacion debera comprobarlo antes de emitir accesos anchos.
//
// nx NO se toca: la alineacion se consigue moviendo el origen del buffer, no
// el stride logico, de modo que idx2d(x, y, nx) y toda la topologia del
// dominio quedan intactas (el kernel no se entera).
// Padding de los TILES DE SHARED (no confundir con kAlignPadElems, que es el de
// los buffers GLOBALES).
//
// Diagnostico del job 6730 (ncu, 4096^2): el mayor stall del kernel WMMA no es
// esperar datos de memoria global (long_scoreboard 9.00, MENOS que el 15.61 de
// GPU_FP32) sino esperar sitio en la cola de la unidad de memoria interna
// (mio_throttle 14.93 contra 0.10 de GPU_FP32, 150x). La causa son los
// conflictos de banco en shared: 1.1 M en lectura y 566 k en escritura, contra
// 9.5 k y 1.9 k de GPU_FP32.
//
// Origen: shared tiene 32 bancos de 4 B. Con ldm = kTile las filas del tile
// quedan a 32 B (T de 16 bits) o 64 B (float), es decir 8 o 16 bancos, y las
// filas 0/4/8/12 -- u 0/2/4/... en float -- arrancan en el MISMO banco. La
// lectura de fragmentos de wmma toca todas las filas a la vez y se serializa.
//
// Separar las filas mas de lo que ocupan desplaza ese arranque y alarga el
// periodo de repeticion. Restriccion dura de wmma: load_matrix_sync y
// store_matrix_sync exigen que ldm * sizeof(elemento) sea multiplo de 16 B,
// luego kLdX debe ser multiplo de 8 (T de 16 bits) y kLdF multiplo de 4
// (float). Valores utiles: kPadTC en {0, 8, 16, 24}, kPadF32 en {0, 4, 8, 12}.
//
// 0/0 reproduce el comportamiento actual BYTE A BYTE: el padding solo cambia
// donde vive cada dato en shared, no que dato es ni en que orden se opera.
// Se dejan como macros para poder barrer valores con -D sin editar el fuente.
#ifndef STENCIL_PAD_TC
#define STENCIL_PAD_TC 0
#endif
#ifndef STENCIL_PAD_F32
#define STENCIL_PAD_F32 0
#endif
constexpr int kPadTC = STENCIL_PAD_TC;
constexpr int kPadF32 = STENCIL_PAD_F32;
constexpr int kLdX = kTile + kPadTC;    // ldm de x_tile, en elementos de T
constexpr int kLdF = kTile + kPadF32;   // ldm de out_tile y comp_tile, en floats
static_assert(kLdX % 8 == 0, "kLdX*sizeof(T) debe ser multiplo de 16 B (wmma)");
static_assert(kLdF % 4 == 0, "kLdF*sizeof(float) debe ser multiplo de 16 B (wmma)");

constexpr size_t kAlignOffsetElems = 15;
// Capacidad fisica extra por buffer. PAD >= OFFSET para que el ultimo elemento
// logico (indice count-1 visto desde el puntero desplazado) siga dentro de la
// asignacion.
constexpr size_t kAlignPadElems = 16;

enum class TensorCoreMode {
    FP16,
    BF16,
    Both
};

// Politica de compensacion del redondeo de almacenamiento a 16 bits en las
// rutas WMMA. Se resuelve en tiempo de COMPILACION (parametro de plantilla del
// kernel, ver compensated_store / stencil2d_wmma_kernel): ninguna de las tres
// paga ramas de las otras dos.
//
//   Off     (--kahan off, por defecto): sin compensacion. Comportamiento
//           historico, byte a byte.
//   Local   (--kahan on): Kahan por celda. comp[idx] guarda el residuo que
//           dejo la celda idx y se reincorpora cuando ESA MISMA celda vuelve
//           a escribir. Comportamiento historico, byte a byte.
//   Spatial (--spatial-comp on): error feedback espacial. Cada celda
//           reincorpora los residuos de sus 4 vecinas Y el propio ANTES de la
//           suma, no solo el propio. Motivacion: en este stencil el valor de
//           una celda se calcula leyendo 5 celdas VECINAS, cada una cargando
//           el residuo que dejo su propia escritura; con compensacion Local
//           ese residuo entra a la suma sin compensar (comp[idx] solo conoce
//           la historia de idx), asi que el error se propaga espacialmente
//           mientras la compensacion es puramente local. Ver la derivacion en
//           compensated_store y el costo en memoria en
//           benchmark_gpu_tensor_core_stencil.
enum class CompMode {
    Off,
    Local,
    Spatial
};

// Operador discreto de 5 puntos. Los dos escalares (coeficiente de vecino y
// coeficiente de centro) son lo UNICO que distingue las dos variantes: kernels,
// rutas, metricas, checkpoints y telemetria son identicos en ambas.
//
//   Stress    (--op-mode stress, por defecto): out = 0.25*(u+d+l+r) - center.
//             Simbolo de von Neumann g(kx,ky) = 0.5*(cos kx + cos ky) - 1, con
//             g(pi,pi) = -2: el modo Nyquist se duplica en cada iteracion. NO es
//             una ecuacion de calor, y su divergencia es INTENCIONAL -- es el
//             fenomeno medido en las fases 2 y 3 (horizonte de overflow, n*).
//             Esta formula no se modifica; el operador difusivo se agrega al
//             lado, no en su lugar.
//   Diffusive (--op-mode diffusive): out = alpha*(u+d+l+r) + (1-4*alpha)*center.
//             g(kx,ky) = 1 - 4*alpha*[sin^2(kx/2) + sin^2(ky/2)]; con el alpha
//             por defecto g(pi,pi) = -0.5 y todos los modos decaen. Es la
//             variante que hace construible el eje de ERROR del frente de Pareto
//             tridimensional: sobre el operador de estres, rel_l2/rel_linf/
//             max_abs quedan en NaN porque toda ruta de precision reducida
//             desborda antes de terminar el horizonte de la campana.
//
// Por que alpha = 3/16 y no 1/8: con alpha = 1/8 el simbolo da
// g(pi,pi) = 1 - 8*alpha = 0 EXACTAMENTE, lo que aniquilaria el modo Nyquist en
// una sola iteracion -- justo el canal por el que se propaga el ruido de
// redondeo que la campana quiere medir.
//
// Los coeficientes de ambas variantes (0.25/-1.0 y 0.1875/0.25) necesitan 2 bits
// de significando, asi que son exactamente representables en FP16, BF16, FP32 y
// FP64: el coeficiente en si no aporta error de representacion, y el error
// medido no se confunde con el de discretizar la constante. Esto NO significa
// que la evaluacion del stencil sea exacta -- las conversiones de ALMACENAMIENTO
// a 16 bits siguen redondeando en cada iteracion, y ese redondeo es precisamente
// el objeto de estudio.
enum class OpMode {
    Stress,
    Diffusive
};

// Condicion inicial.
//   Legacy   (--ci-mode legacy, por defecto): initialize_grid, la CI historica
//            de las fases 1-3. Comportamiento identico al previo.
//   Monomode (--ci-mode monomode): un unico modo propio del Laplaciano discreto
//            (ver initialize_grid_monomode). Necesaria bajo el operador
//            difusivo: la CI legacy es de muy baja frecuencia -- sus modos
//            dominantes son sin(0.01*x) y cos(0.01*y), con k = 0.01 rad/celda
//            FIJO, independiente de la malla -- y decae solo ~1.2 % en 640
//            iteraciones, dejando una solucion cuasi-invariante: un "caso
//            difusivo" degenerado del que no se aprende nada.
enum class CiMode {
    Legacy,
    Monomode
};

static const char* op_mode_label(OpMode mode) {
    return (mode == OpMode::Diffusive) ? "diffusive" : "stress";
}

static const char* ci_mode_label(CiMode mode) {
    return (mode == CiMode::Monomode) ? "monomode" : "legacy";
}

// Forma en que el bucle iterativo de las rutas WMMA entrega el trabajo a la GPU.
// NO cambia el kernel, la formulacion Y = X H + V X, los coeficientes, las
// bandas, la compensacion ni el ping-pong: solo cambia quien paga el costo de
// CPU de poner cada iteracion en el stream.
//
//   Normal (--execution-mode normal, por defecto): un cudaLaunchKernel por
//          iteracion, igual que siempre. Comportamiento historico byte a byte.
//          Es la unica que instrumenta un par de eventos POR ITERACION, asi que
//          es la ruta de analisis fino (t kernel/iter, t no atribuido/iter).
//   Graph  (--execution-mode graph, o --cuda-graph): las iteraciones que no
//          piden checkpoint se agrupan en bloques de --graph-block y se
//          reproducen desde un cudaGraphExec_t ya instanciado. El grid, el
//          bloque, la shared dinamica y los argumentos son EXACTAMENTE los
//          mismos; lo que desaparece es el trabajo de CPU por lanzamiento.
//
// Solo aplica a las rutas WMMA (FP16/BF16). GPU_FP32, GPU_FP64 y las rutas de
// CPU no se tocan y siguen reportando execution_mode=normal.
enum class ExecutionMode {
    Normal,
    Graph
};

static const char* execution_mode_label(ExecutionMode mode) {
    return (mode == ExecutionMode::Graph) ? "cuda_graph" : "normal";
}

// Tamano por defecto del bloque de iteraciones que entra en un grafo. Debe ser
// PAR: tras un numero par de iteraciones el ping-pong (tanto el de estado como
// el de residuos) vuelve a la asignacion de punteros con la que se capturo el
// grafo, que es lo que permite reproducir el MISMO cudaGraphExec_t k veces sin
// reinstanciarlo ni reescribir los argumentos de sus nodos.
constexpr int kDefaultGraphBlock = 32;

struct Options {
    int nx = 2048;
    int ny = 2048;
    int iters = 20;
    TensorCoreMode tc_mode = TensorCoreMode::Both;
    // 0 (por defecto) = sin checkpoints, comportamiento identico al previo.
    // K > 0: cada K iteraciones, cada ruta se compara contra un snapshot FP64
    // de esa misma iteracion (ver CheckpointContext / compute_cpu_stencil_fp64).
    int checkpoint_every = 0;
    // Vacio (por defecto) = sin CSV, comportamiento identico al previo.
    std::string csv_path;
    // false (por defecto) = comportamiento identico al previo. true: omite
    // referencias CPU/FP64 y metricas de error (ver run_profile_only) para
    // que ncu no pague su costo antes de llegar al kernel perfilado.
    bool profile_only = false;
    // false (por defecto, "off") = sin compensacion, comportamiento identico
    // al previo. true ("on"): suma compensada de Kahan del redondeo de
    // ALMACENAMIENTO a 16 bits en las rutas WMMA (FP16/BF16); no aplica a GPU
    // FP32 clasico (acumulador y almacenamiento son ambos FP32 ahi, la
    // compensacion seria un no-op con puro overhead). Ver
    // benchmark_gpu_tensor_core_stencil.
    bool kahan = false;
    // false (por defecto, "off") = comportamiento identico al previo. true
    // ("on"): compensacion ESPACIAL (error feedback de vecinos) en vez de la
    // Kahan local; ver CompMode::Spatial. Mutuamente excluyente con --kahan on
    // (son dos politicas alternativas de la misma compensacion, no dos capas
    // acumulables): parse_args lo rechaza.
    bool spatial_comp = false;
    // true (por defecto, "on"): corre la ruta GPU_FP64 (referencia de maxima
    // precision EN GPU, ver benchmark_gpu_fp64_stencil). Es el denominador
    // GPU-vs-GPU de los speedups de las rutas WMMA compensadas: sin ella la
    // unica referencia FP64 de la corrida es la de CPU, y contrastar una ruta
    // GPU contra un tiempo de CPU mezcla dos dispositivos en una sola razon.
    // "off" la omite (util para recortar el costo de una corrida de smoke test
    // o si la malla no cabe en memoria: duplica los bytes por celda respecto a
    // GPU_FP32, ver el presupuesto en run_stencil_tc.sbatch).
    bool fp64_gpu = true;
    // Ruta CPU_FP64: el patron de oro IEEE 754 (double, serial en CPU) medido
    // como ruta, no solo usado como ground truth. Es la referencia de COSTO del
    // argumento central del proyecto -- "FP64 da la mejor precision, pero se
    // paga en tiempo de ejecucion" --: sin cronometrarla ese costo se asume en
    // vez de medirse, y no hay contra que enunciar la ganancia de las
    // precisiones reducidas. El CSV publica su t_iter_ms/t_total_ms/gflops y su
    // energia crudos; la razon contra cada ruta se calcula fuera, en el
    // analisis, no como columna derivada.
    // "off" la omite: implica una SEGUNDA pasada FP64 sobre la malla (ver
    // benchmark_cpu_fp64_stencil), que a mallas y iters grandes es la parte mas
    // cara de la corrida.
    bool cpu_fp64 = true;
    // Operador y condicion inicial. Los defaults (Stress + Legacy) reproducen
    // byte a byte el comportamiento de las fases 1-3: sin estos flags el binario
    // es el de antes. Ver OpMode / CiMode.
    OpMode op_mode = OpMode::Stress;
    // alpha del operador difusivo. 3/16 = 0.1875 (ver OpMode). Solo tiene efecto
    // con --op-mode diffusive; parse_args rechaza --alpha bajo el operador de
    // estres en vez de aceptarlo en silencio, donde no significaria nada.
    float alpha = 0.1875f;
    CiMode ci_mode = CiMode::Legacy;
    // Numero de periodos completos del modo en cada eje (ver
    // initialize_grid_monomode). p=168 sobre 16384^2 da k ~ 0.0644 y un factor de
    // amplificacion g ~ 0.998444 por iteracion: la norma cae a ~0.61 en 320
    // iteraciones y a ~0.37 en 640 -- dinamica difusiva observable, y denominador
    // de rel_l2 bien condicionado en todo el horizonte.
    int ci_p = 168;
    float ci_amplitude = 1.0f;
    // Vacio (por defecto) = ruta historica intacta. No vacio: los checkpoints
    // caen en ESTAS iteraciones exactas en vez de en los multiplos de
    // checkpoint_every, y la referencia FP64 se vuelca a disco conforme se
    // genera en vez de acumularse en RAM (ver ReferenceSpill). Es lo que hace
    // viable medir drift a 16384^2 con horizonte 640: la ruta de multiplos
    // guardaria un vector<double> por checkpoint, ~1.3 TB con cadencia 1.
    // Mutuamente excluyente con --checkpoint-every.
    std::vector<int> checkpoint_iters;
    // Vacio (por defecto) = sin archivado. No vacio: subconjunto de
    // checkpoint_iters en cuyas iteraciones se vuelca a disco el campo COMPLETO
    // de cada ruta, para poder recalcular offline cualquier metrica futura sin
    // volver a ocupar SLURM.
    std::vector<int> archive_iters;
    // Directorio de los .bin y del manifest.json del archivado. No se crea:
    // debe existir (el sbatch lo prepara). Un mkdir aqui enmascararia un
    // RUN_KIND mal configurado escribiendo gigabytes en el cwd del job.
    std::string archive_dir = "archive";
    // Normal (por defecto) = comportamiento identico al previo, un lanzamiento
    // por iteracion. Graph: las rutas WMMA reproducen bloques de iteraciones
    // desde un CUDA Graph preinstanciado (ver ExecutionMode).
    ExecutionMode execution_mode = ExecutionMode::Normal;
    // Iteraciones por grafo. Solo tiene efecto con --execution-mode graph;
    // parse_args rechaza darlo en modo normal en vez de aceptarlo en silencio,
    // donde no significaria nada (misma regla que --alpha bajo stress). Debe
    // ser par y > 0 (ver kDefaultGraphBlock).
    int graph_block = kDefaultGraphBlock;
};

// Politica efectiva derivada de los dos flags. parse_args ya garantizo que no
// esten ambos activos, asi que el orden de estas ramas no puede ocultar una
// combinacion valida.
static CompMode comp_mode_of(const Options& opt) {
    if (opt.spatial_comp) return CompMode::Spatial;
    if (opt.kahan) return CompMode::Local;
    return CompMode::Off;
}

// El operador activo, reducido a lo que el resto del programa necesita saber de
// el: los dos coeficientes de la formula y su conteo de operaciones por celda.
// Se deriva UNA sola vez, en el punto de entrada de la corrida, y de ahi baja
// como argumento explicito a cada funcion que evalua el stencil; los kernels
// reciben los dos coeficientes sueltos. No hay global ni __constant__ que pueda
// quedar desincronizado con los flags.
struct StencilOperator {
    float neighbor = 0.25f;
    float center = -1.0f;
    // FLOPs por celda interior. Stress: 3 sumas + 1 mult + 1 resta = 5.
    // Diffusive: 3 sumas (u+d+l+r) + 1 mult (alpha*sum) + 1 mult
    // ((1-4alpha)*center) + 1 suma = 6. El conteo entra en gflops y en
    // joules_per_gflop, y por eso mismo esas dos metricas NO son comparables
    // 1:1 entre los dos operadores: ver cell_updates_per_s y
    // energy_per_cell_update_j, que miden el mismo trabajo con un denominador
    // independiente del operador.
    double flops_per_cell = 5.0;
};

static StencilOperator operator_of(const Options& opt) {
    StencilOperator op;
    if (opt.op_mode == OpMode::Diffusive) {
        op.neighbor = opt.alpha;
        op.center = 1.0f - 4.0f * opt.alpha;
        op.flops_per_cell = 6.0;
    }
    return op;
}

// Version double de los mismos coeficientes, para las rutas FP64 (ground truth,
// CPU_FP64 y GPU_FP64). Se PROMUEVE desde el float en vez de recalcularse en
// double a proposito: todas las rutas tienen que evaluar el MISMO operador. Con
// los valores por defecto la promocion es exacta (2 bits de significando); con
// un --alpha arbitrario, el float redondeado ES el operador de referencia, y un
// ground truth que usara un alpha ligeramente distinto meteria un sesgo
// sistematico en todos los rel_l2 publicados.
static double neighbor_coeff_d(const StencilOperator& op) {
    return static_cast<double>(op.neighbor);
}

static double center_coeff_d(const StencilOperator& op) {
    return static_cast<double>(op.center);
}

__host__ __device__ inline int idx2d(int x, int y, int nx) {
    return y * nx + x;
}

// Formatea valores de error/normas en notacion cientifica de 6 cifras: %f con
// rango dinamico de 30 ordenes de magnitud (stencil diverge como 2^n) produce
// literales como "11527513700657988108288.000000" en vez de un numero legible.
static std::string fmt_sci(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6e", v);
    return buf;
}

static std::string fmt_csv_num(double v) {
    return std::isfinite(v) ? fmt_sci(v) : "NaN";
}

static std::string fmt_csv_error_num(const ErrorMetrics& e, double v) {
    return (e.reference_finite && e.solution_finite && std::isfinite(v)) ? fmt_sci(v) : "NaN";
}

static const char* kahan_label(bool kahan) {
    return kahan ? "on" : "off";
}

// Etiquetas de ruta/formato para la variante espacial. La columna kahan de los
// CSV_* sigue siendo off|on (unicos valores que tools/extract_csv.py sabe
// reconocer, ver KAHAN_RE/RUN_RE: un tercer valor no haria match y la fila
// heredaria en silencio el contexto de la corrida anterior); la variante se
// distingue por el SUFIJO de la ruta/formato, que esas herramientas propagan
// tal cual sin interpretarlo. Asi el par (route, kahan) identifica sin
// ambiguedad las tres politicas en un CSV que mezcle corridas:
//   (WMMA_FP16, off) (WMMA_FP16, on) (WMMA_FP16_SP, off)
// Ningun esquema de columnas cambia.
static const char* wmma_route_label(CompMode mode, const char* base, const char* base_spatial) {
    return (mode == CompMode::Spatial) ? base_spatial : base;
}

static const char* fp16_route_label(CompMode mode) {
    return wmma_route_label(mode, "WMMA_FP16", "WMMA_FP16_SP");
}

static const char* bf16_route_label(CompMode mode) {
    return wmma_route_label(mode, "WMMA_BF16", "WMMA_BF16_SP");
}

// Misma convencion para la columna `formato` del CSV de resumen, que va en
// minusculas. Sin esto la fila de spatial salia como (wmma_fp16, kahan=off),
// IDENTICA a la de la politica sin compensar, y al concatenar los CSV del
// bloque A y del bloque B las dos politicas se confundian en silencio: el
// contrato de arriba -- (route, kahan) identifica las tres -- solo se cumplia
// en los marcadores de stdout, no en el fichero.
static const char* fp16_csv_label(CompMode mode) {
    return wmma_route_label(mode, "wmma_fp16", "wmma_fp16_sp");
}

static const char* bf16_csv_label(CompMode mode) {
    return wmma_route_label(mode, "wmma_bf16", "wmma_bf16_sp");
}

static std::string csv_first_nonfinite_field(int first_nf) {
    return std::to_string((first_nf == INT_MAX) ? -1 : first_nf);
}

// Marcador de telemetria para Fase 4: emite por stdout el limite de una
// region cronometrada (begin/end) con timestamp de pared en ns desde epoch.
// Se emite FUERA del par de eventos CUDA que mide t/iter (nunca dentro de lo
// que build_metrics reporta): un muestreador NVML externo alinea ventanas de
// potencia con estos marcadores sin que este archivo tenga que exponer nada
// mas. Solo se usa en las rutas GPU con route_label (GPU_FP32, GPU_FP64,
// WMMA_FP16, WMMA_BF16); CPU FP32 serial no tiene ventana de potencia GPU que
// alinear.
static void emit_csv_region_marker(const char* route, const char* phase) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::cout << "CSV_REGION," << route << "," << phase << "," << ns << "\n";
}

// Un decimal para porcentajes (desglose WMMA/conversion/no atribuido): el
// stream global usa std::fixed con 6 decimales, demasiados para un "%".
static std::string fmt_pct1(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", v);
    return buf;
}

static void print_usage(const char* prog) {
    std::cout
        << "Uso:\n"
        << "  " << prog << " [--nx NX] [--ny NY] [--iters I] [--tc fp16|bf16|both]"
           " [--checkpoint-every K] [--csv RUTA] [--profile-only] [--kahan off|on]"
           " [--spatial-comp off|on] [--op-mode stress|diffusive] [--alpha A]"
           " [--ci-mode legacy|monomode] [--ci-p P] [--ci-amplitude A]"
           " [--execution-mode normal|graph] [--cuda-graph] [--graph-block B]\n\n"
        << "Descripcion:\n"
        << "  Compara CPU FP32, GPU CUDA FP32 y GPU WMMA Tensor Core para stencil 2D.\n"
        << "  La ruta Tensor Core usa operandos FP16/BF16 y acumulacion/salida FP32.\n\n"
        << "  --checkpoint-every K (K entero > 0) mide drift numerico: cada K\n"
        << "  iteraciones, cada ruta se compara contra un snapshot FP64 de esa misma\n"
        << "  iteracion y se emiten filas CSV_DRIFT/CSV_ONSET por stdout. K=0 o ausente\n"
        << "  (por defecto) no activa checkpoints, comportamiento identico al previo.\n"
        << "  Tambien es la UNICA forma de que storage_rel (error relativo de guardar\n"
        << "  en FP16/BF16) sea evaluable cuando una ruta diverge antes de --iters: sin\n"
        << "  checkpoints, si la ruta diverge, storage_rel se reporta como \"NO EVALUABLE\"\n"
        << "  (no hay estado intermedio finito que se pueda recuperar despues del hecho).\n\n"
        << "  --csv RUTA agrega una fila por ruta/configuracion a RUTA (cabecera solo\n"
        << "  si el archivo no existe). Ausente (por defecto) no escribe CSV.\n\n"
        << "  --profile-only ejecuta solo GPU FP32 clasico + la ruta TC de --tc (no\n"
        << "  admite --tc both), sin referencias CPU/FP64 ni metricas de error: para\n"
        << "  perfilar con ncu sin pagar su costo. Ausente (por defecto) no la activa.\n\n"
        << "  --kahan off|on (por defecto off) activa suma compensada de Kahan del\n"
        << "  redondeo de almacenamiento a 16 bits en las rutas WMMA (FP16/BF16); no\n"
        << "  aplica a GPU FP32 clasico. off preserva el comportamiento previo byte a\n"
        << "  byte. La compensacion es LOCAL: comp[idx] solo conoce la historia de la\n"
        << "  celda idx, no la de las 4 vecinas que entran a la suma.\n\n"
        << "  --spatial-comp off|on (por defecto off) usa compensacion ESPACIAL (error\n"
        << "  feedback: cada celda reincorpora los residuos de sus 4 vecinas y el propio\n"
        << "  antes de sumar) en vez de la Kahan local. Mutuamente excluyente con\n"
        << "  --kahan on. Cuesta 5 lecturas globales FP32 y 1 escritura FP32 extra por\n"
        << "  celda por iteracion, y duplica el buffer de residuos (ping-pong): en un\n"
        << "  kernel limitado por memoria eso NO es gratis, ver t/iter reportado. Las\n"
        << "  rutas WMMA se reportan como WMMA_FP16_SP / WMMA_BF16_SP para que sus\n"
        << "  filas CSV no se confundan con las de --kahan off|on.\n\n"
        << "  --fp64-gpu off|on (por defecto on) corre la ruta GPU_FP64: el mismo\n"
        << "  stencil en double sobre GPU, sin Tensor Cores ni compensacion. Es la\n"
        << "  referencia de maxima precision EN GPU, pensada como denominador\n"
        << "  GPU-vs-GPU del speedup de las rutas WMMA (comparar una ruta GPU contra\n"
        << "  el tiempo de CPU mezcla dos dispositivos en una sola razon). Cuesta el\n"
        << "  doble de bytes por celda que GPU_FP32; off la omite por completo.\n\n"
        << "  --cpu-fp64 off|on (por defecto on) corre la ruta CPU_FP64: el mismo\n"
        << "  stencil en double, serial en CPU. El error contra FP64 de CPU ya se\n"
        << "  mide siempre (es el ground truth de todas las rutas); lo que aporta\n"
        << "  esta ruta es su TIEMPO y su ENERGIA: la referencia de costo contra\n"
        << "  la que se enuncia cuanto se gana al bajar de precision frente al\n"
        << "  patron de oro IEEE 754. Implica una segunda pasada FP64 sobre la\n"
        << "  malla, independiente del ground truth y sin su instrumentacion, para\n"
        << "  que su t/iter sea comparable con el de CPU_FP32 (ver\n"
        << "  benchmark_cpu_fp64_stencil); a iters grandes es la parte mas cara de\n"
        << "  la corrida y off la omite.\n\n"
        << "  --op-mode stress|diffusive (por defecto stress) elige el operador de 5\n"
        << "  puntos. stress: out = 0.25*(u+d+l+r) - center, el operador historico de\n"
        << "  las fases 1-3; g(pi,pi) = -2, el modo Nyquist se duplica por iteracion y\n"
        << "  toda ruta de precision reducida termina desbordando (ese es el fenomeno\n"
        << "  que mide el horizonte n*). diffusive: out = alpha*(u+d+l+r) +\n"
        << "  (1-4*alpha)*center, estable, con g(pi,pi) = -0.5 al alpha por defecto:\n"
        << "  es el operador con el que rel_l2 se mantiene finito y el frente de\n"
        << "  Pareto precision-tiempo-energia tiene su tercer eje.\n\n"
        << "  --alpha A (por defecto 0.1875 = 3/16) fija alpha del operador difusivo.\n"
        << "  Solo valido con --op-mode diffusive (con stress es un error, no un\n"
        << "  no-op silencioso). 3/16 y no 1/8 porque con 1/8 el simbolo da\n"
        << "  g(pi,pi) = 0 exacto y el modo Nyquist -- el canal por el que viaja el\n"
        << "  ruido de redondeo que se quiere medir -- se aniquilaria en un paso.\n\n"
        << "  --ci-mode legacy|monomode (por defecto legacy) elige la condicion\n"
        << "  inicial. legacy: la CI historica de las fases 1-3. monomode:\n"
        << "  u0 = A*sin(2*pi*p*x/(nx-1))*sin(2*pi*p*y/(ny-1)), un unico modo propio\n"
        << "  del Laplaciano discreto con frontera Dirichlet 0. Bajo el operador\n"
        << "  difusivo la CI legacy decae solo ~1.2 % en 640 iteraciones (cuasi\n"
        << "  invariante, y el porcentaje no mejora agrandando la malla: su k es\n"
        << "  0.01 rad/celda fijo); el monomodo da una dinamica observable.\n\n"
        << "  --ci-p P (por defecto 168) periodos completos del monomodo en cada eje;\n"
        << "  --ci-amplitude A (por defecto 1.0) su amplitud. Solo con\n"
        << "  --ci-mode monomode.\n\n"
        << "  --checkpoint-iters \"1,2,5,10,...\" mide el error en ESAS iteraciones\n"
        << "  exactas, en vez de en los multiplos de --checkpoint-every (con el que es\n"
        << "  mutuamente excluyente; aquel conserva intacta su semantica para las\n"
        << "  campanas que ya dependen de ella). Emite CSV_CKPT con rel_l2, rel_linf y\n"
        << "  max_abs por ruta, y CSV_NORM con las normas de la referencia. La\n"
        << "  referencia FP64 se vuelca a disco conforme se genera en vez de acumularse\n"
        << "  en RAM, asi que el pico de memoria NO crece con --iters ni con el numero\n"
        << "  de checkpoints (a 16384^2 la ruta de multiplos con cadencia 1 pediria\n"
        << "  ~1.3 TB); el precio es E/S de scratch.\n\n"
        << "  --archive-iters \"320,640\" (subconjunto de --checkpoint-iters) vuelca a\n"
        << "  disco el campo COMPLETO de cada ruta en esas iteraciones: binario crudo\n"
        << "  row-major little-endian mas un manifest.json con dtype, forma y sha256.\n"
        << "  Permite recalcular offline otra norma, otra referencia u otra tolerancia\n"
        << "  sin volver a ocupar el cluster. Se archiva el estado PROPAGADO (buffer de\n"
        << "  16 bits reconvertido, o Q(u)+comp bajo compensacion espacial), nunca el\n"
        << "  acumulador FP32 previo al ultimo redondeo de almacenamiento.\n\n"
        << "  --archive-dir RUTA (por defecto \"archive\") es donde van esos ficheros y\n"
        << "  el spill de referencia. El directorio debe existir: no se crea aqui, para\n"
        << "  no volcar gigabytes en el cwd si la corrida quedo mal configurada.\n\n"
        << "  --execution-mode normal|graph (por defecto normal), con --cuda-graph como\n"
        << "  alias de graph. normal: un lanzamiento de kernel por iteracion,\n"
        << "  comportamiento historico byte a byte. graph: las rutas WMMA agrupan las\n"
        << "  iteraciones que NO piden checkpoint en bloques de --graph-block y las\n"
        << "  reproducen desde un cudaGraphExec_t instanciado ANTES de la region\n"
        << "  cronometrada, para quitar del camino el costo de CPU por lanzamiento. No\n"
        << "  cambia el kernel, la formulacion Y = X H + V X, los coeficientes, la\n"
        << "  compensacion ni el ping-pong: los argumentos de cada nodo son los mismos\n"
        << "  que en modo normal. Solo aplica a FP16/BF16; GPU_FP32, GPU_FP64 y las\n"
        << "  rutas de CPU siguen en normal. En modo graph NO se instrumenta un par de\n"
        << "  eventos por iteracion sino uno por grupo de lanzamiento, asi que para el\n"
        << "  desglose fino por kernel se debe usar --execution-mode normal.\n\n"
        << "  --graph-block B (por defecto 32) iteraciones por grafo. Debe ser PAR: el\n"
        << "  grafo captura una asignacion concreta de los buffers en ping-pong (estado\n"
        << "  y residuos) y solo un numero par de iteraciones la restituye al final de\n"
        << "  cada reproduccion. Solo valido con --execution-mode graph (en modo normal\n"
        << "  es un error, no un no-op silencioso). Si --iters < B no llega a formarse\n"
        << "  ningun bloque y la corrida degenera a lanzamientos normales.\n\n"
        << "Ejemplos:\n"
        << "  " << prog << "\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 20 --tc fp16\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 20 --tc bf16\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 20 --tc both --checkpoint-every 5\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 20 --tc fp16 --kahan on\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 20 --tc fp16 --spatial-comp on\n"
        << "  " << prog << " --nx 16384 --ny 16384 --iters 640 --tc both"
           " --op-mode diffusive --ci-mode monomode\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 320 --tc fp16 --cuda-graph\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 320 --tc fp16"
           " --execution-mode graph --graph-block 64\n";
}

static int parse_int_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Falta valor para " << argv[i] << "\n";
        std::exit(EXIT_FAILURE);
    }
    return std::atoi(argv[++i]);
}

static float parse_float_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Falta valor para " << argv[i] << "\n";
        std::exit(EXIT_FAILURE);
    }
    return static_cast<float>(std::atof(argv[++i]));
}

// Lista de enteros separados por coma ("1,2,5,10"). Devuelve los valores
// ORDENADOS y sin duplicados: el resto del programa asume que el orden de la
// lista es el orden temporal de los checkpoints (ReferenceSpill indexa el
// fichero por posicion), y un duplicado emitiria dos filas CSV_CKPT para la
// misma iteracion.
static std::vector<int> parse_int_list(const char* flag, const char* value) {
    std::vector<int> out;
    const char* p = value;
    while (*p != '\0') {
        while (*p == ' ' || *p == ',') ++p;
        if (*p == '\0') break;
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) {
            std::cerr << "Valor no numerico en " << flag << ": " << value << "\n";
            std::exit(EXIT_FAILURE);
        }
        if (v <= 0 || v > INT_MAX) {
            std::cerr << flag << " admite solo iteraciones >= 1: " << v << "\n";
            std::exit(EXIT_FAILURE);
        }
        out.push_back(static_cast<int>(v));
        p = end;
    }
    if (out.empty()) {
        std::cerr << flag << " no puede quedar vacio.\n";
        std::exit(EXIT_FAILURE);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static std::string join_int_list(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i != 0) s += ",";
        s += std::to_string(v[i]);
    }
    return s;
}

static OpMode parse_op_mode(const char* value) {
    if (std::strcmp(value, "stress") == 0) return OpMode::Stress;
    if (std::strcmp(value, "diffusive") == 0) return OpMode::Diffusive;

    std::cerr << "Operador no reconocido: " << value << " (use stress|diffusive)\n";
    std::exit(EXIT_FAILURE);
}

static CiMode parse_ci_mode(const char* value) {
    if (std::strcmp(value, "legacy") == 0) return CiMode::Legacy;
    if (std::strcmp(value, "monomode") == 0) return CiMode::Monomode;

    std::cerr << "Condicion inicial no reconocida: " << value << " (use legacy|monomode)\n";
    std::exit(EXIT_FAILURE);
}

static ExecutionMode parse_execution_mode(const char* value) {
    if (std::strcmp(value, "normal") == 0) return ExecutionMode::Normal;
    if (std::strcmp(value, "graph") == 0) return ExecutionMode::Graph;

    std::cerr << "Modo de ejecucion no reconocido: " << value << " (use normal|graph)\n";
    std::exit(EXIT_FAILURE);
}

static TensorCoreMode parse_tc_mode(const char* value) {
    if (std::strcmp(value, "fp16") == 0) return TensorCoreMode::FP16;
    if (std::strcmp(value, "bf16") == 0) return TensorCoreMode::BF16;
    if (std::strcmp(value, "both") == 0) return TensorCoreMode::Both;

    std::cerr << "Modo Tensor Core no reconocido: " << value << "\n";
    std::exit(EXIT_FAILURE);
}

static bool parse_on_off_flag(const char* flag, const char* value) {
    if (std::strcmp(value, "off") == 0) return false;
    if (std::strcmp(value, "on") == 0) return true;

    std::cerr << "Valor no reconocido para " << flag << " (use off|on): " << value << "\n";
    std::exit(EXIT_FAILURE);
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    // --alpha / --ci-p / --ci-amplitude solo significan algo bajo el operador o
    // la CI que los usan. Se registra si el usuario los dio EXPLICITAMENTE (no
    // basta comparar contra el default: pasar el default a mano tambien es un
    // malentendido sobre que se esta corriendo) para poder rechazar la
    // combinacion en vez de ignorarla en silencio.
    bool alpha_given = false;
    bool ci_param_given = false;
    bool graph_block_given = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--nx") == 0) {
            opt.nx = parse_int_arg(i, argc, argv);
        } else if (std::strcmp(argv[i], "--ny") == 0) {
            opt.ny = parse_int_arg(i, argc, argv);
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            opt.iters = parse_int_arg(i, argc, argv);
        } else if (std::strcmp(argv[i], "--tc") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --tc\n";
                std::exit(EXIT_FAILURE);
            }
            opt.tc_mode = parse_tc_mode(argv[++i]);
        } else if (std::strcmp(argv[i], "--checkpoint-every") == 0) {
            opt.checkpoint_every = parse_int_arg(i, argc, argv);
        } else if (std::strcmp(argv[i], "--csv") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --csv\n";
                std::exit(EXIT_FAILURE);
            }
            opt.csv_path = argv[++i];
        } else if (std::strcmp(argv[i], "--profile-only") == 0) {
            opt.profile_only = true;
        } else if (std::strcmp(argv[i], "--kahan") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --kahan\n";
                std::exit(EXIT_FAILURE);
            }
            opt.kahan = parse_on_off_flag("--kahan", argv[++i]);
        } else if (std::strcmp(argv[i], "--spatial-comp") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --spatial-comp\n";
                std::exit(EXIT_FAILURE);
            }
            opt.spatial_comp = parse_on_off_flag("--spatial-comp", argv[++i]);
        } else if (std::strcmp(argv[i], "--fp64-gpu") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --fp64-gpu\n";
                std::exit(EXIT_FAILURE);
            }
            opt.fp64_gpu = parse_on_off_flag("--fp64-gpu", argv[++i]);
        } else if (std::strcmp(argv[i], "--cpu-fp64") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --cpu-fp64\n";
                std::exit(EXIT_FAILURE);
            }
            opt.cpu_fp64 = parse_on_off_flag("--cpu-fp64", argv[++i]);
        } else if (std::strcmp(argv[i], "--op-mode") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --op-mode\n";
                std::exit(EXIT_FAILURE);
            }
            opt.op_mode = parse_op_mode(argv[++i]);
        } else if (std::strcmp(argv[i], "--alpha") == 0) {
            opt.alpha = parse_float_arg(i, argc, argv);
            alpha_given = true;
        } else if (std::strcmp(argv[i], "--ci-mode") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --ci-mode\n";
                std::exit(EXIT_FAILURE);
            }
            opt.ci_mode = parse_ci_mode(argv[++i]);
        } else if (std::strcmp(argv[i], "--ci-p") == 0) {
            opt.ci_p = parse_int_arg(i, argc, argv);
            ci_param_given = true;
        } else if (std::strcmp(argv[i], "--ci-amplitude") == 0) {
            opt.ci_amplitude = parse_float_arg(i, argc, argv);
            ci_param_given = true;
        } else if (std::strcmp(argv[i], "--checkpoint-iters") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --checkpoint-iters\n";
                std::exit(EXIT_FAILURE);
            }
            opt.checkpoint_iters = parse_int_list("--checkpoint-iters", argv[++i]);
        } else if (std::strcmp(argv[i], "--archive-iters") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --archive-iters\n";
                std::exit(EXIT_FAILURE);
            }
            opt.archive_iters = parse_int_list("--archive-iters", argv[++i]);
        } else if (std::strcmp(argv[i], "--archive-dir") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --archive-dir\n";
                std::exit(EXIT_FAILURE);
            }
            opt.archive_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--execution-mode") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Falta valor para --execution-mode\n";
                std::exit(EXIT_FAILURE);
            }
            opt.execution_mode = parse_execution_mode(argv[++i]);
        } else if (std::strcmp(argv[i], "--cuda-graph") == 0) {
            // Alias corto y sin valor de --execution-mode graph.
            opt.execution_mode = ExecutionMode::Graph;
        } else if (std::strcmp(argv[i], "--graph-block") == 0) {
            opt.graph_block = parse_int_arg(i, argc, argv);
            graph_block_given = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Argumento no reconocido: " << argv[i] << "\n\n";
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (opt.nx < 3 || opt.ny < 3 || opt.iters <= 0) {
        std::cerr << "nx y ny deben ser >= 3; iters debe ser positivo.\n";
        std::exit(EXIT_FAILURE);
    }
    if (opt.checkpoint_every < 0) {
        std::cerr << "checkpoint-every debe ser >= 0 (0 desactiva los checkpoints).\n";
        std::exit(EXIT_FAILURE);
    }
    if (opt.kahan && opt.spatial_comp) {
        std::cerr << "--kahan on y --spatial-comp on son mutuamente excluyentes: son dos"
                     " politicas alternativas de compensacion del mismo redondeo de\n"
                     "almacenamiento, no dos capas acumulables. Use una u otra.\n";
        std::exit(EXIT_FAILURE);
    }
    if (alpha_given && opt.op_mode != OpMode::Diffusive) {
        std::cerr << "--alpha solo aplica con --op-mode diffusive. Bajo el operador de"
                     " estres los coeficientes son fijos (0.25 / -1.0); aceptar el flag\n"
                     "en silencio dejaria la corrida etiquetada con un alpha que nunca se"
                     " uso.\n";
        std::exit(EXIT_FAILURE);
    }
    // Estabilidad de von Neumann del esquema explicito: g(kx,ky) = 1 -
    // 4*alpha*[sin^2(kx/2) + sin^2(ky/2)] cumple |g| <= 1 para todo modo si y
    // solo si 0 < alpha <= 1/4. Fuera de ese intervalo el operador amplifica y
    // deja de ser difusivo, que es lo unico que justifica esta variante frente
    // al operador de estres, que ya cubre el caso divergente.
    if (opt.op_mode == OpMode::Diffusive && (!(opt.alpha > 0.0f) || opt.alpha > 0.25f)) {
        std::cerr << "--alpha debe estar en (0, 0.25]: fuera de ese intervalo el operador"
                     " de 5 puntos no es difusivo (|g| > 1 para algun modo) y la campana\n"
                     "perderia justo la propiedad por la que existe.\n";
        std::exit(EXIT_FAILURE);
    }
    if (ci_param_given && opt.ci_mode != CiMode::Monomode) {
        std::cerr << "--ci-p / --ci-amplitude solo aplican con --ci-mode monomode.\n";
        std::exit(EXIT_FAILURE);
    }
    if (graph_block_given && opt.execution_mode != ExecutionMode::Graph) {
        std::cerr << "--graph-block solo aplica con --execution-mode graph (o --cuda-graph):"
                     " en modo normal no hay grafo que dimensionar y aceptarlo en silencio\n"
                     "dejaria la corrida etiquetada con un tamano de bloque que nunca se"
                     " uso.\n";
        std::exit(EXIT_FAILURE);
    }
    // Par y > 0: el grafo se captura con una asignacion concreta de punteros del
    // ping-pong y se reproduce k veces; solo con un numero PAR de iteraciones
    // por grafo los punteros vuelven al estado de captura al terminar cada
    // reproduccion (ver kDefaultGraphBlock y build_iteration_graph).
    if (opt.execution_mode == ExecutionMode::Graph &&
        (opt.graph_block <= 0 || opt.graph_block % 2 != 0)) {
        std::cerr << "--graph-block debe ser par y > 0 (recibido " << opt.graph_block
                  << "): el grafo captura una asignacion fija de los buffers en ping-pong\n"
                     "y solo un numero par de iteraciones la restituye al final de cada"
                     " reproduccion.\n";
        std::exit(EXIT_FAILURE);
    }
    // El monomodo u0 = A*sin(2*pi*p*x/(nx-1))*sin(2*pi*p*y/(ny-1)) es el armonico
    // m = 2p del Laplaciano discreto con Dirichlet 0, y una malla de nx puntos
    // solo distingue m = 1..nx-2. Con 2p fuera de ese rango el seno aliasea a
    // otro modo (o al modo nulo) y la CI deja de ser lo que su etiqueta dice.
    if (opt.ci_mode == CiMode::Monomode) {
        const int max_p = (std::min(opt.nx, opt.ny) - 2) / 2;
        if (opt.ci_p < 1 || opt.ci_p > max_p) {
            std::cerr << "--ci-p debe estar en [1, " << max_p << "] para nx=" << opt.nx
                      << ", ny=" << opt.ny << ": el monomodo es el armonico m=2p del\n"
                         "Laplaciano discreto y la malla solo distingue m=1..min(nx,ny)-2."
                         " Fuera de rango el seno aliasea a otro modo.\n";
            std::exit(EXIT_FAILURE);
        }
    }
    // Las dos cadencias de checkpoint son mecanismos alternativos, no capas: la
    // historica indexa fp64_checkpoints por iter/K-1 y la nueva busca la
    // iteracion en una lista. Aceptar ambas obligaria a decidir cual gana en
    // cada llamada y dejaria filas CSV_DRIFT y CSV_CKPT mezcladas para la misma
    // corrida.
    if (opt.checkpoint_every > 0 && !opt.checkpoint_iters.empty()) {
        std::cerr << "--checkpoint-every y --checkpoint-iters son mutuamente excluyentes:"
                     " el primero marca multiplos de K y el segundo iteraciones\n"
                     "exactas. Use uno u otro.\n";
        std::exit(EXIT_FAILURE);
    }
    if (!opt.checkpoint_iters.empty() && opt.checkpoint_iters.back() > opt.iters) {
        std::cerr << "--checkpoint-iters pide la iteracion " << opt.checkpoint_iters.back()
                  << " pero la corrida solo llega a " << opt.iters << ".\n";
        std::exit(EXIT_FAILURE);
    }
    // --archive-iters SUBCONJUNTO de --checkpoint-iters: el archivado se engancha
    // al bloque de checkpoint (mismo D2H, misma pausa del cronometro y de la
    // ventana de energia). Una iteracion de archivado que no fuera checkpoint no
    // tendria donde ejecutarse sin abrir una segunda pausa no contabilizada.
    if (!opt.archive_iters.empty()) {
        if (opt.checkpoint_iters.empty()) {
            std::cerr << "--archive-iters requiere --checkpoint-iters: el archivado se"
                         " engancha al bloque de checkpoint, que es el unico punto donde\n"
                         "el estado ya esta copiado a host y el cronometro esta pausado.\n";
            std::exit(EXIT_FAILURE);
        }
        for (int it : opt.archive_iters) {
            if (std::find(opt.checkpoint_iters.begin(), opt.checkpoint_iters.end(), it) ==
                opt.checkpoint_iters.end()) {
                std::cerr << "--archive-iters debe ser subconjunto de --checkpoint-iters."
                             " La iteracion " << it << " no esta en la lista de checkpoints ("
                          << join_int_list(opt.checkpoint_iters) << ").\n";
                std::exit(EXIT_FAILURE);
            }
        }
    }
    if (opt.profile_only && opt.tc_mode == TensorCoreMode::Both) {
        std::cerr << "--profile-only no admite --tc both: perfila una sola ruta TC"
                     " (fp16 o bf16) por invocacion.\n";
        std::exit(EXIT_FAILURE);
    }
    return opt;
}

static void print_gpu_info() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::cerr << "No se detectaron GPUs CUDA." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    int gpu_clock_khz = 0, mem_clock_khz = 0, mem_bus_width = 0;
    cudaError_t e1 = cudaDeviceGetAttribute(&gpu_clock_khz, cudaDevAttrClockRate, dev);
    cudaError_t e2 = cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaError_t e3 = cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth, dev);

    std::cout << "================ CARACTERISTICAS DE LA GPU ================\n";
    std::cout << "GPUs detectadas            : " << device_count << "\n";
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

static bool device_supports_fp16_tensor_cores() {
    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    return prop.major >= 7;
}

static bool device_supports_bf16_tensor_cores() {
    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDevice(&dev));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    return prop.major >= 8;
}

// Celdas interiores: las unicas que el stencil recalcula (el borde se copia tal
// cual en todas las rutas). Es el denominador de cell_updates_per_s y de
// energy_per_cell_update_j.
static double interior_cells(int nx, int ny) {
    return static_cast<double>(nx - 2) * static_cast<double>(ny - 2);
}

// flops_per_cell deja de ser el literal 5.0 (que asumia el conteo del operador
// de estres: 3 sumas + 1 mult + 1 resta) y pasa a venir del operador activo:
// el difusivo son 6 ops/celda. Consecuencia metodologica: gflops y
// joules_per_gflop dejan de ser comparables 1:1 ENTRE operadores, porque el
// numerador cambia de definicion. Para la comparacion cruzada stress-vs-
// diffusive usar cell_updates_per_s / energy_per_cell_update_j, cuyo
// denominador es el mismo trabajo util en ambos casos.
static double stencil_flops(int nx, int ny, double flops_per_cell) {
    return flops_per_cell * interior_cells(nx, ny);
}

// Nota metodologica: el TFLOPS reportado para la ruta WMMA NO es comparable en
// terminos absolutos al TFLOPS de GEMM (Fase_2/GEMM). Aqui cada tile 16x16
// ejecuta 2 MMA 16x16x16 (Y = X H + V X, ver comentario superior del archivo)
// sobre matrices H y V que son tridiagonales: la mayor parte de sus 4096
// productos son contra ceros estructurales, a diferencia de la GEMM densa. El
// gflops_utiles cuenta el trabajo UTIL del stencil (flops_per_cell por celda
// interior), no las operaciones que el Tensor Core emite. El numero sirve para
// comparar las rutas de Stencil entre si (CPU, GPU FP32, GPU FP64, GPU WMMA),
// no para comparar Stencil contra GEMM.
static Metrics build_metrics(int nx, int ny, double avg_ms, double flops_per_cell) {
    Metrics m;
    m.ms = avg_ms;
    m.gflops = stencil_flops(nx, ny, flops_per_cell) / (m.ms * 1e6);
    m.tflops = m.gflops / 1000.0;
    return m;
}

static void initialize_grid(std::vector<float>& v, int nx, int ny) {
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const float wave = std::sin(0.01f * static_cast<float>(x))
                             + std::cos(0.01f * static_cast<float>(y));
            const int centered = static_cast<int>((x + 3 * y) % 17) - 8;
            v[idx2d(x, y, nx)] = wave + 0.001f * static_cast<float>(centered);
        }
    }
}

// Condicion inicial monomodo (--ci-mode monomode):
//   u0(x,y) = A * sin(2*pi*p*x/(nx-1)) * sin(2*pi*p*y/(ny-1))
// Es exactamente el modo propio m = 2p del Laplaciano discreto de 5 puntos con
// frontera Dirichlet 0, de modo que bajo el operador difusivo el campo se
// amortigua por un factor g uniforme en cada iteracion, sin cambiar de forma:
//   g = 1 - 4*alpha*[sin^2(kx/2) + sin^2(ky/2)],  k = 2*pi*p/(nx-1)
// Con p=168 sobre 16384^2 eso da g ~ 0.998444 (||u||_2 cae a ~0.61 en 320
// iteraciones y a ~0.37 en 640): decaimiento medible, y el denominador de rel_l2
// se mantiene lejos de cero en todo el horizonte.
//
// Los senos se evaluan en double y se convierten a float al escribir: la
// reduccion de argumento de sinf sobre 2*pi*168*16382/16383 perderia digitos
// significativos justo donde la CI tiene que ser reproducible.
//
// La frontera se asigna a 0.0f EXPLICITAMENTE. La formula ya da ~0 ahi (p es
// entero), pero la asignacion elimina el residuo de reduccion de argumento en
// punto flotante; y como el kernel copia el borde hacia adelante sin tocarlo, un
// 0 exacto se mantiene en 0 durante todo el horizonte.
static void initialize_grid_monomode(std::vector<float>& v, int nx, int ny,
                                     int p, float amplitude) {
    constexpr double kPi = 3.14159265358979323846;
    const double two_pi_p = 2.0 * kPi * static_cast<double>(p);
    const double amp = static_cast<double>(amplitude);
    for (int y = 0; y < ny; ++y) {
        const double sy = std::sin(two_pi_p * static_cast<double>(y)
                                   / static_cast<double>(ny - 1));
        for (int x = 0; x < nx; ++x) {
            if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
                v[idx2d(x, y, nx)] = 0.0f;
                continue;
            }
            const double sx = std::sin(two_pi_p * static_cast<double>(x)
                                       / static_cast<double>(nx - 1));
            v[idx2d(x, y, nx)] = static_cast<float>(amp * sx * sy);
        }
    }
}

// Dispatch de condicion inicial. Vive aqui, en el llamador, y no como una rama
// dentro de initialize_grid: esa funcion es la CI historica de las fases 1-3 y
// tiene que seguir siendo exactamente lo que era, sin un flag que pueda
// alterarla por descuido.
static void initialize_input_grid(std::vector<float>& v, const Options& opt) {
    if (opt.ci_mode == CiMode::Monomode) {
        initialize_grid_monomode(v, opt.nx, opt.ny, opt.ci_p, opt.ci_amplitude);
    } else {
        initialize_grid(v, opt.nx, opt.ny);
    }
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------
// Se implementa aqui en vez de enlazar OpenSSL porque el proyecto no agrega
// dependencias externas. Solo se usa fuera de las regiones cronometradas: para
// el hash de la condicion inicial (CSV_CI_SHA256) y para el manifest de
// archivado, que es lo que permite verificar meses despues que un .bin en disco
// sigue siendo el campo que dice ser.
class Sha256 {
public:
    Sha256() { reset(); }

    void reset() {
        static const uint32_t kInit[8] = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        for (int i = 0; i < 8; ++i) h_[i] = kInit[i];
        buf_len_ = 0;
        len_ = 0;
    }

    void update(const void* data, size_t n) {
        const unsigned char* p = static_cast<const unsigned char*>(data);
        len_ += n;
        while (n > 0) {
            const size_t take = std::min(n, sizeof(buf_) - buf_len_);
            std::memcpy(buf_ + buf_len_, p, take);
            buf_len_ += take;
            p += take;
            n -= take;
            if (buf_len_ == sizeof(buf_)) {
                transform(buf_);
                buf_len_ = 0;
            }
        }
    }

    // Cierra el hash y devuelve los 64 caracteres hex. Consume el objeto: para
    // volver a usarlo hay que reset(). No se llama dos veces en este archivo.
    std::string hex() {
        // El largo en bits se captura ANTES del padding: update() sigue
        // incrementando len_, y usarlo despues daria un largo equivocado.
        const uint64_t bitlen = len_ * 8ull;
        const unsigned char one = 0x80;
        update(&one, 1);
        const unsigned char zero = 0x00;
        while (buf_len_ != 56) update(&zero, 1);
        for (int i = 0; i < 8; ++i) {
            buf_[56 + i] = static_cast<unsigned char>((bitlen >> (56 - 8 * i)) & 0xffu);
        }
        transform(buf_);
        buf_len_ = 0;
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + i * 8, 9, "%08x", h_[i]);
        return std::string(out, 64);
    }

private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void transform(const unsigned char* p) {
        static const uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,
            0x923f82a4u,0xab1c5ed5u,0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
            0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,
            0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,
            0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
            0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,
            0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,
            0x5b9cca4fu,0x682e6ff3u,0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
            0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(p[i * 4]) << 24) |
                   (static_cast<uint32_t>(p[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(p[i * 4 + 2]) << 8) |
                   static_cast<uint32_t>(p[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
        h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
    }

    uint32_t h_[8];
    unsigned char buf_[64];
    size_t buf_len_;
    uint64_t len_;
};

template <typename T>
static std::string sha256_of_vector(const std::vector<T>& v) {
    Sha256 h;
    h.update(v.data(), v.size() * sizeof(T));
    return h.hex();
}

// ---------------------------------------------------------------------------
// ReferenceSpill: campos FP64 de referencia en DISCO, no en RAM
// ---------------------------------------------------------------------------
// La ruta historica (--checkpoint-every) guarda un vector<double> COMPLETO por
// checkpoint dentro de fp64_checkpoints. A 16384^2 cada campo son 2 GiB: diez
// checkpoints son 20 GiB, y --checkpoint-every 1 con 640 iteraciones serian
// ~1.3 TB. Aqui la referencia se escribe a un unico fichero conforme se genera
// y se relee cuando cada ruta alcanza su checkpoint, de modo que en RAM solo
// quedan el ping-pong de la referencia y UN buffer de lectura --
// independientemente de cuantos checkpoints y cuantas iteraciones haya.
//
// El precio es E/S, y conviene tenerlo escrito: a 16384^2 con diez checkpoints
// son ~20 GiB de escritura y una relectura de ~20 GiB por cada ruta con
// checkpoints (GPU_FP32, GPU_FP64, FP16, BF16 -> ~80 GiB). Es un intercambio
// deliberado y lo paga UNICAMENTE la corrida numerica (RUN_KIND=numeric, una
// sola ejecucion); la campana energetica de 15 replicas corre sin checkpoints y
// nunca abre un spill.
//
// La alternativa sin disco seria avanzar todas las rutas y la referencia en
// lockstep hasta cada checkpoint, pero eso obliga a partir los cinco bucles de
// benchmark -- con sus ventanas de cronometro y de energia -- en tramos
// reanudables, un cambio mucho mayor que lo que esta tarea admite.
class ReferenceSpill {
public:
    ReferenceSpill() = default;
    ~ReferenceSpill() { close(); }
    ReferenceSpill(const ReferenceSpill&) = delete;
    ReferenceSpill& operator=(const ReferenceSpill&) = delete;

    // El fichero de spill supera 2 GiB en cuanto la malla es grande, asi que
    // fseek/ftell tienen que ser de 64 bits. En Linux x86-64 (el unico objetivo
    // de este proyecto) long lo es; el static_assert lo deja explicito en vez de
    // producir corrupcion silenciosa si eso dejara de cumplirse.
    static_assert(sizeof(long) >= 8,
                  "ReferenceSpill necesita fseek/ftell de 64 bits: el spill supera 2 GiB");

    bool open(const std::string& path, size_t count) {
        close();
        path_ = path;
        count_ = count;
        file_ = std::fopen(path.c_str(), "w+b");
        if (file_ == nullptr) {
            std::cerr << "ERROR: no se pudo crear el spill de referencia en " << path
                      << ". Revise que el directorio exista y tenga espacio.\n";
            return false;
        }
        return true;
    }

    void close() {
        if (file_ != nullptr) {
            std::fclose(file_);
            file_ = nullptr;
        }
        if (!path_.empty()) {
            std::remove(path_.c_str());
            path_.clear();
        }
        index_.clear();
        buf_.clear();
    }

    bool is_open() const { return file_ != nullptr; }

    // Anexa el campo de esta iteracion. Devuelve false si la escritura falla
    // (disco lleno, cuota): el llamador aborta en vez de seguir con una campana
    // cuyos checkpoints quedarian mudos sin aviso.
    bool append(int iter, const std::vector<double>& field) {
        if (file_ == nullptr || field.size() != count_) return false;
        if (std::fseek(file_, record_offset(index_.size()), SEEK_SET) != 0) return false;
        if (std::fwrite(field.data(), sizeof(double), count_, file_) != count_) return false;
        index_.push_back(iter);
        return true;
    }

    // Campo de referencia de esa iteracion, o nullptr si no existe (iteracion
    // fuera de la lista, o referencia ya no finita cuando se llego a ella). El
    // puntero apunta a un buffer interno reutilizado: valido hasta la siguiente
    // llamada a load(), que es todo lo que necesita el unico consumidor
    // (record_checkpoint compara y descarta).
    const std::vector<double>* load(int iter) const {
        if (file_ == nullptr) return nullptr;
        for (size_t i = 0; i < index_.size(); ++i) {
            if (index_[i] != iter) continue;
            if (buf_.size() != count_) buf_.resize(count_);
            if (std::fseek(file_, record_offset(i), SEEK_SET) != 0) return nullptr;
            if (std::fread(buf_.data(), sizeof(double), count_, file_) != count_) return nullptr;
            return &buf_;
        }
        return nullptr;
    }

    size_t stored() const { return index_.size(); }
    // Bytes ocupados en disco: se reporta en el log para que el operador de la
    // campana sepa cuanto scratch consumio la corrida numerica.
    double gib_on_disk() const {
        return static_cast<double>(index_.size()) *
               static_cast<double>(count_ * sizeof(double)) / (1024.0 * 1024.0 * 1024.0);
    }

private:
    long record_offset(size_t i) const {
        return static_cast<long>(i) * static_cast<long>(count_ * sizeof(double));
    }

    std::FILE* file_ = nullptr;
    std::string path_;
    size_t count_ = 0;
    std::vector<int> index_;           // iteracion guardada en cada posicion
    mutable std::vector<double> buf_;  // buffer de lectura reutilizado
};

// ---------------------------------------------------------------------------
// Archivado de campos a disco
// ---------------------------------------------------------------------------
// Formato deliberadamente tonto: binario crudo, row-major, little-endian, sin
// cabecera. Un numpy.fromfile(path, dtype).reshape(ny, nx) lo lee sin una sola
// linea de codigo del proyecto, que es justamente el punto -- poder recalcular
// offline otra norma, otra referencia u otra tolerancia dentro de un ano sin
// volver a ocupar SLURM. Los metadatos (dtype, forma, ruta, iteracion, sha256)
// van en manifest.json al lado.
struct ArchiveEntry {
    std::string route;
    int iter = 0;
    std::string dtype;
    int nx = 0;
    int ny = 0;
    std::string file;
    std::string sha256;
    uint64_t bytes = 0;
};

struct ArchiveContext {
    std::vector<int> iters;
    std::string dir;
    // mutable: las rutas de benchmark reciben el contexto por referencia const
    // (no deben poder cambiar QUE se archiva), pero si tienen que dejar
    // constancia de lo que archivaron.
    mutable std::vector<ArchiveEntry> entries;

    bool due(int iter) const {
        return std::find(iters.begin(), iters.end(), iter) != iters.end();
    }
};

template <typename T>
static void archive_field(const ArchiveContext& arch,
                          const char* route,
                          int iter,
                          int nx,
                          int ny,
                          const char* dtype,
                          const std::vector<T>& field) {
    const std::string name = std::string(route) + "_iter" + std::to_string(iter) + ".bin";
    const std::string full = arch.dir + "/" + name;
    std::FILE* f = std::fopen(full.c_str(), "wb");
    if (f == nullptr) {
        std::cerr << "ERROR: no se pudo abrir " << full << " para archivar. El directorio"
                     " debe existir (lo crea el sbatch); no se crea aqui para no volcar\n"
                     "gigabytes en el cwd si RUN_KIND quedo mal configurado.\n";
        std::exit(EXIT_FAILURE);
    }
    const size_t written = std::fwrite(field.data(), sizeof(T), field.size(), f);
    std::fclose(f);
    if (written != field.size()) {
        std::cerr << "ERROR: escritura incompleta en " << full << " (" << written << " de "
                  << field.size() << " elementos). Probable disco lleno.\n";
        std::exit(EXIT_FAILURE);
    }
    ArchiveEntry e;
    e.route = route;
    e.iter = iter;
    e.dtype = dtype;
    e.nx = nx;
    e.ny = ny;
    e.file = name;
    e.sha256 = sha256_of_vector(field);
    e.bytes = static_cast<uint64_t>(field.size()) * sizeof(T);
    arch.entries.push_back(e);
    std::cout << "CSV_ARCHIVE," << route << "," << iter << "," << dtype << "," << name << ","
              << e.bytes << "," << e.sha256 << "\n";
}

// manifest.json: se escribe UNA vez al final, con todas las entradas. A mano y
// sin libreria de JSON porque son seis campos por entrada y el proyecto no
// agrega dependencias.
static void write_archive_manifest(const ArchiveContext& arch,
                                   const Options& opt,
                                   const std::string& ci_sha256) {
    const std::string path = arch.dir + "/manifest.json";
    std::ofstream m(path);
    if (!m) {
        std::cerr << "ERROR: no se pudo escribir " << path << "\n";
        std::exit(EXIT_FAILURE);
    }
    m << "{\n";
    m << "  \"layout\": \"row-major, little-endian, sin cabecera; "
         "numpy.fromfile(f, dtype).reshape(ny, nx)\",\n";
    m << "  \"nx\": " << opt.nx << ",\n";
    m << "  \"ny\": " << opt.ny << ",\n";
    m << "  \"iters\": " << opt.iters << ",\n";
    m << "  \"op_mode\": \"" << op_mode_label(opt.op_mode) << "\",\n";
    m << "  \"alpha\": " << (opt.op_mode == OpMode::Diffusive ? fmt_sci(opt.alpha)
                                                              : std::string("null")) << ",\n";
    m << "  \"ci_mode\": \"" << ci_mode_label(opt.ci_mode) << "\",\n";
    m << "  \"ci_p\": " << (opt.ci_mode == CiMode::Monomode ? std::to_string(opt.ci_p)
                                                            : std::string("null")) << ",\n";
    m << "  \"ci_sha256\": \"" << ci_sha256 << "\",\n";
    m << "  \"checkpoint_iters\": [" << join_int_list(opt.checkpoint_iters) << "],\n";
    m << "  \"archive_iters\": [" << join_int_list(opt.archive_iters) << "],\n";
    m << "  \"fields\": [\n";
    for (size_t i = 0; i < arch.entries.size(); ++i) {
        const ArchiveEntry& e = arch.entries[i];
        m << "    {\"route\": \"" << e.route << "\", \"iter\": " << e.iter
          << ", \"dtype\": \"" << e.dtype << "\", \"shape\": [" << e.ny << ", " << e.nx
          << "], \"file\": \"" << e.file << "\", \"bytes\": " << e.bytes
          << ", \"sha256\": \"" << e.sha256 << "\"}"
          << (i + 1 < arch.entries.size() ? "," : "") << "\n";
    }
    m << "  ]\n}\n";
    std::cout << "Manifest de archivado : " << path << " (" << arch.entries.size()
              << " campos)\n";
}

// Normas de la referencia FP64 en cada checkpoint. Dan escala a rel_l2: sin
// ellas, un rel_l2 que crece no distingue "la ruta se degrada" de "la
// referencia se encoge" -- y bajo el operador difusivo la referencia SI se
// encoge de forma sistematica (||u^n|| ~ g^n, con g^640 ~ 0.37 en la campana).
//   CSV_NORM,CPU_FP64,<iter>,<l2>,<linf>
static void emit_csv_norm_row(int iter_number, double l2, double linf) {
    std::cout << "CSV_NORM,CPU_FP64," << iter_number << "," << fmt_sci(l2) << ","
              << fmt_sci(linf) << "\n";
}

// Encadenamiento genuino salida(i) -> entrada(i+1) via dos buffers en ping-pong.
// El warm-up (kWarmupIters) tambien encadena, pero sobre buffers propios que se
// descartan: no debe alterar el estado que vera el bucle medido, de modo que
// con --iters 1 el resultado final coincide con Fase_2/Stencil (una sola
// aplicacion sobre el input original).
static Metrics benchmark_cpu_stencil(const std::vector<float>& in,
                                     std::vector<float>& out,
                                     int nx,
                                     int ny,
                                     int iters,
                                     const StencilOperator& op,
                                     int& first_nonfinite_iter,
                                     EnergyMeasurement& out_energy) {
    const float c_neigh = op.neighbor;
    const float c_center = op.center;
    // first_nf == nullptr durante el warm-up: esas iteraciones son descartables
    // y no deben contaminar la medicion (se reinicia antes del bucle medido).
    auto apply = [&](const std::vector<float>& src, std::vector<float>& dst,
                     int iter_number, int* first_nf) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
                    dst[idx2d(x, y, nx)] = src[idx2d(x, y, nx)];
                    continue;
                }

                const float up = src[idx2d(x, y - 1, nx)];
                const float down = src[idx2d(x, y + 1, nx)];
                const float left = src[idx2d(x - 1, y, nx)];
                const float right = src[idx2d(x + 1, y, nx)];
                const float center = src[idx2d(x, y, nx)];
                const float val = c_neigh * (up + down + left + right) + c_center * center;
                dst[idx2d(x, y, nx)] = val;
                if (first_nf != nullptr && *first_nf == INT_MAX && !std::isfinite(val)) {
                    *first_nf = iter_number;
                }
            }
        }
    };

    const RAEnergySnapshot rapl_warmup_before = rapl_snapshot_now();
    {
        std::vector<float> warm_a = in;
        std::vector<float> warm_b = in;
        std::vector<float>* warm_src = &warm_a;
        std::vector<float>* warm_dst = &warm_b;
        for (int i = 0; i < kWarmupIters; ++i) {
            apply(*warm_src, *warm_dst, i + 1, nullptr);
            std::swap(warm_src, warm_dst);
        }
    }

    std::vector<float> buf_a = in;
    std::vector<float> buf_b = in;
    std::vector<float>* src = &buf_a;
    std::vector<float>* dst = &buf_b;

    first_nonfinite_iter = INT_MAX;
    const RAEnergySnapshot rapl_before = rapl_snapshot_now();
    (void)rapl_warmup_before;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        apply(*src, *dst, i + 1, &first_nonfinite_iter);
        std::swap(src, dst);
    }
    auto end = std::chrono::high_resolution_clock::now();

    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    const RAEnergySnapshot rapl_after = rapl_snapshot_now();
    out_energy = EnergyMeasurement{};
    out_energy.time_total_s = std::chrono::duration<double>(end - start).count();
    out_energy.gpu_valid = true;  // La ruta CPU no requiere una lectura NVML.
    out_energy.cpu_valid = rapl_before.valid && rapl_after.valid &&
                           rapl_after.energy_j >= rapl_before.energy_j;
    if (out_energy.cpu_valid) {
        out_energy.energy_cpu_j = rapl_energy_delta(rapl_before, rapl_after);
        out_energy.energy_total_j = out_energy.energy_cpu_j;
        out_energy.edp_j_s = out_energy.energy_total_j * out_energy.time_total_s;
        const double flops_total =
            stencil_flops(nx, ny, op.flops_per_cell) * static_cast<double>(iters);
        out_energy.joules_per_gflop = out_energy.energy_total_j / (flops_total / 1e9);
    }
    out = *src;
    return build_metrics(nx, ny, avg_ms, op.flops_per_cell);
}

// Gemelo FP64 de benchmark_cpu_stencil: MISMA estructura de warm-up, misma
// ventana RAPL y mismo build_metrics, para que t_iter_ms de CPU_FP64 y de
// CPU_FP32 sean comparables sin asteriscos. Es la referencia de costo del
// patron de oro IEEE 754 (ver Options::cpu_fp64).
//
// Es una SEGUNDA pasada FP64 sobre la malla, deliberadamente separada de
// compute_cpu_stencil_fp64, y la duplicacion es intencional. Aquella funcion
// produce el ground truth y por cada iteracion barre la malla dos veces mas
// -- ||u^n||_inf para el modelo de horizonte, y all_finite_fp64 para saber
// hasta donde el checkpoint sigue siendo utilizable -- ademas de copiar
// checkpoints, de modo que su trafico de memoria es del orden de 3x el del
// stencil puro y no tiene warm-up. Cronometrar ESA funcion reportaria un FP64
// de CPU mucho mas caro de lo que realmente es e inflaria artificialmente todo
// speedup calculado contra ella, justo en la direccion que favorece la tesis.
// El costo de repetir la pasada es el precio de una medicion honesta;
// --cpu-fp64 off la omite cuando no hace falta.
static Metrics benchmark_cpu_fp64_stencil(const std::vector<double>& in,
                                          std::vector<double>& out,
                                          int nx,
                                          int ny,
                                          int iters,
                                          const StencilOperator& op,
                                          int& first_nonfinite_iter,
                                          EnergyMeasurement& out_energy) {
    const double c_neigh = neighbor_coeff_d(op);
    const double c_center = center_coeff_d(op);
    // first_nf == nullptr durante el warm-up: mismas razones que en la version
    // FP32 (esas iteraciones son descartables y no deben contaminar la medida).
    auto apply = [&](const std::vector<double>& src, std::vector<double>& dst,
                     int iter_number, int* first_nf) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
                    dst[idx2d(x, y, nx)] = src[idx2d(x, y, nx)];
                    continue;
                }

                const double up = src[idx2d(x, y - 1, nx)];
                const double down = src[idx2d(x, y + 1, nx)];
                const double left = src[idx2d(x - 1, y, nx)];
                const double right = src[idx2d(x + 1, y, nx)];
                const double center = src[idx2d(x, y, nx)];
                const double val = c_neigh * (up + down + left + right) + c_center * center;
                dst[idx2d(x, y, nx)] = val;
                if (first_nf != nullptr && *first_nf == INT_MAX && !std::isfinite(val)) {
                    *first_nf = iter_number;
                }
            }
        }
    };

    {
        std::vector<double> warm_a = in;
        std::vector<double> warm_b = in;
        std::vector<double>* warm_src = &warm_a;
        std::vector<double>* warm_dst = &warm_b;
        for (int i = 0; i < kWarmupIters; ++i) {
            apply(*warm_src, *warm_dst, i + 1, nullptr);
            std::swap(warm_src, warm_dst);
        }
    }

    std::vector<double> buf_a = in;
    std::vector<double> buf_b = in;
    std::vector<double>* src = &buf_a;
    std::vector<double>* dst = &buf_b;

    first_nonfinite_iter = INT_MAX;
    const RAEnergySnapshot rapl_before = rapl_snapshot_now();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        apply(*src, *dst, i + 1, &first_nonfinite_iter);
        std::swap(src, dst);
    }
    auto end = std::chrono::high_resolution_clock::now();

    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    const RAEnergySnapshot rapl_after = rapl_snapshot_now();
    out_energy = EnergyMeasurement{};
    out_energy.time_total_s = std::chrono::duration<double>(end - start).count();
    out_energy.gpu_valid = true;  // La ruta CPU no requiere una lectura NVML.
    out_energy.cpu_valid = rapl_before.valid && rapl_after.valid &&
                           rapl_after.energy_j >= rapl_before.energy_j;
    if (out_energy.cpu_valid) {
        out_energy.energy_cpu_j = rapl_energy_delta(rapl_before, rapl_after);
        out_energy.energy_total_j = out_energy.energy_cpu_j;
        out_energy.edp_j_s = out_energy.energy_total_j * out_energy.time_total_s;
        // stencil_flops cuenta operaciones, no bytes: es el mismo conteo que en
        // FP32 (mismo operador, mismo flops_per_cell), asi que joules_per_gflop
        // es directamente comparable entre ambas rutas de CPU.
        const double flops_total =
            stencil_flops(nx, ny, op.flops_per_cell) * static_cast<double>(iters);
        out_energy.joules_per_gflop = out_energy.energy_total_j / (flops_total / 1e9);
    }
    out = *src;
    return build_metrics(nx, ny, avg_ms, op.flops_per_cell);
}

// Devuelve false si algun elemento de v no es finito (usada solo para decidir
// hasta donde la referencia FP64 sigue siendo utilizable como ground truth
// de checkpoints; ver compute_cpu_stencil_fp64).
static bool all_finite_fp64(const std::vector<double>& v) {
    for (double x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

static bool all_finite_fp32(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

// Referencia FP64 (ground truth): version double de benchmark_cpu_stencil,
// encadenada por el MISMO numero de iteraciones (iters) que las rutas
// comparadas -condicion de aceptacion de Fase 3: sin esto el error vs FP64
// quedaria invalido para iters>1 (N pasos encadenados contra 1 solo paso)-.
// Opera sobre una copia en double del mismo input FP32, sin medir tiempo.
//
// checkpoint_every > 0 activa snapshots para medir drift: al completar cada
// iteracion multiplo de checkpoint_every (K, 2K, 3K, ...) se copia el estado
// actual a checkpoints.push_back(...). En cuanto un checkpoint no es
// finito se deja de tomar snapshots (el operador es lineal: una vez no
// finito, el campo se mantiene no finito el resto de iteraciones), de modo
// que checkpoints.size() ya equivale al numero de checkpoints "validos"
// (referencia finita), sin necesidad de un escaneo posterior sobre todos
// los snapshots guardados. checkpoint_every <= 0 preserva el comportamiento
// original: checkpoints queda vacio, sin copias ni memoria adicional.
//
// linf_per_iter retorna ||u^n||_inf para cada iteracion n=1..iters (usada
// para calibrar el horizonte de overflow desde el patron FP64, no desde la
// proyeccion Nyquist). Se llena siempre (incluso sin checkpoints), hasta que
// la referencia diverga.
static void compute_cpu_stencil_fp64(const std::vector<double>& in,
                                     std::vector<double>& out,
                                     int nx,
                                     int ny,
                                     int iters,
                                     const StencilOperator& op,
                                     int checkpoint_every,
                                     std::vector<std::vector<double>>& checkpoints,
                                     // Ruta de Fase 4 (--checkpoint-iters): en vez de acumular
                                     // un campo por checkpoint en `checkpoints`, cada uno se
                                     // vuelca a `spill` y se libera. Ver ReferenceSpill.
                                     const std::vector<int>* checkpoint_iters,
                                     ReferenceSpill* spill,
                                     const ArchiveContext* archive,
                                     std::vector<double>& linf_per_iter,
                                     int& first_nonfinite_iter) {
    const double c_neigh = neighbor_coeff_d(op);
    const double c_center = center_coeff_d(op);
    auto apply = [&](const std::vector<double>& src, std::vector<double>& dst) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
                    dst[idx2d(x, y, nx)] = src[idx2d(x, y, nx)];
                    continue;
                }

                const double up = src[idx2d(x, y - 1, nx)];
                const double down = src[idx2d(x, y + 1, nx)];
                const double left = src[idx2d(x - 1, y, nx)];
                const double right = src[idx2d(x + 1, y, nx)];
                const double center = src[idx2d(x, y, nx)];
                dst[idx2d(x, y, nx)] = c_neigh * (up + down + left + right) + c_center * center;
            }
        }
    };

    std::vector<double> buf_a = in;
    std::vector<double> buf_b = in;
    std::vector<double>* src = &buf_a;
    std::vector<double>* dst = &buf_b;
    checkpoints.clear();
    linf_per_iter.clear();
    bool reference_diverged = false;
    first_nonfinite_iter = INT_MAX;
    for (int i = 0; i < iters; ++i) {
        apply(*src, *dst);
        std::swap(src, dst);

        const int iter_number = i + 1;

        // Computa ||u^n||_inf: util para calibrar horizon de overflow desde
        // el patron FP64 (ver compute_overflow_horizon_from_reference). Se
        // agrega mientras la referencia sea finita; una vez que diverge se
        // detiene para evitar NaN/inf en el vector.
        double linf = 0.0;
        // sq acompana a linf en el MISMO barrido (coste despreciable frente al
        // stencil): es la norma L2 que CSV_NORM publica en cada checkpoint.
        double sq = 0.0;
        bool src_finite = true;
        for (const auto& x : *src) {
            if (!std::isfinite(x)) {
                src_finite = false;
                break;
            }
            linf = std::max(linf, std::fabs(x));
            sq += x * x;
        }
        if (src_finite) {
            linf_per_iter.push_back(linf);
        }

        if (first_nonfinite_iter == INT_MAX && !all_finite_fp64(*src)) {
            first_nonfinite_iter = iter_number;
        }
        if (checkpoint_every > 0 && !reference_diverged && iter_number % checkpoint_every == 0) {
            if (all_finite_fp64(*src)) {
                checkpoints.push_back(*src);
            } else {
                reference_diverged = true;
            }
        }

        // Ruta por lista: mismo criterio de divergencia que la de multiplos
        // (una vez no finita, la referencia deja de servir de ground truth y no
        // se guarda nada mas), pero el campo va a disco en vez de a RAM.
        const bool on_list =
            (checkpoint_iters != nullptr) &&
            std::find(checkpoint_iters->begin(), checkpoint_iters->end(), iter_number) !=
                checkpoint_iters->end();
        if (on_list && !reference_diverged) {
            if (src_finite) {
                if (spill == nullptr || !spill->append(iter_number, *src)) {
                    std::cerr << "ERROR: fallo al volcar la referencia FP64 de la iteracion "
                              << iter_number << " al spill. Sin ella los checkpoints de esa\n"
                                 "iteracion no tendrian contra que compararse; se aborta en vez"
                                 " de emitir NONFINITE enganosos.\n";
                    std::exit(EXIT_FAILURE);
                }
                emit_csv_norm_row(iter_number, std::sqrt(sq), linf);
                if (archive != nullptr && archive->due(iter_number)) {
                    archive_field(*archive, "CPU_FP64", iter_number, nx, ny, "float64", *src);
                }
            } else {
                reference_diverged = true;
            }
        }
    }
    out = *src;
}

// Marca en *first_nf la PRIMERA iteracion (atomicMin) en que algun punto
// interior del grid completo deja de ser finito. Reduccion en shared: una
// sola atomica por BLOQUE (hilo lider) en vez de una por hilo -- con ~268M
// hilos marcando sobre la misma direccion global al divergir, la version
// por hilo serializaba el kernel (t_div ~constante e independiente del
// kernel/precision, ver contexto). *(volatile int*)first_nf es una lectura
// no atomica: solo es un early-out para evitar atomicMin redundantes una vez
// que ya hay un iter menor o igual registrado; la correccion final depende
// solo de atomicMin, no de esta lectura.
__device__ inline void reduce_and_mark_first_nonfinite(int* first_nf, int iter, int blk_bad) {
    if (blk_bad != 0) {
        if (*(volatile int*)first_nf > iter) {
            atomicMin(first_nf, iter);
        }
    }
}

// Por que fmaf/fma EXPLICITO en los kernels y no en las rutas de CPU:
//
// La forma parametrizada c_n*(u+d+l+r) + c_c*center tiene DOS multiplicaciones
// alimentando una suma, mientras que la forma historica 0.25f*s - center tenia
// una sola. En device nvcc compila con --fmad=true (el sbatch no lo desactiva),
// asi que la forma historica contraia sin ambiguedad a fma(0.25, s, -center);
// con dos multiplicaciones el compilador puede elegir cual fusiona, y las dos
// opciones NO dan el mismo resultado bit a bit (una redondea 0.25*s antes de
// sumar, la otra no). Escribir el fma a mano fija la eleccion en la que
// reproduce la contraccion historica, de modo que --op-mode stress sigue siendo
// byte a byte el kernel de las fases 1-3.
//
// En las rutas de CPU se hace lo contrario -- expresion suelta, sin fmaf --
// porque alli el sbatch compila con -Xcompiler -ffp-contract=off: no hay
// contraccion que reproducir, y un fmaf host-side seria una llamada a libm
// (correctamente redondeada, es decir DISTINTA de las dos redondeos que hacia
// el codigo original, y ademas cara dentro del bucle). Con contraccion apagada,
// c_c*center con c_c = -1.0f es una negacion exacta y a - b === a + (-b) en
// IEEE 754, asi que la expresion suelta ya es bit a bit identica a la anterior.
__global__ static void stencil2d_fp32_kernel(const float* in, float* out, int nx, int ny,
                                             float c_neigh, float c_center,
                                             int iter, int* first_nf) {
    __shared__ int blk_bad;
    if (threadIdx.x == 0 && threadIdx.y == 0) blk_bad = 0;
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const bool in_range = (x < nx && y < ny);
    const bool active = in_range && !(x == 0 || y == 0 || x == nx - 1 || y == ny - 1);

    float val = 0.0f;
    if (in_range) {
        if (active) {
            const float up = in[idx2d(x, y - 1, nx)];
            const float down = in[idx2d(x, y + 1, nx)];
            const float left = in[idx2d(x - 1, y, nx)];
            const float right = in[idx2d(x + 1, y, nx)];
            const float center = in[idx2d(x, y, nx)];
            val = fmaf(c_neigh, up + down + left + right, c_center * center);
            if (!isfinite(val)) blk_bad = 1;    // carrera benigna: todos escriben 1
        } else {
            val = in[idx2d(x, y, nx)];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        reduce_and_mark_first_nonfinite(first_nf, iter, blk_bad);
    }
    if (in_range) out[idx2d(x, y, nx)] = val;
}

// Replica exacta de stencil2d_fp32_kernel en double: misma formula
// (c_neigh*(up+down+left+right) + c_center*center, con el mismo fma explicito),
// los mismos coeficientes del operador activo, mismo mapeo de hilos a celdas, misma
// guarda de borde (las celdas del contorno se copian tal cual) y la misma
// reduccion en shared de la primera iteracion no finita. Lo UNICO que cambia
// es el tipo de dato. Sin Tensor Cores, sin WMMA y sin compensacion: esta ruta
// es la referencia de maxima precision en GPU, no una variante mas del kernel
// mixto -- cualquier divergencia algoritmica respecto al FP32 clasico la
// invalidaria como denominador de speedup GPU-vs-GPU.
__global__ static void stencil2d_fp64_kernel(const double* in, double* out, int nx, int ny,
                                             double c_neigh, double c_center,
                                             int iter, int* first_nf) {
    __shared__ int blk_bad;
    if (threadIdx.x == 0 && threadIdx.y == 0) blk_bad = 0;
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const bool in_range = (x < nx && y < ny);
    const bool active = in_range && !(x == 0 || y == 0 || x == nx - 1 || y == ny - 1);

    double val = 0.0;
    if (in_range) {
        if (active) {
            const double up = in[idx2d(x, y - 1, nx)];
            const double down = in[idx2d(x, y + 1, nx)];
            const double left = in[idx2d(x - 1, y, nx)];
            const double right = in[idx2d(x + 1, y, nx)];
            const double center = in[idx2d(x, y, nx)];
            val = fma(c_neigh, up + down + left + right, c_center * center);
            if (!isfinite(val)) blk_bad = 1;    // carrera benigna: todos escriben 1
        } else {
            val = in[idx2d(x, y, nx)];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        reduce_and_mark_first_nonfinite(first_nf, iter, blk_bad);
    }
    if (in_range) out[idx2d(x, y, nx)] = val;
}

// Contexto compartido de checkpoints para las tres rutas de baja precision:
// snapshots FP64 por checkpoint (iteraciones {K, 2K, ...}, ver
// compute_cpu_stencil_fp64) y el intervalo K que los genero.
// checkpoint_every <= 0 desactiva el mecanismo por completo.
struct CheckpointContext {
    int checkpoint_every = 0;
    const std::vector<std::vector<double>>& fp64_checkpoints;
    // Ruta de Fase 4 (--checkpoint-iters): iteraciones exactas en vez de
    // multiplos, y referencia en disco en vez de en RAM. Cuando este puntero no
    // es nulo manda el, y checkpoint_every vale 0 (parse_args rechaza que
    // ambos esten activos). fp64_checkpoints queda vacio y sin usar.
    const std::vector<int>* checkpoint_iters = nullptr;
    const ReferenceSpill* spill = nullptr;
    // No nulo => en las iteraciones de archive->iters cada ruta vuelca su campo
    // completo a disco. Siempre subconjunto de checkpoint_iters.
    const ArchiveContext* archive = nullptr;
};

// Un unico predicado para las dos cadencias: sin esto la condicion vive
// duplicada en cuatro bucles de benchmark y basta que uno se quede con la
// version vieja para que esa ruta deje de emitir checkpoints en silencio.
static bool checkpoint_due(const CheckpointContext& ckpt, int iter_number) {
    if (ckpt.checkpoint_iters != nullptr) {
        return std::find(ckpt.checkpoint_iters->begin(), ckpt.checkpoint_iters->end(),
                         iter_number) != ckpt.checkpoint_iters->end();
    }
    return ckpt.checkpoint_every > 0 && iter_number % ckpt.checkpoint_every == 0;
}

// Si el mecanismo esta activo en cualquiera de sus dos formas. Decide el
// dimensionado de los buffers de checkpoint y si se excluye la energia del
// bloque de checkpoint de la ventana medida.
static bool checkpoints_enabled(const CheckpointContext& ckpt) {
    return ckpt.checkpoint_every > 0 ||
           (ckpt.checkpoint_iters != nullptr && !ckpt.checkpoint_iters->empty());
}

static bool archive_due(const CheckpointContext& ckpt, int iter_number) {
    return ckpt.archive != nullptr && ckpt.archive->due(iter_number);
}

// Emite una fila CSV_DRIFT parseable para (ruta, checkpoint). Reutiliza las
// guardas de finitud de ErrorMetrics: si la referencia FP64 o la ruta divergen
// en este checkpoint, imprime NONFINITE en los campos afectados en vez de un
// numero, para nunca retener una norma finita obsoleta ante inf/NaN.
static void emit_csv_drift_row(const char* route, int iter_number, const ErrorMetrics& e) {
    std::cout << "CSV_DRIFT," << route << "," << iter_number << ",";
    if (!e.reference_finite) {
        std::cout << "NONFINITE,NONFINITE,NONFINITE,NONFINITE\n";
        return;
    }

    std::cout << fmt_sci(e.ref_l2_norm) << ",";
    if (!e.solution_finite) {
        std::cout << "NONFINITE,NONFINITE,NONFINITE\n";
    } else {
        std::cout << fmt_sci(e.l2_abs) << "," << fmt_sci(e.rel_l2) << "," << fmt_sci(e.max_abs) << "\n";
    }
}

static void emit_csv_drift_nonfinite_reference_row(const char* route, int iter_number) {
    std::cout << "CSV_DRIFT," << route << "," << iter_number
              << ",NONFINITE,NONFINITE,NONFINITE,NONFINITE\n";
}

// Fila de la ruta por lista (--checkpoint-iters). Token PROPIO, distinto de
// CSV_DRIFT, por dos razones: lleva rel_linf (que CSV_DRIFT no tiene) y asi un
// log nunca mezcla las dos cadencias bajo el mismo esquema de columnas. Las
// herramientas de extraccion vigentes ignoran los tokens que no conocen, de
// modo que agregar CSV_CKPT no altera lo que ya leen.
//   CSV_CKPT,<route>,<iter>,<rel_l2>,<rel_linf>,<max_abs>
static void emit_csv_ckpt_row(const char* route, int iter_number, const ErrorMetrics& e) {
    std::cout << "CSV_CKPT," << route << "," << iter_number << ",";
    if (!e.reference_finite || !e.solution_finite) {
        std::cout << "NONFINITE,NONFINITE,NONFINITE\n";
        return;
    }
    std::cout << fmt_sci(e.rel_l2) << "," << fmt_sci(e.rel_linf) << ","
              << fmt_sci(e.max_abs) << "\n";
}

static void emit_csv_ckpt_nonfinite_reference_row(const char* route, int iter_number) {
    std::cout << "CSV_CKPT," << route << "," << iter_number
              << ",NONFINITE,NONFINITE,NONFINITE\n";
}

// Precondicion (garantizada por los llamadores, ver mas abajo): ckpt.checkpoint_every > 0
// y iter_number % ckpt.checkpoint_every == 0. Si el checkpoint cae fuera del
// rango con referencia finita (ver compute_cpu_stencil_fp64), emite CSV_DRIFT
// con ref_l2=NONFINITE y propaga esa no-finitud al resto de columnas de error
// sin cambiar el esquema historico del token. Si no, compara host_buf (ya
// copiado D2H por el llamador, sin copia extra) contra el snapshot FP64
// correspondiente. En ambos casos registra en onset_iter el PRIMER checkpoint
// en que la ruta (no la referencia) deja de ser finita.
static void record_checkpoint(const CheckpointContext& ckpt,
                              const char* route,
                              int iter_number,
                              const std::vector<float>& host_buf,
                              int& onset_iter) {
    // Ruta de Fase 4: la referencia de ESTA iteracion se relee del spill en vez
    // de indexarse en un vector residente. Si no esta guardada (la referencia
    // FP64 dejo de ser finita antes de llegar aqui), se emite NONFINITE con el
    // mismo criterio que la ruta historica.
    if (ckpt.checkpoint_iters != nullptr) {
        const std::vector<double>* ref =
            (ckpt.spill != nullptr) ? ckpt.spill->load(iter_number) : nullptr;
        if (ref == nullptr) {
            emit_csv_ckpt_nonfinite_reference_row(route, iter_number);
            if (!all_finite_fp32(host_buf) && onset_iter < 0) onset_iter = iter_number;
            return;
        }
        const ErrorMetrics e = compare_fp64_ref_vs_fp32(*ref, host_buf);
        emit_csv_ckpt_row(route, iter_number, e);
        if (!e.solution_finite && onset_iter < 0) onset_iter = iter_number;
        return;
    }

    const int ckpt_idx = iter_number / ckpt.checkpoint_every - 1;
    if (ckpt_idx < 0) return;

    if (ckpt_idx >= static_cast<int>(ckpt.fp64_checkpoints.size())) {
        emit_csv_drift_nonfinite_reference_row(route, iter_number);
        if (!all_finite_fp32(host_buf) && onset_iter < 0) {
            onset_iter = iter_number;
        }
        return;
    }

    const ErrorMetrics e = compare_fp64_ref_vs_fp32(ckpt.fp64_checkpoints[ckpt_idx], host_buf);
    emit_csv_drift_row(route, iter_number, e);
    if (!e.solution_finite && onset_iter < 0) {
        onset_iter = iter_number;
    }
}

// Version FP64/FP64 de compare_fp64_ref_vs_fp32 (Fase_2/common.cuh), byte a
// byte igual salvo que `test` ya es double y no hay cast que aplicar. No se
// usa compare_double_vectors -que si existe en common.cuh y compara el mismo
// par de tipos- porque ESA deja l2_abs y ref_l2_norm en 0.0 por diseno (ver su
// comentario): son justamente las dos primeras columnas numericas que
// emit_csv_drift_row imprime, asi que CSV_DRIFT saldria con ref_l2=0 y abs_l2=0
// para toda la ruta GPU_FP64. Se define aqui, y no ampliando common.cuh, para
// no alterar un header compartido con Fase 1 y Fase 2.
static ErrorMetrics compare_fp64_ref_vs_fp64(const std::vector<double>& ref_fp64,
                                             const std::vector<double>& test_fp64) {
    ErrorMetrics out;
    double sq_err = 0.0;
    double sq_ref = 0.0;
    double ref_linf = 0.0;
    for (size_t i = 0; i < ref_fp64.size(); ++i) {
        const double r = ref_fp64[i];
        const double t = test_fp64[i];
        if (!std::isfinite(r)) { out.reference_finite = false; continue; }

        // Misma regla que compare_fp64_ref_vs_fp32: la norma de la REFERENCIA
        // se acumula antes de mirar la solucion, para que no dependa de
        // cuantos puntos de la ruta evaluada siguen finitos.
        sq_ref += r * r;
        ref_linf = std::max(ref_linf, std::abs(r));

        if (!std::isfinite(t)) { out.solution_finite = false; continue; }

        const double diff = r - t;
        out.max_abs = std::max(out.max_abs, std::abs(diff));
        sq_err += diff * diff;
    }
    out.rel_l2 = (out.reference_finite && std::isfinite(sq_ref) && sq_ref > 0.0)
                 ? std::sqrt(sq_err / sq_ref) : 0.0;
    out.l2_abs = (out.reference_finite && std::isfinite(sq_err)) ? std::sqrt(sq_err) : 0.0;
    out.ref_l2_norm = (out.reference_finite && std::isfinite(sq_ref)) ? std::sqrt(sq_ref) : 0.0;
    out.ref_linf = (out.reference_finite && std::isfinite(ref_linf)) ? ref_linf : 0.0;
    out.rel_linf = (out.reference_finite && out.ref_linf > 0.0) ? out.max_abs / out.ref_linf : 0.0;
    return out;
}

// Version FP64 de record_checkpoint: identica en estructura y en el esquema de
// la fila CSV_DRIFT que emite; solo cambian el tipo del estado de la ruta y la
// funcion de comparacion. No se reutiliza la version FP32 convirtiendo el
// estado a float antes de comparar: eso inyectaria el redondeo de FP32 en el
// drift de la unica ruta que no lo tiene, que es precisamente la magnitud que
// esta ruta existe para acotar.
static void record_checkpoint_fp64(const CheckpointContext& ckpt,
                                   const char* route,
                                   int iter_number,
                                   const std::vector<double>& host_buf,
                                   int& onset_iter) {
    // Ver el comentario analogo en record_checkpoint: misma rama por lista, con
    // la comparacion FP64/FP64 en vez de FP64/FP32.
    if (ckpt.checkpoint_iters != nullptr) {
        const std::vector<double>* ref =
            (ckpt.spill != nullptr) ? ckpt.spill->load(iter_number) : nullptr;
        if (ref == nullptr) {
            emit_csv_ckpt_nonfinite_reference_row(route, iter_number);
            if (!all_finite_fp64(host_buf) && onset_iter < 0) onset_iter = iter_number;
            return;
        }
        const ErrorMetrics e = compare_fp64_ref_vs_fp64(*ref, host_buf);
        emit_csv_ckpt_row(route, iter_number, e);
        if (!e.solution_finite && onset_iter < 0) onset_iter = iter_number;
        return;
    }

    const int ckpt_idx = iter_number / ckpt.checkpoint_every - 1;
    if (ckpt_idx < 0) return;

    if (ckpt_idx >= static_cast<int>(ckpt.fp64_checkpoints.size())) {
        emit_csv_drift_nonfinite_reference_row(route, iter_number);
        if (!all_finite_fp64(host_buf) && onset_iter < 0) {
            onset_iter = iter_number;
        }
        return;
    }

    const ErrorMetrics e = compare_fp64_ref_vs_fp64(ckpt.fp64_checkpoints[ckpt_idx], host_buf);
    emit_csv_drift_row(route, iter_number, e);
    if (!e.solution_finite && onset_iter < 0) {
        onset_iter = iter_number;
    }
}

// Construye la medicion de energia a partir de escalares YA depurados del
// consumo de los bloques de checkpoint. make_energy_measurement (en
// tools/power_sampling.h) integra el buffer de muestras COMPLETO, incluido el
// hueco entre parada y reanudacion del muestreo, asi que no puede descontar
// esos tramos; las formulas de aqui son exactamente las suyas, solo cambian
// las entradas. Ver acumulacion por tramos en las rutas GPU de abajo.
static EnergyMeasurement make_energy_measurement_from_segments(bool gpu_valid,
                                                               double energy_gpu_j,
                                                               bool cpu_valid,
                                                               double energy_cpu_j,
                                                               double time_total_s,
                                                               double flops_total,
                                                               int gpu_segment_count) {
    EnergyMeasurement result;
    result.time_total_s = time_total_s;
    result.gpu_valid = gpu_valid;
    result.cpu_valid = cpu_valid;
    // El contador NVML se cuantiza POR TRAMO, no sobre la suma: cada tramo
    // aporta hasta un salto de error, asi que el minimo exigido de ventana se
    // multiplica por el numero de tramos (ver REGIMEN DE VALIDEZ en
    // tools/power_sampling.h). Sin checkpointing hay un solo tramo y esto se
    // reduce a time_total_s >= kEnergyWindowReliableSeconds.
    result.gpu_segment_count = gpu_segment_count;
    result.window_reliable =
        gpu_valid && gpu_segment_count > 0 &&
        time_total_s >= kEnergyWindowReliableSeconds *
                            static_cast<double>(gpu_segment_count);
    if (result.gpu_valid) {
        result.energy_gpu_j = energy_gpu_j;
        result.avg_power_w = (time_total_s > 0.0) ? result.energy_gpu_j / time_total_s : 0.0;
        result.energy_j = result.energy_gpu_j;
    }
    if (result.cpu_valid) {
        result.energy_cpu_j = energy_cpu_j;
    }
    if (result.gpu_valid && result.cpu_valid) {
        result.energy_total_j = result.energy_gpu_j + result.energy_cpu_j;
        result.edp_j_s = result.energy_total_j * time_total_s;
        result.joules_per_gflop = (flops_total > 0.0)
            ? result.energy_total_j / (flops_total / 1e9) : 0.0;
    }
    result.edp = result.energy_gpu_j * time_total_s;
    return result;
}

static Metrics benchmark_gpu_fp32_stencil(const std::vector<float>& in,
                                          std::vector<float>& out,
                                          int nx,
                                          int ny,
                                          int iters,
                                          const StencilOperator& op,
                                          const CheckpointContext& ckpt,
                                          const char* route_label,
                                          int& onset_iter,
                                          int& first_nonfinite_iter,
                                          double& t_checkpoint_ms_out,
                                          EnergyMeasurement& out_energy) {
    const size_t count = in.size();
    float* d_a = nullptr;
    float* d_b = nullptr;
    int* d_first_nf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_first_nf, sizeof(int)));
    // Ambos buffers arrancan como copia completa del input: el kernel nunca
    // escribe las celdas de borde, asi que deben preservarse desde el inicio
    // en cualquier buffer que llegue a jugar el rol de d_out.
    CHECK_CUDA(cudaMemcpy(d_a, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    PowerBuffer* power_buffer = power_buffer_create(0);
    const RAEnergySnapshot rapl_warmup_before = rapl_snapshot_now();
    power_buffer_start_sampling(power_buffer);

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // Warm-up encadenado y descartable: alterna d_a/d_b igual que el bucle
    // medido, pero al terminar se restauran ambos a una copia fresca del
    // input para que el bucle medido siempre arranque desde el estado
    // original (necesario para que --iters 1 coincida con Fase_2/Stencil).
    float* warm_in = d_a;
    float* warm_out = d_b;
    for (int i = 0; i < kWarmupIters; ++i) {
        stencil2d_fp32_kernel<<<grid, block>>>(warm_in, warm_out, nx, ny, op.neighbor,
                                               op.center, i + 1, d_first_nf);
        std::swap(warm_in, warm_out);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(d_a, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    // Reinicia el contador de overflow tras el warm-up: sus iteraciones son
    // descartables y no deben contaminar la medicion del bucle cronometrado.
    {
        const int init_val = INT_MAX;
        CHECK_CUDA(cudaMemcpy(d_first_nf, &init_val, sizeof(int), cudaMemcpyHostToDevice));
    }
    power_buffer_stop_sampling(power_buffer);
    power_buffer_samples_clear(power_buffer);
    const RAEnergySnapshot rapl_before = rapl_snapshot_now();
    (void)rapl_warmup_before;

    // Buffer host reutilizado para las copias D2H de checkpoint; vacio (sin
    // costo) cuando el checkpointing esta desactivado.
    std::vector<float> checkpoint_host_buf;
    if (checkpoints_enabled(ckpt)) {
        checkpoint_host_buf.resize(count);
    }

    float* d_in = d_a;
    float* d_out = d_b;
    CudaEventTimer timer;
    double total_ms = 0.0;
    double checkpoint_ms_total = 0.0;
    // La ventana de energia se mide por TRAMOS, con los mismos cortes que el
    // cronometro: cada bloque de checkpoint cierra el tramo vigente (integra y
    // vacia el buffer de muestras) y abre uno nuevo al terminar. No basta con
    // parar y reanudar el muestreo, porque power_buffer_energy_joules integra
    // el buffer completo y el trapecio que une la ultima muestra de un tramo
    // con la primera del siguiente reintroduciria justamente la energia del
    // checkpoint que se quiere excluir.
    double gpu_energy_j = 0.0;
    double gpu_window_s = 0.0;
    // Numero de tramos acumulados: fija energy_window_reliable junto con la
    // ventana total, porque el contador NVML se cuantiza por tramo y no sobre
    // la suma (cada tramo aporta hasta un salto de error).
    int gpu_segment_count = 0;
    bool gpu_energy_valid = true;
    double checkpoint_cpu_energy_j = 0.0;
    double checkpoint_pause_s = 0.0;
    auto close_energy_segment = [&]() {
        power_buffer_stop_sampling(power_buffer);
        gpu_energy_valid = gpu_energy_valid && power_buffer_capture_valid(power_buffer);
        gpu_energy_j += power_buffer_energy_joules(power_buffer);
        gpu_window_s += power_buffer_window_seconds(power_buffer);
        ++gpu_segment_count;
        power_buffer_samples_clear(power_buffer);
    };
    emit_csv_region_marker(route_label, "begin");
    const auto energy_t0 = std::chrono::steady_clock::now();
    power_buffer_samples_clear(power_buffer);
    power_buffer_start_sampling(power_buffer);
    timer.start();
    // El cronometro se pausa/reanuda alrededor del bloque de checkpoint: sin
    // eso, el tiempo GPU ocioso mientras el host hace el D2H queda
    // contabilizado en total_ms (ver diagnostico analogo en la ruta WMMA).
    for (int i = 0; i < iters; ++i) {
        stencil2d_fp32_kernel<<<grid, block>>>(d_in, d_out, nx, ny, op.neighbor,
                                               op.center, i + 1, d_first_nf);
        std::swap(d_in, d_out);

        // Solo LEE d_in (ya con el swap aplicado, ver comentario mas abajo);
        // no altera el ping-pong.
        if (checkpoint_due(ckpt, i + 1)) {
            total_ms += timer.stop_and_elapsed_ms();
            // Misma pausa que el cronometro, ahora tambien para la energia: el
            // D2H y el escaneo del host dejan la GPU ociosa y, sin excluirlos,
            // energy_gpu_j/edp_j_s medirian sobre todo la instrumentacion (con
            // CHECKPOINT_EVERY=5 llegaba a ~97% de la ventana).
            // pause_t0 se toma ANTES de close_energy_segment(): esa llamada
            // hace pthread_join sobre el hilo de muestreo y puede esperar hasta
            // un intervalo completo (~10 ms). Medirlo despues dejaba esa espera
            // dentro de energy_wall_s pese a estar fuera de la energia
            // integrada, inflando el denominador de avg_power_w/edp_j_s.
            const auto pause_t0 = std::chrono::steady_clock::now();
            close_energy_segment();
            const RAEnergySnapshot rapl_ckpt_before = rapl_snapshot_now();

            const auto ckpt_t0 = std::chrono::high_resolution_clock::now();
            CHECK_CUDA(cudaMemcpy(checkpoint_host_buf.data(), d_in,
                                  count * sizeof(float), cudaMemcpyDeviceToHost));
            record_checkpoint(ckpt, route_label, i + 1, checkpoint_host_buf, onset_iter);
            // En esta ruta el estado propagado ES el buffer FP32 recien copiado:
            // se archiva tal cual, sin conversion intermedia.
            if (archive_due(ckpt, i + 1)) {
                archive_field(*ckpt.archive, route_label, i + 1, nx, ny, "float32",
                              checkpoint_host_buf);
            }
            const auto ckpt_t1 = std::chrono::high_resolution_clock::now();
            checkpoint_ms_total +=
                std::chrono::duration<double, std::milli>(ckpt_t1 - ckpt_t0).count();

            const RAEnergySnapshot rapl_ckpt_after = rapl_snapshot_now();
            checkpoint_cpu_energy_j += rapl_energy_delta(rapl_ckpt_before, rapl_ckpt_after);
            power_buffer_start_sampling(power_buffer);
            checkpoint_pause_s += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pause_t0).count();

            timer.start();
        }
    }
    total_ms += timer.stop_and_elapsed_ms();
    close_energy_segment();
    const RAEnergySnapshot rapl_after = rapl_snapshot_now();
    const auto energy_t1 = std::chrono::steady_clock::now();
    emit_csv_region_marker(route_label, "end");
    // energy_wall_s proviene de los mismos timestamps de muestreo que se
    // integraron en gpu_energy_j (power_buffer_window_seconds acumulado por
    // tramo en close_energy_segment), no de un reloj de pared aparte: asi
    // avg_power_w/edp_j_s quedan derivados del mismo intervalo que la
    // energia.
    const double energy_wall_s = gpu_window_s;
    const double flops_total =
        stencil_flops(nx, ny, op.flops_per_cell) * static_cast<double>(iters);
    const bool cpu_energy_valid = rapl_before.valid && rapl_after.valid &&
                                  rapl_after.energy_j >= rapl_before.energy_j;
    const double cpu_energy_j = std::max(
        0.0, rapl_energy_delta(rapl_before, rapl_after) - checkpoint_cpu_energy_j);
    out_energy = make_energy_measurement_from_segments(
        gpu_energy_valid, gpu_energy_j, cpu_energy_valid, cpu_energy_j,
        energy_wall_s, flops_total, gpu_segment_count);
    power_buffer_destroy(power_buffer);
    t_checkpoint_ms_out = checkpoint_ms_total / iters;
    CHECK_CUDA(cudaGetLastError());
    // Tras el ultimo swap, d_in apunta al buffer con la salida mas reciente.
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&first_nonfinite_iter, d_first_nf, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_first_nf));
    return build_metrics(nx, ny, total_ms / iters, op.flops_per_cell);
}

// Ruta GPU_FP64: replica estructural de benchmark_gpu_fp32_stencil con el tipo
// cambiado a double. Conserva sin excepcion el warm-up descartable
// (kWarmupIters) con restauracion de ambos buffers, el reinicio del contador de
// overflow tras el warm-up, el ping-pong por std::swap, la pausa/reanudacion
// del cronometro alrededor del bloque de checkpoint y -- lo importante para la
// telemetria -- la ventana de energia POR TRAMOS: close_energy_segment() antes
// del D2H de checkpoint y power_buffer_start_sampling() despues, con pause_t0
// tomado ANTES de cerrar el tramo (el mismo orden validado en los jobs
// 4147/4148). Sin ese corte, energy_gpu_j de esta ruta mediria sobre todo la
// instrumentacion y no el kernel, y el speedup/EDP GPU-vs-GPU quedaria medido
// contra un denominador contaminado.
//
// El input llega en FP32 (misma condicion inicial que el resto de rutas) y se
// promueve a double aqui: promover la MISMA condicion inicial, en vez de
// generar una en double, es lo que hace comparables el error y el horizonte de
// esta ruta con los de GPU_FP32 y las WMMA.
//
// La entrada FP32 se promueve elemento a elemento en host y se sube una sola
// vez; el bucle medido no paga ninguna conversion (a diferencia de las rutas
// WMMA, que reconvierten cada iteracion). Por eso esta ruta no reporta
// t_convert_ms: no tiene kernel de conversion que cronometrar.
static Metrics benchmark_gpu_fp64_stencil(const std::vector<float>& in,
                                          std::vector<double>& out,
                                          int nx,
                                          int ny,
                                          int iters,
                                          const StencilOperator& op,
                                          const CheckpointContext& ckpt,
                                          const char* route_label,
                                          int& onset_iter,
                                          int& first_nonfinite_iter,
                                          double& t_checkpoint_ms_out,
                                          EnergyMeasurement& out_energy) {
    const size_t count = in.size();
    std::vector<double> in_fp64(count);
    for (size_t i = 0; i < count; ++i) {
        in_fp64[i] = static_cast<double>(in[i]);
    }

    double* d_a = nullptr;
    double* d_b = nullptr;
    int* d_first_nf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, count * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b, count * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_first_nf, sizeof(int)));
    // Ambos buffers arrancan como copia completa del input: el kernel nunca
    // escribe las celdas de borde, asi que deben preservarse desde el inicio
    // en cualquier buffer que llegue a jugar el rol de d_out.
    CHECK_CUDA(cudaMemcpy(d_a, in_fp64.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, in_fp64.data(), count * sizeof(double), cudaMemcpyHostToDevice));

    PowerBuffer* power_buffer = power_buffer_create(0);
    const RAEnergySnapshot rapl_warmup_before = rapl_snapshot_now();
    power_buffer_start_sampling(power_buffer);

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // Warm-up encadenado y descartable: alterna d_a/d_b igual que el bucle
    // medido, pero al terminar se restauran ambos a una copia fresca del
    // input para que el bucle medido siempre arranque desde el estado
    // original.
    double* warm_in = d_a;
    double* warm_out = d_b;
    for (int i = 0; i < kWarmupIters; ++i) {
        stencil2d_fp64_kernel<<<grid, block>>>(warm_in, warm_out, nx, ny, neighbor_coeff_d(op),
                                               center_coeff_d(op), i + 1, d_first_nf);
        std::swap(warm_in, warm_out);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(d_a, in_fp64.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, in_fp64.data(), count * sizeof(double), cudaMemcpyHostToDevice));

    // Ultimo uso de in_fp64 (la restauracion post-warmup de arriba): se libera
    // ANTES de abrir la ventana medida. A 16384^2 son 2 GiB que de otro modo
    // convivirian con checkpoint_host_buf y con los snapshots FP64 de la
    // referencia durante todo el bucle, en el job con el presupuesto de memoria
    // mas ajustado del barrido (ver run_stencil_tc.sbatch).
    in_fp64.clear();
    in_fp64.shrink_to_fit();

    // Reinicia el contador de overflow tras el warm-up: sus iteraciones son
    // descartables y no deben contaminar la medicion del bucle cronometrado.
    {
        const int init_val = INT_MAX;
        CHECK_CUDA(cudaMemcpy(d_first_nf, &init_val, sizeof(int), cudaMemcpyHostToDevice));
    }
    power_buffer_stop_sampling(power_buffer);
    power_buffer_samples_clear(power_buffer);
    const RAEnergySnapshot rapl_before = rapl_snapshot_now();
    (void)rapl_warmup_before;

    // Buffer host reutilizado para las copias D2H de checkpoint; vacio (sin
    // costo) cuando el checkpointing esta desactivado.
    std::vector<double> checkpoint_host_buf;
    if (checkpoints_enabled(ckpt)) {
        checkpoint_host_buf.resize(count);
    }

    double* d_in = d_a;
    double* d_out = d_b;
    CudaEventTimer timer;
    double total_ms = 0.0;
    double checkpoint_ms_total = 0.0;
    // Ventana de energia por TRAMOS, con los mismos cortes que el cronometro:
    // cada bloque de checkpoint cierra el tramo vigente y abre uno nuevo al
    // terminar, de modo que el consumo del D2H y del escaneo en host queda
    // fuera de gpu_energy_j (ver el mismo patron en benchmark_gpu_fp32_stencil).
    double gpu_energy_j = 0.0;
    double gpu_window_s = 0.0;
    // Numero de tramos acumulados: fija energy_window_reliable junto con la
    // ventana total, porque el contador NVML se cuantiza por tramo y no sobre
    // la suma (cada tramo aporta hasta un salto de error).
    int gpu_segment_count = 0;
    bool gpu_energy_valid = true;
    double checkpoint_cpu_energy_j = 0.0;
    double checkpoint_pause_s = 0.0;
    auto close_energy_segment = [&]() {
        power_buffer_stop_sampling(power_buffer);
        gpu_energy_valid = gpu_energy_valid && power_buffer_capture_valid(power_buffer);
        gpu_energy_j += power_buffer_energy_joules(power_buffer);
        gpu_window_s += power_buffer_window_seconds(power_buffer);
        ++gpu_segment_count;
        power_buffer_samples_clear(power_buffer);
    };
    emit_csv_region_marker(route_label, "begin");
    power_buffer_samples_clear(power_buffer);
    power_buffer_start_sampling(power_buffer);
    timer.start();
    // El cronometro se pausa/reanuda alrededor del bloque de checkpoint: sin
    // eso, el tiempo GPU ocioso mientras el host hace el D2H queda
    // contabilizado en total_ms (mismo diagnostico que en la ruta FP32).
    for (int i = 0; i < iters; ++i) {
        stencil2d_fp64_kernel<<<grid, block>>>(d_in, d_out, nx, ny, neighbor_coeff_d(op),
                                               center_coeff_d(op), i + 1, d_first_nf);
        std::swap(d_in, d_out);

        // Solo LEE d_in (ya con el swap aplicado); no altera el ping-pong.
        if (checkpoint_due(ckpt, i + 1)) {
            total_ms += timer.stop_and_elapsed_ms();
            // Misma pausa que el cronometro, tambien para la energia: el D2H y
            // el escaneo del host dejan la GPU ociosa y, sin excluirlos,
            // energy_gpu_j/edp_j_s medirian sobre todo la instrumentacion.
            // pause_t0 se toma ANTES de close_energy_segment() por el mismo
            // motivo que en la ruta FP32: esa llamada puede consumir tiempo de
            // pared propio, y medirlo despues lo dejaria dentro de
            // energy_wall_s pese a estar fuera de la energia acumulada.
            const auto pause_t0 = std::chrono::steady_clock::now();
            close_energy_segment();
            const RAEnergySnapshot rapl_ckpt_before = rapl_snapshot_now();

            const auto ckpt_t0 = std::chrono::high_resolution_clock::now();
            CHECK_CUDA(cudaMemcpy(checkpoint_host_buf.data(), d_in,
                                  count * sizeof(double), cudaMemcpyDeviceToHost));
            record_checkpoint_fp64(ckpt, route_label, i + 1, checkpoint_host_buf, onset_iter);
            if (archive_due(ckpt, i + 1)) {
                archive_field(*ckpt.archive, route_label, i + 1, nx, ny, "float64",
                              checkpoint_host_buf);
            }
            const auto ckpt_t1 = std::chrono::high_resolution_clock::now();
            checkpoint_ms_total +=
                std::chrono::duration<double, std::milli>(ckpt_t1 - ckpt_t0).count();

            const RAEnergySnapshot rapl_ckpt_after = rapl_snapshot_now();
            checkpoint_cpu_energy_j += rapl_energy_delta(rapl_ckpt_before, rapl_ckpt_after);
            power_buffer_start_sampling(power_buffer);
            checkpoint_pause_s += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pause_t0).count();

            timer.start();
        }
    }
    total_ms += timer.stop_and_elapsed_ms();
    close_energy_segment();
    const RAEnergySnapshot rapl_after = rapl_snapshot_now();
    emit_csv_region_marker(route_label, "end");
    // energy_wall_s proviene de las mismas marcas begin/end que se acumularon
    // en gpu_energy_j (power_buffer_window_seconds por tramo), no de un reloj
    // de pared aparte: avg_power_w/edp_j_s quedan derivados del mismo intervalo
    // que la energia.
    const double energy_wall_s = gpu_window_s;
    const double flops_total =
        stencil_flops(nx, ny, op.flops_per_cell) * static_cast<double>(iters);
    const bool cpu_energy_valid = rapl_before.valid && rapl_after.valid &&
                                  rapl_after.energy_j >= rapl_before.energy_j;
    const double cpu_energy_j = std::max(
        0.0, rapl_energy_delta(rapl_before, rapl_after) - checkpoint_cpu_energy_j);
    out_energy = make_energy_measurement_from_segments(
        gpu_energy_valid, gpu_energy_j, cpu_energy_valid, cpu_energy_j,
        energy_wall_s, flops_total, gpu_segment_count);
    power_buffer_destroy(power_buffer);
    t_checkpoint_ms_out = checkpoint_ms_total / iters;
    CHECK_CUDA(cudaGetLastError());
    // Tras el ultimo swap, d_in apunta al buffer con la salida mas reciente.
    out.resize(count);
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, count * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&first_nonfinite_iter, d_first_nf, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_first_nf));
    // checkpoint_pause_s queda acumulado como diagnostico del tiempo de pared
    // que el bloque de checkpoint mantiene fuera de la ventana medida; igual
    // que en la ruta FP32, no entra en ninguna metrica reportada.
    (void)checkpoint_pause_s;
    return build_metrics(nx, ny, total_ms / iters, op.flops_per_cell);
}

__global__ static void convert_float_to_half_kernel(const float* src, __half* dst, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dst[i] = __float2half(src[i]);
    }
}

__global__ static void convert_float_to_bfloat16_kernel(const float* src,
                                                        __nv_bfloat16* dst,
                                                        int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

__device__ inline float tc_to_float(__half v) {
    return __half2float(v);
}

__device__ inline float tc_to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// Misma funcion de conversion que convert_float_to_half_kernel /
// convert_float_to_bfloat16_kernel (__float2half / __float2bfloat16): el
// kernel WMMA la usa para escribir out_tc directamente, sin pasar por el
// kernel de conversion dentro del bucle. Un redondeo distinto rompería la
// comparabilidad con las fases anteriores.
template <typename T>
__device__ inline T float_to_tc(float v);

template <>
__device__ inline __half float_to_tc<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ inline __nv_bfloat16 float_to_tc<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// Compensacion del redondeo de ALMACENAMIENTO a 16 bits (no de la suma de los
// 5 vecinos, ver metodologia 5.3/4.1.4 y el comentario de
// benchmark_gpu_tensor_core_stencil). comp[idx] persiste en FP32 entre
// iteraciones el residuo indexado por celda; que se guarda ahi y con que signo
// depende de kMode:
//
//   Local (--kahan on): cuantizador con RETROALIMENTACION DE ERROR (error
//     feedback / noise shaping de primer orden). Se PRE-RESTA el residuo de la
//     escritura anterior antes de redondear y se guarda el nuevo residuo con
//     signo Q(y)-y. Intacta byte a byte. El nombre historico de la opcion
//     (--kahan) es enganoso y se conserva solo por compatibilidad de la CLI y
//     de los CSV ya emitidos: NO es suma compensada de Kahan, que exige un
//     acumulador vivo al que sumarle incrementos. Aqui val se recalcula entero
//     desde los 5 vecinos en cada iteracion, asi que no hay tal acumulador --
//     lo que hace la pre-resta es dar forma al espectro del ruido de
//     cuantizacion, empujandolo fuera de la banda donde vive la senal.
//
//     Consecuencia, y es la propiedad que gobierna cuando sirve: su eficacia
//     depende de la CORRELACION TEMPORAL del campo cuantizado. Medido en este
//     mismo codigo (1024^2, 20 iters, --ci-p 168, rel_l2_prop contra el ground
//     truth FP64, frente a --kahan off):
//
//       diffusive (campo suave, residuos correlacionados entre iteraciones):
//         FP16 -26.1 %, BF16 -24.5 % de error. El ruido dado forma cae fuera
//         de la banda de la senal y la retroalimentacion cancela.
//       stress (g(pi,pi) = -2, el modo Nyquist se duplica cada iteracion y
//         decorrelaciona el residuo): FP16 +8.7 %, BF16 +6.0 %. Sin
//         correlacion que explotar, la pre-resta solo inyecta ruido extra.
//
//     Es decir: no es un error algebraico y no debe "corregirse" a la
//     convencion de Spatial. Un intento de hacerlo (sustituirla por
//     comp = val - Q(val), sin reincorporar residuos durante la recurrencia)
//     se midio y quedo indistinguible de --kahan off: entre 0.00004 % y
//     0.003 % de diferencia a 20 iters, porque reconstruir en el readout solo
//     deshace el redondeo de la ULTIMA escritura y ese termino se diluye
//     segun se acumula error (a iters=1 valia -47.8 %, a iters=5 ya -6.3 %).
//     Lo que hay que declarar al interpretar --kahan on es su dependencia del
//     operador, no un supuesto defecto de la formula.
//
//   Spatial (--spatial-comp on): convencion de error feedback. Se guarda lo
//     que el redondeo PERDIO, comp = val - Q(val), de modo que el lector
//     reconstruye el valor FP32 exacto con Q(val) + comp (ver el uso en
//     stencil2d_wmma_kernel). Esa identidad es exacta en FP32 mientras val
//     este dentro del rango normal de T: Q(val) y val difieren en menos de un
//     factor 2, asi que la resta es exacta (Sterbenz) y cabe en la mantisa de
//     24 bits. Fuera de rango (|val| > 65504 en FP16) Q(val) es inf y comp
//     pasa a -inf: la iteracion siguiente produce NaN y first_nf lo marca --
//     es decir, la variante espacial NO extiende el limite de RANGO del
//     formato, solo elimina el error de PRECISION del almacenamiento.
//
// Con kMode == Off, comp no se toca (puede ser nullptr) y esto colapsa a
// float_to_tc<T> sin rama en tiempo de ejecucion (if constexpr, resuelto en
// compilacion): la ruta --kahan off no paga costo alguno.
template <typename T, CompMode kMode>
__device__ inline T compensated_store(float val, float* comp, int idx) {
    if constexpr (kMode == CompMode::Local) {
        // Lazo de retroalimentacion de error (ver el bloque de doc de arriba
        // para por que NO es Kahan y de que depende su eficacia). El residuo
        // de la escritura ANTERIOR de esta misma celda se resta antes de
        // cuantizar, y el nuevo residuo se guarda con signo Q(y)-y para que la
        // proxima iteracion lo reste, no lo sume.
        const float y = val - comp[idx];
        const T s = float_to_tc<T>(y);
        comp[idx] = tc_to_float(s) - y;
        return s;
    } else if constexpr (kMode == CompMode::Spatial) {
        const T s = float_to_tc<T>(val);
        comp[idx] = val - tc_to_float(s);
        return s;
    } else {
        return float_to_tc<T>(val);
    }
}

// Siembra el residuo inicial (solo modo Spatial) con lo que perdio la
// conversion FP32 -> T de la condicion inicial: comp[i] = u0[i] - Q(u0[i]),
// misma convencion de signo que compensated_store<Spatial> (Q(v) + comp
// reconstruye v).
//
// Por que es necesario y no un extra: sin esto la variante espacial
// compensaria TODAS las escrituras menos la primera, y esa primera domina el
// error final. Con lambda~2 el error de una inyeccion en la iteracion k se
// amplifica 2^(n-k), asi que la serie de inyecciones esta dominada por las mas
// tempranas y la conversion inicial (k=0) es el termino mayor de todos. Dejarla
// sin compensar pondria un piso al error que ninguna compensacion posterior
// puede bajar, y la medicion de "sirve la compensacion espacial?" quedaria
// midiendo ese piso en vez del efecto bajo estudio.
//
// Consecuencia a declarar en la interpretacion: el estado propagado de la ruta
// espacial es el PAR (buffer T, buffer comp) = 6 bytes/celda en FP16, no 2. La
// comparacion honesta de costo es contra eso, no contra los 2 bytes de
// --kahan off|on.
//
// Se aplica a TODAS las celdas, borde incluido: el borde nunca se recalcula, de
// modo que su comp queda fijo en el valor sembrado y las celdas interiores
// vecinas al borde reconstruyen su valor FP32 exacto en cada iteracion.
template <typename T>
__global__ static void seed_comp_from_conversion_kernel(const float* __restrict__ src_fp32,
                                                        const T* __restrict__ src_tc,
                                                        float* __restrict__ comp,
                                                        int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        comp[i] = src_fp32[i] - tc_to_float(src_tc[i]);
    }
}

static __half make_tc_value_half(float x) {
    return __float2half(x);
}

static __nv_bfloat16 make_tc_value_bfloat16(float x) {
    return __float2bfloat16(x);
}

template <typename T>
static T make_tc_value(float x);
template <>
__half make_tc_value<__half>(float x) { return make_tc_value_half(x); }
template <>
__nv_bfloat16 make_tc_value<__nv_bfloat16>(float x) { return make_tc_value_bfloat16(x); }

// ---------------------------------------------------------------------------
// Operadores H y V: el stencil como DOS productos matriciales
// ---------------------------------------------------------------------------
//
// Sea X el tile 16x16 del estado, indexado X[i][j] = u(x0+j, y0+i) (i recorre
// y, j recorre x). El stencil de 5 puntos sobre las celdas INTERIORES del tile
// es
//
//   Y[i][j] = c_c*X[i][j] + c_n*( X[i][j-1] + X[i][j+1] + X[i-1][j] + X[i+1][j] )
//
// Los dos primeros vecinos desplazan la COLUMNA (eje x) y los dos ultimos la
// FILA (eje y). Un desplazamiento de columna es una multiplicacion POR LA
// DERECHA; uno de fila, una multiplicacion POR LA IZQUIERDA. De ahi:
//
//   (X H)[i][j] = sum_k X[i][k] H[k][j]
//               = c_c*X[i][j] + c_n*X[i][j-1] + c_n*X[i][j+1]
//     con H[k][j] = c_c si k=j;  c_n si k=j-1;  c_n si k=j+1;  0 en otro caso
//
//   (V X)[i][j] = sum_k V[i][k] X[k][j]
//               = c_n*X[i-1][j] + c_n*X[i+1][j]
//     con V[i][k] = c_n si k=i-1;  c_n si k=i+1;  0 en otro caso
//
// Sumando:  Y = X H + V X.
//
// El termino central va en H y NO en V (poner c_c en ambas lo contaria dos
// veces). Ambas matrices son simetricas y tridiagonales -- V sin diagonal --,
// asi que su transpuesta es irrelevante para el layout.
//
// Esto sustituye a la formulacion anterior, que multiplicaba CINCO tiles
// desplazados por matrices identidad escaladas: cinco mma_sync y cinco tiles
// materializados en shared, donde ahora bastan DOS mma_sync y UN tile. Las
// identidades solo servian para que el Tensor Core copiara y escalara, que es
// trabajo que el hardware no acelera; H y V, en cambio, ponen el stencil dentro
// del propio producto matricial.
//
// Limitacion estructural, resuelta aparte: H y V solo alcanzan a las vecinas
// que caen DENTRO del tile. Las cuatro bandas exteriores (columna x0-1, columna
// x0+16, fila y0-1, fila y0+16) no aparecen en X y se suman despues, en FP32.
// Las esquinas diagonales del halo no hacen falta: el stencil de 5 puntos no
// las usa.
template <typename T>
static void initialize_horizontal_operator(std::vector<T>& mat, const StencilOperator& op) {
    std::fill(mat.begin(), mat.end(), make_tc_value<T>(0.0f));
    for (int k = 0; k < kTile; ++k) {
        for (int j = 0; j < kTile; ++j) {
            float v = 0.0f;
            if (k == j) v = op.center;
            else if (k == j - 1 || k == j + 1) v = op.neighbor;
            if (v != 0.0f) mat[k * kTile + j] = make_tc_value<T>(v);
        }
    }
}

template <typename T>
static void initialize_vertical_operator(std::vector<T>& mat, const StencilOperator& op) {
    std::fill(mat.begin(), mat.end(), make_tc_value<T>(0.0f));
    for (int i = 0; i < kTile; ++i) {
        for (int k = 0; k < kTile; ++k) {
            if (k == i - 1 || k == i + 1) mat[i * kTile + k] = make_tc_value<T>(op.neighbor);
        }
    }
}

// ---------------------------------------------------------------------------
// Presupuesto de shared por WARP
// ---------------------------------------------------------------------------
//
//   offset 0                    X        256*sizeof(T)   (512 B en FP16/BF16)
//   offset 512                  out      256*sizeof(float)          1024 B
//   offset 1536                 bandas   4*16*sizeof(T)              128 B
//   -------------------------------------------------- Off / Local: 1664 B
//   offset 1664                 comp     256*sizeof(float)          1024 B
//   offset 2688                 comp_b   4*16*sizeof(float)          256 B
//   -------------------------------------------------- Spatial:     2944 B
//
// Contra los 3584 B/warp del diseno anterior (cinco tiles + salida). Los dos
// totales son multiplos de 32, de modo que el bloque de CADA warp arranca
// alineado a 32 B y el puntero de X cumple el requisito de alineacion de
// wmma::load_matrix_sync (256 bits). El ldm de X es kTile = 16 elementos = 32
// bytes en 16 bits, multiplo de 16 B, tambien valido.
//
// El espacio de la compensacion espacial solo se reserva en CompMode::Spatial:
// Off y Local no lo pagan.
template <typename T>
__host__ __device__ constexpr size_t wmma_x_tile_bytes() {
    return kTile * kLdX * sizeof(T);
}
__host__ __device__ constexpr size_t wmma_out_tile_bytes() {
    return kTile * kLdF * sizeof(float);
}
template <typename T>
__host__ __device__ constexpr size_t wmma_bands_bytes() {
    return 4 * kTile * sizeof(T);
}
__host__ __device__ constexpr size_t wmma_comp_center_bytes() {
    return kTile * kLdF * sizeof(float);
}
__host__ __device__ constexpr size_t wmma_comp_bands_bytes() {
    return 4 * kTile * sizeof(float);
}
template <typename T>
__host__ __device__ constexpr size_t wmma_warp_shared_bytes(CompMode mode) {
    return wmma_x_tile_bytes<T>() + wmma_out_tile_bytes() + wmma_bands_bytes<T>()
         + ((mode == CompMode::Spatial)
                ? (wmma_comp_center_bytes() + wmma_comp_bands_bytes())
                : 0);
}

// ---------------------------------------------------------------------------
// Tile ANCHO de bloque: 16x64 compartido por los kWarpsPerBlock warps
// ---------------------------------------------------------------------------
//
// Motivo, medido con ncu (job 6730): las bandas IZQUIERDA y DERECHA recorren y
// con paso nx, asi que cada elemento de 2 B cae en su propio sector de 32 B.
// Por cada tile 16x16 son 32 sectores para 64 B utiles -- el 64 % de los
// sectores de carga para el 10 % de los bytes. Ese coste es FIJO por tile, no
// por celda, de modo que cuadruplicar el ancho lo reparte entre 4x celdas:
//
//   por cada 256 celdas utiles     tile 16x16      tile 16x64
//     X                            16 sectores     16 sectores
//     bandas superior/inferior      2               2
//     bandas izquierda/derecha     32               8
//     TOTAL                        50              26          (1.9x menos)
//
// El segundo motivo es que un tile de 16 elementos son 32 B por fila, y con
// solo dos trozos de 16 B por fila NO hay grados de libertad para swizzlear
// (ver el barrido de padding del job 6735, donde ninguna variante gano). Con 64
// elementos la fila mide 128 B = los 32 bancos exactos, y el XOR clasico
// funciona.
constexpr int kTileW = kWarpsPerBlock * kTile;      // 64 columnas por bloque

#ifndef STENCIL_SWIZZLE
#define STENCIL_SWIZZLE 1
#endif
// ldmatrix por defecto ACTIVO. Se decidio con el job 6738, y el motivo no es
// el que parecia: en `off` solo aporta un 1.3 %, pero en `local` aporta un
// 11.4 %. Ese modo esta limitado por instrucciones (256 escrituras escalares a
// comp[] por tile), y calcular swz_x en cada acceso escalar de fragmento le
// anade ALU sin darle nada a cambio -- alli los conflictos de banco no eran el
// cuello. ldmatrix reduce eso a UNA direccion por fragmento en vez de cuatro u
// ocho, y con el swizzle solo (STENCIL_LDMATRIX=0) `local` se hunde un 11 %.
#ifndef STENCIL_LDMATRIX
#define STENCIL_LDMATRIX 1
#endif
constexpr bool kSwizzle = (STENCIL_SWIZZLE != 0);

// Permutacion XOR del tile ancho de X, en trozos de 8 elementos (16 B).
//
// La fila r coloca su trozo c en la posicion (c ^ (r & 7)). Como es una
// PERMUTACION dentro de la fila, no consume un solo byte extra -- a diferencia
// del padding, que fue justo lo que lo hundio.
//
// Grados de conflicto de banco calculados sobre el reparto real de las lanes:
//
//   acceso                     ancho 16 (antes)   ancho 64 plano   ancho 64 swz
//   fragmento A (X en matrix_a)       2                 8               1
//   fragmento B (X en matrix_b)       2                 4               1
//
// Sin swizzle, ensanchar EMPEORA el fragmento A de 2 a 8 vias: las dos cosas no
// son independientes, tienen que ir juntas.
//
// Los pares de columnas contiguas que piden los fragmentos (2*tid, 2*tid+1 y
// 2*tid+8, +9) caen siempre DENTRO del mismo trozo de 8, asi que siguen siendo
// contiguos despues de permutar y se leen con un solo acceso de 32 bits.
__host__ __device__ __forceinline__ int swz_x(int r, int c) {
    return kSwizzle ? (r * kTileW + ((((c >> 3) ^ (r & 7)) << 3) | (c & 7)))
                    : (r * kTileW + c);
}

// ldm del tile ancho de residuos. 64 floats por fila son 256 B = dos vueltas
// completas a los 32 bancos, de modo que TODAS las filas empezarian en el mismo
// banco y el epilogo (que cubre dos filas por paso de warp) chocaria a 2 vias.
// Con 80 el desplazamiento por fila es de 16 bancos y las dos filas de un warp
// quedan disjuntas. Aqui si se paga con bytes porque comp_tile no lo leen los
// fragmentos: no hay un patron rival al que el XOR tenga que servir a la vez.
constexpr int kLdC = kTileW + kTile;                // 80 floats por fila
static_assert(kLdC % 4 == 0, "kLdC*sizeof(float) debe ser multiplo de 16 B");

template <typename T>
__host__ __device__ constexpr size_t wide_x_bytes() { return kTile * kTileW * sizeof(T); }
__host__ __device__ constexpr size_t wide_out_bytes() {
    return static_cast<size_t>(kWarpsPerBlock) * kTile * kLdF * sizeof(float);
}
template <typename T>
__host__ __device__ constexpr size_t wide_bands_bytes() {
    return 2 * static_cast<size_t>(kTile + kTileW) * sizeof(T);
}
__host__ __device__ constexpr size_t wide_comp_bytes() { return kTile * kLdC * sizeof(float); }
__host__ __device__ constexpr size_t wide_comp_bands_bytes() {
    return 2 * static_cast<size_t>(kTile + kTileW) * sizeof(float);
}
template <typename T>
__host__ __device__ constexpr size_t wide_block_shared_bytes(CompMode mode) {
    return wide_x_bytes<T>() + wide_out_bytes() + wide_bands_bytes<T>()
         + ((mode == CompMode::Spatial)
                ? (wide_comp_bytes() + wide_comp_bands_bytes())
                : 0);
}

// El bloque reserva el MAXIMO de los dos layouts porque solo uno de los dos se
// usa en cada bloque: el ancho cuando el tile 16x64 cabe entero, y el de cuatro
// regiones por warp cuando no (ver la rama de respaldo del kernel).
template <typename T>
__host__ __device__ constexpr size_t block_shared_bytes(CompMode mode) {
    return (wide_block_shared_bytes<T>(mode)
                > kWarpsPerBlock * wmma_warp_shared_bytes<T>(mode))
         ? wide_block_shared_bytes<T>(mode)
         : kWarpsPerBlock * wmma_warp_shared_bytes<T>(mode);
}

// Ocho elementos de 16 bits = 16 B: la unidad de acceso de 128 bits de la ruta
// vectorizada. El alignas(16) es lo que autoriza al compilador a emitir un
// LDG.128/STG.128 en vez de ocho accesos escalares; sin el, el cast seria
// legal pero el acceso se desharia en ocho.
template <typename T>
struct alignas(16) TVec8 {
    T v[8];
};

// ---------------------------------------------------------------------------
// mma.sync.aligned.m16n8k16 explicito (sm_80+).
//
// No es una instruccion distinta de la que ya se ejecutaba: wmma m16n16k16 con
// acumulador FP32 se baja a EXACTAMENTE dos mma.m16n8k16 (una por mitad de n),
// con la misma particion del acumulador. Lo que cambia no es la aritmetica sino
// la VISIBILIDAD del reparto lane->celda, que wmma oculta tras load/store_
// matrix_sync y que el ISA de PTX especifica. Esa visibilidad es el requisito
// para swizzlear la shared: store_matrix_sync solo acepta un ldm plano, y un
// ldm plano es justo lo que hizo fracasar al barrido de padding (job 6735,
// ninguna de las 5 variantes gano; el padding CREA conflictos en el epilogo
// escalar, que a ldm = 16 estaba libre de ellos).
//
// Reparto de fragmentos (PTX ISA, "Matrix Fragments for mma.m16n8k16"), con
// gid = lane>>2 (0..7) y tid = lane&3 (0..3):
//
//   A (16x16, row-major)   ra0 -> (gid,   2tid) (gid,   2tid+1)
//                          ra1 -> (gid+8, 2tid) (gid+8, 2tid+1)
//                          ra2 -> (gid,   2tid+8) (gid,   2tid+9)
//                          ra3 -> (gid+8, 2tid+8) (gid+8, 2tid+9)
//   B (16x8,  col-major)   rb0 -> (2tid,   gid) (2tid+1, gid)
//                          rb1 -> (2tid+8, gid) (2tid+9, gid)
//   C/D (16x8)             d0,d1 -> (gid,   2tid) (gid,   2tid+1)
//                          d2,d3 -> (gid+8, 2tid) (gid+8, 2tid+1)
//
// Cada par de A y de D son columnas CONTIGUAS de la misma fila: sobre un origen
// row-major entran y salen con un unico acceso de 32 / 64 bits. El par de B son
// filas contiguas de la misma columna, separadas por ldm: dos accesos de 16
// bits y un empaquetado.
//
// El sufijo .row.col NO describe como esta la matriz en memoria: describe el
// orden interno del fragmento, y es la unica combinacion que m16n8k16 admite en
// sm_80. B[k][n] se sigue leyendo de un origen row-major como src[k*ldm + n],
// igual que hacia wmma::load_matrix_sync sobre un fragment<matrix_b,row_major>.

__device__ __forceinline__ uint32_t tc_pack2(__half lo, __half hi) {
    return (static_cast<uint32_t>(__half_as_ushort(hi)) << 16)
         |  static_cast<uint32_t>(__half_as_ushort(lo));
}
__device__ __forceinline__ uint32_t tc_pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16)
         |  static_cast<uint32_t>(__bfloat16_as_ushort(lo));
}

// Dos elementos de 16 bits contiguos leidos como un unico acceso de 32 bits.
// Todos los usos caen en indices de elemento PARES sobre bases alineadas a 32 B
// (la region de shared de cada warp es multiplo de 32, y cudaMalloc da 256), y
// kLdX / kTile son pares, asi que la direccion es siempre multiplo de 4.
__device__ __forceinline__ uint32_t tc_ld32(const void* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

// El tag por puntero nulo selecciona el sufijo del PTX sin construir nada.
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2],
                                             const float (&c)[4],
                                             const __half*) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    (void)a; (void)b;
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2],
                                             const float (&c)[4],
                                             const __nv_bfloat16*) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
    (void)a; (void)b;
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}

// Aritmetica del epilogo por CELDA. Se extrae a una sola funcion para que las
// dos rutas de acceso del kernel -- la vectorizada y la escalar de respaldo --
// no puedan divergir nunca: hay UNA copia de la formula, no dos.
//
// La secuencia es transcripcion literal de la que tenia el bucle escalar:
// mismo orden de las cuatro correcciones de banda, mismo fmaf explicito, mismos
// casts y misma posicion. Es parte del objeto experimental y no se toca.
//
// Devuelve el valor FP32 corregido y NO almacena nada: la cuantizacion y el
// calculo del residuo siguen siendo responsabilidad exclusiva de
// compensated_store, que es la referencia numerica.
template <typename T, CompMode kMode>
__device__ __forceinline__ float wmma_epilogue_value(
        float val, int local_x, int local_y, int sidx,
        const T* band_l, const T* band_r, const T* band_u, const T* band_d,
        const float* comp_tile,
        const float* comp_band_l, const float* comp_band_r,
        const float* comp_band_u, const float* comp_band_d,
        float c_neigh, float c_center) {
    // H y V solo alcanzan a las vecinas que caen DENTRO del tile: las celdas
    // del borde del tile pierden una vecina cada una, que se suma aqui en FP32
    // desde las bandas. Las cuatro esquinas del tile satisfacen dos de estas
    // condiciones a la vez y reciben, como corresponde, las DOS contribuciones.
    if (local_x == 0)         val += c_neigh * tc_to_float(band_l[local_y]);
    if (local_x == kTile - 1) val += c_neigh * tc_to_float(band_r[local_y]);
    if (local_y == 0)         val += c_neigh * tc_to_float(band_u[local_x]);
    if (local_y == kTile - 1) val += c_neigh * tc_to_float(band_d[local_x]);

    if constexpr (kMode == CompMode::Spatial) {
        // El estado que entro al Tensor Core es de tipo T: sumarle el residuo
        // FP32 antes del mma lo destruiria al reconvertir a 16 bits. Como el
        // operador es LINEAL, la correccion se calcula aparte en FP32 y se suma
        // al acumulador ya volcado:
        //   L(v + c) = L(v) + L(c)
        // donde L es el mismo Laplaciano de 5 puntos, v el estado almacenado en
        // T y c el residuo. Equivale exactamente a leer v+c en cada vecina (que
        // es lo que hace la rama de tile parcial del kernel), sin sacar el
        // trabajo pesado de los Tensor Cores ni tocar la formula del stencil.
        // Los cinco residuos salen de shared: los interiores del tile central y
        // los del borde de las bandas ya cargadas.
        // sidx es el indice en SHARED (fila * kLdF + columna), no el indice
        // logico de celda: con padding las dos cosas dejan de coincidir. El
        // vecino horizontal sigue a distancia 1; el vertical, a kLdF.
        const float cc = comp_tile[sidx];
        const float cl = (local_x > 0)         ? comp_tile[sidx - 1]
                                               : comp_band_l[local_y];
        const float cr = (local_x < kTile - 1) ? comp_tile[sidx + 1]
                                               : comp_band_r[local_y];
        const float cu = (local_y > 0)         ? comp_tile[sidx - kLdF]
                                               : comp_band_u[local_x];
        const float cd = (local_y < kTile - 1) ? comp_tile[sidx + kLdF]
                                               : comp_band_d[local_x];
        val += fmaf(c_neigh, cu + cd + cl + cr, c_center * cc);
    }
    return val;
}

// Gemelo del anterior para el tile ANCHO. Misma secuencia de operaciones en el
// mismo orden -- se duplica en vez de parametrizarse justamente para que se
// pueda leer una al lado de la otra y comprobar que lo es.
//
// Lo unico que cambia es DE DONDE sale cada vecina, no como se combina:
//
//   - las costuras internas del tile (columnas 16, 32 y 48) ya no necesitan una
//     banda traida de global: la vecina esta en el propio x_tile, que los cuatro
//     warps cargaron juntos. Solo los dos bordes EXTERNOS usan band_l/band_r.
//   - el bloque de 16 columnas de cada warp empieza en las mismas posiciones
//     globales que los tiles de 16 de antes (x0 + 16w con x0 = 1 + 64*tile_x),
//     asi que cada celda recibe exactamente las mismas correcciones que recibia.
//     Esa coincidencia es la que hace que el resultado sea identico bit a bit.
template <typename T, CompMode kMode>
__device__ __forceinline__ float wide_epilogue_value(
        float val, int local_x, int local_y, int warp_id,
        const T* x_tile,
        const T* band_l, const T* band_r, const T* band_u, const T* band_d,
        const float* comp_tile,
        const float* comp_band_l, const float* comp_band_r,
        const float* comp_band_u, const float* comp_band_d,
        float c_neigh, float c_center) {
    const int gx = warp_id * kTile + local_x;      // columna dentro del 16x64
    if (local_x == 0)
        val += c_neigh * tc_to_float((warp_id == 0)
                                         ? band_l[local_y]
                                         : x_tile[swz_x(local_y, gx - 1)]);
    if (local_x == kTile - 1)
        val += c_neigh * tc_to_float((warp_id == kWarpsPerBlock - 1)
                                         ? band_r[local_y]
                                         : x_tile[swz_x(local_y, gx + 1)]);
    if (local_y == 0)         val += c_neigh * tc_to_float(band_u[gx]);
    if (local_y == kTile - 1) val += c_neigh * tc_to_float(band_d[gx]);
    if constexpr (kMode == CompMode::Spatial) {
        const int ci = local_y * kLdC + gx;
        const float cc = comp_tile[ci];
        // La compensacion espacial es un stencil de 5 puntos COMPLETO sobre
        // comp_prev, evaluado fuera del mma: sus vecinas se toman por posicion
        // GLOBAL en el tile (gx), no por posicion dentro del bloque del warp.
        const float cl = (gx > 0)           ? comp_tile[ci - 1]    : comp_band_l[local_y];
        const float cr = (gx < kTileW - 1)  ? comp_tile[ci + 1]    : comp_band_r[local_y];
        const float cu = (local_y > 0)      ? comp_tile[ci - kLdC] : comp_band_u[gx];
        const float cd = (local_y < kTile - 1) ? comp_tile[ci + kLdC] : comp_band_d[gx];
        val += fmaf(c_neigh, cu + cd + cl + cr, c_center * cc);
    }
    return val;
}

// Direccion de shared en el espacio de 32 bits que exige ldmatrix.
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// kMode (parametro de plantilla, no runtime): elige la politica de
// compensacion del redondeo de almacenamiento (ver CompMode /
// compensated_store). comp/comp_prev son nullptr y no se tocan cuando
// kMode == Off -- el llamador (benchmark_gpu_tensor_core_stencil) elige la
// instanciacion en tiempo de compilacion segun los flags, asi la ruta off no
// paga rama ni acceso a comp.
//
// comp:      buffer de residuos que ESTA iteracion escribe. En modo Local es
//            tambien el que lee (actualizacion en sitio, cada celda solo toca
//            su propia entrada: no hay carrera).
// comp_prev: solo en modo Spatial, buffer de residuos de la iteracion
//            ANTERIOR (el que corresponde a `in`). Es un buffer DISTINTO de
//            comp, en ping-pong con el, porque aqui cada celda lee las
//            entradas de sus 4 vecinas mientras esas mismas vecinas escriben
//            las suyas: actualizar en sitio seria una carrera lectura/escritura
//            entre bloques, exactamente el mismo motivo por el que in/out ya
//            estan en ping-pong. nullptr en los modos Off/Local, donde nunca se
//            dereferencia (por eso __restrict__ aqui es valido: en Spatial
//            comp y comp_prev nunca apuntan al mismo buffer).
// horizontal_op / vertical_op son las matrices H y V del operador activo (ver
// initialize_horizontal_operator / initialize_vertical_operator): H lleva el
// coeficiente central y los dos vecinos horizontales, V los dos verticales, y
// juntas producen el stencil como Y = X H + V X con DOS mma_sync.
// c_neigh / c_center son los MISMOS dos escalares de los que se derivan H y V,
// necesarios en las ramas que no pasan por los Tensor Cores (tile
// parcial/borde), en la correccion de las bandas exteriores del tile y en la
// compensacion espacial, todas ellas calculadas en FP32 fuera del mma.
template <typename T, CompMode kMode>
__global__ static void stencil2d_wmma_kernel(const T* __restrict__ in,
                                             float* __restrict__ out_fp32,
                                             T* __restrict__ out_tc,
                                             const T* __restrict__ horizontal_op,
                                             const T* __restrict__ vertical_op,
                                             int nx,
                                             int ny,
                                             float c_neigh,
                                             float c_center,
                                             int iter,
                                             bool write_fp32,
                                             int* __restrict__ first_nf,
                                             float* __restrict__ comp,
                                             const float* __restrict__ comp_prev,
                                             const int* __restrict__ iter_offset) {
    // Cada warp procesa un tile 16x16 propio e independiente (shared privada
    // por warp, ver smem_raw mas abajo): el bloque ya no es 1 warp = 1 tile,
    // es kWarpsPerBlock warps = kWarpsPerBlock tiles.
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // El grid ahora es 1D (blockIdx.y no se usa): tiles_x se recalcula con la
    // MISMA formula que el host usa para dimensionar el grid (ver
    // benchmark_gpu_tensor_core_stencil), preservando el mapeo tile->dominio
    // (x0/y0) sin agregar un parametro nuevo a la firma.
    const int tiles_x = (nx - 2 + kTileW - 1) / kTileW;
    const int tile_x = static_cast<int>(blockIdx.x) % tiles_x;
    const int tile_y = static_cast<int>(blockIdx.x) / tiles_x;

    // Un bloque = UN tile de 16x64. Ya no hay warps fantasma: cada bloque mapea
    // a un tile real, y gridDim.x es exactamente tiles_x*tiles_y.
    const int bx0 = 1 + tile_x * kTileW;
    const int by0 = 1 + tile_y * kTile;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const bool block_full = (bx0 + kTileW - 1 < nx - 1) && (by0 + kTile - 1 < ny - 1);
#else
    // sm_70 / sm_75 no tienen mma.sync.m16n8k16, y wmma::load_matrix_sync no
    // puede leer un tile permutado porque solo acepta un ldm plano. En esos
    // objetivos TODOS los bloques toman el respaldo por warp de 16 columnas.
    const bool block_full = false;
#endif

    // block_full es uniforme en todo el BLOQUE (solo depende de blockIdx), que
    // es lo que autoriza el __syncthreads() de dentro del camino ancho. Ningun
    // hilo hace return antes del __syncthreads() final.
    __shared__ int blk_bad;
    if (threadIdx.x == 0) blk_bad = 0;
    __syncthreads();

    if (block_full) {
        // ===================================================================
        // CAMINO ANCHO: los kWarpsPerBlock warps cooperan sobre un tile 16x64.
        // ===================================================================
        // El bloque de 16 columnas de cada warp empieza en bx0 + 16*warp_id, es
        // decir en las MISMAS posiciones globales que los tiles de 16 del
        // diseno anterior (1 + 16*k). Por eso cada celda recibe exactamente las
        // mismas correcciones de borde y el resultado es identico bit a bit: lo
        // unico que cambia es de donde salen las vecinas de las costuras
        // internas (de shared en vez de global) y cuantas veces se paga el
        // coste fijo de las bandas laterales.
        extern __shared__ __align__(32) char smem_raw[];
        T* const x_tile = reinterpret_cast<T*>(smem_raw);
        float* const out_tile = reinterpret_cast<float*>(smem_raw + wide_x_bytes<T>())
                              + warp_id * kTile * kLdF;
        T* const bands = reinterpret_cast<T*>(smem_raw + wide_x_bytes<T>()
                                                       + wide_out_bytes());
        T* const band_l = bands;
        T* const band_r = bands + kTile;
        T* const band_u = bands + 2 * kTile;
        T* const band_d = bands + 2 * kTile + kTileW;

        const int t128 = static_cast<int>(threadIdx.x);
        constexpr int kBlockThreads = kWarpsPerBlock * kWarpThreads;
        const bool vec_ok = ((nx & 7) == 0);

        // --- X: 1024 elementos, 8 por hilo. Ocho hilos cubren una fila entera
        // de 128 B, cuatro sectores contiguos y llenos.
        if (vec_ok) {
            const int r = t128 >> 3;
            const int c0 = (t128 & 7) << 3;
            *reinterpret_cast<TVec8<T>*>(&x_tile[swz_x(r, c0)]) =
                *reinterpret_cast<const TVec8<T>*>(&in[idx2d(bx0 + c0, by0 + r, nx)]);
        } else {
            for (int lin = t128; lin < kTile * kTileW; lin += kBlockThreads) {
                const int r = lin / kTileW;
                const int c = lin - r * kTileW;
                x_tile[swz_x(r, c)] = in[idx2d(bx0 + c, by0 + r, nx)];
            }
        }
        // --- bandas superior e inferior: 64 elementos contiguos cada una ---
        if (vec_ok) {
            if (t128 < 16) {
                const int which = t128 >> 3;              // 0 = superior
                const int c0 = (t128 & 7) << 3;
                const int y_src = (which == 0) ? (by0 - 1) : (by0 + kTile);
                T* const dst = (which == 0) ? band_u : band_d;
                *reinterpret_cast<TVec8<T>*>(&dst[c0]) =
                    *reinterpret_cast<const TVec8<T>*>(&in[idx2d(bx0 + c0, y_src, nx)]);
            }
        } else {
            for (int b = t128; b < kTileW; b += kBlockThreads) {
                band_u[b] = in[idx2d(bx0 + b, by0 - 1,     nx)];
                band_d[b] = in[idx2d(bx0 + b, by0 + kTile, nx)];
            }
        }
        // --- bandas izquierda y derecha: 16 cada una. Siguen siendo escalares
        // (stride nx), pero ahora son 32 lecturas por cada 1024 celdas en vez de
        // por cada 256: es exactamente el coste que este diseno viene a diluir.
        for (int b = t128; b < kTile; b += kBlockThreads) {
            band_l[b] = in[idx2d(bx0 - 1,      by0 + b, nx)];
            band_r[b] = in[idx2d(bx0 + kTileW, by0 + b, nx)];
        }

        float* comp_tile = nullptr;
        float* comp_band_l = nullptr;
        float* comp_band_r = nullptr;
        float* comp_band_u = nullptr;
        float* comp_band_d = nullptr;
        if constexpr (kMode == CompMode::Spatial) {
            comp_tile = reinterpret_cast<float*>(smem_raw + wide_x_bytes<T>()
                                                          + wide_out_bytes()
                                                          + wide_bands_bytes<T>());
            float* const cb = comp_tile + kTile * kLdC;
            comp_band_l = cb;
            comp_band_r = cb + kTile;
            comp_band_u = cb + 2 * kTile;
            comp_band_d = cb + 2 * kTile + kTileW;
            if (vec_ok) {
                // 1024 floats / 4 por float4 = 256 chunks, DOS por hilo, con el
                // reparto por chunks CONTIGUOS ENTRE HILOS: hilos consecutivos
                // piden chunks consecutivos y 16 hilos cubren una fila entera.
                // Es el mismo criterio que corrigio el -2 % del job 6727; el
                // reparto "8 floats por hilo" volveria a entrelazar las mitades
                // de cada fila y a duplicar las peticiones.
                #pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const int c = t128 + t * kBlockThreads;    // chunk 0..255
                    const int r = c >> 4;
                    const int col = (c & 15) << 2;
                    *reinterpret_cast<float4*>(&comp_tile[r * kLdC + col]) =
                        *reinterpret_cast<const float4*>(
                            &comp_prev[idx2d(bx0 + col, by0 + r, nx)]);
                }
            } else {
                for (int lin = t128; lin < kTile * kTileW; lin += kBlockThreads) {
                    const int r = lin / kTileW;
                    const int c = lin - r * kTileW;
                    comp_tile[r * kLdC + c] = comp_prev[idx2d(bx0 + c, by0 + r, nx)];
                }
            }
            for (int b = t128; b < kTileW; b += kBlockThreads) {
                comp_band_u[b] = comp_prev[idx2d(bx0 + b, by0 - 1,     nx)];
                comp_band_d[b] = comp_prev[idx2d(bx0 + b, by0 + kTile, nx)];
            }
            for (int b = t128; b < kTile; b += kBlockThreads) {
                comp_band_l[b] = comp_prev[idx2d(bx0 - 1,      by0 + b, nx)];
                comp_band_r[b] = comp_prev[idx2d(bx0 + kTileW, by0 + b, nx)];
            }
        }
        // x_tile y las bandas las comparten los cuatro warps: aqui __syncwarp()
        // NO basta.
        __syncthreads();

        const int wcol = warp_id * kTile;   // columna base del bloque del warp
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        {
            const int gid = lane >> 2;
            const int tid = lane & 3;

            uint32_t ra_x[4];
#if STENCIL_LDMATRIX
            {
                // Un ldmatrix.x4 en lugar de cuatro accesos de 32 bits. Los
                // ocho primeros lanes dan las direcciones de fila del cuadrante
                // 0, los ocho siguientes las del 1, y asi.
                const int q = lane >> 3;
                const int r = lane & 7;
                const uint32_t dir = smem_u32(
                    &x_tile[swz_x(r + 8 * (q & 1), wcol + 8 * (q >> 1))]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                             "{%0,%1,%2,%3}, [%4];\n"
                             : "=r"(ra_x[0]), "=r"(ra_x[1]), "=r"(ra_x[2]), "=r"(ra_x[3])
                             : "r"(dir));
            }
#else
            ra_x[0] = tc_ld32(&x_tile[swz_x(gid,     wcol + 2 * tid)]);
            ra_x[1] = tc_ld32(&x_tile[swz_x(gid + 8, wcol + 2 * tid)]);
            ra_x[2] = tc_ld32(&x_tile[swz_x(gid,     wcol + 2 * tid + 8)]);
            ra_x[3] = tc_ld32(&x_tile[swz_x(gid + 8, wcol + 2 * tid + 8)]);
#endif
            uint32_t ra_v[4];
            {
                const T* const pv = vertical_op + gid * kTile + 2 * tid;
                ra_v[0] = tc_ld32(pv);
                ra_v[1] = tc_ld32(pv + 8 * kTile);
                ra_v[2] = tc_ld32(pv + 8);
                ra_v[3] = tc_ld32(pv + 8 * kTile + 8);
            }

            const float zero4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            #pragma unroll
            for (int nh = 0; nh < 2; ++nh) {
                const int hcol = nh * 8 + gid;          // columna dentro de H
                const int xcol = wcol + hcol;           // columna dentro del 16x64

                uint32_t rb_h[2];
                rb_h[0] = tc_pack2(horizontal_op[(2 * tid + 0) * kTile + hcol],
                                   horizontal_op[(2 * tid + 1) * kTile + hcol]);
                rb_h[1] = tc_pack2(horizontal_op[(2 * tid + 8) * kTile + hcol],
                                   horizontal_op[(2 * tid + 9) * kTile + hcol]);

                uint32_t rb_x[2];
#if STENCIL_LDMATRIX
                {
                    // .trans entrega el bloque 8x8 transpuesto, que es
                    // exactamente el reparto que pide el operando B.
                    const uint32_t dir = smem_u32(&x_tile[swz_x(lane & 15, wcol + nh * 8)]);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                                 "{%0,%1}, [%2];\n"
                                 : "=r"(rb_x[0]), "=r"(rb_x[1]) : "r"(dir));
                }
#else
                rb_x[0] = tc_pack2(x_tile[swz_x(2 * tid + 0, xcol)],
                                   x_tile[swz_x(2 * tid + 1, xcol)]);
                rb_x[1] = tc_pack2(x_tile[swz_x(2 * tid + 8, xcol)],
                                   x_tile[swz_x(2 * tid + 9, xcol)]);
#endif
                float acc_h[4], acc[4];
                mma_m16n8k16(acc_h, ra_x, rb_h, zero4, static_cast<const T*>(nullptr));
                mma_m16n8k16(acc,   ra_v, rb_x, acc_h, static_cast<const T*>(nullptr));

                *reinterpret_cast<float2*>(&out_tile[gid * kLdF + nh * 8 + 2 * tid]) =
                    make_float2(acc[0], acc[1]);
                *reinterpret_cast<float2*>(&out_tile[(gid + 8) * kLdF + nh * 8 + 2 * tid]) =
                    make_float2(acc[2], acc[3]);
            }
        }
#endif
        // out_tile SI es privado del warp: aqui __syncwarp() es suficiente.
        __syncwarp();

        const bool vec_epilogue = vec_ok && (kMode == CompMode::Off);
        if (vec_epilogue) {
            const int row = lane >> 1;
            const int col0 = 8 * (lane & 1);
            const int lin0 = row * kLdF + col0;
            const int idx0 = idx2d(bx0 + wcol + col0, by0 + row, nx);

            const float4 acc_lo = *reinterpret_cast<const float4*>(&out_tile[lin0]);
            const float4 acc_hi = *reinterpret_cast<const float4*>(&out_tile[lin0 + 4]);
            float vals[8] = {acc_lo.x, acc_lo.y, acc_lo.z, acc_lo.w,
                             acc_hi.x, acc_hi.y, acc_hi.z, acc_hi.w};
            TVec8<T> quantized;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                vals[j] = wide_epilogue_value<T, kMode>(
                    vals[j], col0 + j, row, warp_id, x_tile,
                    band_l, band_r, band_u, band_d,
                    comp_tile, comp_band_l, comp_band_r, comp_band_u, comp_band_d,
                    c_neigh, c_center);
                quantized.v[j] = compensated_store<T, kMode>(vals[j], comp, idx0 + j);
                if (!isfinite(vals[j])) blk_bad = 1;    // carrera benigna
            }

            *reinterpret_cast<TVec8<T>*>(&out_tc[idx0]) = quantized;
            if (write_fp32) {
                *reinterpret_cast<float4*>(&out_fp32[idx0]) =
                    make_float4(vals[0], vals[1], vals[2], vals[3]);
                *reinterpret_cast<float4*>(&out_fp32[idx0 + 4]) =
                    make_float4(vals[4], vals[5], vals[6], vals[7]);
            }
        } else {
            for (int linear = lane; linear < kTile * kTile; linear += kWarpThreads) {
                const int local_x = linear % kTile;
                const int local_y = linear / kTile;
                const int idx = idx2d(bx0 + wcol + local_x, by0 + local_y, nx);
                const float val = wide_epilogue_value<T, kMode>(
                    out_tile[local_y * kLdF + local_x], local_x, local_y, warp_id, x_tile,
                    band_l, band_r, band_u, band_d,
                    comp_tile, comp_band_l, comp_band_r, comp_band_u, comp_band_d,
                    c_neigh, c_center);
                out_tc[idx] = compensated_store<T, kMode>(val, comp, idx);
                if (write_fp32) out_fp32[idx] = val;
                if (!isfinite(val)) blk_bad = 1;        // carrera benigna
            }
        }
    } else {
    // =======================================================================
    // RESPALDO: el tile 16x64 no cabe entero, asi que cada warp toma su bloque
    // de 16 columnas con el codigo ANTERIOR, sin tocar.
    // =======================================================================
    // No es solo comodidad: es lo que impide que el ensanchamiento cambie el
    // EXPERIMENTO. Si los bloques parciales cayeran enteros a la ruta escalar,
    // a 4096^2 las columnas 4033..4080 dejarian de pasar por el tensor core y
    // sus valores cambiarian en los ultimos bits. Con este respaldo cada celda
    // toma exactamente la misma ruta que antes, y el A/B bit a bit sigue siendo
    // exigible en cualquier dominio.
    const int x0 = bx0 + warp_id * kTile;
    const int y0 = by0;
    const bool full_tile = (x0 + kTile - 1 < nx - 1) && (y0 + kTile - 1 < ny - 1);

    if (full_tile) {
        // Particion de la shared dinamica del BLOQUE. Cada warp toma su propia
        // region de wmma_warp_shared_bytes<T>(kMode) bytes, en el orden
        // documentado junto a esos helpers: X, out, bandas y -- solo en
        // Spatial -- los residuos. El tamano por warp es multiplo de 32 B, de
        // modo que el X de cada warp queda alineado a 32 B: es el requisito de
        // wmma::load_matrix_sync, que ademas exige ldm multiplo de 16 B (aqui
        // kTile = 16 elementos = 32 B en 16 bits).
        extern __shared__ __align__(32) char smem_raw[];
        char* const warp_base = smem_raw + warp_id * wmma_warp_shared_bytes<T>(kMode);
        T* x_tile = reinterpret_cast<T*>(warp_base);
        float* out_tile = reinterpret_cast<float*>(warp_base + wmma_x_tile_bytes<T>());
        T* bands = reinterpret_cast<T*>(warp_base + wmma_x_tile_bytes<T>()
                                                  + wmma_out_tile_bytes());
        T* band_l = bands + 0 * kTile;
        T* band_r = bands + 1 * kTile;
        T* band_u = bands + 2 * kTile;
        T* band_d = bands + 3 * kTile;

        // 256 valores del tile central + 4*16 de las bandas exteriores = 320
        // lecturas globales de 16 bits por tile completo. La formulacion
        // anterior, con cinco tiles desplazados, hacia 5*256 = 1280.
        //
        // vec_ok: un acceso de 128 bits solo es representable si cada fila del
        // tile cae en frontera de 16 B. Con el puntero del dominio ya
        // desplazado (ver kAlignOffsetElems) el indice de la primera celda
        // interior de la fila y es 16 + y*nx + 16*tile_x, luego la condicion se
        // reduce a nx multiplo de 8: 16 B son 8 elementos de 16 bits, y para
        // los buffers FP32 son 4 floats, que nx multiplo de 8 tambien implica.
        //
        // La condicion es UNIFORME en todo el grid (nx no depende del tile), asi
        // que la rama no diverge dentro del warp. Con nx no multiplo de 8 -- las
        // mallas 63/511 de la validacion pequena -- se toma la ruta escalar, que
        // es exactamente la de siempre, byte a byte.
        const bool vec_ok = ((nx & 7) == 0);

        if (vec_ok) {
            // 256 elementos / 8 por vector = 32 vectores: exactamente uno por
            // lane. El mapeo lane -> (fila = lane/2, mitad = lane%2) hace que el
            // lane L escriba los bytes [16L, 16L+16) de x_tile, un patron lineal
            // que cubre los 32 bancos de shared sin un solo conflicto.
            const int row = lane >> 1;
            const int col0 = 8 * (lane & 1);
            *reinterpret_cast<TVec8<T>*>(&x_tile[row * kLdX + col0]) =
                *reinterpret_cast<const TVec8<T>*>(&in[idx2d(x0 + col0, y0 + row, nx)]);
        } else {
            for (int linear = lane; linear < kTile * kTile; linear += kWarpThreads) {
                const int local_x = linear % kTile;
                const int local_y = linear / kTile;
                x_tile[local_y * kLdX + local_x] =
                    in[idx2d(x0 + local_x, y0 + local_y, nx)];
            }
        }
        // full_tile garantiza x0 >= 1 y x0+kTile-1 <= nx-2 (idem en y), luego
        // x0-1 >= 0 y x0+kTile <= nx-1: las cuatro bandas caen dentro de la
        // malla y no necesitan guarda de rango. Las esquinas diagonales del
        // halo NO se cargan: el stencil de 5 puntos no las usa.
        //
        // Las bandas SUPERIOR e INFERIOR son 16 elementos contiguos en x y se
        // vectorizan igual que el tile: 2 vectores cada una, 4 lanes en total.
        // Las bandas IZQUIERDA y DERECHA recorren y con paso nx -- un valor por
        // fila, nunca contiguos --, asi que se quedan escalares por
        // construccion. No es una limitacion de la alineacion sino del layout:
        // ningun desplazamiento de puntero las vuelve vectorizables.
        if (vec_ok) {
            if (lane < 4) {
                const int which = lane >> 1;              // 0 = superior, 1 = inferior
                const int col0 = 8 * (lane & 1);
                const int y_src = (which == 0) ? (y0 - 1) : (y0 + kTile);
                T* const dst = (which == 0) ? band_u : band_d;
                *reinterpret_cast<TVec8<T>*>(&dst[col0]) =
                    *reinterpret_cast<const TVec8<T>*>(&in[idx2d(x0 + col0, y_src, nx)]);
            }
            for (int b = lane; b < kTile; b += kWarpThreads) {
                band_l[b] = in[idx2d(x0 - 1,     y0 + b, nx)];
                band_r[b] = in[idx2d(x0 + kTile, y0 + b, nx)];
            }
        } else {
            for (int b = lane; b < kTile; b += kWarpThreads) {
                band_l[b] = in[idx2d(x0 - 1,     y0 + b,     nx)];
                band_r[b] = in[idx2d(x0 + kTile, y0 + b,     nx)];
                band_u[b] = in[idx2d(x0 + b,     y0 - 1,     nx)];
                band_d[b] = in[idx2d(x0 + b,     y0 + kTile, nx)];
            }
        }

        // Residuos de la compensacion espacial: 256 centrales + 64 de banda,
        // cargados UNA vez y reutilizados desde shared por las cinco lecturas
        // que antes iban a global por celda. Siguen en FP32; no se cuantizan.
        float* comp_tile = nullptr;
        float* comp_band_l = nullptr;
        float* comp_band_r = nullptr;
        float* comp_band_u = nullptr;
        float* comp_band_d = nullptr;
        if constexpr (kMode == CompMode::Spatial) {
            comp_tile = reinterpret_cast<float*>(warp_base + wmma_x_tile_bytes<T>()
                                                           + wmma_out_tile_bytes()
                                                           + wmma_bands_bytes<T>());
            float* cb = comp_tile + kTile * kLdF;
            comp_band_l = cb + 0 * kTile;
            comp_band_r = cb + 1 * kTile;
            comp_band_u = cb + 2 * kTile;
            comp_band_d = cb + 3 * kTile;
            if (vec_ok) {
                // 256 floats / 4 por float4 = 64 chunks, DOS por lane.
                //
                // El reparto NO es el mismo que el del tile de X, y la
                // diferencia importa. Con __half caben 8 elementos en 16 B, asi
                // que 8 x 32 lanes = 256 = el tile ENTERO en UNA instruccion
                // perfectamente contigua. Con float solo caben 4: 4 x 32 = 128,
                // media fila-tile, y hacen falta DOS instrucciones. Repartir
                // entonces "8 floats contiguos por lane" (el mapeo de X) hace
                // que cada instruccion cubra mitades ENTRELAZADAS de cada fila:
                // la primera trae los sectores y la segunda los vuelve a pedir.
                // Acierta en L1, pero duplica las peticiones -- medido en el job
                // 6727 como -2 % en spatial, el unico modo con comp_prev, frente
                // al +6 % que las demas cargas vectorizadas dan en local.
                //
                // El reparto correcto es por CHUNKS CONTIGUOS ENTRE LANES: la
                // lane L toma los chunks L y L+32, de modo que las lanes 0..3
                // cubren la fila 0 completa (64 B seguidos), las 4..7 la fila 1,
                // y cada instruccion barre 8 filas enteras sin huecos. En shared
                // el destino es 16*c bytes, lineal, sin conflicto de banco.
                #pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const int c = lane + t * kWarpThreads;   // chunk 0..63
                    const int el = c * 4;                    // elemento local
                    const int lrow = el / kTile;
                    const int lcol = el % kTile;             // 0, 4, 8 o 12
                    *reinterpret_cast<float4*>(&comp_tile[lrow * kLdF + lcol]) =
                        *reinterpret_cast<const float4*>(
                            &comp_prev[idx2d(x0 + lcol, y0 + lrow, nx)]);
                }
            } else {
                for (int linear = lane; linear < kTile * kTile; linear += kWarpThreads) {
                    const int local_x = linear % kTile;
                    const int local_y = linear / kTile;
                    comp_tile[local_y * kLdF + local_x] =
                        comp_prev[idx2d(x0 + local_x, y0 + local_y, nx)];
                }
            }
            // Las cuatro bandas de residuo se quedan escalares: izquierda y
            // derecha por el stride nx (igual que sus homologas en T), y
            // superior/inferior porque son 32 lecturas sobre las 320 del tile y
            // no justifican una tercera ruta de acceso.
            for (int b = lane; b < kTile; b += kWarpThreads) {
                comp_band_l[b] = comp_prev[idx2d(x0 - 1,     y0 + b,     nx)];
                comp_band_r[b] = comp_prev[idx2d(x0 + kTile, y0 + b,     nx)];
                comp_band_u[b] = comp_prev[idx2d(x0 + b,     y0 - 1,     nx)];
                comp_band_d[b] = comp_prev[idx2d(x0 + b,     y0 + kTile, nx)];
            }
        }
        __syncwarp();

        // Y = X H + V X: exactamente DOS mma por mitad de n y tile interior
        // completo, es decir las mismas cuatro HMMA.16816 que emitia wmma. X
        // entra como A en el primero y como B en el segundo; es el mismo tile
        // de shared, leido con los dos repartos de fragmento. El acumulador es
        // FP32 y el orden de acumulacion es identico al de antes:
        //
        //     acc  = X H + 0        (C = cero, como wmma::fill_fragment)
        //     acc' = V X + acc
        //
        // out_tile no solapa a x_tile, asi que el volcado no destruye el tile
        // que los mma acaban de consumir.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        const int gid = lane >> 2;      // 0..7  fila base dentro del fragmento
        const int tid = lane & 3;       // 0..3  par de columnas

        // A = X (shared, row-major, ldm = kLdX). Los cuatro cuadrantes 8x8 en
        // el orden que exige el fragmento: (0,0), (8,0), (0,8), (8,8).
        uint32_t ra_x[4];
        {
            const T* const p = x_tile + gid * kLdX + 2 * tid;
            ra_x[0] = tc_ld32(p);
            ra_x[1] = tc_ld32(p + 8 * kLdX);
            ra_x[2] = tc_ld32(p + 8);
            ra_x[3] = tc_ld32(p + 8 * kLdX + 8);
        }
        // A = V (global, row-major, ldm = kTile: el operador NO lleva padding).
        uint32_t ra_v[4];
        {
            const T* const p = vertical_op + gid * kTile + 2 * tid;
            ra_v[0] = tc_ld32(p);
            ra_v[1] = tc_ld32(p + 8 * kTile);
            ra_v[2] = tc_ld32(p + 8);
            ra_v[3] = tc_ld32(p + 8 * kTile + 8);
        }

        const float zero4[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
            const int col = nh * 8 + gid;      // columna de B que posee la lane

            uint32_t rb_h[2];                  // B = H
            rb_h[0] = tc_pack2(horizontal_op[(2 * tid + 0) * kTile + col],
                               horizontal_op[(2 * tid + 1) * kTile + col]);
            rb_h[1] = tc_pack2(horizontal_op[(2 * tid + 8) * kTile + col],
                               horizontal_op[(2 * tid + 9) * kTile + col]);

            uint32_t rb_x[2];                  // B = X
            rb_x[0] = tc_pack2(x_tile[(2 * tid + 0) * kLdX + col],
                               x_tile[(2 * tid + 1) * kLdX + col]);
            rb_x[1] = tc_pack2(x_tile[(2 * tid + 8) * kLdX + col],
                               x_tile[(2 * tid + 9) * kLdX + col]);

            float acc_h[4], acc[4];
            mma_m16n8k16(acc_h, ra_x, rb_h, zero4, static_cast<const T*>(nullptr));
            mma_m16n8k16(acc,   ra_v, rb_x, acc_h, static_cast<const T*>(nullptr));

            // d0,d1 y d2,d3 son columnas contiguas: dos accesos de 64 bits en
            // vez de cuatro escalares. kLdF es par y el desplazamiento de
            // columna (nh*8 + 2*tid) tambien, asi que la direccion es multiplo
            // de 8 B, que es lo que float2 exige.
            *reinterpret_cast<float2*>(&out_tile[gid * kLdF + nh * 8 + 2 * tid]) =
                make_float2(acc[0], acc[1]);
            *reinterpret_cast<float2*>(&out_tile[(gid + 8) * kLdF + nh * 8 + 2 * tid]) =
                make_float2(acc[2], acc[3]);
        }
#else
        // sm_70 / sm_75 no tienen mma.m16n8k16: se conserva la ruta wmma para
        // que el fuente siga compilando en esos objetivos (ver CUDA_ARCH en los
        // sbatch, que admite 70 y 86 ademas de 80).
        wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, T, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, T, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);

        wmma::load_matrix_sync(a_frag, x_tile, kLdX);            // A = X
        wmma::load_matrix_sync(b_frag, horizontal_op, kTile);    // B = H
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);      // acc  = X H

        wmma::load_matrix_sync(a_frag, vertical_op, kTile);      // A = V
        wmma::load_matrix_sync(b_frag, x_tile, kLdX);            // B = X
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);      // acc += V X

        wmma::store_matrix_sync(out_tile, acc_frag, kLdF, wmma::mem_row_major);
#endif
        __syncwarp();

        // Finitud evaluada sobre el acumulador FP32 (out_tile), antes de
        // convertir a T. out_tc SIEMPRE se escribe (reemplaza al kernel de
        // conversion dentro del bucle); out_fp32 solo cuando write_fp32 (ultima
        // iteracion medida o checkpoint), con la MISMA funcion de conversion
        // que convert_float_to_half_kernel/convert_float_to_bfloat16_kernel.
        // El epilogo vectorizado SOLO se toma en CompMode::Off. Medido en el
        // job 6725 (4096^2, 5 replicas, control GPU_FP32 plano con CV 0.06 %):
        //
        //   off       0.17828 -> 0.17138 ms   (+4.0 %)   <- gana
        //   local     0.37335 -> 0.66152 ms   (-77 %)    <- se hunde
        //   spatial   0.33918 -> 0.58088 ms   (-71 %)    <- se hunde
        //
        // La causa es un conflicto entre dos formas opuestas de repartir las
        // celdas entre lanes:
        //
        //   - un acceso VECTORIAL quiere que cada lane posea celdas contiguas
        //     entre si (8 celdas seguidas = 16 B = un acceso de 128 bits);
        //   - un acceso ESCALAR quiere lo contrario: que las 32 lanes cubran
        //     direcciones consecutivas, para que el warp coalesca en pocos
        //     sectores.
        //
        // compensated_store sigue escribiendo comp[] celda a celda (es la
        // referencia numerica y no se toca). Con el mapeo por-lane-contiguo,
        // para un j fijo las 32 lanes quedan separadas 8 floats = 32 B, o sea
        // una en cada sector: el buffer comp pasa de 32 a 256 sectores por
        // tile, 8x. A 4096^2 son ~470 MB extra por iteracion, es decir
        // +0.30 ms a 1.55 TB/s -- del orden de los +0.25/+0.28 ms medidos.
        //
        // En CompMode::Off no hay buffer comp, no queda ningun acceso escalar
        // que descoalescar, y el mapeo vectorial es puro beneficio.
        //
        // Constante de compilacion: en Local/Spatial la condicion es
        // falsa en tiempo de compilacion y la rama vectorizada desaparece
        // entera (no cuesta ni un registro).
        const bool vec_epilogue = vec_ok && (kMode == CompMode::Off);

        if (vec_epilogue) {
            // Mapeo lane -> 8 celdas CONTIGUAS en x (fila = lane/2, columnas
            // 8*(lane%2)..+7). Es el cambio que hace representables los accesos
            // de 128 bits en la escritura: el bucle anterior (linear = lane,
            // paso kWarpThreads) daba a cada lane celdas dispersas en filas
            // distintas, y ocho celdas dispersas no son un vector.
            //
            // Cambia QUE lane calcula QUE celda, no COMO se calcula. El trabajo
            // de cada celda es independiente del de las demas, asi que la
            // secuencia de operaciones de punto flotante por celda es la misma y
            // el resultado es identico bit a bit -- no "dentro de tolerancia".
            const int row = lane >> 1;
            const int col0 = 8 * (lane & 1);
            const int lin0 = row * kLdF + col0;   // indice en SHARED, con padding
            const int idx0 = idx2d(x0 + col0, y0 + row, nx);

            // El acumulador sale de shared en dos accesos de 128 bits en vez de
            // ocho escalares.
            const float4 acc_lo = *reinterpret_cast<const float4*>(&out_tile[lin0]);
            const float4 acc_hi = *reinterpret_cast<const float4*>(&out_tile[lin0 + 4]);
            float vals[8] = {acc_lo.x, acc_lo.y, acc_lo.z, acc_lo.w,
                             acc_hi.x, acc_hi.y, acc_hi.z, acc_hi.w};
            TVec8<T> quantized;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                vals[j] = wmma_epilogue_value<T, kMode>(
                    vals[j], col0 + j, row, lin0 + j,
                    band_l, band_r, band_u, band_d,
                    comp_tile, comp_band_l, comp_band_r, comp_band_u, comp_band_d,
                    c_neigh, c_center);
                // compensated_store queda INTACTA y sigue escribiendo comp[idx]
                // celda a celda. Vectorizar tambien el residuo obligaria a
                // partirla en lectura / calculo / escritura, y esa funcion es la
                // referencia numerica del experimento: no se toca por
                // rendimiento.
                quantized.v[j] = compensated_store<T, kMode>(vals[j], comp, idx0 + j);
                if (!isfinite(vals[j])) blk_bad = 1;    // carrera benigna
            }

            // out_tc en un unico acceso de 128 bits (8 elementos de 16 bits) y
            // out_fp32 en dos. Sacarlos fuera del bucle no altera ningun valor:
            // out_tc, out_fp32 y comp son buffers distintos (__restrict__ en la
            // firma), asi que el orden entre escrituras independientes es libre.
            *reinterpret_cast<TVec8<T>*>(&out_tc[idx0]) = quantized;
            if (write_fp32) {
                *reinterpret_cast<float4*>(&out_fp32[idx0]) =
                    make_float4(vals[0], vals[1], vals[2], vals[3]);
                *reinterpret_cast<float4*>(&out_fp32[idx0 + 4]) =
                    make_float4(vals[4], vals[5], vals[6], vals[7]);
            }
        } else {
            for (int linear = lane; linear < kTile * kTile; linear += kWarpThreads) {
                const int local_x = linear % kTile;
                const int local_y = linear / kTile;
                const int idx = idx2d(x0 + local_x, y0 + local_y, nx);
                const int sidx = local_y * kLdF + local_x;   // indice en SHARED
                const float val = wmma_epilogue_value<T, kMode>(
                    out_tile[sidx], local_x, local_y, sidx,
                    band_l, band_r, band_u, band_d,
                    comp_tile, comp_band_l, comp_band_r, comp_band_u, comp_band_d,
                    c_neigh, c_center);
                out_tc[idx] = compensated_store<T, kMode>(val, comp, idx);
                if (write_fp32) out_fp32[idx] = val;
                if (!isfinite(val)) blk_bad = 1;    // carrera benigna: todos escriben 1
            }
        }
    } else {
        for (int linear = lane; linear < kTile * kTile; linear += kWarpThreads) {
            const int local_x = linear % kTile;
            const int local_y = linear / kTile;
            const int x = x0 + local_x;
            const int y = y0 + local_y;

            // Esta guarda cubre, a la vez, tres casos: el borde fisico de la
            // grilla (x/y == 0 o == nx-1/ny-1, que nunca se recalcula), los
            // indices que caen fuera de rango porque este es el ultimo tile
            // parcial de la fila/columna (x0/y0 + kTile puede exceder
            // nx-1/ny-1 cuando nx-2 o ny-2 no son multiplos de kTile), y los
            // warps fantasma sin tile real (ver comentario de tile_id mas
            // arriba). Los tres se resuelven igual: no leer ni escribir para
            // ese punto. active (no return) porque el trip count del for es
            // uniforme entre lanes de un warp: todos deben llegar al
            // __syncthreads() de mas abajo.
            const bool active = !(x <= 0 || y <= 0 || x >= nx - 1 || y >= ny - 1);
            if (active) {
                const int i_up = idx2d(x, y - 1, nx);
                const int i_down = idx2d(x, y + 1, nx);
                const int i_left = idx2d(x - 1, y, nx);
                const int i_right = idx2d(x + 1, y, nx);
                const int idx = idx2d(x, y, nx);
                // En modo Spatial cada vecina se reconstruye a su valor FP32
                // exacto (Q(v) + residuo perdido, ver compensated_store) ANTES
                // de entrar a la suma: es la forma directa de lo que la rama
                // full_tile de arriba hace por linealidad.
                float up = tc_to_float(in[i_up]);
                float down = tc_to_float(in[i_down]);
                float left = tc_to_float(in[i_left]);
                float right = tc_to_float(in[i_right]);
                float center = tc_to_float(in[idx]);
                if constexpr (kMode == CompMode::Spatial) {
                    up += comp_prev[i_up];
                    down += comp_prev[i_down];
                    left += comp_prev[i_left];
                    right += comp_prev[i_right];
                    center += comp_prev[idx];
                }
                const float val = fmaf(c_neigh, up + down + left + right, c_center * center);
                out_tc[idx] = compensated_store<T, kMode>(val, comp, idx);
                if (write_fp32) out_fp32[idx] = val;
                if (!isfinite(val)) blk_bad = 1;    // carrera benigna: todos escriben 1
            }
        }
    }
    }   // fin del respaldo por warp

    __syncthreads();
    // El guardia blk_bad != 0 se adelanta aqui (reduce_and_mark_first_nonfinite
    // ya lo repite, y con blk_bad == 0 sigue siendo un no-op) para que la
    // lectura de *iter_offset NO ocurra en el camino comun: solo se paga cuando
    // esta iteracion produjo algo no finito, que es un evento unico por corrida.
    //
    // iter_offset es nullptr en modo normal, donde `iter` YA es la iteracion
    // global (1..N). Bajo CUDA Graph los nodos del grafo llevan `iter` RELATIVO
    // al bloque (1..B, horneado en el nodo y por tanto inmutable entre
    // reproducciones) y *iter_offset lleva cuantas iteraciones globales hay
    // antes del bloque en curso; la suma reconstruye 1..N sin reinstanciar el
    // grafo ni reescribir los argumentos de sus nodos. Ver
    // advance_iteration_offset_kernel.
    if (threadIdx.x == 0 && blk_bad != 0) {
        const int global_iter = (iter_offset != nullptr) ? (*iter_offset + iter) : iter;
        reduce_and_mark_first_nonfinite(first_nf, global_iter, blk_bad);
    }
}

// Ultimo nodo de cada grafo de iteraciones: adelanta el contador global en
// exactamente las iteraciones que el grafo acaba de ejecutar. Va DENTRO del
// grafo (y no como un memcpy del host entre reproducciones) para que reproducir
// el mismo cudaGraphExec_t k veces seguidas produzca 1..k*B sin intervencion de
// la CPU, que es justamente el costo que este modo existe para eliminar. Un
// solo hilo: la escritura no compite con nadie porque la captura en un unico
// stream serializa este nodo despues de los B kernels del bloque.
__global__ static void advance_iteration_offset_kernel(int* offset, int delta) {
    *offset += delta;
}

template <typename T>
static void convert_input_to_tc(const float* d_in_fp32, T* d_in_tc, size_t count);

template <>
void convert_input_to_tc<__half>(const float* d_in_fp32, __half* d_in_tc, size_t count) {
    const int blocks = static_cast<int>((count + kConversionThreads - 1) / kConversionThreads);
    convert_float_to_half_kernel<<<blocks, kConversionThreads>>>(
        d_in_fp32, d_in_tc, static_cast<int>(count));
    CHECK_CUDA(cudaGetLastError());
}

template <>
void convert_input_to_tc<__nv_bfloat16>(const float* d_in_fp32,
                                        __nv_bfloat16* d_in_tc,
                                        size_t count) {
    const int blocks = static_cast<int>((count + kConversionThreads - 1) / kConversionThreads);
    convert_float_to_bfloat16_kernel<<<blocks, kConversionThreads>>>(
        d_in_fp32, d_in_tc, static_cast<int>(count));
    CHECK_CUDA(cudaGetLastError());
}

// --- Conversion host-side de __half / __nv_bfloat16 a float ---
// Se usan unicamente para reportar por stdout, nunca dentro de un kernel.
// __half2float / __bfloat162float son __host__ __device__ desde CUDA 11,
// por lo que son validas aqui sin necesidad de un kernel adicional.
static inline float host_val_to_float(__half v) { return __half2float(v); }
static inline float host_val_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

// Convierte el buffer T (formato de 16 bits) a un vector float elemento a
// elemento, para poder compararlo contra la referencia FP64 con
// compare_fp64_ref_vs_fp32. Se usa para medir el error del ESTADO PROPAGADO
// (ver metrica rel_l2_prop/rel_linf_prop): a diferencia de out_fp32 (el
// acumulador FP32 sin redondear, ancla de no-regresion), lo que realmente se
// propaga entre iteraciones es este buffer en 16 bits.
template <typename T>
static std::vector<float> reduced_to_float(const std::vector<T>& reduced) {
    std::vector<float> out(reduced.size());
    for (size_t i = 0; i < reduced.size(); ++i) {
        out[i] = host_val_to_float(reduced[i]);
    }
    return out;
}

// Reconstruccion espacial sobre el campo COMPLETO: u_efectivo = Q(u) + comp.
//
// Es la MISMA formula que stencil2d_wmma_kernel aplica en su rama Spatial
// (tc_to_float(in[idx]) + comp_prev[idx]), pero evaluada sobre todo el dominio
// y en el host. Hace falta porque aquella la aplica vecino a vecino DENTRO del
// kernel, sobre los cinco puntos que esta leyendo, y nunca materializa el campo
// reconstruido: para archivar el estado que la politica espacial realmente
// propaga hay que armarlo aparte. El kernel no se toca; esta funcion solo lee
// los dos buffers que el ya mantiene.
//
// host_val_to_float es el gemelo host de tc_to_float (misma conversion, misma
// semantica de redondeo), igual que ya hace reduced_to_float.
template <typename T>
static std::vector<float> build_spatial_reconstructed_field(const std::vector<T>& state_tc,
                                                            const std::vector<float>& comp) {
    std::vector<float> out(state_tc.size());
    for (size_t i = 0; i < state_tc.size(); ++i) {
        out[i] = host_val_to_float(state_tc[i]) + comp[i];
    }
    return out;
}

// Campo de LECTURA (readout) del estado propagado, para rel_l2_prop/rel_linf_prop.
//
// Antes de la correccion del 29-ago esto era reduced_to_float(state_tc) a secas
// en los dos sitios de llamada, sin rama por CompMode: rel_l2_prop media el
// buffer T CRUDO incluso bajo compensacion ESPACIAL, que es la unica politica
// cuyo estado efectivo no es el buffer T. Eso ya era inconsistente con lo que
// el archivado hace desde siempre (Q(u)+comp), y subestimaba la precision de
// Spatial en un factor grande: medido a 1024^2, 20 iters, diffusive, la ruta
// FP16_SP pasaba de un 3.03e-04 aparente a 2.02e-05 real, y BF16_SP de
// 1.55e-03 a 2.02e-05. Que ambos formatos converjan al MISMO valor es la
// comprobacion de que la reconstruccion es exacta: recuperado el valor FP32,
// lo que queda es el suelo FP32-vs-FP64, que no depende del formato de 16 bits.
//
// comp llega vacio salvo en Spatial. Para Off porque no hay buffer; para Local
// porque su convencion de error feedback hace que Q(u)+comp sea invalida (ver
// la nota en el bloque de archivado). En ambos casos esto colapsa a
// reduced_to_float, sin cambio numerico respecto al comportamiento historico.
template <typename T>
static std::vector<float> build_readout_field(const std::vector<T>& state_tc,
                                              const std::vector<float>& comp) {
    return comp.empty() ? reduced_to_float(state_tc)
                        : build_spatial_reconstructed_field(state_tc, comp);
}

static inline __half host_float_to_tc_impl(float v, __half*) { return __float2half(v); }
static inline __nv_bfloat16 host_float_to_tc_impl(float v, __nv_bfloat16*) { return __float2bfloat16(v); }

template <typename T>
static T host_float_to_tc(float v) {
    return host_float_to_tc_impl(v, static_cast<T*>(nullptr));
}

// Mide cuanto se pierde SOLO por aplicar el round-trip de almacenamiento
// float -> T -> float al estado propagado u. Importante para --kahan on:
// esto no lee el residuo de compensacion ni compara contra el valor realmente
// desplazado por Kahan antes de almacenar; mide Q(u)-u sobre el estado FP32
// finito que el stencil produjo en la iteracion evaluada.
struct StorageRelResult {
    double rel_norm = std::numeric_limits<double>::quiet_NaN();
    double rel_max_guarded = std::numeric_limits<double>::quiet_NaN();
    size_t excluded_count = 0;
    int eval_iter = 0;  // iter en que se evaluo (util para anotar)
    bool evaluated = false;
};

template <typename T>
static StorageRelResult storage_roundtrip_metrics(const std::vector<float>& state,
                                                  int iter_context) {
    StorageRelResult result;
    result.eval_iter = iter_context;

    double norm_inf = 0.0;
    double sq_state = 0.0;
    double sq_err = 0.0;
    bool all_finite = true;
    for (float xf : state) {
        const double x = static_cast<double>(xf);
        if (!std::isfinite(x)) {
            all_finite = false;
            break;
        }
        const double q = static_cast<double>(host_val_to_float(host_float_to_tc<T>(xf)));
        if (!std::isfinite(q)) {
            all_finite = false;
            break;
        }
        const double diff = q - x;
        norm_inf = std::max(norm_inf, std::fabs(x));
        sq_state += x * x;
        sq_err += diff * diff;
    }

    if (!all_finite || !std::isfinite(norm_inf) || !std::isfinite(sq_state) ||
        !std::isfinite(sq_err) || sq_state <= 0.0) {
        return result;
    }

    result.evaluated = true;
    result.rel_norm = std::sqrt(sq_err / sq_state);

    const double tau = 1.0e-6 * norm_inf;
    bool any_included = false;
    double max_rel = 0.0;
    for (float xf : state) {
        const double x = static_cast<double>(xf);
        const double abs_x = std::fabs(x);
        if (abs_x < tau) {
            result.excluded_count++;
            continue;
        }
        if (abs_x == 0.0) {
            result.excluded_count++;
            continue;
        }
        const double q = static_cast<double>(host_val_to_float(host_float_to_tc<T>(xf)));
        max_rel = std::max(max_rel, std::fabs(q - x) / abs_x);
        any_included = true;
    }
    if (any_included) {
        result.rel_max_guarded = max_rel;
    }
    return result;
}

// Analogo FP64 de storage_roundtrip_metrics. En GPU_FP64 el estado se ALMACENA
// en el mismo formato en que se acumula (double), asi que la cuantizacion Q es
// la identidad y el error de almacenamiento es exactamente 0. No es un relleno
// para llenar la columna: es la cota inferior medida que da escala a los
// store_rel de FP16/BF16 en el mismo CSV_STORE -- sin una fila de referencia,
// un store_rel_norm de 1e-4 no se distingue de "mucho" o "poco".
// excluded_count y eval_iter se calculan con exactamente la misma regla
// (tau = 1e-6 * ||u||_inf, exclusion de ceros) para que las columnas sean
// comparables entre rutas y no reflejen dos criterios distintos.
static StorageRelResult storage_roundtrip_metrics_fp64(const std::vector<double>& state,
                                                       int iter_context) {
    StorageRelResult result;
    result.eval_iter = iter_context;

    double norm_inf = 0.0;
    double sq_state = 0.0;
    for (double x : state) {
        if (!std::isfinite(x)) return result;   // estado no finito: no evaluable
        norm_inf = std::max(norm_inf, std::fabs(x));
        sq_state += x * x;
    }
    if (!std::isfinite(norm_inf) || !std::isfinite(sq_state) || sq_state <= 0.0) {
        return result;
    }

    result.evaluated = true;
    // sqrt(sum (Q(x)-x)^2 / sum x^2) con Q = identidad sobre double.
    result.rel_norm = 0.0;

    const double tau = 1.0e-6 * norm_inf;
    bool any_included = false;
    for (double x : state) {
        const double abs_x = std::fabs(x);
        if (abs_x < tau || abs_x == 0.0) {
            result.excluded_count++;
            continue;
        }
        any_included = true;
    }
    if (any_included) {
        result.rel_max_guarded = 0.0;
    }
    return result;
}

// Encadenamiento genuino salida(i) -> entrada(i+1): el kernel WMMA ahora
// escribe out_tc (T) directamente cada iteracion (misma funcion de
// conversion que antes aplicaba el kernel de conversion aparte, ver
// float_to_tc<T>), asi que el ping-pong entre iteraciones es un simple
// std::swap de punteros T*, igual que benchmark_gpu_fp32_stencil -- ya no
// hace falta relanzar convert_input_to_tc dentro del bucle. out_fp32 sigue
// siendo un buffer FP32 aparte (no participa del ping-pong): el kernel solo
// lo escribe cuando write_fp32 es true (ultima iteracion medida o
// checkpoint), que es cuando algo aguas abajo va a medir error. El warm-up
// encadena de la misma forma (write_fp32=false, descartable) pero al
// terminar se reconvierte d_in_fp32 (nunca modificado) hacia AMBOS buffers T
// -- el kernel nunca escribe las celdas de borde, asi que deben preservarse
// desde el inicio en cualquier buffer que llegue a jugar el rol de entrada
// -- y se restaura out_fp32 con una copia fresca de in via cudaMemcpy, para
// que el bucle medido siempre arranque desde el estado original (necesario
// para que --iters 1 coincida con Fase_2/Stencil).
template <typename T>
static Metrics benchmark_gpu_tensor_core_stencil(const std::vector<float>& in,
                                                 std::vector<float>& out,
                                                 std::vector<T>& out_reduced,
                                                 int nx,
                                                 int ny,
                                                 int iters,
                                                 const StencilOperator& op,
                                                 CompMode comp_mode,
                                                 // Forma de entregar el trabajo a la GPU. No
                                                 // altera nada numerico: mismo kernel, mismos
                                                 // argumentos, mismo orden (ver ExecutionMode).
                                                 ExecutionMode execution_mode,
                                                 int graph_block,
                                                 const CheckpointContext& ckpt,
                                                 const char* route_label,
                                                 int& onset_iter,
                                                 int& first_nonfinite_iter,
                                                 double& t_wmma_ms_out,
                                                 double& t_conv_ms_out,
                                                 int& storage_rel_eval_iter,
                                                 double& t_checkpoint_ms_out,
                                                 std::vector<float>& out_last_finite_o,
                                                 std::vector<T>& out_reduced_last_finite_o,
                                                 EnergyMeasurement& out_energy,
                                                 // Residuo de compensacion que acompana a
                                                 // out_reduced, ya emparejado con el buffer T
                                                 // correcto (ver el volcado al final). Queda
                                                 // VACIO en CompMode::Off: no hay buffer comp.
                                                 std::vector<float>& out_comp) {
    const size_t count = in.size();
    float* d_in_fp32 = nullptr;
    float* d_out_fp32 = nullptr;
    T* d_in_tc = nullptr;
    T* d_out_tc = nullptr;
    T* d_horizontal = nullptr;
    T* d_vertical = nullptr;
    int* d_first_nf = nullptr;
    // Contador de iteraciones ya completadas ANTES del bloque de grafo en curso.
    // Solo se reserva en modo graph; nullptr en modo normal, donde el kernel
    // recibe la iteracion global directamente como inmediato y nunca lo
    // dereferencia (ver stencil2d_wmma_kernel / advance_iteration_offset_kernel).
    int* d_iter_offset = nullptr;
    // d_comp: residuo por celda, en FP32, persistente entre iteraciones (ver
    // compensated_store). Solo se reserva si hay compensacion activa; nullptr
    // en caso contrario (la instanciacion CompMode::Off del kernel nunca lo
    // toca).
    //   Local   : un solo buffer, actualizado en sitio (cada celda solo toca su
    //             propia entrada, no hay carrera). d_comp_prev queda en nullptr.
    //   Spatial : DOS buffers en ping-pong, igual que d_in_tc/d_out_tc. Aqui
    //             cada celda LEE las entradas de sus 4 vecinas mientras esas
    //             vecinas escriben las suyas; en sitio seria una carrera
    //             lectura/escritura entre bloques. Costo de memoria: +2 x 4
    //             bytes por celda frente al 1 x 4 de Local.
    float* d_comp = nullptr;
    float* d_comp_prev = nullptr;
    const bool comp_enabled = (comp_mode != CompMode::Off);
    const bool comp_pingpong = (comp_mode == CompMode::Spatial);

    // Los SEIS buffers que representan el dominio 2D completo se reservan con
    // kAlignPadElems elementos de mas y se usan a traves de un puntero
    // desplazado *_aligned (ver kAlignOffsetElems). d_horizontal, d_vertical,
    // d_first_nf y d_iter_offset NO llevan padding: no son el dominio, sus
    // accesos no se vectorizan y desplazarlos solo anadiria ruido.
    //
    // El puntero base se conserva intacto: es el unico valido para cudaFree
    // (liberar un puntero desplazado es comportamiento indefinido, ver el
    // bloque de cudaFree al final de la funcion).
    CHECK_CUDA(cudaMalloc(&d_in_fp32, (count + kAlignPadElems) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_fp32, (count + kAlignPadElems) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_in_tc, (count + kAlignPadElems) * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_out_tc, (count + kAlignPadElems) * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_horizontal, kTile * kTile * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_vertical, kTile * kTile * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_first_nf, sizeof(int)));

    // Punteros LOGICOS: todo lo que trabaja sobre el dominio (H2D, D2H,
    // memset, conversiones, siembra, kernels, ping-pong y captura de grafo)
    // usa estos; el tamano logico sigue siendo count, nunca count + PAD.
    float* const d_in_fp32_aligned = d_in_fp32 + kAlignOffsetElems;
    float* const d_out_fp32_aligned = d_out_fp32 + kAlignOffsetElems;
    T* const d_in_tc_aligned = d_in_tc + kAlignOffsetElems;
    T* const d_out_tc_aligned = d_out_tc + kAlignOffsetElems;

    // d_comp/d_comp_prev son condicionales: el desplazado se deja en nullptr
    // cuando no hay asignacion (nullptr + 15 seria comportamiento indefinido,
    // y la instanciacion CompMode::Off del kernel nunca lo dereferencia).
    float* d_comp_aligned = nullptr;
    float* d_comp_prev_aligned = nullptr;
    if (comp_enabled) {
        CHECK_CUDA(cudaMalloc(&d_comp, (count + kAlignPadElems) * sizeof(float)));
        d_comp_aligned = d_comp + kAlignOffsetElems;
        CHECK_CUDA(cudaMemset(d_comp_aligned, 0, count * sizeof(float)));
    }
    if (comp_pingpong) {
        CHECK_CUDA(cudaMalloc(&d_comp_prev, (count + kAlignPadElems) * sizeof(float)));
        d_comp_prev_aligned = d_comp_prev + kAlignOffsetElems;
        CHECK_CUDA(cudaMemset(d_comp_prev_aligned, 0, count * sizeof(float)));
    }

    CHECK_CUDA(cudaMemcpy(d_in_fp32_aligned, in.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_out_fp32_aligned, in.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));

    // H y V son la forma en que el operador entra a los Tensor Cores: el mma no
    // toma escalares, asi que los coeficientes viajan como las bandas de dos
    // matrices 16x16 (ver initialize_horizontal_operator /
    // initialize_vertical_operator y la derivacion de Y = X H + V X). Salen de
    // los MISMOS dos escalares que usan las ramas escalares del kernel y las
    // rutas FP32/FP64 (op.neighbor / op.center), no de literales duplicados que
    // pudieran divergir de ellos. Una sola transferencia host->device por
    // benchmark: el operador no cambia entre iteraciones.
    std::vector<T> horizontal(kTile * kTile);
    std::vector<T> vertical(kTile * kTile);
    initialize_horizontal_operator<T>(horizontal, op);
    initialize_vertical_operator<T>(vertical, op);
    CHECK_CUDA(cudaMemcpy(d_horizontal, horizontal.data(),
                          horizontal.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vertical, vertical.data(),
                          vertical.size() * sizeof(T), cudaMemcpyHostToDevice));

    // gridDim.x es 1D y vale exactamente tiles_x*tiles_y, con UN tile de
    // kTile x kTileW (16x64) por bloque; los kWarpsPerBlock warps se reparten
    // sus cuatro bloques de 16 columnas (misma formula de tiles_x/tiles_y que
    // dentro del kernel). Es el mismo numero de bloques que la version de
    // tiles 16x16 -- alli era ceil(4*tiles64/4) -- pero ya no sobran warps
    // fantasma en el ultimo bloque.
    //
    // shared_bytes reserva el maximo de los dos layouts (ancho y respaldo por
    // warp) porque cada bloque usa uno solo: 6656 B en Off/Local y 12224 B en
    // Spatial. Muy por debajo de 48 KiB, asi que no requiere
    // cudaFuncAttributeMaxDynamicSharedMemorySize.
    const int tiles_x = (nx - 2 + kTileW - 1) / kTileW;
    const int tiles_y = (ny - 2 + kTile - 1) / kTile;
    const int total_tiles = tiles_x * tiles_y;
    // El caso mayor (Spatial) es el que tiene que caber; los otros dos son
    // estrictamente menores.
    static_assert(block_shared_bytes<T>(CompMode::Spatial) <= 49152,
                 "shared por bloque excede 48 KiB estaticos/dinamicos");
    dim3 block(kWarpsPerBlock * kWarpThreads);
    // Un bloque = un tile de 16x64 = kWarpsPerBlock bloques de 16 columnas. El
    // numero de bloques es el mismo que antes (antes: ceil(tiles16/4) con
    // tiles16 = 4*tiles64), pero ya no sobran warps fantasma.
    dim3 grid(total_tiles);
    const size_t shared_bytes =
        block_shared_bytes<T>(comp_mode);

    // Siembra los DOS buffers de residuo con el error de la conversion inicial
    // FP32 -> T (ver seed_comp_from_conversion_kernel). Los dos, y no solo uno,
    // por el mismo motivo por el que d_in_tc/d_out_tc se convierten ambos: tras
    // un numero impar de swaps cualquiera de los dos puede ser el que lea la
    // primera iteracion. d_in_tc y d_out_tc tienen contenido identico aqui, asi
    // que basta con leer uno. No-op fuera del modo Spatial: --kahan off|on
    // conservan su residuo inicial en cero, byte a byte.
    auto seed_comp_buffers = [&]() {
        if (!comp_pingpong) return;
        const int blocks = static_cast<int>((count + kConversionThreads - 1) / kConversionThreads);
        seed_comp_from_conversion_kernel<T><<<blocks, kConversionThreads>>>(
            d_in_fp32_aligned, d_in_tc_aligned, d_comp_aligned, static_cast<int>(count));
        seed_comp_from_conversion_kernel<T><<<blocks, kConversionThreads>>>(
            d_in_fp32_aligned, d_in_tc_aligned, d_comp_prev_aligned, static_cast<int>(count));
        CHECK_CUDA(cudaGetLastError());
    };

    // Ambos buffers del ping-pong T arrancan como conversion completa (borde
    // incluido) del input pristino: ver comentario de la funcion.
    convert_input_to_tc<T>(d_in_fp32_aligned, d_in_tc_aligned, count);
    convert_input_to_tc<T>(d_in_fp32_aligned, d_out_tc_aligned, count);
    seed_comp_buffers();
    CHECK_CUDA(cudaDeviceSynchronize());

    // Punteros vivos del ping-pong de residuos (solo en modo Spatial):
    // comp_out es el buffer que la iteracion en curso escribe, comp_in el que
    // dejo la anterior. Se declaran aqui, antes de launch_wmma, porque la
    // lambda los captura por referencia y el swap ocurre junto al de tc_in/
    // tc_out en cada iteracion (incluido el warm-up).
    float* comp_out = d_comp_aligned;
    float* comp_in = d_comp_prev_aligned;

    // Elige la instanciacion CompMode del kernel en tiempo de compilacion
    // segun los flags runtime: comp_mode no cambia dentro de esta llamada, asi
    // que el branch se resuelve una vez por benchmark, no por lanzamiento.
    // Cuando comp_mode es Off, d_comp es nullptr y la instanciacion
    // CompMode::Off nunca lo dereferencia; en Local, comp_prev va en nullptr
    // (esa instanciacion tampoco lo dereferencia).
    // Forma general: los buffers de residuo y el stream son explicitos porque la
    // captura del grafo necesita recorrer el ping-pong sobre punteros propios,
    // sin tocar comp_out/comp_in (que son el estado vivo del bucle medido).
    // iter_off es nullptr en todo lanzamiento normal, donde iter_num ya es la
    // iteracion global; solo los nodos capturados en un grafo lo reciben.
    auto launch_wmma_on = [&](T* in_buf, T* out_buf, float* c_out, const float* c_in,
                              int iter_num, bool write_fp32_flag, const int* iter_off,
                              cudaStream_t stream) {
        switch (comp_mode) {
            case CompMode::Local:
                stencil2d_wmma_kernel<T, CompMode::Local><<<grid, block, shared_bytes, stream>>>(
                    in_buf, d_out_fp32_aligned, out_buf, d_horizontal, d_vertical,
                    nx, ny, op.neighbor, op.center, iter_num, write_fp32_flag, d_first_nf,
                    d_comp_aligned, nullptr, iter_off);
                break;
            case CompMode::Spatial:
                stencil2d_wmma_kernel<T, CompMode::Spatial><<<grid, block, shared_bytes, stream>>>(
                    in_buf, d_out_fp32_aligned, out_buf, d_horizontal, d_vertical,
                    nx, ny, op.neighbor, op.center, iter_num, write_fp32_flag, d_first_nf,
                    c_out, c_in, iter_off);
                break;
            case CompMode::Off:
                stencil2d_wmma_kernel<T, CompMode::Off><<<grid, block, shared_bytes, stream>>>(
                    in_buf, d_out_fp32_aligned, out_buf, d_horizontal, d_vertical,
                    nx, ny, op.neighbor, op.center, iter_num, write_fp32_flag, d_first_nf,
                    nullptr, nullptr, iter_off);
                break;
        }
    };

    // Lanzamiento normal sobre el stream por defecto, con el estado vivo del
    // ping-pong de residuos: es la firma que ya usaban el warm-up y el bucle
    // medido, sin cambios de comportamiento (stream 0 y iter_off nullptr
    // reproducen la configuracion anterior exactamente).
    auto launch_wmma = [&](T* in_buf, T* out_buf, int iter_num, bool write_fp32_flag) {
        launch_wmma_on(in_buf, out_buf, comp_out, comp_in, iter_num, write_fp32_flag,
                       nullptr, /*stream=*/0);
    };

    // Avanza el ping-pong de residuos junto al de los buffers T. No-op fuera
    // del modo Spatial (en Local el unico buffer se actualiza en sitio).
    auto swap_comp = [&]() {
        if (comp_pingpong) std::swap(comp_in, comp_out);
    };

    PowerBuffer* power_buffer = power_buffer_create(0);
    const RAEnergySnapshot rapl_warmup_before = rapl_snapshot_now();
    power_buffer_start_sampling(power_buffer);

    T* tc_in = d_in_tc_aligned;
    T* tc_out = d_out_tc_aligned;
    for (int i = 0; i < kWarmupIters; ++i) {
        launch_wmma(tc_in, tc_out, i + 1, /*write_fp32_flag=*/false);
        std::swap(tc_in, tc_out);
        swap_comp();
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reinicia d_in_tc, d_out_tc y d_out_fp32 al estado original: el warm-up
    // encadenado es descartable y no debe alterar el estado que vera el
    // bucle medido (necesario para que --iters 1 coincida con Fase_2/Stencil).
    convert_input_to_tc<T>(d_in_fp32_aligned, d_in_tc_aligned, count);
    convert_input_to_tc<T>(d_in_fp32_aligned, d_out_tc_aligned, count);
    CHECK_CUDA(cudaMemcpy(d_out_fp32_aligned, in.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reinicia el contador de overflow tras el warm-up: sus iteraciones son
    // descartables y no deben contaminar la medicion del bucle cronometrado.
    {
        const int init_val = INT_MAX;
        CHECK_CUDA(cudaMemcpy(d_first_nf, &init_val, sizeof(int), cudaMemcpyHostToDevice));
    }
    // Reinicia el residuo tras el warm-up, igual que d_first_nf: sin esto los
    // residuos del warm-up (descartable) contaminarian el bucle medido (ver
    // bloque 2 del prompt de correccion). En modo Spatial se ponen a cero los
    // DOS buffers del ping-pong, por el mismo motivo por el que d_in_tc y
    // d_out_tc se reconvierten ambos mas arriba: comp_in/comp_out pueden haber
    // quedado intercambiados tras kWarmupIters swaps, asi que no basta con
    // limpiar uno -- cualquiera de los dos puede ser el que lea la primera
    // iteracion medida.
    if (comp_enabled) {
        CHECK_CUDA(cudaMemset(d_comp_aligned, 0, count * sizeof(float)));
    }
    if (comp_pingpong) {
        CHECK_CUDA(cudaMemset(d_comp_prev_aligned, 0, count * sizeof(float)));
        // Vuelve a sembrar el residuo de la conversion inicial: el bucle medido
        // debe arrancar del MISMO estado (T + residuo) que veria sin warm-up.
        seed_comp_buffers();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ------------------------------------------------------------------
    // CUDA Graphs: captura e instanciacion, FUERA de la region medida.
    // ------------------------------------------------------------------
    // Se hace aqui, despues del warm-up y de la restitucion del estado, y
    // ANTES de rapl_before / del cronometro / de la ventana de energia: la
    // captura y la instanciacion son trabajo de CPU puro (la captura no
    // ejecuta ningun kernel, solo lo registra) y contarlas dentro del tiempo
    // por iteracion falsearia justamente lo que este modo quiere medir.
    //
    // Se instancian DOS grafos, uno por paridad del ping-pong. Un grafo
    // hornea en sus nodos los punteros con los que se capturo; como cada
    // iteracion intercambia (tc_in, tc_out) -- y, en Spatial, (comp_in,
    // comp_out) en el mismo paso, de modo que sus paridades van encadenadas --
    // el estado de punteros solo tiene dos configuraciones posibles. Tras un
    // grafo de graph_block iteraciones (PAR, ver parse_args) los punteros
    // vuelven a la configuracion de captura, asi que reproducir el mismo
    // cudaGraphExec_t k veces seguidas es correcto; el segundo grafo hace falta
    // porque un tramo de lanzamientos normales entre checkpoints puede tener
    // longitud impar y dejar el ping-pong en la otra paridad.
    const bool use_graph = (execution_mode == ExecutionMode::Graph);
    cudaStream_t capture_stream = nullptr;
    cudaGraphExec_t graph_exec[2] = {nullptr, nullptr};
    if (use_graph) {
        CHECK_CUDA(cudaMalloc(&d_iter_offset, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_iter_offset, 0, sizeof(int)));
        CHECK_CUDA(cudaStreamCreate(&capture_stream));

        auto build_iteration_graph = [&](T* first_in, T* first_out,
                                         float* first_comp_out, float* first_comp_in) {
            T* p_in = first_in;
            T* p_out = first_out;
            float* c_out = first_comp_out;
            float* c_in = first_comp_in;
            CHECK_CUDA(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
            for (int k = 0; k < graph_block; ++k) {
                // write_fp32 = false SIEMPRE dentro del grafo: el bucle medido
                // solo mete en un grafo tramos de iteraciones que no piden
                // checkpoint ni son la ultima (ver mas abajo), que son
                // precisamente las que no escriben d_out_fp32.
                // iter_num = k + 1 es RELATIVO al bloque; la iteracion global la
                // reconstruye el kernel con *d_iter_offset.
                launch_wmma_on(p_in, p_out, c_out, c_in, k + 1, /*write_fp32_flag=*/false,
                               d_iter_offset, capture_stream);
                std::swap(p_in, p_out);
                if (comp_pingpong) std::swap(c_in, c_out);
            }
            advance_iteration_offset_kernel<<<1, 1, 0, capture_stream>>>(
                d_iter_offset, graph_block);
            cudaGraph_t graph = nullptr;
            CHECK_CUDA(cudaStreamEndCapture(capture_stream, &graph));
            cudaGraphExec_t exec = nullptr;
            CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
            CHECK_CUDA(cudaGraphDestroy(graph));
            // Sube el grafo al dispositivo por adelantado: sin esto la PRIMERA
            // reproduccion paga la carga, y esa si caeria dentro de la region
            // cronometrada.
            CHECK_CUDA(cudaGraphUpload(exec, 0));
            return exec;
        };

        graph_exec[0] = build_iteration_graph(tc_in, tc_out, comp_out, comp_in);
        graph_exec[1] = build_iteration_graph(tc_out, tc_in, comp_in, comp_out);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    power_buffer_stop_sampling(power_buffer);
    power_buffer_samples_clear(power_buffer);
    const RAEnergySnapshot rapl_before = rapl_snapshot_now();
    (void)rapl_warmup_before;

    // Buffer host reutilizado para las copias D2H de checkpoint; vacio (sin
    // costo) cuando el checkpointing esta desactivado. Tambien se usa para
    // guardar la ultima iteracion FINITA de d_out_fp32 (util para evaluar
    // storage_rel sobre salida no-divergida).
    std::vector<float> checkpoint_host_buf;
    checkpoint_host_buf.resize(count);  // siempre, para guardar ultima finita

    // Almacenan las salidas de la ultima iteracion finita (para evaluar
    // storage_rel correctamente si la ruta diverge).
    std::vector<float> out_last_finite(count);
    std::vector<T> out_reduced_last_finite(count);
    int last_finite_iter = 0;

    // Helper: verifica si un buffer FP32 es completamente finito.
    auto is_finite_buffer = [&](const std::vector<float>& buf) {
        for (const auto& x : buf) {
            if (!std::isfinite(x)) return false;
        }
        return true;
    };

    // Pares de eventos por GRUPO DE LANZAMIENTO, sin sincronizar dentro del
    // bucle: se graban en el stream con cudaEventRecord y solo se leen con
    // cudaEventElapsedTime DESPUES de timer.stop_and_elapsed_ms(), que ya
    // sincronizo una vez al final. Ya no existe un kernel de conversion
    // separado dentro del bucle (out_tc se escribe directamente desde
    // stencil2d_wmma_kernel), asi que t_conv_ms_out queda en 0: no hay nada
    // que medir por separado.
    //
    // En modo normal un grupo es exactamente una iteracion, asi que hay iters
    // grupos y la instrumentacion es la de siempre, iteracion por iteracion.
    // En modo graph un grupo es una tanda de reproducciones consecutivas del
    // mismo cudaGraphExec_t (>= graph_block iteraciones): instrumentar por
    // iteracion exigiria nodos de evento dentro del grafo, que reintroducirian
    // por la puerta de atras el sobrecoste que el modo quiere quitar. Por eso
    // el desglose fino por kernel es la ruta normal, y en graph esta columna
    // mide el tiempo de GPU dentro de los grupos (que es lo comparable con
    // t/iter total). El vector se dimensiona a iters porque un grupo cubre al
    // menos una iteracion: nunca puede haber mas grupos que iteraciones.
    std::vector<cudaEvent_t> wmma_start(iters), wmma_stop(iters);
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaEventCreate(&wmma_start[i]));
        CHECK_CUDA(cudaEventCreate(&wmma_stop[i]));
    }
    int timed_groups = 0;

    CudaEventTimer timer;
    double total_ms = 0.0;
    double checkpoint_ms_total = 0.0;
    // Igual que en la ruta GPU_FP32: energia acumulada por tramos, con cortes
    // en los mismos bloques que pausan el cronometro (ver comentario alli
    // sobre por que no basta con parar/reanudar el muestreo).
    //
    // El corte es INCONDICIONAL, no depende de checkpoints_enabled(ckpt). Esta
    // ruta entra al bloque por write_fp32, que es cierto TAMBIEN en la ultima
    // iteracion medida aunque no haya checkpointing: sin corte, ese D2H final
    // (~1 s a 16384^2) quedaba dentro de la ventana de energia mientras el
    // cronometro si lo excluia, y energy_j medía un trabajo distinto del que
    // medía t_ms_iter. Las rutas GPU_FP32/GPU_FP64 no tenian el fallo porque
    // entran por checkpoint_due() y cierran el tramo sin condicion.
    // Coste: una frontera de tramo extra (un salto de cuantizacion NVML, ~5 J)
    // a cambio de los ~44 J de contaminacion que elimina.
    double gpu_energy_j = 0.0;
    double gpu_window_s = 0.0;
    // Numero de tramos acumulados: fija energy_window_reliable junto con la
    // ventana total, porque el contador NVML se cuantiza por tramo y no sobre
    // la suma (cada tramo aporta hasta un salto de error).
    int gpu_segment_count = 0;
    bool gpu_energy_valid = true;
    double checkpoint_cpu_energy_j = 0.0;
    double checkpoint_pause_s = 0.0;
    auto close_energy_segment = [&]() {
        power_buffer_stop_sampling(power_buffer);
        gpu_energy_valid = gpu_energy_valid && power_buffer_capture_valid(power_buffer);
        gpu_energy_j += power_buffer_energy_joules(power_buffer);
        gpu_window_s += power_buffer_window_seconds(power_buffer);
        ++gpu_segment_count;
        power_buffer_samples_clear(power_buffer);
    };
    emit_csv_region_marker(route_label, "begin");
    const auto energy_t0 = std::chrono::steady_clock::now();
    power_buffer_samples_clear(power_buffer);
    power_buffer_start_sampling(power_buffer);
    timer.start();
    // El cronometro se pausa/reanuda alrededor del bloque de checkpoint (ver
    // mas abajo): sin eso, el tiempo GPU ocioso mientras el host hace el D2H
    // y escanea is_finite_buffer queda contabilizado en total_ms (ver
    // diagnostico: 98% "no atribuido" a 16384^2 con checkpoints activos).
    // Paridad viva del ping-pong respecto al estado con el que se capturaron
    // los grafos: 0 = (tc_in, tc_out) coincide con la captura de graph_exec[0].
    // Cada lanzamiento normal la invierte; una tanda de grafos NO la toca,
    // porque graph_block es par.
    int pingpong_parity = 0;
    for (int i = 0; i < iters;) {
        // iter_number es la iteracion GLOBAL 1..N que se va a ejecutar en esta
        // vuelta. Se fija antes del avance de i porque el bloque de checkpoint
        // de mas abajo corre DESPUES de ese avance y debe seguir hablando de la
        // iteracion recien ejecutada, no de la siguiente.
        const int iter_number = i + 1;
        // write_fp32 solo en la ultima iteracion medida o en un checkpoint:
        // es lo unico que necesita d_out_fp32 (comparacion final de error,
        // o CSV_DRIFT contra el snapshot FP64 de esta iteracion).
        const bool write_fp32 = (iter_number == iters) || checkpoint_due(ckpt, iter_number);

        // Tramo de grafo: solo cuando la iteracion actual no pide write_fp32.
        // Se mide cuantas iteraciones consecutivas desde aqui tampoco lo piden
        // y se cubren con floor(run / graph_block) reproducciones; el resto del
        // tramo y la propia iteracion de checkpoint caen por la via normal de
        // abajo. Esto produce exactamente el patron pedido:
        //   grafo -> iteraciones hasta el checkpoint -> checkpoint -> grafo...
        // y nunca mete una iteracion write_fp32 dentro de un grafo.
        if (use_graph && !write_fp32) {
            int run = 0;
            while (i + run < iters) {
                const int it = i + run + 1;
                if (it == iters || checkpoint_due(ckpt, it)) break;
                ++run;
            }
            const int replays = run / graph_block;
            if (replays > 0) {
                // Fija el origen global del primer bloque. Copia sincrona (4 B)
                // y una sola vez por tanda -- no por reproduccion --, asi que su
                // costo no escala con iters. El kernel advance_iteration_offset
                // se encarga del resto desde dentro del grafo.
                const int base = i;
                CHECK_CUDA(cudaMemcpy(d_iter_offset, &base, sizeof(int),
                                      cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaEventRecord(wmma_start[timed_groups]));
                for (int r = 0; r < replays; ++r) {
                    CHECK_CUDA(cudaGraphLaunch(graph_exec[pingpong_parity], /*stream=*/0));
                }
                CHECK_CUDA(cudaEventRecord(wmma_stop[timed_groups]));
                ++timed_groups;
                // graph_block par => numero par de swaps por reproduccion => el
                // ping-pong queda como estaba, tanto en punteros como en
                // paridad. No hay swap que replicar en el host.
                i += replays * graph_block;
                continue;
            }
        }

        CHECK_CUDA(cudaEventRecord(wmma_start[timed_groups]));
        launch_wmma(tc_in, tc_out, iter_number, write_fp32);
        CHECK_CUDA(cudaEventRecord(wmma_stop[timed_groups]));
        ++timed_groups;
        std::swap(tc_in, tc_out);
        swap_comp();
        pingpong_parity ^= 1;
        ++i;

        if (write_fp32) {
            // Cierra el tramo cronometrado antes de tocar el host con
            // cudaMemcpy/is_finite_buffer (REGLA CRITICA: nada de eso puede
            // quedar dentro de la region que mide t/iter).
            total_ms += timer.stop_and_elapsed_ms();
            // Misma pausa que el cronometro, ahora tambien para la energia.
            // Cubre TODO bloque write_fp32 (no solo los multiplos de
            // checkpoint_every): el D2H y el escaneo del host son identicos en
            // ambos casos, y dejar el ultimo fuera haria que energy_gpu_j
            // dependiera de si iters es multiplo de la cadencia.
            // pause_t0 antes de close_energy_segment(), por el pthread_join
            // que esa llamada hace sobre el hilo de muestreo (ver la misma
            // nota en benchmark_gpu_fp32_stencil).
            const auto pause_t0 = std::chrono::steady_clock::now();
            close_energy_segment();
            const RAEnergySnapshot rapl_ckpt_before = rapl_snapshot_now();

            const auto ckpt_t0 = std::chrono::high_resolution_clock::now();
            // Una sola copia D2H de d_out_fp32, reutilizada tanto para
            // record_checkpoint (si esta iteracion es multiplo de
            // checkpoint_every) como para el rastreo de ultima-iteracion-finita
            // de abajo: antes eran dos copias identicas seguidas al mismo buffer.
            CHECK_CUDA(cudaMemcpy(checkpoint_host_buf.data(), d_out_fp32_aligned,
                                  count * sizeof(float), cudaMemcpyDeviceToHost));
            if (checkpoint_due(ckpt, iter_number)) {
                record_checkpoint(ckpt, route_label, iter_number, checkpoint_host_buf, onset_iter);
            }
            if (archive_due(ckpt, iter_number)) {
                // NO se archiva checkpoint_host_buf: ese es d_out_fp32, el
                // acumulador ANTES del ultimo redondeo de almacenamiento. Lo que
                // el algoritmo propaga a la iteracion siguiente es el buffer T
                // (tc_in tras el swap), que es la misma convencion que ya usa
                // rel_l2_prop via reduced_to_float.
                std::vector<T> state_tc(count);
                CHECK_CUDA(cudaMemcpy(state_tc.data(), tc_in, count * sizeof(T),
                                      cudaMemcpyDeviceToHost));
                // Bajo compensacion espacial el estado efectivo no es Q(u)
                // sino Q(u)+comp: el residuo se reincorpora al leer, asi que
                // archivar solo el buffer T perderia exactamente la parte
                // que esa politica existe para conservar. comp_in es el
                // buffer recien escrito (swap_comp ya corrio), el que
                // corresponde a tc_in.
                // Por que Local queda FUERA de la reconstruccion, y no es un descuido:
                //
                // La identidad Q(u) + comp == u vale SOLO con la convencion de signo de
                // Spatial, comp = val - Q(val). Local usa retroalimentacion de error y guarda
                // comp = Q(y) - y, con el signo opuesto y referido a y = val - comp_anterior,
                // no a val. Sumarlo da Q(y) + (Q(y) - y) = 2Q(y) - y, que se PASA de largo en
                // vez de corregir. Medido en diffusive con --kahan on, rel_l2_prop empeora
                // 1.78x a 1 iteracion, 1.37x a 2, 1.20x a 5, y a 20 queda diluido a 1.000x --
                // invisible, pero igual de incorrecto.
                //
                // Tampoco hay una reconstruccion alternativa valida: Q(y) - comp recupera y,
                // no val, y val = y + comp_anterior con comp_anterior ya sobrescrito por esta
                // misma escritura. El estado que Local propaga ES el buffer T crudo, asi que
                // leerlo crudo no es una perdida de precision: es la lectura correcta.
                if (comp_mode == CompMode::Spatial) {
                    std::vector<float> comp_host(count);
                    CHECK_CUDA(cudaMemcpy(comp_host.data(), comp_in, count * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                    archive_field(*ckpt.archive, route_label, iter_number, nx, ny, "float32",
                                  build_spatial_reconstructed_field(state_tc, comp_host));
                } else {
                    archive_field(*ckpt.archive, route_label, iter_number, nx, ny, "float32",
                                  reduced_to_float(state_tc));
                }
            }
            if (is_finite_buffer(checkpoint_host_buf)) {
                // std::swap en vez de out_last_finite = checkpoint_host_buf:
                // evita copiar el vector completo (~1 GB a 16384^2) en cada
                // checkpoint. checkpoint_host_buf queda con el contenido
                // anterior de out_last_finite, que el proximo checkpoint
                // sobrescribe de todas formas con el cudaMemcpy de arriba.
                std::swap(out_last_finite, checkpoint_host_buf);
                out_reduced_last_finite.resize(count);
                CHECK_CUDA(cudaMemcpy(out_reduced_last_finite.data(), tc_in, count * sizeof(T),
                                      cudaMemcpyDeviceToHost));
                last_finite_iter = iter_number;
            }
            const auto ckpt_t1 = std::chrono::high_resolution_clock::now();
            checkpoint_ms_total +=
                std::chrono::duration<double, std::milli>(ckpt_t1 - ckpt_t0).count();

            const RAEnergySnapshot rapl_ckpt_after = rapl_snapshot_now();
            checkpoint_cpu_energy_j += rapl_energy_delta(rapl_ckpt_before, rapl_ckpt_after);
            power_buffer_start_sampling(power_buffer);
            checkpoint_pause_s += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pause_t0).count();

            timer.start();
        }
    }
    total_ms += timer.stop_and_elapsed_ms();
    close_energy_segment();
    const RAEnergySnapshot rapl_after = rapl_snapshot_now();
    const auto energy_t1 = std::chrono::steady_clock::now();
    emit_csv_region_marker(route_label, "end");
    // energy_wall_s proviene de los mismos timestamps de muestreo que se
    // integraron en gpu_energy_j (power_buffer_window_seconds acumulado por
    // tramo en close_energy_segment), no de un reloj de pared aparte: asi
    // avg_power_w/edp_j_s quedan derivados del mismo intervalo que la
    // energia.
    const double energy_wall_s = gpu_window_s;
    const double flops_total =
        stencil_flops(nx, ny, op.flops_per_cell) * static_cast<double>(iters);
    const bool cpu_energy_valid = rapl_before.valid && rapl_after.valid &&
                                  rapl_after.energy_j >= rapl_before.energy_j;
    const double cpu_energy_j = std::max(
        0.0, rapl_energy_delta(rapl_before, rapl_after) - checkpoint_cpu_energy_j);
    out_energy = make_energy_measurement_from_segments(
        gpu_energy_valid, gpu_energy_j, cpu_energy_valid, cpu_energy_j,
        energy_wall_s, flops_total, gpu_segment_count);
    power_buffer_destroy(power_buffer);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(out.data(), d_out_fp32_aligned, count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&first_nonfinite_iter, d_first_nf, sizeof(int), cudaMemcpyDeviceToHost));

    // out/out_reduced quedan SIEMPRE con la ultima iteracion medida cruda,
    // aunque contenga inf/NaN: el llamador los compara contra la referencia
    // FP64 de esa MISMA iteracion (sustituir por un estado de una iteracion
    // anterior invalidaba la metrica de error, ver diagnostico del bloque 1).
    // Quien necesite un estado recuperable (storage_rel) usa
    // out_last_finite_o / out_reduced_last_finite_o, expuestos aparte.
    out_last_finite_o = std::move(out_last_finite);
    out_reduced_last_finite_o = std::move(out_reduced_last_finite);

    // Solo se leen los timed_groups pares realmente grabados (en modo normal
    // timed_groups == iters y esto es el bucle de siempre); los sobrantes se
    // destruyen sin consultarlos, porque cudaEventElapsedTime sobre un evento
    // nunca grabado devuelve cudaErrorInvalidResourceHandle.
    double t_wmma_sum_ms = 0.0;
    for (int i = 0; i < timed_groups; ++i) {
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, wmma_start[i], wmma_stop[i]));
        t_wmma_sum_ms += ms;
    }
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaEventDestroy(wmma_start[i]));
        CHECK_CUDA(cudaEventDestroy(wmma_stop[i]));
    }
    // Mismo denominador (iters) que build_metrics usa para total_ms: asi
    // t_wmma_ms_out + t_conv_ms_out + no_atribuido reproduce exactamente el
    // t/iter total sin redondeos cruzados entre distintos denominadores.
    t_wmma_ms_out = t_wmma_sum_ms / iters;
    t_conv_ms_out = 0.0;
    t_checkpoint_ms_out = checkpoint_ms_total / iters;

    // Registra en que iteracion se evaluo storage_rel (util para anotar si
    // la ruta divergio antes de iters). Sentinela -1: la ruta divergio y
    // ningun checkpoint (ni la iteracion final) alcanzo a capturar un estado
    // COMPLETAMENTE finito antes de eso -- d_out_fp32 solo se escribe en
    // iteraciones write_fp32 (checkpoints o la ultima), asi que sin
    // --checkpoint-every no hay forma de recuperar un estado finito posterior
    // a la divergencia; storage_rel no es evaluable de forma confiable en
    // ese caso (ver StorageRelResult y el llamador, que debe mostrar
    // "NO EVALUABLE" en vez de imprimir un numero calculado sobre datos
    // parcialmente no finitos).
    if (first_nonfinite_iter != INT_MAX && last_finite_iter == 0) {
        storage_rel_eval_iter = -1;
    } else {
        storage_rel_eval_iter = (last_finite_iter > 0) ? last_finite_iter : iters;
    }

    // out_reduced (formato T) SIEMPRE se toma de tc_in, crudo: tras el ultimo
    // swap, tc_in apunta al buffer con la salida mas reciente (float_to_tc<T>,
    // misma funcion que antes aplicaba convert_input_to_tc). Ya no se
    // sustituye por out_reduced_last_finite (ver comentario de out/out_reduced
    // mas arriba).
    out_reduced.resize(count);
    CHECK_CUDA(cudaMemcpy(out_reduced.data(), tc_in, count * sizeof(T), cudaMemcpyDeviceToHost));

    // Residuo que corresponde a ese out_reduced, con el MISMO criterio que el
    // bloque de archivado: SOLO Spatial, y de comp_in, que es el buffer recien
    // escrito (el swap_comp del final de la ultima iteracion ya corrio, igual
    // que el std::swap de tc_in/tc_out del que sale el memcpy de arriba).
    //
    // comp_pingpong es exactamente (comp_mode == CompMode::Spatial). En Local y
    // en Off out_comp queda VACIO y build_readout_field colapsa a
    // reduced_to_float: para Off porque no hay buffer, y para Local porque su
    // convencion de signo hace que Q(u)+comp sea una reconstruccion invalida
    // (ver la nota extensa en el bloque de archivado de checkpoints).
    out_comp.clear();
    if (comp_pingpong) {
        out_comp.resize(count);
        CHECK_CUDA(cudaMemcpy(out_comp.data(), comp_in, count * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaFree(d_in_fp32));
    CHECK_CUDA(cudaFree(d_out_fp32));
    CHECK_CUDA(cudaFree(d_in_tc));
    CHECK_CUDA(cudaFree(d_out_tc));
    CHECK_CUDA(cudaFree(d_horizontal));
    CHECK_CUDA(cudaFree(d_vertical));
    CHECK_CUDA(cudaFree(d_first_nf));
    if (d_comp != nullptr) {
        CHECK_CUDA(cudaFree(d_comp));
    }
    if (d_comp_prev != nullptr) {
        CHECK_CUDA(cudaFree(d_comp_prev));
    }
    for (cudaGraphExec_t exec : graph_exec) {
        if (exec != nullptr) CHECK_CUDA(cudaGraphExecDestroy(exec));
    }
    if (capture_stream != nullptr) {
        CHECK_CUDA(cudaStreamDestroy(capture_stream));
    }
    if (d_iter_offset != nullptr) {
        CHECK_CUDA(cudaFree(d_iter_offset));
    }

    return build_metrics(nx, ny, total_ms / iters, op.flops_per_cell);
}

// Imprime max_abs/rel_l2, o un mensaje explicito si la referencia o la
// solucion no son finitas (evita imprimir "0.000000"/"nan" como si fuera
// una medicion valida). first_nf es la primera iteracion no finita de LA
// RUTA evaluada (no de la referencia); se concatena como causa solo cuando
// la ruta (no la referencia) es la que diverge.
static void print_error_metrics(const char* label_max, const char* label_l2,
                                 const char* label_linf, const ErrorMetrics& e, int first_nf) {
    if (!e.reference_finite) {
        std::cout << label_max
                   << "REFERENCIA NO FINITA: la solucion diverguio; "
                      "L2/Linf no medibles en esta configuracion\n";
        return;
    }
    if (!e.solution_finite) {
        std::cout << label_max
                   << "SOLUCION NO FINITA: la ruta diverguio; "
                      "L2/Linf no medibles en esta configuracion";
        if (first_nf != INT_MAX) {
            std::cout << " (desbordamiento de exponente en iteracion " << first_nf << ")";
        }
        std::cout << "\n";
        return;
    }
    std::cout << label_max  << fmt_sci(e.max_abs)  << "\n";
    std::cout << label_l2   << fmt_sci(e.rel_l2)   << "\n";
    std::cout << label_linf << fmt_sci(e.rel_linf) << "\n";
}

// Analogo a print_error_metrics pero solo para rel_l2/rel_linf del ESTADO
// PROPAGADO (el buffer T en 16 bits, no el acumulador FP32 sin redondear que
// ya reporta print_error_metrics contra out_fp32): responde si Kahan acerca
// FP16/BF16 a la exactitud de FP32 en lo que realmente encadena la siguiente
// iteracion (metodologia 5.3/4.1.4). Mismas guardas de finitud; sin
// max_abs/linf_abs porque el bloque solo pide L2/Linf relativos aqui.
static void print_propagated_error_metrics(const ErrorMetrics& e, int first_nf) {
    if (!e.reference_finite) {
        std::cout << "Error relativo L2 (estado propagado, 16 bits)   : "
                     "REFERENCIA NO FINITA: la solucion diverguio; L2/Linf no medibles\n";
        return;
    }
    if (!e.solution_finite) {
        std::cout << "Error relativo L2 (estado propagado, 16 bits)   : "
                     "SOLUCION NO FINITA: la ruta diverguio; L2/Linf no medibles";
        if (first_nf != INT_MAX) {
            std::cout << " (desbordamiento de exponente en iteracion " << first_nf << ")";
        }
        std::cout << "\n";
        return;
    }
    std::cout << "Error relativo L2 (estado propagado, 16 bits)   : " << fmt_sci(e.rel_l2) << "\n";
    std::cout << "Error rel Linf (estado propagado, 16 bits)      : " << fmt_sci(e.rel_linf) << "\n";
}

// n == INT_MAX (sentinel de "nunca se marco") se reporta como "ninguna".
static void print_first_nonfinite(const char* label, int first_nf, int iters) {
    std::cout << label;
    if (first_nf == INT_MAX) {
        std::cout << "ninguna (finito hasta iters=" << iters << ")\n";
    } else {
        std::cout << first_nf << "\n";
    }
}

constexpr double kFp16StorageUlp = 4.8828125e-4;  // 2^-11
constexpr double kBf16StorageUlp = 3.90625e-3;    // 2^-8
// 2^-53: unidad de redondeo de double, el formato en que GPU_FP64 acumula Y
// almacena. Da la escala contra la que se leen los store_rel de FP16/BF16 en
// el mismo CSV_STORE.
constexpr double kFp64StorageUlp = 1.1102230246251565e-16;  // 2^-53

static void append_storage_eval_annotation(const StorageRelResult& storage, int iters) {
    if (storage.eval_iter > 0 && storage.eval_iter < iters) {
        std::cout << "  (eval. en iter " << storage.eval_iter << ")";
    }
}

static void print_storage_metrics(const char* format_label,
                                  const StorageRelResult& storage,
                                  bool storage_evaluable,
                                  int iters,
                                  double warning_threshold) {
    const std::string prefix = std::string(" en ") + format_label;
    if (!storage_evaluable || !storage.evaluated) {
        const char* msg = storage_evaluable
            ? "NO EVALUABLE (estado de evaluacion no finito o norma nula)"
            : "NO EVALUABLE (la ruta divergio antes de cualquier checkpoint finito;"
              " use --checkpoint-every para medir store_rel de forma confiable)";
        std::cout << "Error relativo L2 al guardar" << prefix
                  << " (store_rel_norm)      : " << msg << "\n";
        std::cout << "Error relativo max por elemento al guardar" << prefix
                  << " (store_rel_max_guarded): " << msg << "\n";
        std::cout << "Elementos excluidos al guardar" << prefix
                  << " (store_excluded_count): " << msg << "\n";
        return;
    }

    std::cout << "Error relativo L2 al guardar" << prefix
              << " (store_rel_norm)      : " << fmt_csv_num(storage.rel_norm);
    append_storage_eval_annotation(storage, iters);
    std::cout << "\n";

    std::cout << "Error relativo max por elemento al guardar" << prefix
              << " (store_rel_max_guarded): " << fmt_csv_num(storage.rel_max_guarded);
    append_storage_eval_annotation(storage, iters);
    std::cout << "\n";

    std::cout << "Elementos excluidos al guardar" << prefix
              << " (store_excluded_count): " << storage.excluded_count;
    append_storage_eval_annotation(storage, iters);
    std::cout << "\n";

    if (std::isfinite(storage.rel_max_guarded) && storage.rel_max_guarded > warning_threshold) {
        std::cout << "ADVERTENCIA: store_rel_max_guarded=" << fmt_sci(storage.rel_max_guarded)
                  << " supera 2 ulp en iter " << storage.eval_iter << "\n";
    }
}

static std::string storage_num_field(const StorageRelResult& storage,
                                     bool storage_evaluable,
                                     double value) {
    return (storage_evaluable && storage.evaluated && std::isfinite(value)) ? fmt_sci(value) : "NaN";
}

static std::string storage_count_field(const StorageRelResult& storage, bool storage_evaluable) {
    return (storage_evaluable && storage.evaluated) ? std::to_string(storage.excluded_count) : "NaN";
}

static std::string storage_eval_iter_field(const StorageRelResult& storage, bool storage_evaluable) {
    return (storage_evaluable && storage.evaluated && storage.eval_iter >= 0)
           ? std::to_string(storage.eval_iter) : "NaN";
}

static std::string energy_csv_field(bool valid, double value) {
    return (valid && std::isfinite(value)) ? fmt_sci(value) : "NaN";
}

// Metricas de trabajo util. A diferencia de gflops y joules_per_gflop, su
// denominador -- la celda interior actualizada -- es el MISMO trabajo bajo los
// dos operadores, asi que son las unicas dos que se pueden comparar 1:1 entre
// --op-mode stress y --op-mode diffusive (el conteo de FLOPs cambia de 5 a 6
// ops/celda, ver StencilOperator::flops_per_cell). En la comparacion cruzada
// entre operadores son las metricas primarias de rendimiento y de energia;
// GFLOP/s pasa a ser una metrica interna de cada operador.
static double cell_updates_per_s(int nx, int ny, double t_iter_ms) {
    return interior_cells(nx, ny) / (t_iter_ms * 1e-3);
}

static double energy_per_cell_update_j(int nx, int ny, int iters, double energy_total_j) {
    return energy_total_j / (interior_cells(nx, ny) * static_cast<double>(iters));
}

// La misma metrica, ya formateada para el CSV opcional (--csv). Usa a proposito
// el MISMO numerador (energy_total_j) y la misma guarda de validez que la
// columna homonima de CSV_SUMMARY: la columna energy_j vecina de ese archivo es
// solo la GPU, y publicar dos magnitudes distintas bajo el mismo nombre en dos
// salidas del mismo binario seria una trampa para el analisis. under_ncu la
// anula por la misma razon que anula las otras columnas de energia -- bajo el
// perfilador el tiempo, y con el la energia, estan inflados.
static std::string energy_per_cell_csv_field(bool under_ncu, const EnergyMeasurement& e,
                                             int nx, int ny, int iters) {
    return energy_field(!under_ncu && e.gpu_valid && e.cpu_valid,
                        energy_per_cell_update_j(nx, ny, iters, e.energy_total_j));
}

// Papel de la ruta como referencia de error, explicito en el CSV para que el
// analisis no tenga que reconstruirlo desde el nombre de la ruta:
//   CPU_FP64_canonical: la verdad IEEE 754 contra la que se mide todo rel_l2
//                       publicado (exigido por el marco de la tesis).
//   GPU_FP64_control  : control de implementacion -- el piso de error
//                       alcanzable en GPU -- y a la vez un punto propio del
//                       frente de Pareto, no una referencia de error.
// El resto de rutas son observaciones, no referencias: "NA".
static const char* kReferenceRoleNone = "NA";
static const char* kReferenceRoleCpuFp64 = "CPU_FP64_canonical";
static const char* kReferenceRoleGpuFp64 = "GPU_FP64_control";

static void print_energy_metrics(const EnergyMeasurement& energy) {
    std::cout << "Energy GPU    : " << energy_csv_field(energy.gpu_valid, energy.energy_gpu_j) << " J\n";
    std::cout << "Energy CPU    : " << energy_csv_field(energy.cpu_valid, energy.energy_cpu_j) << " J\n";
    const bool total_valid = energy.gpu_valid && energy.cpu_valid;
    std::cout << "Energy total  : " << energy_csv_field(total_valid, energy.energy_total_j) << " J\n";
    std::cout << "EDP           : " << energy_csv_field(total_valid, energy.edp_j_s) << " J s\n";
    std::cout << "Joules/GFLOP  : " << energy_csv_field(total_valid, energy.joules_per_gflop) << "\n";
}

// gpu_route distingue las rutas que de verdad midieron NVML de la ruta CPU,
// que fija gpu_valid=true sin leer el contador (no hay ventana de GPU que
// medir, ver benchmark_cpu_stencil). Las dos ultimas columnas salen NaN en esa
// ruta en vez de 0/0: un 0 simularia una medicion de GPU que nunca se hizo.
static void emit_csv_energy_row(const char* route,
                                int nx,
                                int ny,
                                int iters,
                                bool kahan,
                                const EnergyMeasurement& energy,
                                double flops_total,
                                bool gpu_route) {
    const bool total_valid = energy.gpu_valid && energy.cpu_valid;
    // Metrica de comparacion GPU-vs-GPU entre formatos: la energia absoluta no
    // es comparable entre corridas con ITERS distintos, la energia por
    // iteracion si -- siempre que la ventana sea fiable, que es lo que informa
    // la columna siguiente.
    const bool per_iter_valid = gpu_route && energy.gpu_valid && iters > 0;
    const double energy_gpu_j_per_iter =
        per_iter_valid ? energy.energy_gpu_j / static_cast<double>(iters) : 0.0;
    std::cout << "CSV_ENERGY," << route << "," << nx << "," << ny << "," << iters << ","
              << kahan_label(kahan) << ","
              << energy_csv_field(energy.gpu_valid, energy.energy_gpu_j) << ","
              << energy_csv_field(energy.cpu_valid, energy.energy_cpu_j) << ","
              << energy_csv_field(total_valid, energy.energy_total_j) << ","
              << energy_csv_field(total_valid, energy.edp_j_s) << ","
              << energy_csv_field(total_valid, energy.joules_per_gflop) << ","
              << energy_csv_field(std::isfinite(energy.time_total_s), energy.time_total_s) << ","
              << energy_csv_field(std::isfinite(flops_total), flops_total / 1e9) << ","
              << energy_csv_field(per_iter_valid, energy_gpu_j_per_iter) << ","
              << (gpu_route ? (energy.window_reliable ? "1" : "0") : "NaN") << "\n";
}

// opt entra entero (y no como cuatro escalares mas) porque las cuatro columnas
// de contexto -- op_mode, alpha, ci_mode, ci_p -- son las MISMAS para todas las
// filas de una corrida: pasarlas sueltas invitaria a que un solo call site las
// desincronizara. Las columnas nuevas van al FINAL, detras de joules_per_gflop,
// para no correr ninguna posicion del esquema anterior.
static void emit_csv_summary_row(const Options& opt,
                                 const char* route,
                                 int nx,
                                 int ny,
                                 int iters,
                                 bool kahan,
                                 double t_iter_ms,
                                 double gflops,
                                 const std::string& speedup_cpu,
                                 const std::string& speedup_fp32,
                                 const std::string& t_kernel_ms,
                                 const std::string& t_convert_ms,
                                 const std::string& t_checkpoint_ms,
                                 const ErrorMetrics& err,
                                 int first_nf,
                                 const std::string& rel_l2_prop,
                                 const std::string& rel_linf_prop,
                                 const std::string& store_rel_norm,
                                 const std::string& store_rel_max_guarded,
                                 const std::string& store_excluded_count,
                                 const std::string& store_eval_iter,
                                 const EnergyMeasurement& energy,
                                 const char* reference_role) {
    const bool total_valid_energy = energy.gpu_valid && energy.cpu_valid;
    std::cout << "CSV_SUMMARY," << route << "," << nx << "," << ny << "," << iters << ","
              << kahan_label(kahan) << "," << fmt_csv_num(t_iter_ms) << ","
              << fmt_csv_num(t_iter_ms * iters) << "," << fmt_csv_num(gflops) << ","
              << speedup_cpu << "," << speedup_fp32 << "," << t_kernel_ms << ","
              << t_convert_ms << "," << t_checkpoint_ms << ","
              << fmt_csv_error_num(err, err.rel_l2) << ","
              << fmt_csv_error_num(err, err.rel_linf) << ","
              << fmt_csv_error_num(err, err.max_abs) << ","
              << rel_l2_prop << "," << rel_linf_prop << ","
              << csv_first_nonfinite_field(first_nf) << ","
              << store_rel_norm << "," << store_rel_max_guarded << ","
              << store_excluded_count << "," << store_eval_iter << ","
              << energy_csv_field(energy.gpu_valid, energy.energy_gpu_j) << ","
              << energy_csv_field(energy.cpu_valid, energy.energy_cpu_j) << ","
              << energy_csv_field(energy.gpu_valid && energy.cpu_valid, energy.energy_total_j) << ","
              << energy_csv_field(energy.gpu_valid && energy.cpu_valid, energy.edp_j_s) << ","
              << energy_csv_field(energy.gpu_valid && energy.cpu_valid, energy.joules_per_gflop)
              // --- columnas nuevas de Fase 4 (Pareto 3D), siempre al final ---
              << "," << op_mode_label(opt.op_mode) << ","
              << (opt.op_mode == OpMode::Diffusive ? fmt_sci(opt.alpha) : std::string("NA")) << ","
              << ci_mode_label(opt.ci_mode) << ","
              << (opt.ci_mode == CiMode::Monomode ? std::to_string(opt.ci_p) : std::string("NA")) << ","
              << fmt_csv_num(cell_updates_per_s(nx, ny, t_iter_ms)) << ","
              << energy_csv_field(total_valid_energy,
                                  energy_per_cell_update_j(nx, ny, iters, energy.energy_total_j))
              << "," << reference_role
              // execution_mode: se AGREGA al final para no correr ningun indice
              // existente (los scripts de post-proceso siguen leyendo $1..$36
              // igual que antes). Se deriva del prefijo de la ruta y no se pasa
              // por parametro porque --execution-mode graph solo tiene efecto
              // en las rutas WMMA: CPU_FP32, CPU_FP64, GPU_FP32 y GPU_FP64
              // corren siempre con un lanzamiento por iteracion, y etiquetarlas
              // con el modo pedido diria que usaron un grafo que nunca existio.
              << "," << ((std::strncmp(route, "WMMA", 4) == 0)
                             ? execution_mode_label(opt.execution_mode)
                             : execution_mode_label(ExecutionMode::Normal))
              << "\n";
}

static void emit_csv_store_row(const char* route,
                               int nx,
                               int ny,
                               int iters,
                               bool kahan,
                               const StorageRelResult& storage,
                               bool storage_evaluable,
                               double format_ulp) {
    std::cout << "CSV_STORE," << route << "," << nx << "," << ny << "," << iters << ","
              << kahan_label(kahan) << ","
              << storage_num_field(storage, storage_evaluable, storage.rel_norm) << ","
              << storage_num_field(storage, storage_evaluable, storage.rel_max_guarded) << ","
              << storage_count_field(storage, storage_evaluable) << ","
              << storage_eval_iter_field(storage, storage_evaluable) << ","
              << fmt_csv_num(format_ulp) << "\n";
}

static void print_reference_comparison(const char* label,
                                       const Metrics& m,
                                       double ref_ms,
                                       const ErrorMetrics& e_fp64,
                                       const ErrorMetrics& e_cpu,
                                       int first_nf,
                                       int iters,
                                       double t_checkpoint_ms) {
    std::cout << label << " - tiempo/iter (media) : " << m.ms << " ms\n";
    std::cout << label << " - tiempo total        : " << m.ms * iters << " ms\n";
    std::cout << label << " - rendimiento    : " << m.gflops << " GFLOP/s ("
              << m.tflops << " TFLOP/s efectivos)\n";
    std::cout << "Speedup vs CPU             : " << ref_ms / m.ms << "x\n";
    std::cout << "t checkpoints/iter  : " << t_checkpoint_ms
              << " ms  (excluido del t/iter reportado)\n";
    print_error_metrics("Error max abs vs FP64      : ", "Error relativo L2 vs FP64  : ",
                        "Error rel Linf vs FP64     : ", e_fp64, first_nf);
    print_error_metrics("Error max abs vs CPU FP32  : ", "Error rel L2 vs CPU FP32   : ",
                        "Error rel Linf vs CPU FP32 : ", e_cpu, first_nf);
    print_first_nonfinite("Primera iteracion no finita : ", first_nf, iters);
    std::cout << "\n";
}

// Imprime, una sola vez por corrida, las normas de la referencia FP64: dan
// escala al error absoluto (Linf/L2 sin normalizar no dicen nada por si
// solos, ver comentario de ErrorMetrics::rel_linf).
static void print_fp64_reference_norms(const std::vector<double>& y_ref, int first_nf_fp64_ref) {
    if (first_nf_fp64_ref != INT_MAX) {
        std::cout << "Norma ||u^n||_inf (ref FP64) : REFERENCIA NO FINITA\n";
        std::cout << "Norma ||u^n||_2   (ref FP64) : REFERENCIA NO FINITA\n";
        return;
    }
    double norm_inf = 0.0;
    double sq = 0.0;
    for (double x : y_ref) {
        norm_inf = std::max(norm_inf, std::abs(x));
        sq += x * x;
    }
    std::cout << "Norma ||u^n||_inf (ref FP64) : " << fmt_sci(norm_inf) << "\n";
    std::cout << "Norma ||u^n||_2   (ref FP64) : " << fmt_sci(std::sqrt(sq)) << "\n";
}

// Abre el CSV en modo append; escribe la cabecera solo si el archivo aun no
// existe (probeado antes de abrir en modo append, que no trunca ni crea con
// contenido previo visible al ifstream).
// Cabecera CSV FINAL de Fase 3 (incluye las 3 columnas de Fase 4: energy_j,
// avg_power_w, edp -- llenadas in-process por PowerBuffer/RAPL en la misma
// ventana begin/end que emit_csv_region_marker delimita; "NA" si la sonda
// no es valida para la ruta. Este CSV opcional conserva su esquema historico;
// el stdout parseable de Fase 3 usa CSV_SUMMARY y CSV_ENERGY.
// Las 7 columnas finales las agrega Fase 4 (campana de Pareto 3D). Van al FINAL
// justo por la validacion de open_csv de mas abajo: cualquier CSV previo tiene
// una cabecera distinta y el programa aborta en vez de appendear filas
// desalineadas, que es el comportamiento deseado -- la campana de Fase 4 escribe
// a un archivo NUEVO, nunca a los de campanas anteriores.
static const char* kCsvHeader =
    "kernel,formato,kahan,nx,ny,iters,t_ms_iter,t_ms_total,t_ms_iter_wmma,t_ms_iter_conv,"
    "t_ms_iter_ckpt,gflops_utiles,rel_l2,rel_linf,linf_abs,ref_linf,rel_l2_prop,"
    "rel_linf_prop,n_star,storage_rel_err,energy_j,avg_power_w,edp,"
    "op_mode,alpha,ci_mode,ci_p,cell_updates_per_s,energy_per_cell_update_j,"
    "reference_role,execution_mode\n";

// Abre el CSV en modo append; escribe la cabecera solo si el archivo aun no
// existe. Si el archivo YA existe con una cabecera distinta a kCsvHeader
// (p.ej. el esquema anterior a este bloque, sin columna kahan/rel_l2_prop/
// energia), aborta en vez de appendear: mezclar filas con esquemas de
// columnas distintos corrompe el CSV sin ningun aviso visible.
static std::ofstream open_csv(const std::string& path) {
    std::ifstream probe(path);
    const bool exists = probe.good();
    std::string existing_header;
    if (exists) {
        std::getline(probe, existing_header);
        existing_header += '\n';
    }
    probe.close();
    if (exists && existing_header != kCsvHeader) {
        std::cerr << "ERROR: " << path << " ya existe con una cabecera CSV distinta al "
                     "esquema vigente (ver bloque 3, esquema final de Fase 3). No se puede "
                     "appendear filas con columnas desalineadas: use un archivo --csv nuevo.\n"
                  << "  Cabecera esperada  : " << kCsvHeader
                  << "  Cabecera encontrada: " << existing_header;
        std::exit(EXIT_FAILURE);
    }
    std::ofstream csv(path, std::ios::app);
    if (!exists) {
        csv << kCsvHeader;
    }
    return csv;
}

// Una fila por ruta/configuracion. n_star es -1 cuando la ruta se mantuvo
// finita; storage_rel_err, t_ms_iter_wmma, t_ms_iter_conv, rel_l2_prop y
// rel_linf_prop son "NA" en cpu_fp32/gpu_fp32 (no aplica: esas rutas no pasan
// por 16 bits ni separan kernel WMMA de conversion). t_ms_iter_ckpt es "NA"
// solo en cpu_fp32 (unica ruta sin CheckpointContext). energy_j/avg_power_w/
// edp llegan ya formateados (energy_field(), "NA" si la sonda no es valida
// para esa ruta) porque el llamador es quien decide, por ruta, si corresponde
// medir GPU (cpu_fp32 no tiene sonda; NCU_* la fuerza a "NA" porque el
// perfilador infla tiempos y energia igual que ya hace con t_ms_iter).
static void write_csv_row(std::ofstream& csv, const Options& opt, const std::string& formato,
                          bool kahan, int nx, int ny,
                          int iters, double t_ms_iter, double gflops, const ErrorMetrics& e,
                          int first_nf, const std::string& storage_rel_err,
                          const char* reference_role,
                          const std::string& energy_per_cell_update = "NA",
                          const std::string& t_ms_iter_wmma = "NA",
                          const std::string& t_ms_iter_conv = "NA",
                          const std::string& t_ms_iter_ckpt = "NA",
                          const std::string& rel_l2_prop = "NA",
                          const std::string& rel_linf_prop = "NA",
                          const std::string& energy_j = "NA",
                          const std::string& avg_power_w = "NA",
                          const std::string& edp = "NA",
                          // Solo las rutas WMMA pueden correr bajo CUDA Graph; las
                          // demas quedan con el valor por defecto porque etiquetarlas
                          // con el modo pedido diria que usaron un grafo inexistente.
                          const char* execution_mode = "normal") {
    const int n_star = (first_nf == INT_MAX) ? -1 : first_nf;
    csv << "stencil," << formato << "," << (kahan ? 1 : 0) << "," << nx << "," << ny << ","
        << iters << "," << fmt_sci(t_ms_iter) << "," << fmt_sci(t_ms_iter * iters) << ","
        << t_ms_iter_wmma << "," << t_ms_iter_conv << "," << t_ms_iter_ckpt << ","
        << fmt_sci(gflops) << "," << fmt_sci(e.rel_l2) << "," << fmt_sci(e.rel_linf) << ","
        << fmt_sci(e.max_abs) << "," << fmt_sci(e.ref_linf) << "," << rel_l2_prop << ","
        << rel_linf_prop << "," << n_star << "," << storage_rel_err << ","
        << energy_j << "," << avg_power_w << "," << edp << ","
        // --- columnas nuevas de Fase 4, en el mismo orden que kCsvHeader ---
        << op_mode_label(opt.op_mode) << ","
        << (opt.op_mode == OpMode::Diffusive ? fmt_sci(opt.alpha) : std::string("NA")) << ","
        << ci_mode_label(opt.ci_mode) << ","
        << (opt.ci_mode == CiMode::Monomode ? std::to_string(opt.ci_p) : std::string("NA")) << ","
        << fmt_sci(cell_updates_per_s(nx, ny, t_ms_iter)) << ","
        << energy_per_cell_update << "," << reference_role << ","
        << execution_mode << "\n";
}

// Umbral de overflow por formato (maximo valor finito representable), solo
// para la PREDICCION del horizonte. FP32/FP64 salen de std::numeric_limits;
// FP16/BF16 se dejan explicitos porque numeric_limits<__half/__nv_bfloat16>
// no esta garantizado en compilacion host. BF16 comparte los 8 bits de
// exponente de FP32 (mismo rango, distinta mantisa), por eso su umbral es
// del mismo orden que el de FP32.
// Piso de siembra por formato, usado para acotar la semilla minima de cada
// formato en compute_overflow_horizon_from_reference (semilla_T = max(A,
// piso_T * ||u0||_inf)). El valor es 1/4 de la ULP real de T (unidad de
// redondeo/media ULP: 2^-11 FP16, 2^-8 BF16, 2^-24 FP32, 2^-53 FP64) -- NO es
// "media ULP" pese al nombre historico de la constante: es una calibracion
// empirica del modelo de siembra del ajuste log-lineal (fit_overflow_model),
// no una cota derivada de redondeo. Se conserva en 1/4 de ULP porque ajusta
// mejor la prediccion de n* contra el horizonte medido que la ULP completa
// (-3.6% de error de prediccion vs -7.1%).
constexpr double kFp16Max = 65504.0;
constexpr double kFp16SeedFloor = 2.44140625e-4;      // 2^-12 (calibrado, 0.5 x unidad de redondeo 2^-11)
constexpr double kBf16Max = 3.38953139e38;
constexpr double kBf16SeedFloor = 1.953125e-3;        // 2^-9  (calibrado, 0.5 x unidad de redondeo 2^-8)
constexpr double kFp32Max = 3.4028235e38;
constexpr double kFp32SeedFloor = 2.98023225e-8;      // 2^-25 (calibrado, 0.5 x unidad de redondeo 2^-24)
constexpr double kFp64Max = 1.7976931348623157e308;
constexpr double kFp64SeedFloor = 5.5511151e-17;      // 2^-54 (calibrado, 0.5 x unidad de redondeo 2^-53)

// Proyecta la condicion inicial u^0 (sin modificarla) sobre el modo de
// Nyquist (pi,pi): a_nyq = |<u^0, e_nyq>| / N, con e_nyq(i,j) = (-1)^(i+j).
// El operador discreto 0.25*(up+down+left+right)-center tiene simbolo
// 0.5*(cos(tx)+cos(ty)) - 1, que en (tx,ty)=(pi,pi) vale exactamente -2
// (|lambda|=2, propiedad del operador, no un parametro ajustable): bajo esa
// condicion inicial la componente Nyquist crece como a_nyq * 2^n hasta
// desbordar el formato en n* = log2(fmt_max / a_nyq). Se calcula en FP64,
// una sola vez, antes de cualquier region cronometrada.
// NOTA: la proyeccion directa es grid-dependent (incluye bordes). Para
// prediccion robusta de horizonte, se calibra desde la referencia FP64
// (ver compute_overflow_horizon_from_reference).
static double compute_nyquist_component(const std::vector<float>& u0, int nx, int ny) {
    double a_nyq = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            a_nyq += static_cast<double>(u0[idx2d(i, j, nx)]) * (((i + j) & 1) ? -1.0 : 1.0);
        }
    }
    return std::fabs(a_nyq) / (static_cast<double>(nx) * static_cast<double>(ny));
}

// Resultado del ajuste log-lineal: valid distingue "no hay ajuste" de
// "lambda_medido diverge del teorico" (antes ambos casos colapsaban en
// lambda == 0.0, y la guarda de advertencia en print_overflow_horizon exigia
// lambda > 0.0 -- el centinela apagaba su propia advertencia, ver diagnostico
// del bloque 4). n_points/r_squared cuantifican la calidad del ajuste.
struct OverflowFitResult {
    double A = 0.0;
    double lambda = 0.0;
    bool valid = false;
    int n_points = 0;
    double r_squared = 0.0;
};

// Minimo de puntos finitos en la ventana asintotica para aceptar el ajuste:
// por debajo de esto el ajuste degenera sobre el transitorio inicial (ver
// diagnostico: con iters=10 el ajuste tomo 4 puntos de las iteraciones 6-9 y
// A vario 5 ordenes de magnitud segun iters, de 2.008 a 1.02e-5).
constexpr int kMinOverflowFitPoints = 30;

// Ajusta modelo log-lineal log2(||u^n||_inf) = log2(A) + n*log2(lambda)
// sobre la referencia FP64 en el regimen asintotico (60%-90% de las iters
// finitas). result.valid es false si la ventana tiene menos de
// kMinOverflowFitPoints puntos finitos o la matriz de minimos cuadrados es
// singular; en ese caso A/lambda/r_squared no deben usarse ni imprimirse.
// Se calcula una sola vez, fuera de regiones cronometradas.
static OverflowFitResult fit_overflow_model(const std::vector<double>& linf_per_iter) {
    OverflowFitResult result;
    if (linf_per_iter.size() < 3) {
        return result;  // datos insuficientes
    }

    const size_t n_total = linf_per_iter.size();
    const size_t idx_start = std::max(size_t(1), size_t(0.60 * n_total));
    const size_t idx_end = std::max(idx_start + 1, size_t(0.90 * n_total));

    // Ajuste por minimos cuadrados en espacio log: sum_i (log2(u_i) - c - k*i)^2 minimo
    // => c = log2(A), k = log2(lambda)
    double sum_i = 0.0, sum_log_u = 0.0, sum_i2 = 0.0, sum_i_log_u = 0.0, sum_log_u2 = 0.0;
    size_t count = 0;
    for (size_t i = idx_start; i < idx_end && i < n_total; ++i) {
        if (linf_per_iter[i] > 0.0) {
            const double log_u = std::log2(linf_per_iter[i]);
            const double n_iter = static_cast<double>(i + 1);
            sum_i += n_iter;
            sum_log_u += log_u;
            sum_i2 += n_iter * n_iter;
            sum_i_log_u += n_iter * log_u;
            sum_log_u2 += log_u * log_u;
            count++;
        }
    }

    result.n_points = static_cast<int>(count);
    if (count < static_cast<size_t>(kMinOverflowFitPoints)) {
        return result;  // ventana asintotica insuficiente: ajuste invalido
    }

    const double n_dbl = static_cast<double>(count);
    const double denom = n_dbl * sum_i2 - sum_i * sum_i;
    if (std::fabs(denom) < 1e-12) {
        return result;  // matriz singular
    }

    const double k = (n_dbl * sum_i_log_u - sum_i * sum_log_u) / denom;
    const double c = (sum_log_u - k * sum_i) / n_dbl;
    result.A = std::pow(2.0, c);
    result.lambda = std::pow(2.0, k);

    // R^2 de la regresion lineal simple: (n*Sxy - Sx*Sy)^2 / (Sxx'*Syy'), con
    // Sxy-Sx*Sy/n = k*denom/n (ya despejado arriba) y Syy' = n*sum_log_u2 - sum_log_u^2.
    const double ss_tot = n_dbl * sum_log_u2 - sum_log_u * sum_log_u;
    result.r_squared = (std::fabs(ss_tot) < 1e-12) ? 1.0 : (k * k * denom) / ss_tot;
    result.valid = true;
    return result;
}

// Amplificacion por iteracion del modo Nyquist (kx = ky = pi) bajo el operador
// activo. Es monomode_amplification evaluada en el vertice de la zona de
// Brillouin: cos(pi) = -1, asi que g(pi,pi) = c_center - 4*c_neigh.
//   estres   : -1 - 4*(0.25)      = -2.0     -> |g| = 2, el modo CRECE
//   difusivo : (1-4a) - 4a        = 1 - 8a   -> |g| <= 1 para todo a en (0, 1/4]
static double nyquist_amplification(const StencilOperator& op) {
    return center_coeff_d(op) - 4.0 * neighbor_coeff_d(op);
}

// El horizonte de overflow solo esta definido para operadores que AMPLIFICAN el
// modo Nyquist: es el canal por el que el ruido de redondeo crece hasta
// desbordar el formato, y toda la prediccion de
// compute_overflow_horizon_from_reference se apoya en ese crecimiento.
//
// Con |g(pi,pi)| <= 1 el modo decae y no hay nada que desbordar. Peor: el
// ajuste log-lineal no se queda sin datos, se engancha al modo dominante que
// SI sobreviva -- bajo la CI monomodo, el propio monomodo, cuyo lambda no tiene
// relacion con el que la formula supone. Medido a 512^2 con alpha=3/16 y p=5:
// lambda=0.9986 con R^2=1.000000, que es exactamente el g del monomodo
// (0.998583) y no el g(pi,pi) = -0.5 del operador. La prediccion resultante
// (horizonte de 16 iteraciones para FP16) es un numero bien formado y sin
// significado, y salia etiquetado 'ok' en CSV_HORIZON.
static bool horizon_applies(const StencilOperator& op) {
    return std::fabs(nyquist_amplification(op)) > 1.0;
}

// Prediccion de horizonte de overflow por formato, mas el ajuste que la
// calibro (ver OverflowFitResult). Si fit.valid es false no hay prediccion:
// pred_* quedan en 0.0 y el llamador (print_overflow_horizon) no debe
// imprimirlos. Si applicable es false tampoco la hay, por una razon distinta:
// el operador no amplifica el modo Nyquist (ver horizon_applies).
struct OverflowHorizonPrediction {
    OverflowFitResult fit;
    double pred_fp16 = 0.0;
    double pred_bf16 = 0.0;
    double pred_fp32 = 0.0;
    double pred_fp64 = 0.0;
    bool applicable = true;
    double lambda_nyquist = 0.0;
};

// Calcula horizonte de overflow predicho para cada formato, basandose en el
// ajuste de la referencia FP64. Modelo:
//   semilla_T = max(A, halfUlp_T * ||u0||_inf)
//   n*_T = log2(FMT_MAX_T) - log2(semilla_T)
// con A la semilla efectiva del ajuste log-lineal (fit_overflow_model) y
// halfUlp_T media unidad en el ultimo lugar del formato T. Usa A, NO la
// proyeccion Nyquist directa de u^0 (a_nyq_ic): esta ultima es grid-dependent
// (varia >1000x entre mallas con el mismo n* medido, ver diagnostico del
// bloque 3) y solo se imprime aparte, como diagnostico, en
// print_overflow_horizon.
static OverflowHorizonPrediction compute_overflow_horizon_from_reference(
        const std::vector<double>& linf_per_iter,
        double u0_linf,
        const StencilOperator& op) {
    OverflowHorizonPrediction result;
    result.lambda_nyquist = nyquist_amplification(op);
    result.applicable = horizon_applies(op);
    // El ajuste se calcula siempre: aunque el operador sea contractivo, lambda
    // y R^2 son un diagnostico util (delatan a que modo se engancho). Lo que no
    // se calcula en ese caso es la prediccion, que es la parte que careceria de
    // significado.
    result.fit = fit_overflow_model(linf_per_iter);
    if (!result.applicable || !result.fit.valid) {
        return result;  // sin horizonte definido, o sin ajuste confiable
    }

    const double A = result.fit.A;
    const double semilla_fp16 = std::max(A, kFp16SeedFloor * u0_linf);
    const double semilla_bf16 = std::max(A, kBf16SeedFloor * u0_linf);
    const double semilla_fp32 = std::max(A, kFp32SeedFloor * u0_linf);
    const double semilla_fp64 = std::max(A, kFp64SeedFloor * u0_linf);

    // Calculos en espacio logaritmico, nunca divide (previene overflow en FP64).
    // La formula asume implicitamente log2(lambda) = 1, es decir lambda = 2.0:
    // el g(pi,pi) del operador de estres, unico operador en alcance que
    // amplifica el modo Nyquist (el difusivo tiene |g(pi,pi)| = |1-8a| <= 1 y
    // horizon_applies ya lo filtro arriba). Si el ajuste diverge de ese 2.0,
    // print_overflow_horizon emite ADVERTENCIA pero la prediccion sigue usando
    // el valor teorico, nunca el lambda_medido fuera de rango.
    result.pred_fp16 = std::log2(kFp16Max) - std::log2(semilla_fp16);
    result.pred_bf16 = std::log2(kBf16Max) - std::log2(semilla_bf16);
    result.pred_fp32 = std::log2(kFp32Max) - std::log2(semilla_fp32);
    result.pred_fp64 = std::log2(kFp64Max) - std::log2(semilla_fp64);
    return result;
}

static std::string fmt_horizon_row(const char* label, double predicted, int measured_n) {
    // measured_n == INT_MAX (nunca diverguio, o la ruta ni se corrio): -1,
    // mismo centinela que n_star en el CSV.
    const int shown = (measured_n == INT_MAX) ? -1 : measured_n;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "  %-5s : %6.1f / %4d\n", label, predicted, shown);
    return buf;
}

// Predicho (calibrado desde la referencia FP64 mediante ajuste de mínimos
// cuadrados del modo log-lineal) vs medido (primera iteracion no finita de
// cada ruta, INT_MAX si nunca diverguio). Si el ajuste no es valido (ventana
// asintotica con menos de kMinOverflowFitPoints puntos finitos, ver
// fit_overflow_model) no imprime la tabla ni A/lambda: el centinela anterior
// (lambda == 0.0) colapsaba "sin ajuste" y "lambda diverge" en el mismo valor
// y por eso su propia guarda (lambda > 0.0) suprimia la advertencia que debia
// encender (ver diagnostico del bloque 4).
static void print_overflow_horizon(const OverflowHorizonPrediction& horizon,
                                   double a_nyq_ic,
                                   int n_fp16, int n_bf16, int n_gpu_fp32, int n_fp64,
                                   bool ran_gpu_fp64, int n_gpu_fp64) {
    std::cout << "=========== HORIZONTE DE OVERFLOW (Fase 3) ===========\n";
    const OverflowFitResult& fit = horizon.fit;
    if (!horizon.applicable) {
        std::cout << "NO APLICA: el operador contrae el modo Nyquist, g(pi,pi) = "
                  << fmt_sci(horizon.lambda_nyquist) << " (|g| <= 1).\n"
                     "El horizonte de overflow solo esta definido para operadores que lo\n"
                     "AMPLIFICAN: sin crecimiento no hay formato que desbordar. No se predice\n"
                     "horizonte; las filas CSV_HORIZON salen con NaN y estado\n"
                     "'contractive_operator'.\n";
        if (fit.valid) {
            // Se imprime como diagnostico, con la etiqueta que impide leerlo
            // como el modo Nyquist: bajo un operador contractivo el ajuste se
            // engancha al modo dominante que sobreviva, no a Nyquist.
            std::cout << "  Ajuste asintotico (modo dominante, NO Nyquist): n=" << fit.n_points
                      << " puntos, R^2=" << fmt_sci(fit.r_squared)
                      << ", lambda=" << fmt_sci(fit.lambda) << "\n";
        }
        std::cout << "=======================================================\n\n";
        return;
    }
    if (!fit.valid) {
        std::cout << "AJUSTE NO DISPONIBLE (se requieren >=" << kMinOverflowFitPoints
                  << " iteraciones finitas de referencia FP64 en la ventana asintotica"
                     " 60%-90%; disponibles: " << fit.n_points << "). "
                     "Horizonte predicho no calculado.\n";
        std::cout << "=======================================================\n\n";
        return;
    }

    std::cout << "Horizonte de overflow (predicho / medido)\n";
    std::cout << fmt_horizon_row("FP16", horizon.pred_fp16, n_fp16);
    std::cout << fmt_horizon_row("BF16", horizon.pred_bf16, n_bf16);
    std::cout << fmt_horizon_row("FP32", horizon.pred_fp32, n_gpu_fp32);
    std::cout << fmt_horizon_row("FP64", horizon.pred_fp64, n_fp64);
    // La fila FP64 de arriba es la referencia FP64 de CPU; esta es la ruta
    // FP64 de GPU. Comparten prediccion (mismo formato, misma semilla) pero
    // NO el horizonte medido: son dos implementaciones distintas del mismo
    // operador y sus n* pueden separarse (p.ej. por contraccion FMA en el
    // kernel), que es justamente lo que esta fila permite ver.
    if (ran_gpu_fp64) {
        std::cout << fmt_horizon_row("FP64g", horizon.pred_fp64, n_gpu_fp64);
    }
    std::cout << "  Semilla efectiva A (fit FP64)       : " << fmt_sci(fit.A) << "\n";
    std::cout << "  Ajuste asintotico: n=" << fit.n_points << " puntos, R^2="
              << std::fixed << std::setprecision(6) << fit.r_squared
              << ", lambda=" << std::setprecision(4) << fit.lambda << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Contenido Nyquist exacto de u^0 (~0 para CI suave) : " << fmt_sci(a_nyq_ic) << "\n";
    std::cout << "  Piso de siembra por formato (calibrado, 0.5 x unidad de redondeo):\n";
    std::cout << "    FP16                             : " << std::scientific << kFp16SeedFloor << "\n";
    std::cout << "    BF16                             : " << std::scientific << kBf16SeedFloor << "\n";
    std::cout << "    FP32                             : " << std::scientific << kFp32SeedFloor << "\n";
    std::cout << "    FP64                             : " << std::scientific << kFp64SeedFloor << "\n";

    // El valor teorico se toma del operador activo y no de un 2.0 fijo. Con el
    // 2.0 cableado, el operador difusivo disparaba esta advertencia diciendo
    // "revisar la formula del stencil" cuando la formula era correcta y lo
    // erroneo era la referencia contra la que se comparaba.
    const double lambda_teorico = std::fabs(horizon.lambda_nyquist);
    if (std::fabs(fit.lambda - lambda_teorico) / lambda_teorico > 0.05) {
        std::cout << "  ADVERTENCIA: lambda_medido diverge >5% del valor teorico "
                  << fmt_sci(lambda_teorico) << "\n"
                  << "  Revisar condicion inicial o formula del stencil.\n";
    }
    std::cout << "=======================================================\n\n";
}

static std::string csv_measured_horizon_field(int measured_n) {
    return std::to_string((measured_n == INT_MAX) ? -1 : measured_n);
}

static void emit_csv_horizon_row(const char* format,
                                 int nx,
                                 int ny,
                                 int iters,
                                 bool kahan,
                                 double predicted,
                                 int measured_n,
                                 const OverflowFitResult& fit,
                                 bool applicable,
                                 double a_nyq_ic,
                                 double seed_floor) {
    const bool fit_ok = fit.valid;
    // h_predicho solo se emite si ADEMAS el horizonte esta definido para el
    // operador activo. lambda/R^2/A si se emiten aunque no lo este: son el
    // ajuste medido, no una prediccion derivada de el. El numero de columnas no
    // cambia en ningun caso -- extract_csv.py cuenta campos.
    const bool pred_ok = fit_ok && applicable;
    const char* estado = !applicable ? "contractive_operator"
                                     : (fit_ok ? "ok" : "insufficient_points");
    std::cout << "CSV_HORIZON," << format << "," << nx << "," << ny << "," << iters << ","
              << kahan_label(kahan) << ","
              << (pred_ok ? fmt_csv_num(predicted) : "NaN") << ","
              << csv_measured_horizon_field(measured_n) << ","
              << (fit_ok ? fmt_csv_num(fit.lambda) : "NaN") << ","
              << (fit_ok ? fmt_csv_num(fit.r_squared) : "NaN") << ","
              << fit.n_points << ","
              << (fit_ok ? fmt_csv_num(fit.A) : "NaN") << ","
              << fmt_csv_num(a_nyq_ic) << ","
              << fmt_csv_num(seed_floor) << ","
              << estado << "\n";
}

static void emit_csv_horizon_rows(const OverflowHorizonPrediction& horizon,
                                  double a_nyq_ic,
                                  int nx,
                                  int ny,
                                  int iters,
                                  bool kahan,
                                  CompMode comp_mode,
                                  int n_fp16,
                                  int n_bf16,
                                  int n_gpu_fp32,
                                  int n_fp64,
                                  bool ran_gpu_fp64,
                                  int n_gpu_fp64) {
    const OverflowFitResult& fit = horizon.fit;
    // Mismo criterio de sufijo que las rutas (ver wmma_route_label): h_medido
    // de estas dos filas SI depende de la politica de compensacion (es el
    // first_nonfinite de la ruta WMMA correspondiente), asi que la variante
    // espacial no puede compartir la etiqueta de formato con --kahan off|on.
    // Las filas FP32/FP64 no dependen de la compensacion y conservan su
    // etiqueta.
    emit_csv_horizon_row(wmma_route_label(comp_mode, "FP16", "FP16_SP"),
                         nx, ny, iters, kahan, horizon.pred_fp16, n_fp16,
                         fit, horizon.applicable, a_nyq_ic, kFp16SeedFloor);
    emit_csv_horizon_row(wmma_route_label(comp_mode, "BF16", "BF16_SP"),
                         nx, ny, iters, kahan, horizon.pred_bf16, n_bf16,
                         fit, horizon.applicable, a_nyq_ic, kBf16SeedFloor);
    emit_csv_horizon_row("FP32", nx, ny, iters, kahan, horizon.pred_fp32, n_gpu_fp32,
                         fit, horizon.applicable, a_nyq_ic, kFp32SeedFloor);
    emit_csv_horizon_row("FP64", nx, ny, iters, kahan, horizon.pred_fp64, n_fp64,
                         fit, horizon.applicable, a_nyq_ic, kFp64SeedFloor);
    // Fila de la RUTA GPU_FP64, distinta de la fila "FP64" de arriba (que es la
    // referencia FP64 de CPU). Se etiqueta con el nombre de ruta completo y no
    // con un nombre de formato porque es lo que la desambigua de esa otra fila
    // al unir CSVs; h_predicho es el mismo (mismo formato) y h_medido es el
    // first_nonfinite propio de la ruta GPU. Solo se emite si la ruta corrio:
    // una fila con h_medido=-1 seria indistinguible de "corrio y nunca
    // diverguio".
    if (ran_gpu_fp64) {
        emit_csv_horizon_row("GPU_FP64", nx, ny, iters, kahan, horizon.pred_fp64, n_gpu_fp64,
                             fit, horizon.applicable, a_nyq_ic, kFp64SeedFloor);
    }
}

// Factor de amplificacion por iteracion del monomodo bajo el operador activo:
//   g = c_center + 2*c_neigh*(cos kx + cos ky),  k = 2*pi*p/(n-1)
// (aplicar el stencil a un modo propio devuelve u+d+l+r = 2(cos kx + cos ky)
// veces el centro). Para el operador de estres eso da la forma documentada
// 0.5*(cos kx + cos ky) - 1, con g(pi,pi) = -2; para el difusivo,
// 1 - 4*alpha*[sin^2(kx/2) + sin^2(ky/2)], con g(pi,pi) = -0.5 al alpha
// por defecto.
//
// La CI monomodo es exactamente ese modo propio del Laplaciano discreto, asi que
// g^n es la razon de normas EXACTA en aritmetica infinita. Se imprime para que
// quien lanza la campana vea de inmediato si el caso quedo degenerado -- g^iters
// ~ 1 (campo cuasi invariante, no hay dinamica que medir) o g^iters ~ 0 (campo
// colapsado, el denominador de rel_l2 se hunde y el error relativo se dispara
// sin que eso signifique nada sobre la precision de la ruta).
static double monomode_amplification(const Options& opt, const StencilOperator& op) {
    constexpr double kPi = 3.14159265358979323846;
    const double kx = 2.0 * kPi * static_cast<double>(opt.ci_p) / static_cast<double>(opt.nx - 1);
    const double ky = 2.0 * kPi * static_cast<double>(opt.ci_p) / static_cast<double>(opt.ny - 1);
    return center_coeff_d(op)
         + 2.0 * neighbor_coeff_d(op) * (std::cos(kx) + std::cos(ky));
}

static void print_configuration(const Options& opt, const StencilOperator& op) {
    std::cout << "================== CONFIGURACION ==================\n";
    std::cout << "Stencil                    : 2D 5-puntos\n";
    std::cout << "Dimensiones (nx, ny)       : " << opt.nx << ", " << opt.ny << "\n";
    std::cout << "Puntos interiores          : "
              << static_cast<long long>(opt.nx - 2) * static_cast<long long>(opt.ny - 2)
              << "\n";
    std::cout << "Iteraciones                : " << opt.iters << "\n";
    std::cout << "Tile Tensor Core           : 16x16 con WMMA\n";
    std::cout << "Acumulacion TC             : FP32\n";
    // Esta linea es la que tools/extract_csv.py usa (KAHAN_RE) para poblar la
    // columna kahan cuando el log no trae la cabecera "Corrida:" del sbatch:
    // su valor debe seguir siendo exactamente off|on. La politica espacial se
    // reporta en una linea APARTE, no reemplazando este token.
    std::cout << "Kahan (residuo almacen.)   : " << (opt.kahan ? "on" : "off") << "\n";
    std::cout << "Compensacion espacial      : " << (opt.spatial_comp ? "on" : "off") << "\n";
    if (opt.spatial_comp) {
        std::cout << "  (rutas WMMA reportadas como WMMA_FP16_SP / WMMA_BF16_SP)\n";
    }
    // Solo describe a las rutas WMMA: las demas no tienen ruta de grafo (ver
    // ExecutionMode). La columna execution_mode de CSV_SUMMARY sigue la misma
    // regla, fila por fila.
    std::cout << "Modo de ejecucion (WMMA)   : " << execution_mode_label(opt.execution_mode)
              << "\n";
    if (opt.execution_mode == ExecutionMode::Graph) {
        std::cout << "  Iteraciones por grafo    : " << opt.graph_block << "\n";
        if (opt.iters < opt.graph_block) {
            std::cout << "  AVISO: iters (" << opt.iters << ") < graph-block ("
                      << opt.graph_block << "): no se forma ningun bloque y la corrida\n"
                      << "         degenera a lanzamientos normales.\n";
        }
    }
    std::cout << "Ruta GPU FP64 (referencia) : " << (opt.fp64_gpu ? "on" : "off") << "\n";
    std::cout << "Ruta CPU FP64 (cronometro) : " << (opt.cpu_fp64 ? "on" : "off") << "\n";
    std::cout << "Operador                   : " << op_mode_label(opt.op_mode) << "\n";
    std::cout << "Coef vecino / centro       : " << fmt_sci(op.neighbor) << " / "
              << fmt_sci(op.center) << "\n";
    std::cout << "FLOPs por celda interior   : " << op.flops_per_cell << "\n";
    if (opt.op_mode == OpMode::Diffusive) {
        std::cout << "Alpha                      : " << fmt_sci(opt.alpha) << "\n";
    }
    std::cout << "Checkpoints                : "
              << (!opt.checkpoint_iters.empty()
                      ? ("lista [" + join_int_list(opt.checkpoint_iters) + "]")
                      : (opt.checkpoint_every > 0
                             ? ("cada " + std::to_string(opt.checkpoint_every))
                             : std::string("off")))
              << "\n";
    if (!opt.archive_iters.empty()) {
        std::cout << "Archivado de campos        : [" << join_int_list(opt.archive_iters)
                  << "] en " << opt.archive_dir << "\n";
    }
    std::cout << "Condicion inicial          : " << ci_mode_label(opt.ci_mode) << "\n";
    if (opt.ci_mode == CiMode::Monomode) {
        // Diagnostico de diseno, no una medicion: g^iters es la razon de normas
        // exacta del modo propio en aritmetica infinita. Si sale ~1 el caso es
        // cuasi invariante y si sale ~0 el campo colapso; en ambos extremos la
        // corrida no sirve para el eje de error del Pareto (ver
        // monomode_amplification).
        const double g = monomode_amplification(opt, op);
        std::cout << "  p / amplitud             : " << opt.ci_p << " / "
                  << fmt_sci(opt.ci_amplitude) << "\n";
        std::cout << "  g (por iteracion)        : " << fmt_sci(g) << "\n";
        std::cout << "  |g|^iters (razon normas) : "
                  << fmt_sci(std::pow(std::fabs(g), static_cast<double>(opt.iters))) << "\n";
    }
    std::cout << "===================================================\n\n";
}

static const char* tc_mode_to_string(TensorCoreMode mode) {
    switch (mode) {
        case TensorCoreMode::FP16: return "fp16";
        case TensorCoreMode::BF16: return "bf16";
        case TensorCoreMode::Both: return "both";
    }
    return "both";
}

// Metricas y regex deben coincidir con NCU_QUICK_METRICS / NCU_KERNEL_REGEX_WMMA
// en tools/common_ncu.sh y run_stencil_tc.sbatch (antes este hint mostraba solo
// 2 metricas mientras la corrida real usa las 12 de NCU_QUICK_METRICS).
// --launch-skip se deriva de kWarmupIters (no un literal) para no desincronizarse.
static void print_nsight_hint(const char* exe_name, int nx, int ny, int iters,
                              TensorCoreMode tc_mode, bool kahan, bool spatial_comp) {
    std::cout << "Validacion Nsight Compute (coincide con NCU_QUICK_METRICS):\n";
    std::cout << "  ncu --kernel-name regex:.*stencil2d_wmma_kernel.* \\\n";
    std::cout << "      --launch-skip " << kWarmupIters << " --launch-count 1 \\\n";
    std::cout << "      --metrics sm__inst_executed_pipe_tensor_op_hmma.sum,"
                 "sm__inst_executed_pipe_tensor_op_hmma_type_hfma2.sum,"
                 "sm__ops_path_tensor_src_fp16_dst_fp32.sum,"
                 "sm__ops_path_tensor_src_bf16_dst_fp32.sum,"
                 "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
                 "sm__warps_active.avg.pct_of_peak_sustained_active,"
                 "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
                 "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,"
                 "dram__bytes_read.sum,dram__bytes_write.sum,"
                 "l1tex__t_sector_hit_rate.pct,"
                 "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,"
                 "launch__registers_per_thread,"
                 "sm__sass_thread_inst_executed_op_fadd_pred_on.sum,"
                 "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,"
                 "launch__occupancy_limit_registers \\\n";
    std::cout << "      " << exe_name << " --nx " << nx << " --ny " << ny
              << " --iters " << iters << " --tc " << tc_mode_to_string(tc_mode)
              << " --kahan " << (kahan ? "on" : "off")
              << (spatial_comp ? " --spatial-comp on" : "") << " --profile-only\n";
}

// Modo --profile-only: los ~1723 s de pared por llamada a ncu eran, sobre
// todo, la aplicacion recalculando la referencia CPU FP32 (~742s) y la FP64
// encadenada (~900s) ANTES de llegar al kernel bajo perfil (que en si tarda
// segundos con --launch-skip/--launch-count). Aqui se omiten ambas
// referencias y todo el calculo/impresion de metricas de error, conservando
// condicion inicial, warm-up (kWarmupIters) y el bucle de iters con el mismo
// ping-pong que benchmark_gpu_fp32_stencil / benchmark_gpu_tensor_core_stencil
// usan en la corrida completa (sin --profile-only). GPU FP32 clasico se
// mantiene (no es un "reference": es la ruta que perfila stencil2d_fp32_kernel).
static void run_profile_only(const Options& opt) {
    std::cout << "*** MODO --profile-only: sin referencia CPU FP32 ni FP64 encadenada,"
                 " sin metricas de error. GPU FP32 clasico + TC "
              << tc_mode_to_string(opt.tc_mode) << " ***\n\n";

    const StencilOperator op = operator_of(opt);
    const size_t count = static_cast<size_t>(opt.nx) * static_cast<size_t>(opt.ny);
    std::vector<float> input(count);
    initialize_input_grid(input, opt);

    const std::vector<std::vector<double>> no_checkpoints;
    const CheckpointContext ckpt{0, no_checkpoints};

    std::vector<float> y_gpu(count, 0.0f);
    int onset_gpu_fp32 = -1;
    int first_nf_gpu_fp32 = INT_MAX;
    double t_checkpoint_ms_unused_fp32 = 0.0;
    EnergyMeasurement e_unused_fp32;
    benchmark_gpu_fp32_stencil(input, y_gpu, opt.nx, opt.ny, opt.iters, op,
                               ckpt, "GPU_FP32", onset_gpu_fp32, first_nf_gpu_fp32,
                               t_checkpoint_ms_unused_fp32, e_unused_fp32);

    double t_wmma_ms_unused = 0.0, t_conv_ms_unused = 0.0, t_checkpoint_ms_unused = 0.0;
    int storage_rel_eval_iter_unused = 0;
    if (opt.tc_mode == TensorCoreMode::FP16) {
        std::vector<float> y_tc_fp16(count, 0.0f);
        std::vector<__half> y_tc_fp16_reduced;
        std::vector<float> y_tc_fp16_last_finite_unused;
        std::vector<__half> y_tc_fp16_reduced_last_finite_unused;
        int onset_fp16 = -1;
        int first_nf_fp16 = INT_MAX;
        EnergyMeasurement e_unused_fp16;
        std::vector<float> y_tc_fp16_comp_unused;
        benchmark_gpu_tensor_core_stencil<__half>(input, y_tc_fp16, y_tc_fp16_reduced, opt.nx, opt.ny,
                                                  opt.iters, op, comp_mode_of(opt),
                                                  opt.execution_mode, opt.graph_block, ckpt,
                                                  fp16_route_label(comp_mode_of(opt)), onset_fp16, first_nf_fp16,
                                                  t_wmma_ms_unused, t_conv_ms_unused, storage_rel_eval_iter_unused,
                                                  t_checkpoint_ms_unused, y_tc_fp16_last_finite_unused,
                                                  y_tc_fp16_reduced_last_finite_unused, e_unused_fp16,
                                                  y_tc_fp16_comp_unused);
    } else {
        std::vector<float> y_tc_bf16(count, 0.0f);
        std::vector<__nv_bfloat16> y_tc_bf16_reduced;
        std::vector<float> y_tc_bf16_last_finite_unused;
        std::vector<__nv_bfloat16> y_tc_bf16_reduced_last_finite_unused;
        int onset_bf16 = -1;
        int first_nf_bf16 = INT_MAX;
        EnergyMeasurement e_unused_bf16;
        std::vector<float> y_tc_bf16_comp_unused;
        benchmark_gpu_tensor_core_stencil<__nv_bfloat16>(input, y_tc_bf16, y_tc_bf16_reduced, opt.nx, opt.ny,
                                                         opt.iters, op, comp_mode_of(opt),
                                                         opt.execution_mode, opt.graph_block, ckpt,
                                                         bf16_route_label(comp_mode_of(opt)), onset_bf16, first_nf_bf16,
                                                         t_wmma_ms_unused, t_conv_ms_unused, storage_rel_eval_iter_unused,
                                                         t_checkpoint_ms_unused, y_tc_bf16_last_finite_unused,
                                                         y_tc_bf16_reduced_last_finite_unused, e_unused_bf16,
                                                         y_tc_bf16_comp_unused);
    }
}

static void run_benchmark(const Options& opt, const char* exe_name) {
    // Operador activo, derivado UNA sola vez para toda la corrida: de aqui baja
    // como argumento a cada ruta, a cada kernel y a cada metrica de FLOPs.
    const StencilOperator op = operator_of(opt);
    print_configuration(opt, op);

    if (!device_supports_fp16_tensor_cores()) {
        std::cerr << "La GPU activa no reporta soporte minimo para Tensor Cores FP16 (SM >= 70).\n";
        std::exit(EXIT_FAILURE);
    }
    if ((opt.tc_mode == TensorCoreMode::BF16 || opt.tc_mode == TensorCoreMode::Both) &&
        !device_supports_bf16_tensor_cores()) {
        std::cerr << "BF16 Tensor Core requiere arquitectura Ampere o superior (SM >= 80).\n";
        std::exit(EXIT_FAILURE);
    }

    if (opt.profile_only) {
        run_profile_only(opt);
        return;
    }

    // NCU_PROFILING lo exporta run_stencil_tc.sbatch (via common_ncu.sh)
    // unicamente al invocar ncu: los tiempos bajo perfilado quedan inflados
    // (ver contexto: 22.96 ms FP16 bajo ncu vs 15.90 ms limpio) y no deben
    // confundirse con una corrida normal.
    const bool under_ncu = std::getenv("NCU_PROFILING") != nullptr;
    if (under_ncu) {
        std::cout << "\n*** CORRIDA BAJO NSIGHT COMPUTE — TIEMPOS NO VALIDOS ***\n";
    }

    std::ofstream csv;
    const bool csv_enabled = !opt.csv_path.empty();
    if (csv_enabled) {
        csv = open_csv(opt.csv_path);
    }

    // Fijado aqui (y no justo antes de "RESULTADOS...", como antes) para que
    // las filas CSV_DRIFT -que pueden emitirse desde dentro de
    // benchmark_gpu_fp32_stencil, antes de llegar a esa seccion- usen el
    // mismo formato numerico que el resto de la salida.
    std::cout << std::fixed << std::setprecision(6);

    const size_t count = static_cast<size_t>(opt.nx) * static_cast<size_t>(opt.ny);
    std::vector<float> input(count);
    std::vector<float> y_cpu(count, 0.0f);
    std::vector<float> y_gpu(count, 0.0f);
    std::vector<float> y_tc_fp16(count, 0.0f);
    std::vector<float> y_tc_bf16(count, 0.0f);
    std::vector<__half> y_tc_fp16_reduced;
    std::vector<__nv_bfloat16> y_tc_bf16_reduced;

    initialize_input_grid(input, opt);

    // Huella de la condicion inicial. Es lo que permite afirmar, meses despues,
    // que dos corridas partieron del MISMO campo: el error de este stencil es
    // determinista, asi que si dos ejecuciones con la misma CI y el mismo
    // binario dan rel_l2 distinto, el problema esta en otro sitio.
    const std::string ci_sha256 = sha256_of_vector(input);
    std::cout << "CSV_CI_SHA256," << ci_sha256 << "\n";

    // Spill de referencia y contexto de archivado (--checkpoint-iters /
    // --archive-iters). Ambos quedan inertes si los flags no se pasaron, de modo
    // que una corrida energetica no abre fichero alguno.
    ReferenceSpill reference_spill;
    const bool use_iter_list = !opt.checkpoint_iters.empty();
    if (use_iter_list) {
        // El spill vive junto al archivado (mismo scratch dimensionado por el
        // sbatch) y se borra solo al cerrarse; el pid lo hace unico si dos jobs
        // comparten directorio.
        const char* job_env = std::getenv("SLURM_JOB_ID");
        const std::string spill_path = opt.archive_dir + "/reference_spill_" +
                                       (job_env != nullptr ? job_env : "manual") + ".bin";
        if (!reference_spill.open(spill_path, count)) std::exit(EXIT_FAILURE);
        std::cout << "Spill de referencia   : " << spill_path << " ("
                  << opt.checkpoint_iters.size() << " checkpoints, "
                  << (static_cast<double>(opt.checkpoint_iters.size()) *
                      static_cast<double>(count * sizeof(double)) / (1024.0 * 1024.0 * 1024.0))
                  << " GiB de disco)\n";
    }
    ArchiveContext archive_ctx;
    const bool use_archive = !opt.archive_iters.empty();
    if (use_archive) {
        archive_ctx.iters = opt.archive_iters;
        archive_ctx.dir = opt.archive_dir;
    }

    // Prediccion del horizonte de overflow (ver comentario de la funcion):
    // se mide la condicion inicial ya generada, no se la modifica; corre
    // antes de cualquier CudaEventTimer/std::chrono de las rutas medidas.
    const double a_nyq = compute_nyquist_component(input, opt.nx, opt.ny);

    // Norma infinito de la condicion inicial (usada en calibracion del modelo
    // de overflow por formato: semilla_T = max(a_nyq, u_T * ||u0||_inf)).
    double u0_linf = 0.0;
    for (const auto& x : input) {
        u0_linf = std::max(u0_linf, std::fabs(static_cast<double>(x)));
    }

    // Referencia FP64 (ground truth): opt.iters aplicaciones encadenadas del
    // stencil en double sobre una copia en double del mismo input, mismo
    // numero de iteraciones que las rutas comparadas (ver comentario en
    // compute_cpu_stencil_fp64).
    std::vector<double> input_fp64(count);
    std::vector<double> y_ref(count, 0.0);
    for (size_t i = 0; i < count; ++i) {
        input_fp64[i] = static_cast<double>(input[i]);
    }
    // Con --checkpoint-every K > 0, fp64_checkpoints recibe un snapshot por
    // cada iteracion multiplo de K (ver compute_cpu_stencil_fp64); su tamano
    // final ya es el numero de checkpoints "validos" (referencia finita).
    std::vector<std::vector<double>> fp64_checkpoints;
    std::vector<double> linf_per_iter;  // ||u^n||_inf para cada iteracion
    int first_nf_fp64_ref = INT_MAX;
    compute_cpu_stencil_fp64(input_fp64, y_ref, opt.nx, opt.ny, opt.iters, op,
                             opt.checkpoint_every, fp64_checkpoints,
                             use_iter_list ? &opt.checkpoint_iters : nullptr,
                             use_iter_list ? &reference_spill : nullptr,
                             use_archive ? &archive_ctx : nullptr,
                             linf_per_iter, first_nf_fp64_ref);

    if (opt.checkpoint_every > 0) {
        const int got = static_cast<int>(fp64_checkpoints.size());
        const int expected = opt.iters / opt.checkpoint_every;
        if (got < expected) {
            const int divergence_iter = (got + 1) * opt.checkpoint_every;
            std::cout << "Referencia FP64 no finita desde iter " << divergence_iter
                      << "; CSV_DRIFT marcara NONFINITE desde ese checkpoint "
                      << "para todas las rutas (" << got << " de " << expected
                      << " checkpoints validos).\n\n";
        }
    }
    if (use_iter_list) {
        const size_t got = reference_spill.stored();
        const size_t expected = opt.checkpoint_iters.size();
        if (got < expected) {
            std::cout << "Referencia FP64 no finita antes de completar la lista de checkpoints: "
                      << got << " de " << expected << " guardados. Las iteraciones restantes"
                         " saldran como NONFINITE en CSV_CKPT.\n";
        }
        std::cout << "Spill de referencia   : " << got << " campos, "
                  << reference_spill.gib_on_disk() << " GiB en disco\n\n";
    }
    const CheckpointContext ckpt{opt.checkpoint_every, fp64_checkpoints,
                                 use_iter_list ? &opt.checkpoint_iters : nullptr,
                                 use_iter_list ? &reference_spill : nullptr,
                                 use_archive ? &archive_ctx : nullptr};

    int first_nf_cpu = INT_MAX;
    EnergyMeasurement e_cpu;
    const Metrics cpu = benchmark_cpu_stencil(input, y_cpu, opt.nx, opt.ny, opt.iters, op,
                                              first_nf_cpu, e_cpu);

    // Ruta CPU_FP64: se MIDE aqui, junto a la otra ruta de CPU y antes de
    // cualquier ruta GPU, para que ambas referencias de CPU compartan el mismo
    // estado termico y de cache de la maquina, y para no meter una corrida
    // larga de CPU en medio de las rutas GPU. Se REPORTA justo despues de
    // CPU_FP32. El error contra FP64 de CPU ya lo tenian todas las rutas via
    // y_ref; lo que aporta esta ruta es su tiempo y su energia.
    bool ran_cpu_fp64 = false;
    int first_nf_cpu_fp64 = INT_MAX;
    std::vector<double> y_cpu_fp64;
    EnergyMeasurement e_cpu_fp64;
    Metrics cpu_fp64;
    if (opt.cpu_fp64) {
        ran_cpu_fp64 = true;
        cpu_fp64 = benchmark_cpu_fp64_stencil(input_fp64, y_cpu_fp64, opt.nx, opt.ny,
                                              opt.iters, op, first_nf_cpu_fp64, e_cpu_fp64);
    }

    int onset_gpu_fp32 = -1;
    int first_nf_gpu_fp32 = INT_MAX;
    double t_checkpoint_ms_gpu_fp32 = 0.0;
    EnergyMeasurement e_gpu_fp32;
    const Metrics gpu = benchmark_gpu_fp32_stencil(input, y_gpu, opt.nx, opt.ny, opt.iters, op,
                                                    ckpt, "GPU_FP32", onset_gpu_fp32,
                                                    first_nf_gpu_fp32, t_checkpoint_ms_gpu_fp32,
                                                    e_gpu_fp32);
    // Metrica primaria: contra el ground truth FP64 (objetivo especifico #3);
    // secundaria: contra la CPU FP32 (trazabilidad con corridas previas).
    const ErrorMetrics cpu_err        = compare_fp64_ref_vs_fp32(y_ref, y_cpu);
    const ErrorMetrics gpu_err        = compare_fp64_ref_vs_fp32(y_ref, y_gpu);
    const ErrorMetrics gpu_vs_cpu_err = compare_float_vectors(y_cpu, y_gpu);

    std::cout << "=========== RESULTADOS STENCIL 2D FASE 3 ===========\n";
    print_first_nonfinite("Primera iteracion no finita (ref FP64)     : ", first_nf_fp64_ref, opt.iters);
    print_fp64_reference_norms(y_ref, first_nf_fp64_ref);
    std::cout << "\n";
    std::cout << "CPU FP32 serial - tiempo/iter (media) : " << cpu.ms << " ms\n";
    std::cout << "CPU FP32 serial - tiempo total        : " << cpu.ms * opt.iters << " ms\n";
    std::cout << "CPU FP32 serial - rend.    : " << cpu.gflops << " GFLOP/s ("
              << cpu.tflops << " TFLOP/s efectivos)\n";
    print_error_metrics("Error max abs vs FP64      : ", "Error relativo L2 vs FP64  : ",
                        "Error rel Linf vs FP64     : ", cpu_err, first_nf_cpu);
    print_first_nonfinite("Primera iteracion no finita : ", first_nf_cpu, opt.iters);
    print_energy_metrics(e_cpu);
    std::cout << "\n";
    // speedup_fp32 es una razon GPU-vs-GPU: en la fila de la ruta de CPU sale
    // NaN, no un numero calculado contra otro dispositivo.
    emit_csv_summary_row(opt, "CPU_FP32", opt.nx, opt.ny, opt.iters, opt.kahan,
                         cpu.ms, cpu.gflops, fmt_csv_num(1.0), "NaN",
                         "NaN", "NaN", "NaN", cpu_err, first_nf_cpu,
                         "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", e_cpu,
                         kReferenceRoleNone);
    emit_csv_energy_row("CPU_FP32", opt.nx, opt.ny, opt.iters, opt.kahan, e_cpu,
                        stencil_flops(opt.nx, opt.ny, op.flops_per_cell) * static_cast<double>(opt.iters),
                        /*gpu_route=*/false);
    if (csv_enabled) {
        write_csv_row(csv, opt, under_ncu ? "NCU_cpu_fp32" : "cpu_fp32", opt.kahan, opt.nx, opt.ny,
                      opt.iters, cpu.ms, cpu.gflops, cpu_err, first_nf_cpu, "NA",
                      kReferenceRoleNone,
                      energy_per_cell_csv_field(under_ncu, e_cpu, opt.nx, opt.ny, opt.iters));
    }

    if (ran_cpu_fp64) {
        // rel_l2 sale 0 por CONSTRUCCION: esta ruta ejecuta la misma aritmetica,
        // en el mismo orden, que produjo y_ref. No es una metrica de calidad
        // sino una verificacion cruzada -- si alguna vez sale distinta de 0, la
        // pasada cronometrada se separo del ground truth y hay un bug. Lo que
        // esta fila aporta de verdad son t_iter_ms, gflops y la energia RAPL.
        const ErrorMetrics cpu_fp64_err = compare_fp64_ref_vs_fp64(y_ref, y_cpu_fp64);
        std::cout << "CPU FP64 serial - tiempo/iter (media) : " << cpu_fp64.ms << " ms\n";
        std::cout << "CPU FP64 serial - tiempo total        : " << cpu_fp64.ms * opt.iters << " ms\n";
        std::cout << "CPU FP64 serial - rend.    : " << cpu_fp64.gflops << " GFLOP/s ("
                  << cpu_fp64.tflops << " TFLOP/s efectivos)\n";
        std::cout << "Sobrecosto FP64 vs FP32 en CPU : " << cpu_fp64.ms / cpu.ms << "x\n";
        print_error_metrics("Error max abs vs FP64      : ", "Error relativo L2 vs FP64  : ",
                            "Error rel Linf vs FP64     : ", cpu_fp64_err, first_nf_cpu_fp64);
        print_first_nonfinite("Primera iteracion no finita : ", first_nf_cpu_fp64, opt.iters);
        print_energy_metrics(e_cpu_fp64);
        std::cout << "\n";
        // speedup_fp32 es una razon GPU-vs-GPU: NaN aqui, por la misma
        // disciplina que en la fila CPU_FP32. speedup_cpu si aplica y muestra
        // el sobrecosto de double frente a CPU_FP32, ambas en el mismo device.
        emit_csv_summary_row(opt, "CPU_FP64", opt.nx, opt.ny, opt.iters, opt.kahan,
                             cpu_fp64.ms, cpu_fp64.gflops,
                             fmt_csv_num(cpu.ms / cpu_fp64.ms), "NaN",
                             "NaN", "NaN", "NaN", cpu_fp64_err, first_nf_cpu_fp64,
                             "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", e_cpu_fp64,
                             kReferenceRoleCpuFp64);
        emit_csv_energy_row("CPU_FP64", opt.nx, opt.ny, opt.iters, opt.kahan, e_cpu_fp64,
                            stencil_flops(opt.nx, opt.ny, op.flops_per_cell) * static_cast<double>(opt.iters),
                            /*gpu_route=*/false);
        if (csv_enabled) {
            write_csv_row(csv, opt, under_ncu ? "NCU_cpu_fp64" : "cpu_fp64", opt.kahan,
                          opt.nx, opt.ny, opt.iters, cpu_fp64.ms, cpu_fp64.gflops,
                          cpu_fp64_err, first_nf_cpu_fp64, "NA", kReferenceRoleCpuFp64,
                          energy_per_cell_csv_field(under_ncu, e_cpu_fp64, opt.nx, opt.ny,
                                                    opt.iters));
        }
    }

    print_reference_comparison("GPU CUDA FP32 clasico", gpu, cpu.ms, gpu_err, gpu_vs_cpu_err,
                               first_nf_gpu_fp32, opt.iters, t_checkpoint_ms_gpu_fp32);
    emit_csv_summary_row(opt, "GPU_FP32", opt.nx, opt.ny, opt.iters, opt.kahan,
                         gpu.ms, gpu.gflops, fmt_csv_num(cpu.ms / gpu.ms), fmt_csv_num(1.0),
                         "NaN", "NaN", fmt_csv_num(t_checkpoint_ms_gpu_fp32),
                         gpu_err, first_nf_gpu_fp32,
                         "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", e_gpu_fp32,
                         kReferenceRoleNone);
    print_energy_metrics(e_gpu_fp32);
    emit_csv_energy_row("GPU_FP32", opt.nx, opt.ny, opt.iters, opt.kahan, e_gpu_fp32,
                        stencil_flops(opt.nx, opt.ny, op.flops_per_cell) * static_cast<double>(opt.iters),
                        /*gpu_route=*/true);
    if (csv_enabled) {
        // under_ncu fuerza "NA" en las 3 columnas de energia igual que ya
        // fuerza el prefijo NCU_ en el nombre de ruta: bajo el perfilador
        // t_ms_iter (y por tanto energia*tiempo) esta inflado y no es
        // comparable con una corrida limpia.
        write_csv_row(csv, opt, under_ncu ? "NCU_gpu_fp32" : "gpu_fp32", opt.kahan, opt.nx, opt.ny,
                     opt.iters, gpu.ms, gpu.gflops, gpu_err, first_nf_gpu_fp32, "NA",
                     kReferenceRoleNone,
                     energy_per_cell_csv_field(under_ncu, e_gpu_fp32, opt.nx, opt.ny, opt.iters),
                     "NA", "NA",
                     fmt_sci(t_checkpoint_ms_gpu_fp32), "NA", "NA",
                     energy_field(!under_ncu && e_gpu_fp32.gpu_valid, e_gpu_fp32.energy_j),
                     energy_field(!under_ncu && e_gpu_fp32.gpu_valid, e_gpu_fp32.avg_power_w),
                     energy_field(!under_ncu && e_gpu_fp32.gpu_valid, e_gpu_fp32.edp));
    }

    // Ruta GPU_FP64: referencia de maxima precision EN GPU. Corre justo despues
    // de GPU_FP32 y antes de las rutas WMMA para que su ventana de energia caiga
    // en el mismo regimen termico que el resto de rutas GPU de la corrida (una
    // referencia medida al final, con la GPU ya caliente, sesgaria a la baja
    // todo speedup y toda razon de energia calculada contra ella).
    //
    // El error se mide con compare_fp64_ref_vs_fp64 contra el MISMO ground truth
    // FP64 de CPU que usan las demas rutas: no es una comparacion trivial contra
    // si misma, porque el kernel y el bucle de CPU no son bit a bit identicos
    // (el kernel puede contraer 0.25*s - c en un FMA). Lo que mide es
    // exactamente el piso de error alcanzable en GPU para este operador.
    bool ran_gpu_fp64 = false;
    int onset_gpu_fp64 = -1;
    int first_nf_gpu_fp64 = INT_MAX;
    if (opt.fp64_gpu) {
        ran_gpu_fp64 = true;
        std::vector<double> y_gpu_fp64;
        double t_checkpoint_ms_gpu_fp64 = 0.0;
        EnergyMeasurement e_gpu_fp64;
        const Metrics gpu_fp64 = benchmark_gpu_fp64_stencil(
            input, y_gpu_fp64, opt.nx, opt.ny, opt.iters, op, ckpt, "GPU_FP64",
            onset_gpu_fp64, first_nf_gpu_fp64, t_checkpoint_ms_gpu_fp64, e_gpu_fp64);
        const ErrorMetrics gpu_fp64_err = compare_fp64_ref_vs_fp64(y_ref, y_gpu_fp64);
        // Estado final promovido a FP32 solo para la comparacion contra la CPU
        // FP32 (columna de trazabilidad); no toca y_gpu_fp64 ni ninguna metrica
        // vs FP64.
        std::vector<float> y_gpu_fp64_as_fp32(y_gpu_fp64.size());
        for (size_t i = 0; i < y_gpu_fp64.size(); ++i) {
            y_gpu_fp64_as_fp32[i] = static_cast<float>(y_gpu_fp64[i]);
        }
        const ErrorMetrics gpu_fp64_vs_cpu_err = compare_float_vectors(y_cpu, y_gpu_fp64_as_fp32);

        // store_rel de esta ruta: Q = identidad sobre double (ver
        // storage_roundtrip_metrics_fp64). Se evalua sobre el estado final, que
        // es el unico que esta ruta retiene; si diverguio, no es evaluable.
        const bool fp64_storage_evaluable = (first_nf_gpu_fp64 == INT_MAX);
        const StorageRelResult fp64_storage_result = fp64_storage_evaluable
            ? storage_roundtrip_metrics_fp64(y_gpu_fp64, opt.iters)
            : StorageRelResult{};

        print_reference_comparison("GPU CUDA FP64 (referencia GPU)", gpu_fp64, cpu.ms,
                                   gpu_fp64_err, gpu_fp64_vs_cpu_err, first_nf_gpu_fp64,
                                   opt.iters, t_checkpoint_ms_gpu_fp64);
        std::cout << "Speedup GPU FP32 vs GPU FP64       : " << gpu_fp64.ms / gpu.ms << "x\n";
        print_storage_metrics("FP64", fp64_storage_result, fp64_storage_evaluable,
                              opt.iters, 1.0e-15);
        emit_csv_summary_row(opt, "GPU_FP64", opt.nx, opt.ny, opt.iters, opt.kahan,
                             gpu_fp64.ms, gpu_fp64.gflops,
                             fmt_csv_num(cpu.ms / gpu_fp64.ms), fmt_csv_num(gpu.ms / gpu_fp64.ms),
                             "NaN", "NaN", fmt_csv_num(t_checkpoint_ms_gpu_fp64),
                             gpu_fp64_err, first_nf_gpu_fp64,
                             "NaN", "NaN",
                             storage_num_field(fp64_storage_result, fp64_storage_evaluable,
                                               fp64_storage_result.rel_norm),
                             storage_num_field(fp64_storage_result, fp64_storage_evaluable,
                                               fp64_storage_result.rel_max_guarded),
                             storage_count_field(fp64_storage_result, fp64_storage_evaluable),
                             storage_eval_iter_field(fp64_storage_result, fp64_storage_evaluable),
                             e_gpu_fp64, kReferenceRoleGpuFp64);
        emit_csv_store_row("GPU_FP64", opt.nx, opt.ny, opt.iters, opt.kahan,
                           fp64_storage_result, fp64_storage_evaluable, kFp64StorageUlp);
        print_energy_metrics(e_gpu_fp64);
        emit_csv_energy_row("GPU_FP64", opt.nx, opt.ny, opt.iters, opt.kahan, e_gpu_fp64,
                            stencil_flops(opt.nx, opt.ny, op.flops_per_cell) * static_cast<double>(opt.iters),
                            /*gpu_route=*/true);
        if (csv_enabled) {
            write_csv_row(csv, opt, under_ncu ? "NCU_gpu_fp64" : "gpu_fp64", opt.kahan,
                          opt.nx, opt.ny, opt.iters, gpu_fp64.ms, gpu_fp64.gflops,
                          gpu_fp64_err, first_nf_gpu_fp64,
                          storage_num_field(fp64_storage_result, fp64_storage_evaluable,
                                            fp64_storage_result.rel_max_guarded),
                          kReferenceRoleGpuFp64,
                          energy_per_cell_csv_field(under_ncu, e_gpu_fp64, opt.nx, opt.ny,
                                                    opt.iters),
                          "NA", "NA", fmt_sci(t_checkpoint_ms_gpu_fp64), "NA", "NA",
                          energy_field(!under_ncu && e_gpu_fp64.gpu_valid, e_gpu_fp64.energy_j),
                          energy_field(!under_ncu && e_gpu_fp64.gpu_valid, e_gpu_fp64.avg_power_w),
                          energy_field(!under_ncu && e_gpu_fp64.gpu_valid, e_gpu_fp64.edp));
        }
        std::cout << "\n";
    }

    bool ran_fp16 = false;
    bool ran_bf16 = false;
    int onset_fp16 = -1;
    int onset_bf16 = -1;
    int first_nf_fp16 = INT_MAX;
    int first_nf_bf16 = INT_MAX;

    // Politica de compensacion y etiquetas de ruta derivadas: con
    // --spatial-comp on las rutas WMMA se reportan como WMMA_FP16_SP /
    // WMMA_BF16_SP en TODAS sus filas CSV (DRIFT, SUMMARY, STORE, ENERGY,
    // REGION, ONSET) para que no se confundan con las de --kahan off|on al
    // mezclar corridas. La columna kahan de esas filas sigue siendo off (ver
    // wmma_route_label): el esquema no cambia.
    const CompMode comp_mode = comp_mode_of(opt);
    const char* route_fp16 = fp16_route_label(comp_mode);
    const char* route_bf16 = bf16_route_label(comp_mode);

    if (opt.tc_mode == TensorCoreMode::FP16 || opt.tc_mode == TensorCoreMode::Both) {
        ran_fp16 = true;
        double t_wmma_ms_fp16 = 0.0, t_conv_ms_fp16 = 0.0, t_checkpoint_ms_fp16 = 0.0;
        int storage_rel_eval_iter_fp16 = 0;
        std::vector<float> y_tc_fp16_last_finite;
        std::vector<__half> y_tc_fp16_reduced_last_finite;
        EnergyMeasurement e_fp16;
        std::vector<float> y_tc_fp16_comp;
        const Metrics tc_fp16 = benchmark_gpu_tensor_core_stencil<__half>(
            input, y_tc_fp16, y_tc_fp16_reduced, opt.nx, opt.ny, opt.iters, op, comp_mode,
            opt.execution_mode, opt.graph_block, ckpt, route_fp16, onset_fp16, first_nf_fp16, t_wmma_ms_fp16, t_conv_ms_fp16,
            storage_rel_eval_iter_fp16, t_checkpoint_ms_fp16, y_tc_fp16_last_finite,
            y_tc_fp16_reduced_last_finite, e_fp16, y_tc_fp16_comp);
        const ErrorMetrics tc_fp16_err        = compare_fp64_ref_vs_fp32(y_ref, y_tc_fp16);
        const ErrorMetrics tc_fp16_vs_cpu_err = compare_float_vectors(y_cpu, y_tc_fp16);
        // eval_iter == -1: la ruta divergio sin que ningun checkpoint
        // capturara un estado finito antes (ver comentario en
        // benchmark_gpu_tensor_core_stencil). y_tc_fp16/y_tc_fp16_reduced son
        // SIEMPRE la ultima iteracion cruda (pueden contener inf/NaN, ver
        // bloque 1); store_rel se evalua con Q(u)-u sobre
        // y_tc_fp16_last_finite, el estado FP32 recuperable mas reciente.
        const bool fp16_storage_evaluable = (storage_rel_eval_iter_fp16 != -1);
        const StorageRelResult fp16_storage_result = fp16_storage_evaluable
            ? storage_roundtrip_metrics<__half>(y_tc_fp16_last_finite, storage_rel_eval_iter_fp16)
            : StorageRelResult{};
        // Estado PROPAGADO (buffer T crudo, no out_fp32): responde si Kahan
        // acerca lo que realmente se encadena entre iteraciones a la
        // exactitud de FP32 (ver print_propagated_error_metrics/bloque 2).
        const ErrorMetrics tc_fp16_prop_err =
            compare_fp64_ref_vs_fp32(y_ref, build_readout_field(y_tc_fp16_reduced, y_tc_fp16_comp));
        // Sin instrumentar por separado el lector asume que el cuello de
        // botella es el Tensor Core; en realidad convert_float_to_half_kernel
        // (reconversion de d_out a T en cada iteracion) explica buena parte
        // del t/iter total. no_atribuido cubre overhead de lanzamiento
        // (1048576 bloques) no capturado por ninguno de los dos eventos.
        const double t_unattrib_fp16 = tc_fp16.ms - t_wmma_ms_fp16 - t_conv_ms_fp16;

        std::cout << "GPU WMMA FP16 Tensor Core - tiempo/iter (media) : " << tc_fp16.ms << " ms\n";
        std::cout << "GPU WMMA FP16 Tensor Core - tiempo total        : " << tc_fp16.ms * opt.iters << " ms\n";
        std::cout << "GPU WMMA FP16 Tensor Core - rend.  : " << tc_fp16.gflops
                  << " GFLOP/s (" << tc_fp16.tflops << " TFLOP/s efectivos)\n";
        std::cout << "Speedup TC FP16 vs CPU             : " << cpu.ms / tc_fp16.ms << "x\n";
        std::cout << "Speedup TC FP16 vs GPU FP32        : " << gpu.ms / tc_fp16.ms << "x\n";
        std::cout << "t kernel WMMA/iter  : " << t_wmma_ms_fp16 << " ms ("
                  << fmt_pct1(100.0 * t_wmma_ms_fp16 / tc_fp16.ms) << " %)\n";
        std::cout << "t conversion/iter   : " << t_conv_ms_fp16 << " ms ("
                  << fmt_pct1(100.0 * t_conv_ms_fp16 / tc_fp16.ms) << " %)\n";
        std::cout << "t no atribuido/iter : " << t_unattrib_fp16 << " ms ("
                  << fmt_pct1(100.0 * t_unattrib_fp16 / tc_fp16.ms) << " %)\n";
        std::cout << "t checkpoints/iter  : " << t_checkpoint_ms_fp16
                  << " ms  (excluido del t/iter reportado)\n";
        std::cout << "Modo de ejecucion   : " << execution_mode_label(opt.execution_mode)
                  << (opt.execution_mode == ExecutionMode::Graph
                          ? "  (" + std::to_string(opt.graph_block) + " iteraciones por grafo)"
                          : std::string())
                  << "\n";
        print_error_metrics("Error max abs vs FP64              : ", "Error relativo L2 vs FP64          : ",
                            "Error rel Linf vs FP64             : ", tc_fp16_err, first_nf_fp16);
        print_propagated_error_metrics(tc_fp16_prop_err, first_nf_fp16);
        print_error_metrics("Error max abs vs CPU FP32          : ", "Error relativo L2 vs CPU FP32      : ",
                            "Error rel Linf vs CPU FP32         : ", tc_fp16_vs_cpu_err, first_nf_fp16);
        print_first_nonfinite("Primera iteracion no finita        : ", first_nf_fp16, opt.iters);
        print_storage_metrics("FP16", fp16_storage_result, fp16_storage_evaluable,
                              opt.iters, 1.0e-3);
        std::cout << "\n\n";
        emit_csv_summary_row(opt, route_fp16, opt.nx, opt.ny, opt.iters, opt.kahan,
                             tc_fp16.ms, tc_fp16.gflops,
                             fmt_csv_num(cpu.ms / tc_fp16.ms), fmt_csv_num(gpu.ms / tc_fp16.ms),
                             fmt_csv_num(t_wmma_ms_fp16), fmt_csv_num(t_conv_ms_fp16),
                             fmt_csv_num(t_checkpoint_ms_fp16), tc_fp16_err, first_nf_fp16,
                             fmt_csv_error_num(tc_fp16_prop_err, tc_fp16_prop_err.rel_l2),
                             fmt_csv_error_num(tc_fp16_prop_err, tc_fp16_prop_err.rel_linf),
                             storage_num_field(fp16_storage_result, fp16_storage_evaluable,
                                               fp16_storage_result.rel_norm),
                             storage_num_field(fp16_storage_result, fp16_storage_evaluable,
                                               fp16_storage_result.rel_max_guarded),
                             storage_count_field(fp16_storage_result, fp16_storage_evaluable),
                             storage_eval_iter_field(fp16_storage_result, fp16_storage_evaluable), e_fp16,
                             kReferenceRoleNone);
        emit_csv_store_row(route_fp16, opt.nx, opt.ny, opt.iters, opt.kahan,
                           fp16_storage_result, fp16_storage_evaluable, kFp16StorageUlp);
        print_energy_metrics(e_fp16);
        emit_csv_energy_row(route_fp16, opt.nx, opt.ny, opt.iters, opt.kahan, e_fp16,
                            stencil_flops(opt.nx, opt.ny, op.flops_per_cell) * static_cast<double>(opt.iters),
                            /*gpu_route=*/true);
        if (csv_enabled) {
            write_csv_row(csv, opt,
                         std::string(under_ncu ? "NCU_" : "") + fp16_csv_label(comp_mode),
                         opt.kahan, opt.nx, opt.ny,
                         opt.iters, tc_fp16.ms, tc_fp16.gflops, tc_fp16_err, first_nf_fp16,
                         storage_num_field(fp16_storage_result, fp16_storage_evaluable,
                                           fp16_storage_result.rel_max_guarded),
                         kReferenceRoleNone,
                         energy_per_cell_csv_field(under_ncu, e_fp16, opt.nx, opt.ny, opt.iters),
                         fmt_sci(t_wmma_ms_fp16), fmt_sci(t_conv_ms_fp16), fmt_sci(t_checkpoint_ms_fp16),
                         fmt_sci(tc_fp16_prop_err.rel_l2), fmt_sci(tc_fp16_prop_err.rel_linf),
                         energy_field(!under_ncu && e_fp16.gpu_valid, e_fp16.energy_j),
                         energy_field(!under_ncu && e_fp16.gpu_valid, e_fp16.avg_power_w),
                         energy_field(!under_ncu && e_fp16.gpu_valid, e_fp16.edp),
                         execution_mode_label(opt.execution_mode));
        }
    }

    if (opt.tc_mode == TensorCoreMode::BF16 || opt.tc_mode == TensorCoreMode::Both) {
        ran_bf16 = true;
        double t_wmma_ms_bf16 = 0.0, t_conv_ms_bf16 = 0.0, t_checkpoint_ms_bf16 = 0.0;
        int storage_rel_eval_iter_bf16 = 0;
        std::vector<float> y_tc_bf16_last_finite;
        std::vector<__nv_bfloat16> y_tc_bf16_reduced_last_finite;
        EnergyMeasurement e_bf16;
        std::vector<float> y_tc_bf16_comp;
        const Metrics tc_bf16 = benchmark_gpu_tensor_core_stencil<__nv_bfloat16>(
            input, y_tc_bf16, y_tc_bf16_reduced, opt.nx, opt.ny, opt.iters, op, comp_mode,
            opt.execution_mode, opt.graph_block, ckpt, route_bf16, onset_bf16, first_nf_bf16, t_wmma_ms_bf16, t_conv_ms_bf16,
            storage_rel_eval_iter_bf16, t_checkpoint_ms_bf16, y_tc_bf16_last_finite,
            y_tc_bf16_reduced_last_finite, e_bf16, y_tc_bf16_comp);
        const ErrorMetrics tc_bf16_err        = compare_fp64_ref_vs_fp32(y_ref, y_tc_bf16);
        const ErrorMetrics tc_bf16_vs_cpu_err = compare_float_vectors(y_cpu, y_tc_bf16);
        // Ver comentario analogo en el bloque FP16: store_rel se evalua con
        // Q(u)-u sobre y_tc_bf16_last_finite, no sobre y_tc_bf16/y_tc_bf16_reduced
        // (que son la ultima iteracion cruda).
        const bool bf16_storage_evaluable = (storage_rel_eval_iter_bf16 != -1);
        const StorageRelResult bf16_storage_result = bf16_storage_evaluable
            ? storage_roundtrip_metrics<__nv_bfloat16>(y_tc_bf16_last_finite, storage_rel_eval_iter_bf16)
            : StorageRelResult{};
        // Ver comentario analogo en el bloque FP16: estado PROPAGADO (buffer
        // T crudo), no out_fp32.
        const ErrorMetrics tc_bf16_prop_err =
            compare_fp64_ref_vs_fp32(y_ref, build_readout_field(y_tc_bf16_reduced, y_tc_bf16_comp));
        // Ver comentario analogo en el bloque FP16: sin este desglose el
        // 2.3x de t/iter frente a GPU FP32 clasico se le atribuiria por
        // error al Tensor Core en vez de a convert_float_to_bfloat16_kernel.
        const double t_unattrib_bf16 = tc_bf16.ms - t_wmma_ms_bf16 - t_conv_ms_bf16;

        std::cout << "GPU WMMA BF16 Tensor Core - tiempo/iter (media) : " << tc_bf16.ms << " ms\n";
        std::cout << "GPU WMMA BF16 Tensor Core - tiempo total        : " << tc_bf16.ms * opt.iters << " ms\n";
        std::cout << "GPU WMMA BF16 Tensor Core - rend.  : " << tc_bf16.gflops
                  << " GFLOP/s (" << tc_bf16.tflops << " TFLOP/s efectivos)\n";
        std::cout << "Speedup TC BF16 vs CPU             : " << cpu.ms / tc_bf16.ms << "x\n";
        std::cout << "Speedup TC BF16 vs GPU FP32        : " << gpu.ms / tc_bf16.ms << "x\n";
        std::cout << "t kernel WMMA/iter  : " << t_wmma_ms_bf16 << " ms ("
                  << fmt_pct1(100.0 * t_wmma_ms_bf16 / tc_bf16.ms) << " %)\n";
        std::cout << "t conversion/iter   : " << t_conv_ms_bf16 << " ms ("
                  << fmt_pct1(100.0 * t_conv_ms_bf16 / tc_bf16.ms) << " %)\n";
        std::cout << "t no atribuido/iter : " << t_unattrib_bf16 << " ms ("
                  << fmt_pct1(100.0 * t_unattrib_bf16 / tc_bf16.ms) << " %)\n";
        std::cout << "t checkpoints/iter  : " << t_checkpoint_ms_bf16
                  << " ms  (excluido del t/iter reportado)\n";
        std::cout << "Modo de ejecucion   : " << execution_mode_label(opt.execution_mode)
                  << (opt.execution_mode == ExecutionMode::Graph
                          ? "  (" + std::to_string(opt.graph_block) + " iteraciones por grafo)"
                          : std::string())
                  << "\n";
        print_error_metrics("Error max abs vs FP64              : ", "Error relativo L2 vs FP64          : ",
                            "Error rel Linf vs FP64             : ", tc_bf16_err, first_nf_bf16);
        print_propagated_error_metrics(tc_bf16_prop_err, first_nf_bf16);
        print_error_metrics("Error max abs vs CPU FP32          : ", "Error relativo L2 vs CPU FP32      : ",
                            "Error rel Linf vs CPU FP32         : ", tc_bf16_vs_cpu_err, first_nf_bf16);
        print_first_nonfinite("Primera iteracion no finita        : ", first_nf_bf16, opt.iters);
        print_storage_metrics("BF16", bf16_storage_result, bf16_storage_evaluable,
                              opt.iters, 8.0e-3);
        std::cout << "\n\n";
        emit_csv_summary_row(opt, route_bf16, opt.nx, opt.ny, opt.iters, opt.kahan,
                             tc_bf16.ms, tc_bf16.gflops,
                             fmt_csv_num(cpu.ms / tc_bf16.ms), fmt_csv_num(gpu.ms / tc_bf16.ms),
                             fmt_csv_num(t_wmma_ms_bf16), fmt_csv_num(t_conv_ms_bf16),
                             fmt_csv_num(t_checkpoint_ms_bf16), tc_bf16_err, first_nf_bf16,
                             fmt_csv_error_num(tc_bf16_prop_err, tc_bf16_prop_err.rel_l2),
                             fmt_csv_error_num(tc_bf16_prop_err, tc_bf16_prop_err.rel_linf),
                             storage_num_field(bf16_storage_result, bf16_storage_evaluable,
                                               bf16_storage_result.rel_norm),
                             storage_num_field(bf16_storage_result, bf16_storage_evaluable,
                                               bf16_storage_result.rel_max_guarded),
                             storage_count_field(bf16_storage_result, bf16_storage_evaluable),
                             storage_eval_iter_field(bf16_storage_result, bf16_storage_evaluable), e_bf16,
                             kReferenceRoleNone);
        emit_csv_store_row(route_bf16, opt.nx, opt.ny, opt.iters, opt.kahan,
                           bf16_storage_result, bf16_storage_evaluable, kBf16StorageUlp);
        print_energy_metrics(e_bf16);
        emit_csv_energy_row(route_bf16, opt.nx, opt.ny, opt.iters, opt.kahan, e_bf16,
                            stencil_flops(opt.nx, opt.ny, op.flops_per_cell) * static_cast<double>(opt.iters),
                            /*gpu_route=*/true);
        if (csv_enabled) {
            write_csv_row(csv, opt,
                         std::string(under_ncu ? "NCU_" : "") + bf16_csv_label(comp_mode),
                         opt.kahan, opt.nx, opt.ny,
                         opt.iters, tc_bf16.ms, tc_bf16.gflops, tc_bf16_err, first_nf_bf16,
                         storage_num_field(bf16_storage_result, bf16_storage_evaluable,
                                           bf16_storage_result.rel_max_guarded),
                         kReferenceRoleNone,
                         energy_per_cell_csv_field(under_ncu, e_bf16, opt.nx, opt.ny, opt.iters),
                         fmt_sci(t_wmma_ms_bf16), fmt_sci(t_conv_ms_bf16), fmt_sci(t_checkpoint_ms_bf16),
                         fmt_sci(tc_bf16_prop_err.rel_l2), fmt_sci(tc_bf16_prop_err.rel_linf),
                         energy_field(!under_ncu && e_bf16.gpu_valid, e_bf16.energy_j),
                         energy_field(!under_ncu && e_bf16.gpu_valid, e_bf16.avg_power_w),
                         energy_field(!under_ncu && e_bf16.gpu_valid, e_bf16.edp),
                         execution_mode_label(opt.execution_mode));
        }
    }

    std::cout << "====================================================\n\n";

    // Calibracion del horizonte de overflow desde la referencia FP64: ajuste
    // de modelo log-lineal en el regimen asintotico (ver OverflowFitResult).
    const OverflowHorizonPrediction horizon =
        compute_overflow_horizon_from_reference(linf_per_iter, u0_linf, op);

    print_overflow_horizon(horizon, a_nyq,
                          first_nf_fp16, first_nf_bf16, first_nf_gpu_fp32, first_nf_fp64_ref,
                          ran_gpu_fp64, first_nf_gpu_fp64);
    emit_csv_horizon_rows(horizon, a_nyq, opt.nx, opt.ny, opt.iters, opt.kahan, comp_mode,
                          first_nf_fp16, first_nf_bf16, first_nf_gpu_fp32, first_nf_fp64_ref,
                          ran_gpu_fp64, first_nf_gpu_fp64);

    if (checkpoints_enabled(ckpt)) {
        std::cout << "=========== RESUMEN ONSET DE DIVERGENCIA ===========\n";
        std::cout << "CSV_ONSET,GPU_FP32," << onset_gpu_fp32 << "\n";
        if (ran_gpu_fp64) {
            std::cout << "CSV_ONSET,GPU_FP64," << onset_gpu_fp64 << "\n";
        }
        if (ran_fp16) {
            std::cout << "CSV_ONSET," << route_fp16 << "," << onset_fp16 << "\n";
        }
        if (ran_bf16) {
            std::cout << "CSV_ONSET," << route_bf16 << "," << onset_bf16 << "\n";
        }
        std::cout << "=====================================================\n\n";
    }

    if (use_archive) {
        write_archive_manifest(archive_ctx, opt, ci_sha256);
    }

    print_nsight_hint(exe_name, opt.nx, opt.ny, opt.iters, opt.tc_mode, opt.kahan,
                      opt.spatial_comp);
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);
    print_gpu_info();
    // La comprobacion ocurre despues de cudaGetDeviceProperties (dentro de
    // print_gpu_info), antes de iniciar cualquier benchmark.
    telemetry_nvml_initialize(0);
    if (!rapl_available()) {
        std::fprintf(stderr,
                     "ADVERTENCIA: RAPL no esta disponible o no es legible en "
                     "/sys/class/powercap. Energia CPU sera NaN.\n");
    }
    run_benchmark(opt, argv[0]);
    return 0;
}
