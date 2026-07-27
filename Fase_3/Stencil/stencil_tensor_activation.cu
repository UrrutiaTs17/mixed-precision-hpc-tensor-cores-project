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
// La ruta WMMA reescribe cada tile interior 16x16 como cinco operaciones MMA:
// left*0.25I + right*0.25I + 0.25I*up + 0.25I*down + center*(-I).
// Es una adaptacion didactica para validar activacion de Tensor Cores en stencil;
// no pretende ser el stencil mas eficiente posible en memoria.
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
// warps/SM (50%) pese a que registros (64 warps/SM) y shared (36 bloques) lo
// permitirian; con 4 warps/bloque, shared por bloque = 4*3584 B = 14 KiB
// (10 bloques/SM x 4 warps = 40 warps/SM, 62.5%) y los CTAs bajan de 1.05M a
// 262144 (gridDim.x = ceil(total_tiles / kWarpsPerBlock)).
constexpr int kWarpsPerBlock = 4;

enum class TensorCoreMode {
    FP16,
    BF16,
    Both
};

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
};

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

static std::string csv_first_nonfinite_field(int first_nf) {
    return std::to_string((first_nf == INT_MAX) ? -1 : first_nf);
}

// Marcador de telemetria para Fase 4: emite por stdout el limite de una
// region cronometrada (begin/end) con timestamp de pared en ns desde epoch.
// Se emite FUERA del par de eventos CUDA que mide t/iter (nunca dentro de lo
// que build_metrics reporta): un muestreador NVML externo alinea ventanas de
// potencia con estos marcadores sin que este archivo tenga que exponer nada
// mas. Solo se usa en las rutas GPU con route_label (GPU_FP32, WMMA_FP16,
// WMMA_BF16); CPU FP32 serial no tiene ventana de potencia GPU que alinear.
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
           " [--checkpoint-every K] [--csv RUTA] [--profile-only] [--kahan off|on]\n\n"
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
        << "  byte.\n\n"
        << "Ejemplos:\n"
        << "  " << prog << "\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 20 --tc fp16\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 20 --tc bf16\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 20 --tc both --checkpoint-every 5\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 20 --tc fp16 --kahan on\n";
}

static int parse_int_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Falta valor para " << argv[i] << "\n";
        std::exit(EXIT_FAILURE);
    }
    return std::atoi(argv[++i]);
}

static TensorCoreMode parse_tc_mode(const char* value) {
    if (std::strcmp(value, "fp16") == 0) return TensorCoreMode::FP16;
    if (std::strcmp(value, "bf16") == 0) return TensorCoreMode::BF16;
    if (std::strcmp(value, "both") == 0) return TensorCoreMode::Both;

    std::cerr << "Modo Tensor Core no reconocido: " << value << "\n";
    std::exit(EXIT_FAILURE);
}

static bool parse_kahan_flag(const char* value) {
    if (std::strcmp(value, "off") == 0) return false;
    if (std::strcmp(value, "on") == 0) return true;

    std::cerr << "Valor no reconocido para --kahan (use off|on): " << value << "\n";
    std::exit(EXIT_FAILURE);
}

static Options parse_args(int argc, char** argv) {
    Options opt;
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
            opt.kahan = parse_kahan_flag(argv[++i]);
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

static double stencil_flops(int nx, int ny) {
    return 5.0 * static_cast<double>(nx - 2) * static_cast<double>(ny - 2);
}

// Nota metodologica: el TFLOPS reportado para la ruta WMMA NO es comparable en
// terminos absolutos al TFLOPS de GEMM (Fase_2/GEMM). Aqui cada tile 16x16 solo
// ejecuta 5 MMA 16x16x16 para reproducir un escalado elemento a elemento via
// matrices identidad (ver comentario superior del archivo); el costo esta
// dominado por el movimiento de datos a shared memory (carga de 5 tiles T y
// escritura del tile de salida), no por computo dense real como en GEMM. El
// numero sirve para comparar las tres rutas de Stencil entre si (CPU, GPU FP32,
// GPU WMMA), no para comparar Stencil contra GEMM.
static Metrics build_metrics(int nx, int ny, double avg_ms) {
    Metrics m;
    m.ms = avg_ms;
    m.gflops = stencil_flops(nx, ny) / (m.ms * 1e6);
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
                                     int& first_nonfinite_iter,
                                     EnergyMeasurement& out_energy) {
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
                const float val = 0.25f * (up + down + left + right) - center;
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
        const double flops_total = 9.0 * static_cast<double>(nx - 2) *
                                   static_cast<double>(ny - 2) * iters;
        out_energy.joules_per_gflop = out_energy.energy_total_j / (flops_total / 1e9);
    }
    out = *src;
    return build_metrics(nx, ny, avg_ms);
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
                                     int checkpoint_every,
                                     std::vector<std::vector<double>>& checkpoints,
                                     std::vector<double>& linf_per_iter,
                                     int& first_nonfinite_iter) {
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
                dst[idx2d(x, y, nx)] = 0.25 * (up + down + left + right) - center;
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
        bool src_finite = true;
        for (const auto& x : *src) {
            if (!std::isfinite(x)) {
                src_finite = false;
                break;
            }
            linf = std::max(linf, std::fabs(x));
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

__global__ static void stencil2d_fp32_kernel(const float* in, float* out, int nx, int ny,
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
            val = 0.25f * (up + down + left + right) - center;
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
};

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
                                                               double flops_total) {
    EnergyMeasurement result;
    result.time_total_s = time_total_s;
    result.gpu_valid = gpu_valid;
    result.cpu_valid = cpu_valid;
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
        stencil2d_fp32_kernel<<<grid, block>>>(warm_in, warm_out, nx, ny, i + 1, d_first_nf);
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
    if (ckpt.checkpoint_every > 0) {
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
    bool gpu_energy_valid = true;
    double checkpoint_cpu_energy_j = 0.0;
    double checkpoint_pause_s = 0.0;
    auto close_energy_segment = [&]() {
        power_buffer_stop_sampling(power_buffer);
        gpu_energy_valid = gpu_energy_valid && power_buffer_capture_valid(power_buffer);
        gpu_energy_j += power_buffer_energy_joules(power_buffer);
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
        stencil2d_fp32_kernel<<<grid, block>>>(d_in, d_out, nx, ny, i + 1, d_first_nf);
        std::swap(d_in, d_out);

        // Solo LEE d_in (ya con el swap aplicado, ver comentario mas abajo);
        // no altera el ping-pong.
        if (ckpt.checkpoint_every > 0 && (i + 1) % ckpt.checkpoint_every == 0) {
            total_ms += timer.stop_and_elapsed_ms();
            // Misma pausa que el cronometro, ahora tambien para la energia: el
            // D2H y el escaneo del host dejan la GPU ociosa y, sin excluirlos,
            // energy_gpu_j/edp_j_s medirian sobre todo la instrumentacion (con
            // CHECKPOINT_EVERY=5 llegaba a ~97% de la ventana).
            close_energy_segment();
            const auto pause_t0 = std::chrono::steady_clock::now();
            const RAEnergySnapshot rapl_ckpt_before = rapl_snapshot_now();

            const auto ckpt_t0 = std::chrono::high_resolution_clock::now();
            CHECK_CUDA(cudaMemcpy(checkpoint_host_buf.data(), d_in,
                                  count * sizeof(float), cudaMemcpyDeviceToHost));
            record_checkpoint(ckpt, route_label, i + 1, checkpoint_host_buf, onset_iter);
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
    // Tiempo de la ventana de energia = pared total menos los tramos de
    // checkpoint, para que avg_power_w/edp_j_s usen el mismo intervalo sobre
    // el que se integro gpu_energy_j.
    const double energy_wall_s = std::max(
        0.0, std::chrono::duration<double>(energy_t1 - energy_t0).count() - checkpoint_pause_s);
    const double flops_total = 9.0 * static_cast<double>(nx - 2) *
                               static_cast<double>(ny - 2) * iters;
    const bool cpu_energy_valid = rapl_before.valid && rapl_after.valid &&
                                  rapl_after.energy_j >= rapl_before.energy_j;
    const double cpu_energy_j = std::max(
        0.0, rapl_energy_delta(rapl_before, rapl_after) - checkpoint_cpu_energy_j);
    out_energy = make_energy_measurement_from_segments(
        gpu_energy_valid, gpu_energy_j, cpu_energy_valid, cpu_energy_j,
        energy_wall_s, flops_total);
    power_buffer_destroy(power_buffer);
    t_checkpoint_ms_out = checkpoint_ms_total / iters;
    CHECK_CUDA(cudaGetLastError());
    // Tras el ultimo swap, d_in apunta al buffer con la salida mas reciente.
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&first_nonfinite_iter, d_first_nf, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_first_nf));
    return build_metrics(nx, ny, total_ms / iters);
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

// Suma compensada de Kahan del redondeo de ALMACENAMIENTO a 16 bits (no de la
// suma de los 5 vecinos, ver metodologia 5.3/4.1.4 y el comentario de
// benchmark_gpu_tensor_core_stencil): comp[idx] persiste en FP32 entre
// iteraciones el residuo del redondeo anterior, indexado por celda. Cuando
// kKahan es false, comp no se toca (puede ser nullptr) y esto colapsa a
// float_to_tc<T> sin rama en tiempo de ejecucion (if constexpr, resuelto en
// compilacion): la ruta --kahan off no paga costo alguno.
template <typename T, bool kKahan>
__device__ inline T compensated_store(float val, float* comp, int idx) {
    if constexpr (kKahan) {
        const float y = val - comp[idx];
        const T s = float_to_tc<T>(y);
        comp[idx] = tc_to_float(s) - y;
        return s;
    } else {
        return float_to_tc<T>(val);
    }
}

static __half make_tc_value_half(float x) {
    return __float2half(x);
}

static __nv_bfloat16 make_tc_value_bfloat16(float x) {
    return __float2bfloat16(x);
}

template <typename T>
static void initialize_scaled_identity(std::vector<T>& mat, float scale);

template <>
void initialize_scaled_identity<__half>(std::vector<__half>& mat, float scale) {
    std::fill(mat.begin(), mat.end(), make_tc_value_half(0.0f));
    for (int i = 0; i < kTile; ++i) {
        mat[i * kTile + i] = make_tc_value_half(scale);
    }
}

template <>
void initialize_scaled_identity<__nv_bfloat16>(std::vector<__nv_bfloat16>& mat, float scale) {
    std::fill(mat.begin(), mat.end(), make_tc_value_bfloat16(0.0f));
    for (int i = 0; i < kTile; ++i) {
        mat[i * kTile + i] = make_tc_value_bfloat16(scale);
    }
}

// Bytes de shared por WARP (no por bloque): tc_tiles[5][16][16] en T +
// out_tile[16][16] en float. Con T de 2 bytes da 2560 + 1024 = 3584 B/warp
// (confirmado por NCU: tamano estatico de shared con 1 warp/bloque). Usadas
// tanto por el kernel (para particionar smem_raw) como por el host (para
// dimensionar el shared dinamico del lanzamiento) -- una sola definicion
// evita que ambos lados se desincronicen.
template <typename T>
__host__ __device__ constexpr size_t wmma_tc_tiles_bytes() {
    return 5 * kTile * kTile * sizeof(T);
}
__host__ __device__ constexpr size_t wmma_out_tile_bytes() {
    return kTile * kTile * sizeof(float);
}
template <typename T>
__host__ __device__ constexpr size_t wmma_warp_shared_bytes() {
    return wmma_tc_tiles_bytes<T>() + wmma_out_tile_bytes();
}

// kKahan (parametro de plantilla, no runtime): activa la compensacion de
// Kahan del redondeo de almacenamiento (ver compensated_store). comp es
// nullptr y no se toca cuando kKahan es false -- el llamador (benchmark_
// gpu_tensor_core_stencil) elige la instanciacion en tiempo de compilacion
// segun el flag --kahan, asi la ruta off no paga rama ni acceso a comp.
template <typename T, bool kKahan>
__global__ static void stencil2d_wmma_kernel(const T* __restrict__ in,
                                             float* __restrict__ out_fp32,
                                             T* __restrict__ out_tc,
                                             const T* __restrict__ identity_pos,
                                             const T* __restrict__ identity_neg,
                                             int nx,
                                             int ny,
                                             int iter,
                                             bool write_fp32,
                                             int* __restrict__ first_nf,
                                             float* __restrict__ comp) {
    // Cada warp procesa un tile 16x16 propio e independiente (shared privada
    // por warp, ver smem_raw mas abajo): el bloque ya no es 1 warp = 1 tile,
    // es kWarpsPerBlock warps = kWarpsPerBlock tiles.
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // El grid ahora es 1D (blockIdx.y no se usa): tiles_x se recalcula con la
    // MISMA formula que el host usa para dimensionar el grid (ver
    // benchmark_gpu_tensor_core_stencil), preservando el mapeo tile->dominio
    // (x0/y0) sin agregar un parametro nuevo a la firma.
    const int tiles_x = (nx - 2 + kTile - 1) / kTile;
    const int tile_id = blockIdx.x * kWarpsPerBlock + warp_id;
    const int tile_x = tile_id % tiles_x;
    const int tile_y = tile_id / tiles_x;

    const int x0 = 1 + tile_x * kTile;
    const int y0 = 1 + tile_y * kTile;
    const bool full_tile = (x0 + kTile - 1 < nx - 1) && (y0 + kTile - 1 < ny - 1);

    // full_tile es uniforme POR WARP (mismo tile_id para las 32 lanes de un
    // warp), asi que __syncwarp() dentro de cada rama es valido aunque
    // distintos warps del bloque tomen ramas distintas. Ningun hilo hace
    // return antes del __syncthreads() final: los warps "fantasma" que
    // gridDim.x = ceil(total_tiles/kWarpsPerBlock) agrega al ultimo bloque
    // (tile_id >= total_tiles) caen en tile_y >= tiles_y, lo que fuerza
    // full_tile = false y active = false para sus 256 puntos (ver rama
    // else): no leen ni escriben nada, pero igual llegan al final.
    __shared__ int blk_bad;
    if (threadIdx.x == 0) blk_bad = 0;
    __syncthreads();

    if (full_tile) {
        // smem_raw es la shared dinamica de TODO el bloque (kWarpsPerBlock *
        // wmma_warp_shared_bytes<T>(), fijada por el host al lanzar); cada
        // warp toma su propia porcion via warp_id, sin solaparse con las
        // demas: tc_tiles primero, out_tile justo despues, igual que las
        // dos declaraciones estaticas que reemplazan (mismo tamano/layout).
        extern __shared__ __align__(32) char smem_raw[];
        T* tc_tiles = reinterpret_cast<T*>(smem_raw + warp_id * wmma_warp_shared_bytes<T>());
        float* out_tile = reinterpret_cast<float*>(smem_raw + warp_id * wmma_warp_shared_bytes<T>()
                                                    + wmma_tc_tiles_bytes<T>());

        T* left_tile = tc_tiles + 0 * kTile * kTile;
        T* right_tile = tc_tiles + 1 * kTile * kTile;
        T* up_tile = tc_tiles + 2 * kTile * kTile;
        T* down_tile = tc_tiles + 3 * kTile * kTile;
        T* center_tile = tc_tiles + 4 * kTile * kTile;

        for (int linear = lane; linear < kTile * kTile; linear += kWarpThreads) {
            const int local_x = linear % kTile;
            const int local_y = linear / kTile;
            const int x = x0 + local_x;
            const int y = y0 + local_y;

            left_tile[linear] = in[idx2d(x - 1, y, nx)];
            right_tile[linear] = in[idx2d(x + 1, y, nx)];
            up_tile[linear] = in[idx2d(x, y - 1, nx)];
            down_tile[linear] = in[idx2d(x, y + 1, nx)];
            center_tile[linear] = in[idx2d(x, y, nx)];
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, T, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, T, wmma::row_major> b_frag;
        wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, T, wmma::row_major> id_a_frag;
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, T, wmma::row_major> id_pos_b_frag;
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, T, wmma::row_major> id_neg_b_frag;
        wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);

        wmma::load_matrix_sync(id_pos_b_frag, identity_pos, kTile);
        wmma::load_matrix_sync(id_neg_b_frag, identity_neg, kTile);
        wmma::load_matrix_sync(id_a_frag, identity_pos, kTile);

        wmma::load_matrix_sync(a_frag, left_tile, kTile);
        wmma::mma_sync(acc_frag, a_frag, id_pos_b_frag, acc_frag);

        wmma::load_matrix_sync(a_frag, right_tile, kTile);
        wmma::mma_sync(acc_frag, a_frag, id_pos_b_frag, acc_frag);

        wmma::load_matrix_sync(b_frag, up_tile, kTile);
        wmma::mma_sync(acc_frag, id_a_frag, b_frag, acc_frag);

        wmma::load_matrix_sync(b_frag, down_tile, kTile);
        wmma::mma_sync(acc_frag, id_a_frag, b_frag, acc_frag);

        wmma::load_matrix_sync(a_frag, center_tile, kTile);
        wmma::mma_sync(acc_frag, a_frag, id_neg_b_frag, acc_frag);

        wmma::store_matrix_sync(out_tile, acc_frag, kTile, wmma::mem_row_major);
        __syncwarp();

        // Finitud evaluada sobre el acumulador FP32 (out_tile), antes de
        // convertir a T. out_tc SIEMPRE se escribe (reemplaza al kernel de
        // conversion dentro del bucle); out_fp32 solo cuando write_fp32 (ultima
        // iteracion medida o checkpoint), con la MISMA funcion de conversion
        // que convert_float_to_half_kernel/convert_float_to_bfloat16_kernel.
        for (int linear = lane; linear < kTile * kTile; linear += kWarpThreads) {
            const int local_x = linear % kTile;
            const int local_y = linear / kTile;
            const float val = out_tile[linear];
            const int idx = idx2d(x0 + local_x, y0 + local_y, nx);
            out_tc[idx] = compensated_store<T, kKahan>(val, comp, idx);
            if (write_fp32) out_fp32[idx] = val;
            if (!isfinite(val)) blk_bad = 1;    // carrera benigna: todos escriben 1
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
                const float up = tc_to_float(in[idx2d(x, y - 1, nx)]);
                const float down = tc_to_float(in[idx2d(x, y + 1, nx)]);
                const float left = tc_to_float(in[idx2d(x - 1, y, nx)]);
                const float right = tc_to_float(in[idx2d(x + 1, y, nx)]);
                const float center = tc_to_float(in[idx2d(x, y, nx)]);
                const float val = 0.25f * (up + down + left + right) - center;
                const int idx = idx2d(x, y, nx);
                out_tc[idx] = compensated_store<T, kKahan>(val, comp, idx);
                if (write_fp32) out_fp32[idx] = val;
                if (!isfinite(val)) blk_bad = 1;    // carrera benigna: todos escriben 1
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        reduce_and_mark_first_nonfinite(first_nf, iter, blk_bad);
    }
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
                                                 bool kahan_enabled,
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
                                                 EnergyMeasurement& out_energy) {
    const size_t count = in.size();
    float* d_in_fp32 = nullptr;
    float* d_out_fp32 = nullptr;
    T* d_in_tc = nullptr;
    T* d_out_tc = nullptr;
    T* d_identity_pos = nullptr;
    T* d_identity_neg = nullptr;
    int* d_first_nf = nullptr;
    // d_comp: residuo de Kahan por celda, en FP32, persistente entre
    // iteraciones (NO participa del ping-pong: se actualiza en sitio, ver
    // compensated_store). Solo se reserva si --kahan on; nullptr en caso
    // contrario (la instanciacion kKahan=false del kernel nunca lo toca).
    float* d_comp = nullptr;

    CHECK_CUDA(cudaMalloc(&d_in_fp32, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_fp32, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_in_tc, count * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_out_tc, count * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_identity_pos, kTile * kTile * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_identity_neg, kTile * kTile * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_first_nf, sizeof(int)));
    if (kahan_enabled) {
        CHECK_CUDA(cudaMalloc(&d_comp, count * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_comp, 0, count * sizeof(float)));
    }

    CHECK_CUDA(cudaMemcpy(d_in_fp32, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_out_fp32, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<T> identity_pos(kTile * kTile);
    std::vector<T> identity_neg(kTile * kTile);
    initialize_scaled_identity<T>(identity_pos, 0.25f);
    initialize_scaled_identity<T>(identity_neg, -1.0f);
    CHECK_CUDA(cudaMemcpy(d_identity_pos, identity_pos.data(), identity_pos.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_identity_neg, identity_neg.data(), identity_neg.size() * sizeof(T),
                          cudaMemcpyHostToDevice));

    // gridDim.x ya no es tiles_x/tiles_y (2D, 1 tile = 1 bloque): es 1D,
    // ceil(total_tiles / kWarpsPerBlock), con kWarpsPerBlock tiles por
    // bloque (ver derivacion de tile_id/tiles_x dentro del kernel, misma
    // formula de tiles_x/tiles_y que aqui). shared_bytes es dinamica (antes
    // era estatica, __shared__ T tc_tiles[...]/float out_tile[...] por
    // bloque): kWarpsPerBlock * wmma_warp_shared_bytes<T>() = 4*3584 = 14336
    // B, muy por debajo de 48 KiB (no requiere
    // cudaFuncAttributeMaxDynamicSharedMemorySize).
    const int tiles_x = (nx - 2 + kTile - 1) / kTile;
    const int tiles_y = (ny - 2 + kTile - 1) / kTile;
    const int total_tiles = tiles_x * tiles_y;
    static_assert(kWarpsPerBlock * wmma_warp_shared_bytes<T>() <= 49152,
                 "shared por bloque excede 48 KiB estaticos/dinamicos");
    dim3 block(kWarpsPerBlock * kWarpThreads);
    dim3 grid((total_tiles + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const size_t shared_bytes = static_cast<size_t>(kWarpsPerBlock) * wmma_warp_shared_bytes<T>();

    // Ambos buffers del ping-pong T arrancan como conversion completa (borde
    // incluido) del input pristino: ver comentario de la funcion.
    convert_input_to_tc<T>(d_in_fp32, d_in_tc, count);
    convert_input_to_tc<T>(d_in_fp32, d_out_tc, count);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Elige la instanciacion kKahan={true,false} del kernel en tiempo de
    // compilacion segun el flag runtime --kahan: kahan_enabled no cambia
    // dentro de esta llamada, asi que el branch se resuelve una vez por
    // benchmark, no por lanzamiento. Cuando kahan_enabled es false, d_comp es
    // nullptr y la instanciacion kKahan=false nunca lo dereferencia.
    auto launch_wmma = [&](T* in_buf, T* out_buf, int iter_num, bool write_fp32_flag) {
        if (kahan_enabled) {
            stencil2d_wmma_kernel<T, true><<<grid, block, shared_bytes>>>(
                in_buf, d_out_fp32, out_buf, d_identity_pos, d_identity_neg,
                nx, ny, iter_num, write_fp32_flag, d_first_nf, d_comp);
        } else {
            stencil2d_wmma_kernel<T, false><<<grid, block, shared_bytes>>>(
                in_buf, d_out_fp32, out_buf, d_identity_pos, d_identity_neg,
                nx, ny, iter_num, write_fp32_flag, d_first_nf, nullptr);
        }
    };

    PowerBuffer* power_buffer = power_buffer_create(0);
    const RAEnergySnapshot rapl_warmup_before = rapl_snapshot_now();
    power_buffer_start_sampling(power_buffer);

    T* tc_in = d_in_tc;
    T* tc_out = d_out_tc;
    for (int i = 0; i < kWarmupIters; ++i) {
        launch_wmma(tc_in, tc_out, i + 1, /*write_fp32_flag=*/false);
        std::swap(tc_in, tc_out);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reinicia d_in_tc, d_out_tc y d_out_fp32 al estado original: el warm-up
    // encadenado es descartable y no debe alterar el estado que vera el
    // bucle medido (necesario para que --iters 1 coincida con Fase_2/Stencil).
    convert_input_to_tc<T>(d_in_fp32, d_in_tc, count);
    convert_input_to_tc<T>(d_in_fp32, d_out_tc, count);
    CHECK_CUDA(cudaMemcpy(d_out_fp32, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reinicia el contador de overflow tras el warm-up: sus iteraciones son
    // descartables y no deben contaminar la medicion del bucle cronometrado.
    {
        const int init_val = INT_MAX;
        CHECK_CUDA(cudaMemcpy(d_first_nf, &init_val, sizeof(int), cudaMemcpyHostToDevice));
    }
    // Reinicia el residuo de Kahan tras el warm-up, igual que d_first_nf: sin
    // esto los residuos del warm-up (descartable) contaminarian el bucle
    // medido (ver bloque 2 del prompt de correccion).
    if (kahan_enabled) {
        CHECK_CUDA(cudaMemset(d_comp, 0, count * sizeof(float)));
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

    // Pares de eventos por lanzamiento del kernel WMMA (iters), sin
    // sincronizar dentro del bucle: se graban en el stream con
    // cudaEventRecord y solo se leen con cudaEventElapsedTime DESPUES de
    // timer.stop_and_elapsed_ms(), que ya sincronizo una vez al final. Ya no
    // existe un kernel de conversion separado dentro del bucle (out_tc se
    // escribe directamente desde stencil2d_wmma_kernel), asi que
    // t_conv_ms_out queda en 0: no hay nada que medir por separado.
    std::vector<cudaEvent_t> wmma_start(iters), wmma_stop(iters);
    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaEventCreate(&wmma_start[i]));
        CHECK_CUDA(cudaEventCreate(&wmma_stop[i]));
    }

    CudaEventTimer timer;
    double total_ms = 0.0;
    double checkpoint_ms_total = 0.0;
    // Igual que en la ruta GPU_FP32: energia acumulada por tramos, con cortes
    // en los mismos bloques que pausan el cronometro (ver comentario alli
    // sobre por que no basta con parar/reanudar el muestreo). La exclusion
    // solo se activa con checkpointing encendido; con checkpoint_every<=0 el
    // bucle recorre un unico tramo y el resultado es identico al anterior.
    const bool exclude_checkpoint_energy = (ckpt.checkpoint_every > 0);
    double gpu_energy_j = 0.0;
    bool gpu_energy_valid = true;
    double checkpoint_cpu_energy_j = 0.0;
    double checkpoint_pause_s = 0.0;
    auto close_energy_segment = [&]() {
        power_buffer_stop_sampling(power_buffer);
        gpu_energy_valid = gpu_energy_valid && power_buffer_capture_valid(power_buffer);
        gpu_energy_j += power_buffer_energy_joules(power_buffer);
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
    for (int i = 0; i < iters; ++i) {
        // write_fp32 solo en la ultima iteracion medida o en un checkpoint:
        // es lo unico que necesita d_out_fp32 (comparacion final de error,
        // o CSV_DRIFT contra el snapshot FP64 de esta iteracion).
        const bool write_fp32 = (i + 1 == iters) ||
                                (ckpt.checkpoint_every > 0 && (i + 1) % ckpt.checkpoint_every == 0);
        CHECK_CUDA(cudaEventRecord(wmma_start[i]));
        launch_wmma(tc_in, tc_out, i + 1, write_fp32);
        CHECK_CUDA(cudaEventRecord(wmma_stop[i]));
        std::swap(tc_in, tc_out);

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
            std::chrono::steady_clock::time_point pause_t0;
            RAEnergySnapshot rapl_ckpt_before{};
            if (exclude_checkpoint_energy) {
                close_energy_segment();
                pause_t0 = std::chrono::steady_clock::now();
                rapl_ckpt_before = rapl_snapshot_now();
            }

            const auto ckpt_t0 = std::chrono::high_resolution_clock::now();
            // Una sola copia D2H de d_out_fp32, reutilizada tanto para
            // record_checkpoint (si esta iteracion es multiplo de
            // checkpoint_every) como para el rastreo de ultima-iteracion-finita
            // de abajo: antes eran dos copias identicas seguidas al mismo buffer.
            CHECK_CUDA(cudaMemcpy(checkpoint_host_buf.data(), d_out_fp32,
                                  count * sizeof(float), cudaMemcpyDeviceToHost));
            if (ckpt.checkpoint_every > 0 && (i + 1) % ckpt.checkpoint_every == 0) {
                record_checkpoint(ckpt, route_label, i + 1, checkpoint_host_buf, onset_iter);
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
                last_finite_iter = i + 1;
            }
            const auto ckpt_t1 = std::chrono::high_resolution_clock::now();
            checkpoint_ms_total +=
                std::chrono::duration<double, std::milli>(ckpt_t1 - ckpt_t0).count();

            if (exclude_checkpoint_energy) {
                const RAEnergySnapshot rapl_ckpt_after = rapl_snapshot_now();
                checkpoint_cpu_energy_j += rapl_energy_delta(rapl_ckpt_before, rapl_ckpt_after);
                power_buffer_start_sampling(power_buffer);
                checkpoint_pause_s += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - pause_t0).count();
            }

            timer.start();
        }
    }
    total_ms += timer.stop_and_elapsed_ms();
    close_energy_segment();
    const RAEnergySnapshot rapl_after = rapl_snapshot_now();
    const auto energy_t1 = std::chrono::steady_clock::now();
    emit_csv_region_marker(route_label, "end");
    // Tiempo de la ventana de energia = pared total menos los tramos de
    // checkpoint, para que avg_power_w/edp_j_s usen el mismo intervalo sobre
    // el que se integro gpu_energy_j.
    const double energy_wall_s = std::max(
        0.0, std::chrono::duration<double>(energy_t1 - energy_t0).count() - checkpoint_pause_s);
    const double flops_total = 9.0 * static_cast<double>(nx - 2) *
                               static_cast<double>(ny - 2) * iters;
    const bool cpu_energy_valid = rapl_before.valid && rapl_after.valid &&
                                  rapl_after.energy_j >= rapl_before.energy_j;
    const double cpu_energy_j = std::max(
        0.0, rapl_energy_delta(rapl_before, rapl_after) - checkpoint_cpu_energy_j);
    out_energy = make_energy_measurement_from_segments(
        gpu_energy_valid, gpu_energy_j, cpu_energy_valid, cpu_energy_j,
        energy_wall_s, flops_total);
    power_buffer_destroy(power_buffer);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(out.data(), d_out_fp32, count * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&first_nonfinite_iter, d_first_nf, sizeof(int), cudaMemcpyDeviceToHost));

    // out/out_reduced quedan SIEMPRE con la ultima iteracion medida cruda,
    // aunque contenga inf/NaN: el llamador los compara contra la referencia
    // FP64 de esa MISMA iteracion (sustituir por un estado de una iteracion
    // anterior invalidaba la metrica de error, ver diagnostico del bloque 1).
    // Quien necesite un estado recuperable (storage_rel) usa
    // out_last_finite_o / out_reduced_last_finite_o, expuestos aparte.
    out_last_finite_o = std::move(out_last_finite);
    out_reduced_last_finite_o = std::move(out_reduced_last_finite);

    double t_wmma_sum_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, wmma_start[i], wmma_stop[i]));
        t_wmma_sum_ms += ms;
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

    CHECK_CUDA(cudaFree(d_in_fp32));
    CHECK_CUDA(cudaFree(d_out_fp32));
    CHECK_CUDA(cudaFree(d_in_tc));
    CHECK_CUDA(cudaFree(d_out_tc));
    CHECK_CUDA(cudaFree(d_identity_pos));
    CHECK_CUDA(cudaFree(d_identity_neg));
    CHECK_CUDA(cudaFree(d_first_nf));
    if (d_comp != nullptr) {
        CHECK_CUDA(cudaFree(d_comp));
    }

    return build_metrics(nx, ny, total_ms / iters);
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

static void print_energy_metrics(const EnergyMeasurement& energy) {
    std::cout << "Energy GPU    : " << energy_csv_field(energy.gpu_valid, energy.energy_gpu_j) << " J\n";
    std::cout << "Energy CPU    : " << energy_csv_field(energy.cpu_valid, energy.energy_cpu_j) << " J\n";
    const bool total_valid = energy.gpu_valid && energy.cpu_valid;
    std::cout << "Energy total  : " << energy_csv_field(total_valid, energy.energy_total_j) << " J\n";
    std::cout << "EDP           : " << energy_csv_field(total_valid, energy.edp_j_s) << " J s\n";
    std::cout << "Joules/GFLOP  : " << energy_csv_field(total_valid, energy.joules_per_gflop) << "\n";
}

static void emit_csv_energy_row(const char* route,
                                int nx,
                                int ny,
                                int iters,
                                bool kahan,
                                const EnergyMeasurement& energy,
                                double flops_total) {
    const bool total_valid = energy.gpu_valid && energy.cpu_valid;
    std::cout << "CSV_ENERGY," << route << "," << nx << "," << ny << "," << iters << ","
              << kahan_label(kahan) << ","
              << energy_csv_field(energy.gpu_valid, energy.energy_gpu_j) << ","
              << energy_csv_field(energy.cpu_valid, energy.energy_cpu_j) << ","
              << energy_csv_field(total_valid, energy.energy_total_j) << ","
              << energy_csv_field(total_valid, energy.edp_j_s) << ","
              << energy_csv_field(total_valid, energy.joules_per_gflop) << ","
              << energy_csv_field(std::isfinite(energy.time_total_s), energy.time_total_s) << ","
              << energy_csv_field(std::isfinite(flops_total), flops_total / 1e9) << "\n";
}

static void emit_csv_summary_row(const char* route,
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
                                 const EnergyMeasurement& energy) {
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
static const char* kCsvHeader =
    "kernel,formato,kahan,nx,ny,iters,t_ms_iter,t_ms_total,t_ms_iter_wmma,t_ms_iter_conv,"
    "t_ms_iter_ckpt,gflops_utiles,rel_l2,rel_linf,linf_abs,ref_linf,rel_l2_prop,"
    "rel_linf_prop,n_star,storage_rel_err,energy_j,avg_power_w,edp\n";

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
static void write_csv_row(std::ofstream& csv, const std::string& formato, bool kahan, int nx, int ny,
                          int iters, double t_ms_iter, double gflops, const ErrorMetrics& e,
                          int first_nf, const std::string& storage_rel_err,
                          const std::string& t_ms_iter_wmma = "NA",
                          const std::string& t_ms_iter_conv = "NA",
                          const std::string& t_ms_iter_ckpt = "NA",
                          const std::string& rel_l2_prop = "NA",
                          const std::string& rel_linf_prop = "NA",
                          const std::string& energy_j = "NA",
                          const std::string& avg_power_w = "NA",
                          const std::string& edp = "NA") {
    const int n_star = (first_nf == INT_MAX) ? -1 : first_nf;
    csv << "stencil," << formato << "," << (kahan ? 1 : 0) << "," << nx << "," << ny << ","
        << iters << "," << fmt_sci(t_ms_iter) << "," << fmt_sci(t_ms_iter * iters) << ","
        << t_ms_iter_wmma << "," << t_ms_iter_conv << "," << t_ms_iter_ckpt << ","
        << fmt_sci(gflops) << "," << fmt_sci(e.rel_l2) << "," << fmt_sci(e.rel_linf) << ","
        << fmt_sci(e.max_abs) << "," << fmt_sci(e.ref_linf) << "," << rel_l2_prop << ","
        << rel_linf_prop << "," << n_star << "," << storage_rel_err << ","
        << energy_j << "," << avg_power_w << "," << edp << "\n";
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

// Prediccion de horizonte de overflow por formato, mas el ajuste que la
// calibro (ver OverflowFitResult). Si fit.valid es false no hay prediccion:
// pred_* quedan en 0.0 y el llamador (print_overflow_horizon) no debe
// imprimirlos.
struct OverflowHorizonPrediction {
    OverflowFitResult fit;
    double pred_fp16 = 0.0;
    double pred_bf16 = 0.0;
    double pred_fp32 = 0.0;
    double pred_fp64 = 0.0;
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
        double u0_linf) {
    OverflowHorizonPrediction result;
    result.fit = fit_overflow_model(linf_per_iter);
    if (!result.fit.valid) {
        return result;  // sin ajuste: ninguna prediccion es confiable
    }

    const double A = result.fit.A;
    const double semilla_fp16 = std::max(A, kFp16SeedFloor * u0_linf);
    const double semilla_bf16 = std::max(A, kBf16SeedFloor * u0_linf);
    const double semilla_fp32 = std::max(A, kFp32SeedFloor * u0_linf);
    const double semilla_fp64 = std::max(A, kFp64SeedFloor * u0_linf);

    // Calculos en espacio logaritmico, nunca divide (previene overflow en FP64).
    // La formula asume implicitamente log2(lambda) = 1 (lambda = 2.0, el
    // valor teorico del operador en (pi,pi)): si el ajuste diverge de eso,
    // print_overflow_horizon emite ADVERTENCIA pero la prediccion sigue
    // usando 2.0, nunca el lambda_medido fuera de rango.
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
                                   int n_fp16, int n_bf16, int n_gpu_fp32, int n_fp64) {
    std::cout << "=========== HORIZONTE DE OVERFLOW (Fase 3) ===========\n";
    const OverflowFitResult& fit = horizon.fit;
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

    if (std::fabs(fit.lambda - 2.0) / 2.0 > 0.05) {
        std::cout << "  ADVERTENCIA: lambda_medido diverge >5% del valor teorico 2.0\n"
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
                                 double a_nyq_ic,
                                 double seed_floor) {
    const bool fit_ok = fit.valid;
    std::cout << "CSV_HORIZON," << format << "," << nx << "," << ny << "," << iters << ","
              << kahan_label(kahan) << ","
              << (fit_ok ? fmt_csv_num(predicted) : "NaN") << ","
              << csv_measured_horizon_field(measured_n) << ","
              << (fit_ok ? fmt_csv_num(fit.lambda) : "NaN") << ","
              << (fit_ok ? fmt_csv_num(fit.r_squared) : "NaN") << ","
              << fit.n_points << ","
              << (fit_ok ? fmt_csv_num(fit.A) : "NaN") << ","
              << fmt_csv_num(a_nyq_ic) << ","
              << fmt_csv_num(seed_floor) << ","
              << (fit_ok ? "ok" : "insufficient_points") << "\n";
}

static void emit_csv_horizon_rows(const OverflowHorizonPrediction& horizon,
                                  double a_nyq_ic,
                                  int nx,
                                  int ny,
                                  int iters,
                                  bool kahan,
                                  int n_fp16,
                                  int n_bf16,
                                  int n_gpu_fp32,
                                  int n_fp64) {
    const OverflowFitResult& fit = horizon.fit;
    emit_csv_horizon_row("FP16", nx, ny, iters, kahan, horizon.pred_fp16, n_fp16,
                         fit, a_nyq_ic, kFp16SeedFloor);
    emit_csv_horizon_row("BF16", nx, ny, iters, kahan, horizon.pred_bf16, n_bf16,
                         fit, a_nyq_ic, kBf16SeedFloor);
    emit_csv_horizon_row("FP32", nx, ny, iters, kahan, horizon.pred_fp32, n_gpu_fp32,
                         fit, a_nyq_ic, kFp32SeedFloor);
    emit_csv_horizon_row("FP64", nx, ny, iters, kahan, horizon.pred_fp64, n_fp64,
                         fit, a_nyq_ic, kFp64SeedFloor);
}

static void print_configuration(const Options& opt) {
    std::cout << "================== CONFIGURACION ==================\n";
    std::cout << "Stencil                    : 2D 5-puntos\n";
    std::cout << "Dimensiones (nx, ny)       : " << opt.nx << ", " << opt.ny << "\n";
    std::cout << "Puntos interiores          : "
              << static_cast<long long>(opt.nx - 2) * static_cast<long long>(opt.ny - 2)
              << "\n";
    std::cout << "Iteraciones                : " << opt.iters << "\n";
    std::cout << "Tile Tensor Core           : 16x16 con WMMA\n";
    std::cout << "Acumulacion TC             : FP32\n";
    std::cout << "Kahan (residuo almacen.)   : " << (opt.kahan ? "on" : "off") << "\n";
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
                              TensorCoreMode tc_mode, bool kahan) {
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
              << " --kahan " << (kahan ? "on" : "off") << " --profile-only\n";
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

    const size_t count = static_cast<size_t>(opt.nx) * static_cast<size_t>(opt.ny);
    std::vector<float> input(count);
    initialize_grid(input, opt.nx, opt.ny);

    const std::vector<std::vector<double>> no_checkpoints;
    const CheckpointContext ckpt{0, no_checkpoints};

    std::vector<float> y_gpu(count, 0.0f);
    int onset_gpu_fp32 = -1;
    int first_nf_gpu_fp32 = INT_MAX;
    double t_checkpoint_ms_unused_fp32 = 0.0;
    EnergyMeasurement e_unused_fp32;
    benchmark_gpu_fp32_stencil(input, y_gpu, opt.nx, opt.ny, opt.iters,
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
        benchmark_gpu_tensor_core_stencil<__half>(input, y_tc_fp16, y_tc_fp16_reduced, opt.nx, opt.ny,
                                                  opt.iters, opt.kahan, ckpt, "WMMA_FP16", onset_fp16, first_nf_fp16,
                                                  t_wmma_ms_unused, t_conv_ms_unused, storage_rel_eval_iter_unused,
                                                  t_checkpoint_ms_unused, y_tc_fp16_last_finite_unused,
                                                  y_tc_fp16_reduced_last_finite_unused, e_unused_fp16);
    } else {
        std::vector<float> y_tc_bf16(count, 0.0f);
        std::vector<__nv_bfloat16> y_tc_bf16_reduced;
        std::vector<float> y_tc_bf16_last_finite_unused;
        std::vector<__nv_bfloat16> y_tc_bf16_reduced_last_finite_unused;
        int onset_bf16 = -1;
        int first_nf_bf16 = INT_MAX;
        EnergyMeasurement e_unused_bf16;
        benchmark_gpu_tensor_core_stencil<__nv_bfloat16>(input, y_tc_bf16, y_tc_bf16_reduced, opt.nx, opt.ny,
                                                         opt.iters, opt.kahan, ckpt, "WMMA_BF16", onset_bf16, first_nf_bf16,
                                                         t_wmma_ms_unused, t_conv_ms_unused, storage_rel_eval_iter_unused,
                                                         t_checkpoint_ms_unused, y_tc_bf16_last_finite_unused,
                                                         y_tc_bf16_reduced_last_finite_unused, e_unused_bf16);
    }
}

static void run_benchmark(const Options& opt, const char* exe_name) {
    print_configuration(opt);

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

    initialize_grid(input, opt.nx, opt.ny);

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
    compute_cpu_stencil_fp64(input_fp64, y_ref, opt.nx, opt.ny, opt.iters,
                             opt.checkpoint_every, fp64_checkpoints, linf_per_iter,
                             first_nf_fp64_ref);

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
    const CheckpointContext ckpt{opt.checkpoint_every, fp64_checkpoints};

    int first_nf_cpu = INT_MAX;
    EnergyMeasurement e_cpu;
    const Metrics cpu = benchmark_cpu_stencil(input, y_cpu, opt.nx, opt.ny, opt.iters,
                                              first_nf_cpu, e_cpu);
    int onset_gpu_fp32 = -1;
    int first_nf_gpu_fp32 = INT_MAX;
    double t_checkpoint_ms_gpu_fp32 = 0.0;
    EnergyMeasurement e_gpu_fp32;
    const Metrics gpu = benchmark_gpu_fp32_stencil(input, y_gpu, opt.nx, opt.ny, opt.iters,
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
    emit_csv_summary_row("CPU_FP32", opt.nx, opt.ny, opt.iters, opt.kahan,
                         cpu.ms, cpu.gflops, fmt_csv_num(1.0), "NaN",
                         "NaN", "NaN", "NaN", cpu_err, first_nf_cpu,
                         "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", e_cpu);
    emit_csv_energy_row("CPU_FP32", opt.nx, opt.ny, opt.iters, opt.kahan, e_cpu,
                        9.0 * static_cast<double>(opt.nx - 2) *
                        static_cast<double>(opt.ny - 2) * opt.iters);
    if (csv_enabled) {
        write_csv_row(csv, under_ncu ? "NCU_cpu_fp32" : "cpu_fp32", opt.kahan, opt.nx, opt.ny, opt.iters,
                     cpu.ms, cpu.gflops, cpu_err, first_nf_cpu, "NA");
    }

    print_reference_comparison("GPU CUDA FP32 clasico", gpu, cpu.ms, gpu_err, gpu_vs_cpu_err,
                               first_nf_gpu_fp32, opt.iters, t_checkpoint_ms_gpu_fp32);
    emit_csv_summary_row("GPU_FP32", opt.nx, opt.ny, opt.iters, opt.kahan,
                         gpu.ms, gpu.gflops, fmt_csv_num(cpu.ms / gpu.ms), fmt_csv_num(1.0),
                         "NaN", "NaN", fmt_csv_num(t_checkpoint_ms_gpu_fp32),
                         gpu_err, first_nf_gpu_fp32,
                         "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", e_gpu_fp32);
    print_energy_metrics(e_gpu_fp32);
    emit_csv_energy_row("GPU_FP32", opt.nx, opt.ny, opt.iters, opt.kahan, e_gpu_fp32,
                        9.0 * static_cast<double>(opt.nx - 2) *
                        static_cast<double>(opt.ny - 2) * opt.iters);
    if (csv_enabled) {
        // under_ncu fuerza "NA" en las 3 columnas de energia igual que ya
        // fuerza el prefijo NCU_ en el nombre de ruta: bajo el perfilador
        // t_ms_iter (y por tanto energia*tiempo) esta inflado y no es
        // comparable con una corrida limpia.
        write_csv_row(csv, under_ncu ? "NCU_gpu_fp32" : "gpu_fp32", opt.kahan, opt.nx, opt.ny, opt.iters,
                     gpu.ms, gpu.gflops, gpu_err, first_nf_gpu_fp32, "NA", "NA", "NA",
                     fmt_sci(t_checkpoint_ms_gpu_fp32), "NA", "NA",
                     energy_field(!under_ncu && e_gpu_fp32.gpu_valid, e_gpu_fp32.energy_j),
                     energy_field(!under_ncu && e_gpu_fp32.gpu_valid, e_gpu_fp32.avg_power_w),
                     energy_field(!under_ncu && e_gpu_fp32.gpu_valid, e_gpu_fp32.edp));
    }

    bool ran_fp16 = false;
    bool ran_bf16 = false;
    int onset_fp16 = -1;
    int onset_bf16 = -1;
    int first_nf_fp16 = INT_MAX;
    int first_nf_bf16 = INT_MAX;

    if (opt.tc_mode == TensorCoreMode::FP16 || opt.tc_mode == TensorCoreMode::Both) {
        ran_fp16 = true;
        double t_wmma_ms_fp16 = 0.0, t_conv_ms_fp16 = 0.0, t_checkpoint_ms_fp16 = 0.0;
        int storage_rel_eval_iter_fp16 = 0;
        std::vector<float> y_tc_fp16_last_finite;
        std::vector<__half> y_tc_fp16_reduced_last_finite;
        EnergyMeasurement e_fp16;
        const Metrics tc_fp16 = benchmark_gpu_tensor_core_stencil<__half>(
            input, y_tc_fp16, y_tc_fp16_reduced, opt.nx, opt.ny, opt.iters, opt.kahan,
            ckpt, "WMMA_FP16", onset_fp16, first_nf_fp16, t_wmma_ms_fp16, t_conv_ms_fp16,
            storage_rel_eval_iter_fp16, t_checkpoint_ms_fp16, y_tc_fp16_last_finite,
            y_tc_fp16_reduced_last_finite, e_fp16);
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
            compare_fp64_ref_vs_fp32(y_ref, reduced_to_float(y_tc_fp16_reduced));
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
        print_error_metrics("Error max abs vs FP64              : ", "Error relativo L2 vs FP64          : ",
                            "Error rel Linf vs FP64             : ", tc_fp16_err, first_nf_fp16);
        print_propagated_error_metrics(tc_fp16_prop_err, first_nf_fp16);
        print_error_metrics("Error max abs vs CPU FP32          : ", "Error relativo L2 vs CPU FP32      : ",
                            "Error rel Linf vs CPU FP32         : ", tc_fp16_vs_cpu_err, first_nf_fp16);
        print_first_nonfinite("Primera iteracion no finita        : ", first_nf_fp16, opt.iters);
        print_storage_metrics("FP16", fp16_storage_result, fp16_storage_evaluable,
                              opt.iters, 1.0e-3);
        std::cout << "\n\n";
        emit_csv_summary_row("WMMA_FP16", opt.nx, opt.ny, opt.iters, opt.kahan,
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
                             storage_eval_iter_field(fp16_storage_result, fp16_storage_evaluable), e_fp16);
        emit_csv_store_row("WMMA_FP16", opt.nx, opt.ny, opt.iters, opt.kahan,
                           fp16_storage_result, fp16_storage_evaluable, kFp16StorageUlp);
        print_energy_metrics(e_fp16);
        emit_csv_energy_row("WMMA_FP16", opt.nx, opt.ny, opt.iters, opt.kahan, e_fp16,
                            9.0 * static_cast<double>(opt.nx - 2) *
                            static_cast<double>(opt.ny - 2) * opt.iters);
        if (csv_enabled) {
            write_csv_row(csv, under_ncu ? "NCU_wmma_fp16" : "wmma_fp16", opt.kahan, opt.nx, opt.ny, opt.iters,
                         tc_fp16.ms, tc_fp16.gflops, tc_fp16_err, first_nf_fp16,
                         storage_num_field(fp16_storage_result, fp16_storage_evaluable,
                                           fp16_storage_result.rel_max_guarded),
                         fmt_sci(t_wmma_ms_fp16), fmt_sci(t_conv_ms_fp16), fmt_sci(t_checkpoint_ms_fp16),
                         fmt_sci(tc_fp16_prop_err.rel_l2), fmt_sci(tc_fp16_prop_err.rel_linf),
                         energy_field(!under_ncu && e_fp16.gpu_valid, e_fp16.energy_j),
                         energy_field(!under_ncu && e_fp16.gpu_valid, e_fp16.avg_power_w),
                         energy_field(!under_ncu && e_fp16.gpu_valid, e_fp16.edp));
        }
    }

    if (opt.tc_mode == TensorCoreMode::BF16 || opt.tc_mode == TensorCoreMode::Both) {
        ran_bf16 = true;
        double t_wmma_ms_bf16 = 0.0, t_conv_ms_bf16 = 0.0, t_checkpoint_ms_bf16 = 0.0;
        int storage_rel_eval_iter_bf16 = 0;
        std::vector<float> y_tc_bf16_last_finite;
        std::vector<__nv_bfloat16> y_tc_bf16_reduced_last_finite;
        EnergyMeasurement e_bf16;
        const Metrics tc_bf16 = benchmark_gpu_tensor_core_stencil<__nv_bfloat16>(
            input, y_tc_bf16, y_tc_bf16_reduced, opt.nx, opt.ny, opt.iters, opt.kahan,
            ckpt, "WMMA_BF16", onset_bf16, first_nf_bf16, t_wmma_ms_bf16, t_conv_ms_bf16,
            storage_rel_eval_iter_bf16, t_checkpoint_ms_bf16, y_tc_bf16_last_finite,
            y_tc_bf16_reduced_last_finite, e_bf16);
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
            compare_fp64_ref_vs_fp32(y_ref, reduced_to_float(y_tc_bf16_reduced));
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
        print_error_metrics("Error max abs vs FP64              : ", "Error relativo L2 vs FP64          : ",
                            "Error rel Linf vs FP64             : ", tc_bf16_err, first_nf_bf16);
        print_propagated_error_metrics(tc_bf16_prop_err, first_nf_bf16);
        print_error_metrics("Error max abs vs CPU FP32          : ", "Error relativo L2 vs CPU FP32      : ",
                            "Error rel Linf vs CPU FP32         : ", tc_bf16_vs_cpu_err, first_nf_bf16);
        print_first_nonfinite("Primera iteracion no finita        : ", first_nf_bf16, opt.iters);
        print_storage_metrics("BF16", bf16_storage_result, bf16_storage_evaluable,
                              opt.iters, 8.0e-3);
        std::cout << "\n\n";
        emit_csv_summary_row("WMMA_BF16", opt.nx, opt.ny, opt.iters, opt.kahan,
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
                             storage_eval_iter_field(bf16_storage_result, bf16_storage_evaluable), e_bf16);
        emit_csv_store_row("WMMA_BF16", opt.nx, opt.ny, opt.iters, opt.kahan,
                           bf16_storage_result, bf16_storage_evaluable, kBf16StorageUlp);
        print_energy_metrics(e_bf16);
        emit_csv_energy_row("WMMA_BF16", opt.nx, opt.ny, opt.iters, opt.kahan, e_bf16,
                            9.0 * static_cast<double>(opt.nx - 2) *
                            static_cast<double>(opt.ny - 2) * opt.iters);
        if (csv_enabled) {
            write_csv_row(csv, under_ncu ? "NCU_wmma_bf16" : "wmma_bf16", opt.kahan, opt.nx, opt.ny, opt.iters,
                         tc_bf16.ms, tc_bf16.gflops, tc_bf16_err, first_nf_bf16,
                         storage_num_field(bf16_storage_result, bf16_storage_evaluable,
                                           bf16_storage_result.rel_max_guarded),
                         fmt_sci(t_wmma_ms_bf16), fmt_sci(t_conv_ms_bf16), fmt_sci(t_checkpoint_ms_bf16),
                         fmt_sci(tc_bf16_prop_err.rel_l2), fmt_sci(tc_bf16_prop_err.rel_linf),
                         energy_field(!under_ncu && e_bf16.gpu_valid, e_bf16.energy_j),
                         energy_field(!under_ncu && e_bf16.gpu_valid, e_bf16.avg_power_w),
                         energy_field(!under_ncu && e_bf16.gpu_valid, e_bf16.edp));
        }
    }

    std::cout << "====================================================\n\n";

    // Calibracion del horizonte de overflow desde la referencia FP64: ajuste
    // de modelo log-lineal en el regimen asintotico (ver OverflowFitResult).
    const OverflowHorizonPrediction horizon =
        compute_overflow_horizon_from_reference(linf_per_iter, u0_linf);

    print_overflow_horizon(horizon, a_nyq,
                          first_nf_fp16, first_nf_bf16, first_nf_gpu_fp32, first_nf_fp64_ref);
    emit_csv_horizon_rows(horizon, a_nyq, opt.nx, opt.ny, opt.iters, opt.kahan,
                          first_nf_fp16, first_nf_bf16, first_nf_gpu_fp32, first_nf_fp64_ref);

    if (opt.checkpoint_every > 0) {
        std::cout << "=========== RESUMEN ONSET DE DIVERGENCIA ===========\n";
        std::cout << "CSV_ONSET,GPU_FP32," << onset_gpu_fp32 << "\n";
        if (ran_fp16) {
            std::cout << "CSV_ONSET,WMMA_FP16," << onset_fp16 << "\n";
        }
        if (ran_bf16) {
            std::cout << "CSV_ONSET,WMMA_BF16," << onset_bf16 << "\n";
        }
        std::cout << "=====================================================\n\n";
    }

    print_nsight_hint(exe_name, opt.nx, opt.ny, opt.iters, opt.tc_mode, opt.kahan);
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
