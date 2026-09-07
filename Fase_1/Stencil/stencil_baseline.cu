// Fase_1/Stencil/stencil_baseline.cu
//
// Compilar con:
//   nvcc -std=c++17 -O3 stencil_baseline.cu -o stencil_baseline \
//        -gencode arch=compute_80,code=sm_80
//
// Ejecutar:
//   ./stencil_baseline --nx 4096 --ny 4096 --iters 20
//   ./stencil_baseline --double --nx 8192 --ny 8192 --iters 20
//   ./stencil_baseline 16384 16384 10      (forma posicional: nx ny iters)
//
// Que compara este programa:
//   1. CPU serial (FP32 o FP64, segun --double) como referencia numerica.
//   2. GPU CUDA clasico (mismo tipo T que la CPU), sin Tensor Cores.
// No hay ruta de biblioteca (cuBLAS/cuDNN) ni ruta Tensor Core aqui: esas
// comparaciones empiezan en Fase 2 (ver Fase_2/Stencil). Fase 1 solo
// establece la linea base CPU-vs-GPU-clasico para el stencil 2D de 5 puntos.
//
// Migracion: version limpia y documentada de
// old/Fase_1/Stencil2D/stencil2d_baseline.cu, sin cambios de logica numerica.
// El unico cambio de comportamiento respecto al original es de nombres/
// organizacion (ver comentarios "MIGRACION" abajo donde aplica), no de
// aritmetica: mismos kernels, mismo esquema, mismos valores para las mismas
// entradas.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

// MIGRACION: el binario original definia su propia macro CUDA_CHECK,
// identica en mensaje y comportamiento (mismo formato de error, mismo abort)
// a CHECK_CUDA de common/cuda_checks.cuh. Se usa aqui la version compartida
// para no duplicar esa macro entre Fase 1, Fase 2 y las fases siguientes;
// no es un cambio de comportamiento, solo de donde vive la definicion.
#include "../../common/cuda_checks.cuh"
// Se usa unicamente CudaEventTimer de este header (el cronometro de eventos
// CUDA que reemplaza el manejo manual de cudaEvent_t del original). Los
// structs Metrics/ErrorMetrics de aqui NO se usan: Fase 1 necesita campos
// propios (total_ms, iters -- ver StencilMetrics abajo) y una formula de
// error relativo con un epsilon distinto (ver rel_l2_error), asi que se
// conserva su propia estructura en vez de forzarla dentro de la generica.
#include "../../common/metrics.cuh"

// Parametros de un experimento, resueltos por parse_args() a partir de flags
// de linea de comandos. Nada de esto esta fijo en el codigo: todo tamano,
// numero de iteraciones y precision se elige al invocar el binario.
struct Options {
    int nx = 4096;
    int ny = 4096;
    int iters = 10;
    bool use_double = false;
};

// Metricas de una corrida (CPU o GPU): tiempo medio de una aplicacion del
// stencil, rendimiento derivado, y el total crudo + iteraciones detras del
// promedio. Se llama StencilMetrics (no Metrics, como en common/metrics.cuh)
// porque este archivo tambien incluye common/metrics.cuh por CudaEventTimer,
// y los campos no coinciden con los de la version generica (ver total_ms/
// iters abajo, que compare_* de common no necesita).
struct StencilMetrics {
    double milliseconds = 0.0;
    double gflops = 0.0;
    // Total crudo del cronometro y numero de iteraciones promediadas. Se
    // imprimen junto al promedio para poder distinguir una coincidencia real de
    // tiempos entre binarios de un error de transcripcion al comparar tablas:
    // el promedio solo, redondeado, no permite hacer esa distincion.
    double total_ms = 0.0;
    int iters = 0;
};

// Indice lineal (row-major) de la celda (x, y) en una grilla de ancho nx.
// __host__ __device__ porque se usa tanto en el kernel GPU como en la
// referencia CPU y en la inicializacion.
__host__ __device__ inline int idx(int x, int y, int nx) {
    return y * nx + x;
}

void print_usage(const char* prog) {
    std::cout << "Uso: " << prog
              << " [--nx NX] [--ny NY] [--iters I] [--double]\n"
              << "Tambien se acepta: " << prog << " [nx] [ny] [iters]\n";
}

// Parsea argv en un Options. Acepta flags con nombre (--nx, --ny, --iters,
// --double) y, para compatibilidad con el binario original, hasta tres
// argumentos posicionales (nx ny iters). Aborta con print_usage() ante
// cualquier argumento no reconocido o exceso de posicionales.
Options parse_args(int argc, char** argv) {
    Options opt;
    int positional = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--nx") == 0 && i + 1 < argc) {
            opt.nx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ny") == 0 && i + 1 < argc) {
            opt.ny = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            opt.iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--double") == 0) {
            opt.use_double = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (argv[i][0] != '-') {
            int value = std::atoi(argv[i]);
            if (positional == 0) {
                opt.nx = value;
            } else if (positional == 1) {
                opt.ny = value;
            } else if (positional == 2) {
                opt.iters = value;
            } else {
                std::cerr << "Demasiados argumentos posicionales.\n";
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            ++positional;
        } else {
            std::cerr << "Argumento no reconocido: " << argv[i] << "\n";
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    opt.nx = std::max(3, opt.nx);
    opt.ny = std::max(3, opt.ny);
    opt.iters = std::max(1, opt.iters);
    return opt;
}

// Imprime las caracteristicas de la GPU activa (device 0). Sirve para dejar
// constancia en el log de en que hardware corrio cada resultado, sin tener
// que cruzar referencias con el nombre del nodo SLURM.
void print_gpu_info() {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::cerr << "No se detectaron GPUs CUDA." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int dev = 0;
    CHECK_CUDA(cudaGetDevice(&dev));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    std::cout << "================ CARACTERISTICAS DE LA GPU ================\n";
    std::cout << "GPUs detectadas           : " << device_count << "\n";
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
    std::cout << "Bus de memoria             : " << prop.memoryBusWidth << " bits\n";
    std::cout << "Memoria compartida/bloque  : " << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
    std::cout << "===========================================================\n\n";
}

// Stencil 2D de 5 puntos (Laplaciano discreto): cada celda interior se
// reemplaza por 0.25*(vecinos N/S/E/O) - centro. Los bordes de la grilla se
// copian sin modificar (condicion de frontera "identidad"). Templado sobre T
// para compartir el mismo codigo entre FP32 y FP64 (--double).
template <typename T>
__global__ void stencil2d_kernel(const T* in, T* out, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) {
        return;
    }

    if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
        out[idx(x, y, nx)] = in[idx(x, y, nx)];
        return;
    }

    T up = in[idx(x, y - 1, nx)];
    T down = in[idx(x, y + 1, nx)];
    T left = in[idx(x - 1, y, nx)];
    T right = in[idx(x + 1, y, nx)];
    T center = in[idx(x, y, nx)];

    out[idx(x, y, nx)] = static_cast<T>(0.25) * (up + down + left + right) - center;
}

// Genera la grilla de entrada: una onda seno/coseno de baja frecuencia mas
// una pequena perturbacion aleatoria (semilla fija = 42, para que dos
// corridas con el mismo nx/ny sean bit-a-bit comparables). No se expone la
// semilla como flag: es un dato de entrada fijo del experimento, no un
// parametro que cambie el escenario que se esta midiendo.
template <typename T>
void initialize_grid(std::vector<T>& grid, int nx, int ny) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            double wave = std::sin(0.01 * x) + std::cos(0.01 * y);
            grid[idx(x, y, nx)] = static_cast<T>(wave + 0.01 * dist(gen));
        }
    }
}

// Numero de FLOPs de una aplicacion completa del stencil: 5 operaciones
// (3 sumas + 1 multiplicacion + 1 resta) por cada punto interior. Los puntos
// de borde no cuentan porque solo se copian, no se recalculan.
double stencil_operations(int nx, int ny) {
    double interior = static_cast<double>(nx - 2) * static_cast<double>(ny - 2);
    return interior * 5.0;
}

// Referencia CPU serial: aplica el stencil `iters` veces sobre la misma
// entrada (no encadena la salida de una iteracion como entrada de la
// siguiente -- cada pasada parte de `in`), mide el tiempo total con
// std::chrono y devuelve el promedio por iteracion. Antes de cronometrar se
// hace una pasada de calentamiento para que el resultado en `out` quede
// listo para comparar sin que esa primera pasada (con cache fria) sesgue el
// promedio.
template <typename T>
StencilMetrics run_cpu_stencil(const std::vector<T>& in, std::vector<T>& out,
                        int nx, int ny, int iters) {
    auto apply_stencil = [&]() {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
                    out[idx(x, y, nx)] = in[idx(x, y, nx)];
                    continue;
                }

                T up = in[idx(x, y - 1, nx)];
                T down = in[idx(x, y + 1, nx)];
                T left = in[idx(x - 1, y, nx)];
                T right = in[idx(x + 1, y, nx)];
                T center = in[idx(x, y, nx)];
                out[idx(x, y, nx)] = static_cast<T>(0.25) * (up + down + left + right) - center;
            }
        }
    };

    apply_stencil();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        apply_stencil();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ms = total_ms / iters;
    double gflops = stencil_operations(nx, ny) / (ms * 1.0e6);
    return {ms, gflops, total_ms, iters};
}

// Ruta GPU CUDA clasica (sin Tensor Cores): copia la entrada a device, hace
// una pasada de calentamiento (para que la primera compilacion JIT/carga de
// contexto no contamine el tiempo medido), y cronometra `iters` lanzamientos
// consecutivos con CudaEventTimer (tiempo de GPU puro, sin la latencia de
// sincronizacion del lado del host que tendria envolver el lanzamiento con
// std::chrono). Las copias H2D/D2H quedan fuera de la medicion.
template <typename T>
StencilMetrics run_gpu_stencil(const std::vector<T>& in, std::vector<T>& out,
                        int nx, int ny, int iters) {
    size_t elements = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    size_t bytes = elements * sizeof(T);

    T* d_in = nullptr;
    T* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));

    // 32 hilos en x: cada fila de hilos cubre una linea de cache de 128 B (FP32)
    // o 256 B (FP64). Con el (16,16) original una fila pedia solo media linea,
    // desperdiciando la mitad de cada transaccion en un kernel que esta limitado
    // por ancho de banda de memoria.
    dim3 block(32, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    stencil2d_kernel<T><<<grid, block>>>(d_in, d_out, nx, ny);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        stencil2d_kernel<T><<<grid, block>>>(d_in, d_out, nx, ny);
    }
    const float total_ms = timer.stop_and_elapsed_ms();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    double ms = static_cast<double>(total_ms) / iters;
    double gflops = stencil_operations(nx, ny) / (ms * 1.0e6);
    return {ms, gflops, static_cast<double>(total_ms), iters};
}

// Error maximo absoluto |ref - test| sobre toda la grilla.
template <typename T>
double max_abs_diff(const std::vector<T>& ref, const std::vector<T>& test) {
    double max_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = std::abs(static_cast<double>(ref[i]) - static_cast<double>(test[i]));
        max_err = std::max(max_err, diff);
    }
    return max_err;
}

// Error relativo en norma L2: ||ref - test||_2 / ||ref||_2. El epsilon
// 1e-30 en el denominador evita una division por cero si la referencia
// fuera identicamente nula (no ocurre con la grilla de initialize_grid,
// pero deja la funcion segura para cualquier entrada).
template <typename T>
double rel_l2_error(const std::vector<T>& ref, const std::vector<T>& test) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double r = static_cast<double>(ref[i]);
        double t = static_cast<double>(test[i]);
        double d = r - t;
        num += d * d;
        den += r * r;
    }
    return std::sqrt(num) / (std::sqrt(den) + 1e-30);
}

void print_experiment_header(const Options& opt, const char* precision_name) {
    std::cout << "Configuracion del experimento\n";
    std::cout << "Precision                 : " << precision_name << "\n";
    std::cout << "Stencil                   : 2D 5-puntos\n";
    std::cout << "Dimensiones               : " << opt.nx << " x " << opt.ny << "\n";
    std::cout << "Puntos interiores         : "
              << static_cast<long long>(opt.nx - 2) * static_cast<long long>(opt.ny - 2)
              << "\n";
    std::cout << "Iteraciones promedio      : " << opt.iters << "\n";
    std::cout << "Medicion GPU              : tiempo de kernel, sin copias H2D/D2H\n";
}

// Corre el experimento completo para un tipo T (float o double): genera la
// entrada, ejecuta la referencia CPU y la ruta GPU, compara resultados y
// reporta tiempos/rendimiento/error por stdout.
template <typename T>
void run_experiment(const Options& opt, const char* precision_name) {
    size_t elements = static_cast<size_t>(opt.nx) * static_cast<size_t>(opt.ny);
    std::vector<T> input(elements);
    std::vector<T> output_cpu(elements, static_cast<T>(0));
    std::vector<T> output_gpu(elements, static_cast<T>(0));

    initialize_grid(input, opt.nx, opt.ny);
    print_experiment_header(opt, precision_name);

    StencilMetrics cpu = run_cpu_stencil(input, output_cpu, opt.nx, opt.ny, opt.iters);
    StencilMetrics gpu = run_gpu_stencil(input, output_gpu, opt.nx, opt.ny, opt.iters);

    double max_err = max_abs_diff(output_cpu, output_gpu);
    double rel_err = rel_l2_error(output_cpu, output_gpu);

    std::cout << std::fixed << std::setprecision(7);
    std::cout << "---------------- RESULTADOS ----------------\n";
    std::cout << "CPU serial - tiempo medio : " << cpu.milliseconds << " ms\n";
    std::cout << "CPU serial - rendimiento  : " << cpu.gflops << " GFLOP/s\n";
    std::cout << "GPU CUDA   - tiempo medio : " << gpu.milliseconds << " ms\n";
    std::cout << "GPU CUDA   - tiempo total : " << gpu.total_ms << " ms en "
              << gpu.iters << " iteraciones\n";
    std::cout << "GPU CUDA   - rendimiento  : " << gpu.gflops << " GFLOP/s\n";
    std::cout << "Speedup GPU/CPU           : " << (cpu.milliseconds / gpu.milliseconds) << "x\n";
    std::cout << "Error max abs             : " << std::scientific << max_err << "\n";
    std::cout << "Error relativo L2         : " << rel_err << std::fixed << "\n";
    std::cout << "--------------------------------------------\n";
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    print_gpu_info();

    if (opt.use_double) {
        run_experiment<double>(opt, "FP64");
    } else {
        run_experiment<float>(opt, "FP32");
    }

    return 0;
}
