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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

#include "../../Fase_2/common.cuh"

using namespace nvcuda;

constexpr int kTile = 16;
constexpr int kWarpThreads = 32;
constexpr int kWarmupIters = 3;
constexpr int kConversionThreads = 256;

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
};

__host__ __device__ inline int idx2d(int x, int y, int nx) {
    return y * nx + x;
}

static void print_usage(const char* prog) {
    std::cout
        << "Uso:\n"
        << "  " << prog << " [--nx NX] [--ny NY] [--iters I] [--tc fp16|bf16|both]\n\n"
        << "Descripcion:\n"
        << "  Compara CPU FP32, GPU CUDA FP32 y GPU WMMA Tensor Core para stencil 2D.\n"
        << "  La ruta Tensor Core usa operandos FP16/BF16 y acumulacion/salida FP32.\n\n"
        << "Ejemplos:\n"
        << "  " << prog << "\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 20 --tc fp16\n"
        << "  " << prog << " --nx 4096 --ny 4096 --iters 20 --tc bf16\n";
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
                                     int iters) {
    auto apply = [&](const std::vector<float>& src, std::vector<float>& dst) {
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
                dst[idx2d(x, y, nx)] = 0.25f * (up + down + left + right) - center;
            }
        }
    };

    {
        std::vector<float> warm_a = in;
        std::vector<float> warm_b = in;
        std::vector<float>* warm_src = &warm_a;
        std::vector<float>* warm_dst = &warm_b;
        for (int i = 0; i < kWarmupIters; ++i) {
            apply(*warm_src, *warm_dst);
            std::swap(warm_src, warm_dst);
        }
    }

    std::vector<float> buf_a = in;
    std::vector<float> buf_b = in;
    std::vector<float>* src = &buf_a;
    std::vector<float>* dst = &buf_b;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        apply(*src, *dst);
        std::swap(src, dst);
    }
    auto end = std::chrono::high_resolution_clock::now();

    const double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    out = *src;
    return build_metrics(nx, ny, avg_ms);
}

// Referencia FP64 (ground truth): version double de benchmark_cpu_stencil,
// encadenada por el MISMO numero de iteraciones (iters) que las rutas
// comparadas -condicion de aceptacion de Fase 3: sin esto el error vs FP64
// quedaria invalido para iters>1 (N pasos encadenados contra 1 solo paso)-.
// Opera sobre una copia en double del mismo input FP32, sin medir tiempo.
static void compute_cpu_stencil_fp64(const std::vector<double>& in,
                                     std::vector<double>& out,
                                     int nx,
                                     int ny,
                                     int iters) {
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
    for (int i = 0; i < iters; ++i) {
        apply(*src, *dst);
        std::swap(src, dst);
    }
    out = *src;
}

__global__ static void stencil2d_fp32_kernel(const float* in, float* out, int nx, int ny) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
        out[idx2d(x, y, nx)] = in[idx2d(x, y, nx)];
        return;
    }

    const float up = in[idx2d(x, y - 1, nx)];
    const float down = in[idx2d(x, y + 1, nx)];
    const float left = in[idx2d(x - 1, y, nx)];
    const float right = in[idx2d(x + 1, y, nx)];
    const float center = in[idx2d(x, y, nx)];
    out[idx2d(x, y, nx)] = 0.25f * (up + down + left + right) - center;
}

static Metrics benchmark_gpu_fp32_stencil(const std::vector<float>& in,
                                          std::vector<float>& out,
                                          int nx,
                                          int ny,
                                          int iters) {
    const size_t count = in.size();
    float* d_a = nullptr;
    float* d_b = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, count * sizeof(float)));
    // Ambos buffers arrancan como copia completa del input: el kernel nunca
    // escribe las celdas de borde, asi que deben preservarse desde el inicio
    // en cualquier buffer que llegue a jugar el rol de d_out.
    CHECK_CUDA(cudaMemcpy(d_a, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // Warm-up encadenado y descartable: alterna d_a/d_b igual que el bucle
    // medido, pero al terminar se restauran ambos a una copia fresca del
    // input para que el bucle medido siempre arranque desde el estado
    // original (necesario para que --iters 1 coincida con Fase_2/Stencil).
    float* warm_in = d_a;
    float* warm_out = d_b;
    for (int i = 0; i < kWarmupIters; ++i) {
        stencil2d_fp32_kernel<<<grid, block>>>(warm_in, warm_out, nx, ny);
        std::swap(warm_in, warm_out);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(d_a, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    float* d_in = d_a;
    float* d_out = d_b;
    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        stencil2d_fp32_kernel<<<grid, block>>>(d_in, d_out, nx, ny);
        std::swap(d_in, d_out);
    }
    const float total_ms = timer.stop_and_elapsed_ms();
    CHECK_CUDA(cudaGetLastError());
    // Tras el ultimo swap, d_in apunta al buffer con la salida mas reciente.
    CHECK_CUDA(cudaMemcpy(out.data(), d_in, count * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    return build_metrics(nx, ny, static_cast<double>(total_ms) / iters);
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

template <typename T>
__global__ static void stencil2d_wmma_kernel(const T* in,
                                             float* out,
                                             const T* identity_pos,
                                             const T* identity_neg,
                                             int nx,
                                             int ny) {
    const int x0 = 1 + blockIdx.x * kTile;
    const int y0 = 1 + blockIdx.y * kTile;
    const bool full_tile = (x0 + kTile - 1 < nx - 1) && (y0 + kTile - 1 < ny - 1);

    if (full_tile) {
        __shared__ __align__(32) T tc_tiles[5 * kTile * kTile];
        __shared__ __align__(32) float out_tile[kTile * kTile];

        T* left_tile = tc_tiles + 0 * kTile * kTile;
        T* right_tile = tc_tiles + 1 * kTile * kTile;
        T* up_tile = tc_tiles + 2 * kTile * kTile;
        T* down_tile = tc_tiles + 3 * kTile * kTile;
        T* center_tile = tc_tiles + 4 * kTile * kTile;

        for (int linear = threadIdx.x; linear < kTile * kTile; linear += blockDim.x) {
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

        for (int linear = threadIdx.x; linear < kTile * kTile; linear += blockDim.x) {
            const int local_x = linear % kTile;
            const int local_y = linear / kTile;
            out[idx2d(x0 + local_x, y0 + local_y, nx)] = out_tile[linear];
        }
        return;
    }

    for (int linear = threadIdx.x; linear < kTile * kTile; linear += blockDim.x) {
        const int local_x = linear % kTile;
        const int local_y = linear / kTile;
        const int x = x0 + local_x;
        const int y = y0 + local_y;

        // Esta guarda cubre, a la vez, dos casos distintos: el borde fisico
        // de la grilla (x/y == 0 o == nx-1/ny-1, que nunca se recalcula) y
        // los indices que caen fuera de rango porque este es el ultimo tile
        // parcial de la fila/columna (x0/y0 + kTile puede exceder nx-1/ny-1
        // cuando nx-2 o ny-2 no son multiplos de kTile). Ambos casos se
        // resuelven igual: no escribir out[] para ese hilo.
        if (x <= 0 || y <= 0 || x >= nx - 1 || y >= ny - 1) {
            continue;
        }

        const float up = tc_to_float(in[idx2d(x, y - 1, nx)]);
        const float down = tc_to_float(in[idx2d(x, y + 1, nx)]);
        const float left = tc_to_float(in[idx2d(x - 1, y, nx)]);
        const float right = tc_to_float(in[idx2d(x + 1, y, nx)]);
        const float center = tc_to_float(in[idx2d(x, y, nx)]);
        out[idx2d(x, y, nx)] = 0.25f * (up + down + left + right) - center;
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

// Mide cuanto se pierde SOLO por el hecho de guardar el resultado WMMA
// (que internamente ya vive en float, via el acumulador de Tensor Cores)
// en 16 bits. Antes de este cambio, ese resultado nunca se guardaba de
// verdad en __half/__nv_bfloat16 -- por eso no se podia medir.
template <typename T>
static double storage_roundtrip_max_abs(const std::vector<float>& computed,
                                        const std::vector<T>& stored) {
    double max_abs = 0.0;
    for (size_t i = 0; i < computed.size(); ++i) {
        double diff = std::fabs(static_cast<double>(computed[i]) -
                                static_cast<double>(host_val_to_float(stored[i])));
        max_abs = std::max(max_abs, diff);
    }
    return max_abs;
}

// Encadenamiento genuino salida(i) -> entrada(i+1): el kernel WMMA consume T
// (half/bfloat16) pero produce float (acumulador Tensor Core), asi que cada
// iteracion > 1 debe reconvertir la salida float de la iteracion anterior a T
// antes de usarla como entrada de la siguiente (reutiliza convert_input_to_tc).
// El warm-up encadena de la misma forma pero es descartable: al terminar se
// reconvierte d_in_fp32 (nunca modificado) para reiniciar d_in_tc, y se
// restaura d_out con una copia fresca de in via cudaMemcpy, igual que en
// benchmark_gpu_fp32_stencil, para que el bucle medido siempre arranque
// desde el estado original (necesario para que --iters 1 coincida con
// Fase_2/Stencil). Este reset explicito no depende de que el kernel deje
// intactas las celdas de borde: es robusto aunque esa logica cambie.
template <typename T>
static Metrics benchmark_gpu_tensor_core_stencil(const std::vector<float>& in,
                                                 std::vector<float>& out,
                                                 std::vector<T>& out_reduced,
                                                 int nx,
                                                 int ny,
                                                 int iters) {
    const size_t count = in.size();
    float* d_in_fp32 = nullptr;
    float* d_out = nullptr;
    T* d_in_tc = nullptr;
    T* d_out_reduced = nullptr;
    T* d_identity_pos = nullptr;
    T* d_identity_neg = nullptr;

    CHECK_CUDA(cudaMalloc(&d_in_fp32, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_in_tc, count * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_out_reduced, count * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_identity_pos, kTile * kTile * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_identity_neg, kTile * kTile * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(d_in_fp32, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_out, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<T> identity_pos(kTile * kTile);
    std::vector<T> identity_neg(kTile * kTile);
    initialize_scaled_identity<T>(identity_pos, 0.25f);
    initialize_scaled_identity<T>(identity_neg, -1.0f);
    CHECK_CUDA(cudaMemcpy(d_identity_pos, identity_pos.data(), identity_pos.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_identity_neg, identity_neg.data(), identity_neg.size() * sizeof(T),
                          cudaMemcpyHostToDevice));

    dim3 block(kWarpThreads);
    dim3 grid((nx - 2 + kTile - 1) / kTile, (ny - 2 + kTile - 1) / kTile);

    convert_input_to_tc<T>(d_in_fp32, d_in_tc, count);
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < kWarmupIters; ++i) {
        stencil2d_wmma_kernel<T><<<grid, block>>>(d_in_tc, d_out, d_identity_pos,
                                                  d_identity_neg, nx, ny);
        if (i + 1 < kWarmupIters) {
            convert_input_to_tc<T>(d_out, d_in_tc, count);
        }
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reinicia d_in_tc y d_out al estado original: el warm-up encadenado es
    // descartable y no debe alterar el estado que vera el bucle medido
    // (necesario para que --iters 1 coincida con Fase_2/Stencil).
    convert_input_to_tc<T>(d_in_fp32, d_in_tc, count);
    CHECK_CUDA(cudaMemcpy(d_out, in.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.start();
    for (int i = 0; i < iters; ++i) {
        stencil2d_wmma_kernel<T><<<grid, block>>>(d_in_tc, d_out, d_identity_pos,
                                                  d_identity_neg, nx, ny);
        if (i + 1 < iters) {
            convert_input_to_tc<T>(d_out, d_in_tc, count);
        }
    }
    const float total_ms = timer.stop_and_elapsed_ms();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost));

    // NUEVO: el resultado WMMA solo existia en float (d_out). Se reutiliza
    // el mismo conversor ya usado para la entrada (convert_input_to_tc)
    // para tambien convertir y almacenar la salida en el tipo reducido:
    // asi el resultado queda realmente guardado en FP16/BF16, no solo
    // calculado internamente en float.
    convert_input_to_tc<T>(d_out, d_out_reduced, count);
    CHECK_CUDA(cudaDeviceSynchronize());
    out_reduced.resize(count);
    CHECK_CUDA(cudaMemcpy(out_reduced.data(), d_out_reduced, count * sizeof(T), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_in_fp32));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_in_tc));
    CHECK_CUDA(cudaFree(d_out_reduced));
    CHECK_CUDA(cudaFree(d_identity_pos));
    CHECK_CUDA(cudaFree(d_identity_neg));

    return build_metrics(nx, ny, static_cast<double>(total_ms) / iters);
}

static void print_reference_comparison(const char* label,
                                       const Metrics& m,
                                       double ref_ms,
                                       const ErrorMetrics& e_fp64,
                                       const ErrorMetrics& e_cpu) {
    std::cout << label << " - tiempo         : " << m.ms << " ms\n";
    std::cout << label << " - rendimiento    : " << m.gflops << " GFLOP/s ("
              << m.tflops << " TFLOP/s efectivos)\n";
    std::cout << "Speedup vs CPU             : " << ref_ms / m.ms << "x\n";
    std::cout << "Error max abs vs FP64      : " << e_fp64.max_abs << "\n";
    std::cout << "Error relativo L2 vs FP64  : " << e_fp64.rel_l2 << "\n";
    std::cout << "Error max abs vs CPU FP32  : " << e_cpu.max_abs << "\n";
    std::cout << "Error rel L2 vs CPU FP32   : " << e_cpu.rel_l2 << "\n\n";
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
    std::cout << "===================================================\n\n";
}

static void print_nsight_hint(const char* exe_name) {
    std::cout << "Validacion Nsight Compute:\n";
    std::cout << "  ncu --kernel-name regex:.*stencil2d_wmma_kernel.* \\\n";
    std::cout << "      --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed,"
              << "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed \\\n";
    std::cout << "      " << exe_name << " --nx 4096 --ny 4096 --iters 20 --tc fp16\n";
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

    const size_t count = static_cast<size_t>(opt.nx) * static_cast<size_t>(opt.ny);
    std::vector<float> input(count);
    std::vector<float> y_cpu(count, 0.0f);
    std::vector<float> y_gpu(count, 0.0f);
    std::vector<float> y_tc_fp16(count, 0.0f);
    std::vector<float> y_tc_bf16(count, 0.0f);
    std::vector<__half> y_tc_fp16_reduced;
    std::vector<__nv_bfloat16> y_tc_bf16_reduced;

    initialize_grid(input, opt.nx, opt.ny);

    // Referencia FP64 (ground truth): opt.iters aplicaciones encadenadas del
    // stencil en double sobre una copia en double del mismo input, mismo
    // numero de iteraciones que las rutas comparadas (ver comentario en
    // compute_cpu_stencil_fp64).
    std::vector<double> input_fp64(count);
    std::vector<double> y_ref(count, 0.0);
    for (size_t i = 0; i < count; ++i) {
        input_fp64[i] = static_cast<double>(input[i]);
    }
    compute_cpu_stencil_fp64(input_fp64, y_ref, opt.nx, opt.ny, opt.iters);

    const Metrics cpu = benchmark_cpu_stencil(input, y_cpu, opt.nx, opt.ny, opt.iters);
    const Metrics gpu = benchmark_gpu_fp32_stencil(input, y_gpu, opt.nx, opt.ny, opt.iters);
    // Metrica primaria: contra el ground truth FP64 (objetivo especifico #3);
    // secundaria: contra la CPU FP32 (trazabilidad con corridas previas).
    const ErrorMetrics cpu_err        = compare_fp64_ref_vs_fp32(y_ref, y_cpu);
    const ErrorMetrics gpu_err        = compare_fp64_ref_vs_fp32(y_ref, y_gpu);
    const ErrorMetrics gpu_vs_cpu_err = compare_float_vectors(y_cpu, y_gpu);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=========== RESULTADOS STENCIL 2D FASE 3 ===========\n";
    std::cout << "CPU FP32 serial - tiempo   : " << cpu.ms << " ms\n";
    std::cout << "CPU FP32 serial - rend.    : " << cpu.gflops << " GFLOP/s ("
              << cpu.tflops << " TFLOP/s efectivos)\n";
    std::cout << "Error max abs vs FP64      : " << cpu_err.max_abs << "\n";
    std::cout << "Error relativo L2 vs FP64  : " << cpu_err.rel_l2 << "\n\n";

    print_reference_comparison("GPU CUDA FP32 clasico", gpu, cpu.ms, gpu_err, gpu_vs_cpu_err);

    if (opt.tc_mode == TensorCoreMode::FP16 || opt.tc_mode == TensorCoreMode::Both) {
        const Metrics tc_fp16 = benchmark_gpu_tensor_core_stencil<__half>(
            input, y_tc_fp16, y_tc_fp16_reduced, opt.nx, opt.ny, opt.iters);
        const ErrorMetrics tc_fp16_err        = compare_fp64_ref_vs_fp32(y_ref, y_tc_fp16);
        const ErrorMetrics tc_fp16_vs_cpu_err = compare_float_vectors(y_cpu, y_tc_fp16);
        const double fp16_storage_err = storage_roundtrip_max_abs<__half>(y_tc_fp16, y_tc_fp16_reduced);

        std::cout << "GPU WMMA FP16 Tensor Core - tiempo : " << tc_fp16.ms << " ms\n";
        std::cout << "GPU WMMA FP16 Tensor Core - rend.  : " << tc_fp16.gflops
                  << " GFLOP/s (" << tc_fp16.tflops << " TFLOP/s efectivos)\n";
        std::cout << "Speedup TC FP16 vs CPU             : " << cpu.ms / tc_fp16.ms << "x\n";
        std::cout << "Speedup TC FP16 vs GPU FP32        : " << gpu.ms / tc_fp16.ms << "x\n";
        std::cout << "Error max abs vs FP64              : " << tc_fp16_err.max_abs << "\n";
        std::cout << "Error relativo L2 vs FP64          : " << tc_fp16_err.rel_l2 << "\n";
        std::cout << "Error max abs vs CPU FP32          : " << tc_fp16_vs_cpu_err.max_abs << "\n";
        std::cout << "Error relativo L2 vs CPU FP32      : " << tc_fp16_vs_cpu_err.rel_l2 << "\n";
        std::cout << "Error por guardar en FP16 (16 bits): " << fp16_storage_err << "\n\n";
    }

    if (opt.tc_mode == TensorCoreMode::BF16 || opt.tc_mode == TensorCoreMode::Both) {
        const Metrics tc_bf16 = benchmark_gpu_tensor_core_stencil<__nv_bfloat16>(
            input, y_tc_bf16, y_tc_bf16_reduced, opt.nx, opt.ny, opt.iters);
        const ErrorMetrics tc_bf16_err        = compare_fp64_ref_vs_fp32(y_ref, y_tc_bf16);
        const ErrorMetrics tc_bf16_vs_cpu_err = compare_float_vectors(y_cpu, y_tc_bf16);
        const double bf16_storage_err = storage_roundtrip_max_abs<__nv_bfloat16>(y_tc_bf16, y_tc_bf16_reduced);

        std::cout << "GPU WMMA BF16 Tensor Core - tiempo : " << tc_bf16.ms << " ms\n";
        std::cout << "GPU WMMA BF16 Tensor Core - rend.  : " << tc_bf16.gflops
                  << " GFLOP/s (" << tc_bf16.tflops << " TFLOP/s efectivos)\n";
        std::cout << "Speedup TC BF16 vs CPU             : " << cpu.ms / tc_bf16.ms << "x\n";
        std::cout << "Speedup TC BF16 vs GPU FP32        : " << gpu.ms / tc_bf16.ms << "x\n";
        std::cout << "Error max abs vs FP64              : " << tc_bf16_err.max_abs << "\n";
        std::cout << "Error relativo L2 vs FP64          : " << tc_bf16_err.rel_l2 << "\n";
        std::cout << "Error max abs vs CPU FP32          : " << tc_bf16_vs_cpu_err.max_abs << "\n";
        std::cout << "Error relativo L2 vs CPU FP32      : " << tc_bf16_vs_cpu_err.rel_l2 << "\n";
        std::cout << "Error por guardar en BF16 (16 bits): " << bf16_storage_err << "\n\n";
    }

    std::cout << "====================================================\n\n";
    print_nsight_hint(exe_name);
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);
    print_gpu_info();
    run_benchmark(opt, argv[0]);
    return 0;
}
