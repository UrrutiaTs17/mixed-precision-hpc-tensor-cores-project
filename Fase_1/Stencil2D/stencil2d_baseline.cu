// nvcc stencil2d_baseline.cu -o stencil2d_baseline
// ./stencil2d_baseline --nx 4096 --ny 4096 --iters 20
// ./stencil2d_baseline --double --nx 4096 --ny 4096 --iters 20

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(err__) << std::endl;     \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

struct Options {
    int nx = 2048;
    int ny = 2048;
    int iters = 10;
    bool use_double = false;
};

struct Metrics {
    double milliseconds = 0.0;
    double gflops = 0.0;
};

__host__ __device__ inline int idx(int x, int y, int nx) {
    return y * nx + x;
}

void print_usage(const char* prog) {
    std::cout << "Uso: " << prog
              << " [--nx NX] [--ny NY] [--iters I] [--double]\n"
              << "Tambien se acepta: " << prog << " [nx] [ny] [iters]\n";
}

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

void print_gpu_info() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::cerr << "No se detectaron GPUs CUDA." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

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

double stencil_operations(int nx, int ny) {
    double interior = static_cast<double>(nx - 2) * static_cast<double>(ny - 2);
    return interior * 5.0;
}

template <typename T>
Metrics run_cpu_stencil(const std::vector<T>& in, std::vector<T>& out,
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

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double gflops = stencil_operations(nx, ny) / (ms * 1.0e6);
    return {ms, gflops};
}

template <typename T>
Metrics run_gpu_stencil(const std::vector<T>& in, std::vector<T>& out,
                        int nx, int ny, int iters) {
    size_t elements = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    size_t bytes = elements * sizeof(T);

    T* d_in = nullptr;
    T* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    stencil2d_kernel<T><<<grid, block>>>(d_in, d_out, nx, ny);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        stencil2d_kernel<T><<<grid, block>>>(d_in, d_out, nx, ny);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    double ms = static_cast<double>(total_ms) / iters;
    double gflops = stencil_operations(nx, ny) / (ms * 1.0e6);
    return {ms, gflops};
}

template <typename T>
double max_abs_diff(const std::vector<T>& ref, const std::vector<T>& test) {
    double max_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = std::abs(static_cast<double>(ref[i]) - static_cast<double>(test[i]));
        max_err = std::max(max_err, diff);
    }
    return max_err;
}

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

template <typename T>
void run_experiment(const Options& opt, const char* precision_name) {
    size_t elements = static_cast<size_t>(opt.nx) * static_cast<size_t>(opt.ny);
    std::vector<T> input(elements);
    std::vector<T> output_cpu(elements, static_cast<T>(0));
    std::vector<T> output_gpu(elements, static_cast<T>(0));

    initialize_grid(input, opt.nx, opt.ny);
    print_experiment_header(opt, precision_name);

    Metrics cpu = run_cpu_stencil(input, output_cpu, opt.nx, opt.ny, opt.iters);
    Metrics gpu = run_gpu_stencil(input, output_gpu, opt.nx, opt.ny, opt.iters);

    double max_err = max_abs_diff(output_cpu, output_gpu);
    double rel_err = rel_l2_error(output_cpu, output_gpu);

    std::cout << std::fixed << std::setprecision(7);
    std::cout << "---------------- RESULTADOS ----------------\n";
    std::cout << "CPU serial - tiempo medio : " << cpu.milliseconds << " ms\n";
    std::cout << "CPU serial - rendimiento  : " << cpu.gflops << " GFLOP/s\n";
    std::cout << "GPU CUDA   - tiempo medio : " << gpu.milliseconds << " ms\n";
    std::cout << "GPU CUDA   - rendimiento  : " << gpu.gflops << " GFLOP/s\n";
    std::cout << "Speedup GPU/CPU           : " << (cpu.milliseconds / gpu.milliseconds) << "x\n";
    std::cout << "Error max abs             : " << std::scientific << max_err << "\n";
    std::cout << "Error relativo L2         : " << rel_err << std::fixed << "\n";
    std::cout << "--------------------------------------------\n";
}

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
