#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                  << " en linea " << __LINE__ << std::endl;        \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)

__host__ __device__ inline int idx(int x, int y, int nx) {
    return y * nx + x;
}

__global__ void stencil2d_fp32_kernel(const float* in, float* out, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) return;

    if (x == 0 || y == 0 || x == nx - 1 || y == ny - 1) {
        out[idx(x, y, nx)] = in[idx(x, y, nx)];
        return;
    }

    float up     = in[idx(x, y - 1, nx)];
    float down   = in[idx(x, y + 1, nx)];
    float left   = in[idx(x - 1, y, nx)];
    float right  = in[idx(x + 1, y, nx)];
    float center = in[idx(x, y, nx)];

    out[idx(x, y, nx)] = 0.25f * (up + down + left + right) - center;
}

int main() {
    std::cout << "INICIO DEL PROGRAMA" << std::endl;

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "Error al consultar GPUs: "
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Numero de GPUs detectadas: " << deviceCount << std::endl;

    if (deviceCount == 0) {
        std::cerr << "No hay GPU CUDA disponible." << std::endl;
        return 1;
    }

    int nx = 512;
    int ny = 512;
    int size = nx * ny;
    size_t bytes = size * sizeof(float);

    std::vector<float> h_in(size), h_out(size);

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            h_in[idx(x, y, nx)] = 0.001f * (x + y);
        }
    }

    float* d_in = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    std::cout << "Antes del kernel" << std::endl;

    stencil2d_fp32_kernel<<<grid, block>>>(d_in, d_out, nx, ny);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "Despues del kernel" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "out[100,100] = " << h_out[idx(100, 100, nx)] << std::endl;
    std::cout << "FIN DEL PROGRAMA" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}