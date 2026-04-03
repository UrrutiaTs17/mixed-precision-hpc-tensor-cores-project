#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cudnn.h>

// Macros de control de errores
#define checkCUDART(expression) {                                           \
cudaError_t status = (expression);                                      \
if (status != cudaSuccess) {                                            \
    std::cerr << "Error CUDA: " << cudaGetErrorString(status)           \
    << " en " << __FILE__ << ":" << __LINE__ << std::endl;    \
    std::exit(1);                                                       \
}                                                                       \
}

#define checkCudnn(expression) {                                            \
cudnnStatus_t status = (expression);                                    \
if (status != CUDNN_STATUS_SUCCESS) {                                   \
    std::cerr << "Error cuDNN: " << cudnnGetErrorString(status)         \
    << " en " << __FILE__ << ":" << __LINE__ << std::endl;    \
    std::exit(1);                                                       \
}                                                                       \
}

int main() {
    cudnnHandle_t cudnn;
    checkCudnn(cudnnCreate(&cudnn));

    // Dimensiones
    const int N = 1;
    const int C = 1;
    const int side = 1024;
    const int filter_s = 3;
    const int K = 1;

    // Descriptores
    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnFilterDescriptor_t filt_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCudnn(cudnnCreateTensorDescriptor(&in_desc));
    checkCudnn(cudnnCreateTensorDescriptor(&out_desc));
    checkCudnn(cudnnCreateFilterDescriptor(&filt_desc));
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc));

    checkCudnn(cudnnSetTensor4dDescriptor(
        in_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        N, C, side, side
    ));

    checkCudnn(cudnnSetFilter4dDescriptor(
        filt_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        K, C, filter_s, filter_s
    ));

    checkCudnn(cudnnSetConvolution2dDescriptor(
        conv_desc,
        1, 1,   // padding
        1, 1,   // stride
        1, 1,   // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    // Obtener dimensiones de salida
    int n, c, h, w;
    checkCudnn(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc, &n, &c, &h, &w
    ));

    checkCudnn(cudnnSetTensor4dDescriptor(
        out_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        n, c, h, w
    ));

    // Algoritmo de convolución
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    size_t workspace_size = 0;
    checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        in_desc,
        filt_desc,
        conv_desc,
        out_desc,
        algo,
        &workspace_size
    ));

    // Memoria en host para inicializar datos
    size_t in_size = static_cast<size_t>(N) * C * side * side;
    size_t filt_size = static_cast<size_t>(K) * C * filter_s * filter_s;
    size_t out_size = static_cast<size_t>(n) * c * h * w;

    std::vector<float> h_in(in_size);
    std::vector<float> h_filt(filt_size);
    std::vector<float> h_out(out_size, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < in_size; ++i) {
        h_in[i] = dist(rng);
    }

    for (size_t i = 0; i < filt_size; ++i) {
        h_filt[i] = dist(rng);
    }

    // Memoria en device
    float *d_in = nullptr, *d_filt = nullptr, *d_out = nullptr;
    void *d_ws = nullptr;

    checkCUDART(cudaMalloc(&d_in, in_size * sizeof(float)));
    checkCUDART(cudaMalloc(&d_filt, filt_size * sizeof(float)));
    checkCUDART(cudaMalloc(&d_out, out_size * sizeof(float)));

    if (workspace_size > 0) {
        checkCUDART(cudaMalloc(&d_ws, workspace_size));
    }

    // Copiar datos al device
    checkCUDART(cudaMemcpy(d_in, h_in.data(), in_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCUDART(cudaMemcpy(d_filt, h_filt.data(), filt_size * sizeof(float), cudaMemcpyHostToDevice));
    checkCUDART(cudaMemcpy(d_out, h_out.data(), out_size * sizeof(float), cudaMemcpyHostToDevice));

    // Ejecución
    float alpha = 1.0f, beta = 0.0f;
    checkCudnn(cudnnConvolutionForward(
        cudnn,
        &alpha,
        in_desc, d_in,
        filt_desc, d_filt,
        conv_desc,
        algo,
        d_ws, workspace_size,
        &beta,
        out_desc, d_out
    ));

    checkCUDART(cudaDeviceSynchronize());

    // Copiar salida al host
    checkCUDART(cudaMemcpy(h_out.data(), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "¡Éxito! Convolución de 1024x1024 completada.\n";
    std::cout << "Salida: N=" << n << ", C=" << c << ", H=" << h << ", W=" << w << "\n";
    std::cout << "Workspace usado: " << workspace_size << " bytes\n";
    std::cout << "Primeros 10 valores de salida:\n";

    for (int i = 0; i < 10 && i < static_cast<int>(h_out.size()); ++i) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    // Limpieza
    if (d_ws) cudaFree(d_ws);
    cudaFree(d_in);
    cudaFree(d_filt);
    cudaFree(d_out);

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}
