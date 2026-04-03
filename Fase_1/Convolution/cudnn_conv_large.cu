#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CHECK_CUDNN(call)                                                     \
    do {                                                                      \
        cudnnStatus_t status = (call);                                        \
        if (status != CUDNN_STATUS_SUCCESS) {                                 \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(status)       \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

int main() {
    // =========================================================
    // 1. Parámetros del problema
    // =========================================================
    const int N = 1;      // batch size
    const int C = 3;      // canales de entrada
    const int H = 1024;    // alto de entrada
    const int W = 1024;    // ancho de entrada

    const int K = 16;     // número de filtros / canales de salida
    const int R = 5;      // alto del filtro
    const int S = 5;      // ancho del filtro

    const int pad_h = 1;
    const int pad_w = 1;
    const int stride_h = 1;
    const int stride_w = 1;
    const int dilation_h = 1;
    const int dilation_w = 1;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // =========================================================
    // 2. Crear y llenar tensores en host
    // =========================================================
    const size_t xCount = static_cast<size_t>(N) * C * H * W;
    const size_t wCount = static_cast<size_t>(K) * C * R * S;

    std::vector<float> h_x(xCount);
    std::vector<float> h_w(wCount);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < xCount; ++i) {
        h_x[i] = dist(rng);
    }

    for (size_t i = 0; i < wCount; ++i) {
        h_w[i] = dist(rng);
    }

    // =========================================================
    // 3. Crear handle y descriptores de cuDNN
    // =========================================================
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreate(&handle));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // =========================================================
    // 4. Configurar descriptores de entrada y filtro
    // =========================================================
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        xDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        N, C, H, W
    ));

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        wDesc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        K, C, R, S
    ));

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        convDesc,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    // =========================================================
    // 5. Consultar dimensiones de salida
    // =========================================================
    int outN, outC, outH, outW;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        convDesc,
        xDesc,
        wDesc,
        &outN, &outC, &outH, &outW
    ));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        yDesc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        outN, outC, outH, outW
    ));

    const size_t yCount = static_cast<size_t>(outN) * outC * outH * outW;

    std::vector<float> h_y(yCount, 0.0f);

    std::cout << "==============================\n";
    std::cout << "Configuracion de la convolucion\n";
    std::cout << "==============================\n";
    std::cout << "Entrada : N=" << N << ", C=" << C
              << ", H=" << H << ", W=" << W << "\n";
    std::cout << "Filtro  : K=" << K << ", C=" << C
              << ", R=" << R << ", S=" << S << "\n";
    std::cout << "Salida  : N=" << outN << ", C=" << outC
              << ", H=" << outH << ", W=" << outW << "\n";
    std::cout << "alpha=" << alpha << ", beta=" << beta << "\n\n";

    // =========================================================
    // 6. Reservar memoria en GPU
    // =========================================================
    float *d_x = nullptr;
    float *d_w = nullptr;
    float *d_y = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x, xCount * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, wCount * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, yCount * sizeof(float)));

    // =========================================================
    // 7. Copiar datos al device
    // =========================================================
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), xCount * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), wCount * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), yCount * sizeof(float), cudaMemcpyHostToDevice));

    // =========================================================
    // 8. Elegir algoritmo de convolucion
    // =========================================================
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    // =========================================================
    // 9. Consultar y reservar workspace
    // =========================================================
    size_t workspaceBytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        xDesc,
        wDesc,
        convDesc,
        yDesc,
        algo,
        &workspaceBytes
    ));

    void* d_workspace = nullptr;
    if (workspaceBytes > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspaceBytes));
    }

    std::cout << "Workspace requerido: " << workspaceBytes << " bytes\n\n";

    // =========================================================
    // 10. Warm-up
    // =========================================================
    CHECK_CUDNN(cudnnConvolutionForward(
        handle,
        &alpha,
        xDesc, d_x,
        wDesc, d_w,
        convDesc,
        algo,
        d_workspace, workspaceBytes,
        &beta,
        yDesc, d_y
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // =========================================================
    // 11. Medir tiempo de la convolucion
    // =========================================================
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    CHECK_CUDA(cudaEventRecord(startEvent));

    CHECK_CUDNN(cudnnConvolutionForward(
        handle,
        &alpha,
        xDesc, d_x,
        wDesc, d_w,
        convDesc,
        algo,
        d_workspace, workspaceBytes,
        &beta,
        yDesc, d_y
    ));

    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));

    // =========================================================
    // 12. Copiar resultado al host
    // =========================================================
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, yCount * sizeof(float), cudaMemcpyDeviceToHost));

    // =========================================================
    // 13. Calcular FLOPs aproximadas y rendimiento
    // =========================================================
    const double flops =
        2.0 * static_cast<double>(N) *
        static_cast<double>(outH) *
        static_cast<double>(outW) *
        static_cast<double>(K) *
        static_cast<double>(C) *
        static_cast<double>(R) *
        static_cast<double>(S);

    const double gflops = flops / (elapsedMs * 1e6);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Tiempo de convolucion: " << elapsedMs << " ms\n";
    std::cout << "Rendimiento aproximado: " << gflops << " GFLOP/s\n\n";

    // =========================================================
    // 14. Mostrar algunos valores de salida
    // =========================================================
    std::cout << "Primeros 10 valores de salida:\n";
    for (int i = 0; i < 10 && i < static_cast<int>(h_y.size()); ++i) {
        std::cout << h_y[i] << " ";
    }
    std::cout << "\n";

    // =========================================================
    // 15. Liberar recursos
    // =========================================================
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));

    if (d_workspace) {
        CHECK_CUDA(cudaFree(d_workspace));
    }

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroy(handle));

    return 0;
}
