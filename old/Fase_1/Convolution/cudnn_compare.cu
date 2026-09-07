#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA(call)                                                       \
do {                                                                           \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        std::cerr << "CUDA error en " << __FILE__ << ":" << __LINE__           \
                  << " -> " << cudaGetErrorString(err) << std::endl;           \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while(0)

#define CHECK_CUDNN(call)                                                      \
do {                                                                           \
    cudnnStatus_t status = (call);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
        std::cerr << "cuDNN error en " << __FILE__ << ":" << __LINE__          \
                  << " -> " << cudnnGetErrorString(status) << std::endl;       \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while(0)

void fillVectorFloat(std::vector<float>& v, float value) {
    for (size_t i = 0; i < v.size(); i++) v[i] = value;
}

void fillVectorDouble(std::vector<double>& v, double value) {
    for (size_t i = 0; i < v.size(); i++) v[i] = value;
}

void benchmark_convolution_fp32(
    cudnnHandle_t cudnn,
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int iters,
    int warmup
) {
    std::cout << "\n========================================\n";
    std::cout << "Benchmark cuDNN - Precision: FLOAT (FP32)\n";

    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W
    ));

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S
    ));

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        convDesc,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    int outN, outC, outH, outW;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        convDesc, xDesc, wDesc, &outN, &outC, &outH, &outW
    ));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outN, outC, outH, outW
    ));

    cudnnConvolutionFwdAlgoPerf_t perfResults[10];
    int returnedAlgoCount = 0;

    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn, xDesc, wDesc, convDesc, yDesc,
        10, &returnedAlgoCount, perfResults
    ));

    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;

    size_t workspaceBytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, xDesc, wDesc, convDesc, yDesc, algo, &workspaceBytes
    ));

    size_t xSize = (size_t)N * C * H * W;
    size_t wSize = (size_t)K * C * R * S;
    size_t ySize = (size_t)outN * outC * outH * outW;

    std::vector<float> h_x(xSize), h_w(wSize), h_y(ySize);
    fillVectorFloat(h_x, 1.0f);
    fillVectorFloat(h_w, 1.0f);
    fillVectorFloat(h_y, 0.0f);

    float *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
    void* d_workspace = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x, xSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, wSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, ySize * sizeof(float)));
    if (workspaceBytes > 0) CHECK_CUDA(cudaMalloc(&d_workspace, workspaceBytes));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), xSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), wSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), ySize * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta  = 0.0f;

    for (int i = 0; i < warmup; i++) {
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            xDesc, d_x,
            wDesc, d_w,
            convDesc, algo,
            d_workspace, workspaceBytes,
            &beta,
            yDesc, d_y
        ));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            xDesc, d_x,
            wDesc, d_w,
            convDesc, algo,
            d_workspace, workspaceBytes,
            &beta,
            yDesc, d_y
        ));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, start, stop));
    double avgMs = totalMs / iters;

    double flops = 2.0 * N * outC * outH * outW * C * R * S;
    double gflops = (flops / (avgMs * 1e-3)) / 1e9;

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, ySize * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Input   : (" << N << ", " << C << ", " << H << ", " << W << ")\n";
    std::cout << "Filter  : (" << K << ", " << C << ", " << R << ", " << S << ")\n";
    std::cout << "Output  : (" << outN << ", " << outC << ", " << outH << ", " << outW << ")\n";
    std::cout << "Tiempo promedio: " << avgMs << " ms\n";
    std::cout << "Rendimiento    : " << gflops << " GFLOP/s\n";
    std::cout << "Salida[0]      : " << h_y[0] << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));
    if (d_workspace) CHECK_CUDA(cudaFree(d_workspace));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
}

void benchmark_convolution_fp64(
    cudnnHandle_t cudnn,
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int iters,
    int warmup
) {
    std::cout << "\n========================================\n";
    std::cout << "Benchmark cuDNN - Precision: DOUBLE (FP64)\n";

    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, H, W
    ));

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        wDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, R, S
    ));

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        convDesc,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_DOUBLE
    ));

    int outN, outC, outH, outW;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        convDesc, xDesc, wDesc, &outN, &outC, &outH, &outW
    ));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, outN, outC, outH, outW
    ));

    cudnnConvolutionFwdAlgoPerf_t perfResults[10];
    int returnedAlgoCount = 0;

    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn, xDesc, wDesc, convDesc, yDesc,
        10, &returnedAlgoCount, perfResults
    ));

    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;

    size_t workspaceBytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, xDesc, wDesc, convDesc, yDesc, algo, &workspaceBytes
    ));

    size_t xSize = (size_t)N * C * H * W;
    size_t wSize = (size_t)K * C * R * S;
    size_t ySize = (size_t)outN * outC * outH * outW;

    std::vector<double> h_x(xSize), h_w(wSize), h_y(ySize);
    fillVectorDouble(h_x, 1.0);
    fillVectorDouble(h_w, 1.0);
    fillVectorDouble(h_y, 0.0);

    double *d_x = nullptr, *d_w = nullptr, *d_y = nullptr;
    void* d_workspace = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x, xSize * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_w, wSize * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_y, ySize * sizeof(double)));
    if (workspaceBytes > 0) CHECK_CUDA(cudaMalloc(&d_workspace, workspaceBytes));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), xSize * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), wSize * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), ySize * sizeof(double), cudaMemcpyHostToDevice));

    double alpha = 1.0;
    double beta  = 0.0;

    for (int i = 0; i < warmup; i++) {
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            xDesc, d_x,
            wDesc, d_w,
            convDesc, algo,
            d_workspace, workspaceBytes,
            &beta,
            yDesc, d_y
        ));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            xDesc, d_x,
            wDesc, d_w,
            convDesc, algo,
            d_workspace, workspaceBytes,
            &beta,
            yDesc, d_y
        ));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, start, stop));
    double avgMs = totalMs / iters;

    double flops = 2.0 * N * outC * outH * outW * C * R * S;
    double gflops = (flops / (avgMs * 1e-3)) / 1e9;

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, ySize * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "Input   : (" << N << ", " << C << ", " << H << ", " << W << ")\n";
    std::cout << "Filter  : (" << K << ", " << C << ", " << R << ", " << S << ")\n";
    std::cout << "Output  : (" << outN << ", " << outC << ", " << outH << ", " << outW << ")\n";
    std::cout << "Tiempo promedio: " << avgMs << " ms\n";
    std::cout << "Rendimiento    : " << gflops << " GFLOP/s\n";
    std::cout << "Salida[0]      : " << h_y[0] << "\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_w));
    CHECK_CUDA(cudaFree(d_y));
    if (d_workspace) CHECK_CUDA(cudaFree(d_workspace));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(wDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
}

int main() {
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    int N = 1, C = 1, H = 1024, W = 1024;
    int K = 64, R = 5, S = 5;
    int pad_h = 1, pad_w = 1;
    int stride_h = 1, stride_w = 1;
    int dilation_h = 1, dilation_w = 1;
    int warmup = 10;
    int iters = 100;

    benchmark_convolution_fp32(
        cudnn, N, C, H, W, K, R, S,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, iters, warmup
    );

    benchmark_convolution_fp64(
        cudnn, N, C, H, W, K, R, S,
        pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, iters, warmup
    );

    CHECK_CUDNN(cudnnDestroy(cudnn));
    return 0;
}
