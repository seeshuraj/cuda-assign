#include "exponentialIntegralCuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <algorithm>

#define CHECK(call) { \
    const cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

__constant__ float float_max = 3.402823466e+38f;
__constant__ float float_nan = 0x7fc00000;
__constant__ double double_max = 1.7976931348623157e+308;
__constant__ double double_nan = 0x7ff8000000000000;

__device__ float exponentialIntegralFloatDevice(int n, float x, int maxIterations) {
    const float eulerConstant = 0.5772156649015329f;
    const float epsilon = 1.E-30f;
    const int nm1 = n - 1;
    float ans = 0.0f;

    if (n < 0 || x < 0.0f || (x == 0.0f && (n == 0 || n == 1))) return float_nan;
    
    if (n == 0) {
        ans = expf(-x)/x;
    } else if (x > 1.0f) {
        float b = x + n;
        float c = float_max;
        float d = 1.0f/b;
        float h = d;
        
        for (int i = 1; i <= maxIterations; i++) {
            const float a = -i*(nm1 + i);
            b += 2.0f;
            d = 1.0f/(a*d + b);
            c = b + a/c;
            const float del = c*d;
            h *= del;
            if (fabsf(del - 1.0f) <= epsilon) return h*expf(-x);
        }
        return h*expf(-x);
    } else {
        ans = (nm1 != 0) ? 1.0f/nm1 : -logf(x)-eulerConstant;
        float fact = 1.0f;
        
        for (int i = 1; i <= maxIterations; i++) {
            fact *= -x/i;
            float del;
            
            if (i != nm1) {
                del = -fact/(i-nm1);
            } else {
                float psi = -eulerConstant;
                for (int ii = 1; ii <= nm1; ii++) psi += 1.0f/ii;
                del = fact*(-logf(x) + psi);
            }
            
            ans += del;
            if (fabsf(del) < fabsf(ans)*epsilon) return ans;
        }
    }
    return ans;
}

__device__ double exponentialIntegralDoubleDevice(int n, double x, int maxIterations) {
    const double eulerConstant = 0.5772156649015329;
    const double epsilon = 1.E-30;
    const int nm1 = n - 1;
    double ans = 0.0;

    if (n < 0 || x < 0.0 || (x == 0.0 && (n == 0 || n == 1))) return double_nan;
    
    if (n == 0) {
        ans = exp(-x)/x;
    } else if (x > 1.0) {
        double b = x + n;
        double c = double_max;
        double d = 1.0/b;
        double h = d;
        
        for (int i = 1; i <= maxIterations; i++) {
            const double a = -i*(nm1 + i);
            b += 2.0;
            d = 1.0/(a*d + b);
            c = b + a/c;
            const double del = c*d;
            h *= del;
            if (fabs(del - 1.0) <= epsilon) return h*exp(-x);
        }
        return h*exp(-x);
    } else {
        ans = (nm1 != 0) ? 1.0/nm1 : -log(x)-eulerConstant;
        double fact = 1.0;
        
        for (int i = 1; i <= maxIterations; i++) {
            fact *= -x/i;
            double del;
            
            if (i != nm1) {
                del = -fact/(i-nm1);
            } else {
                double psi = -eulerConstant;
                for (int ii = 1; ii <= nm1; ii++) psi += 1.0/ii;
                del = fact*(-log(x) + psi);
            }
            
            ans += del;
            if (fabs(del) < fabs(ans)*epsilon) return ans;
        }
    }
    return ans;
}

__global__ void exponentialIntegralFloatKernel(float* results, int max_n, int max_samples, 
                                             double a, double b, int maxIterations) {
    const int order = blockIdx.y*blockDim.y + threadIdx.y + 1;
    const int sampleIdx = blockIdx.x*blockDim.x + threadIdx.x;

    if (order <= max_n && sampleIdx < max_samples) {
        const double x_val = a + (sampleIdx + 1)*((b - a)/max_samples);
        results[(order-1)*max_samples + sampleIdx] = exponentialIntegralFloatDevice(order, (float)x_val, maxIterations);
    }
}

__global__ void exponentialIntegralDoubleKernel(double* results, int max_n, int max_samples, 
                                              double a, double b, int maxIterations) {
    const int order = blockIdx.y*blockDim.y + threadIdx.y + 1;
    const int sampleIdx = blockIdx.x*blockDim.x + threadIdx.x;

    if (order <= max_n && sampleIdx < max_samples) {
        const double x = a + (sampleIdx + 1)*((b - a)/max_samples);
        results[(order-1)*max_samples + sampleIdx] = exponentialIntegralDoubleDevice(order, x, maxIterations);
    }
}

void runGPU(std::vector<std::vector<float>>& resultsFloatGpu,
            std::vector<std::vector<double>>& resultsDoubleGpu,
            unsigned int n, unsigned int numberOfSamples,
            double a, double b, int maxIterations,
            double& timeTotalGpu,
            double& timeFloatKernel,
            double& timeDoubleKernel,
            bool verbose) {
    
    // Check memory requirements
    size_t free_mem, total_mem;
    CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    size_t required_mem = n * numberOfSamples * (sizeof(float) + sizeof(double));
    if (required_mem > free_mem * 0.9) {
        std::cerr << "Error: Not enough GPU memory. Required: " 
                 << required_mem/1e9 << "GB, Available: " << free_mem/1e9 << "GB\n";
        exit(1);
    }

    // Allocate pinned host memory
    float* h_results_float;
    double* h_results_double;
    CHECK(cudaMallocHost((void**)&h_results_float, n * numberOfSamples * sizeof(float)));
    CHECK(cudaMallocHost((void**)&h_results_double, n * numberOfSamples * sizeof(double)));

    // Device memory
    float* d_results_float;
    double* d_results_double;
    CHECK(cudaMalloc((void**)&d_results_float, n * numberOfSamples * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_results_double, n * numberOfSamples * sizeof(double)));

    // Configure kernel
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    
    dim3 block(32, 8);  // 256 threads per block
    dim3 grid((numberOfSamples + block.x - 1)/block.x, 
              (n + block.y - 1)/block.y);
    grid.x = std::min(grid.x, (unsigned int)prop.maxGridSize[0]);
    grid.y = std::min(grid.y, (unsigned int)prop.maxGridSize[1]);

    if (verbose) {
        std::cout << "\nGPU Configuration:\n";
        std::cout << "  Grid: [" << grid.x << ", " << grid.y << "]\n";
        std::cout << "  Block: [" << block.x << ", " << block.y << "]\n";
    }

    // Timing events
    cudaEvent_t start, stop, startFloat, stopFloat, startDouble, stopDouble;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventCreate(&startFloat));
    CHECK(cudaEventCreate(&stopFloat));
    CHECK(cudaEventCreate(&startDouble));
    CHECK(cudaEventCreate(&stopDouble));

    CHECK(cudaEventRecord(start));

    // Launch kernels
    CHECK(cudaEventRecord(startFloat));
    exponentialIntegralFloatKernel<<<grid, block>>>(d_results_float, n, numberOfSamples, a, b, maxIterations);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stopFloat));
    
    CHECK(cudaEventRecord(startDouble));
    exponentialIntegralDoubleKernel<<<grid, block>>>(d_results_double, n, numberOfSamples, a, b, maxIterations);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stopDouble));

    // Copy results
    CHECK(cudaMemcpy(h_results_float, d_results_float, n * numberOfSamples * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_results_double, d_results_double, n * numberOfSamples * sizeof(double), cudaMemcpyDeviceToHost));
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    // Get timings
    float tempTime;
    CHECK(cudaEventElapsedTime(&tempTime, startFloat, stopFloat));
    timeFloatKernel = tempTime/1000.0;
    CHECK(cudaEventElapsedTime(&tempTime, startDouble, stopDouble));
    timeDoubleKernel = tempTime/1000.0;
    CHECK(cudaEventElapsedTime(&tempTime, start, stop));
    timeTotalGpu = tempTime/1000.0;

    // Store results
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < numberOfSamples; j++) {
            resultsFloatGpu[i][j] = h_results_float[i * numberOfSamples + j];
            resultsDoubleGpu[i][j] = h_results_double[i * numberOfSamples + j];
        }
    }

    // Cleanup
    CHECK(cudaFreeHost(h_results_float));
    CHECK(cudaFreeHost(h_results_double));
    CHECK(cudaFree(d_results_float));
    CHECK(cudaFree(d_results_double));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaEventDestroy(startFloat));
    CHECK(cudaEventDestroy(stopFloat));
    CHECK(cudaEventDestroy(startDouble));
    CHECK(cudaEventDestroy(stopDouble));
}
