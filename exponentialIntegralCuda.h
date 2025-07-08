#ifndef EXPONENTIAL_INTEGRAL_CUDA_H
#define EXPONENTIAL_INTEGRAL_CUDA_H

#include <vector>

void runGPU(std::vector<std::vector<float>>& resultsFloatGpu,
            std::vector<std::vector<double>>& resultsDoubleGpu,
            unsigned int n, unsigned int numberOfSamples,
            double a, double b, int maxIterations,
            double& timeTotalGpu,
            double& timeFloatKernel,
            double& timeDoubleKernel,
            bool verbose);

#endif
