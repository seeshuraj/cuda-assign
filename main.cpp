#include <time.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <unistd.h>
#include "exponentialIntegralCuda.h"

using namespace std;

float exponentialIntegralFloat(const int n, const float x);
double exponentialIntegralDouble(const int n, const double x);
void outputResults(const vector<vector<float>>& resultsFloat, 
                 const vector<vector<double>>& resultsDouble,
                 bool isCpu = true);
int parseArguments(int argc, char **argv);
void printUsage(void);

bool verbose, timing, cpu, gpu;
int maxIterations;
unsigned int n, numberOfSamples;
double a, b;

int main(int argc, char *argv[]) {
    cpu = true;
    gpu = true;
    verbose = false;
    timing = false;
    n = 10;
    numberOfSamples = 10;
    a = 0.0;
    b = 10.0;
    maxIterations = 2000000000;

    parseArguments(argc, argv);

    if (verbose) {
        cout << "Parameters:" << endl;
        cout << "  n=" << n << endl;
        cout << "  numberOfSamples=" << numberOfSamples << endl;
        cout << "  a=" << a << endl;
        cout << "  b=" << b << endl;
        cout << "  maxIterations=" << maxIterations << endl;
        cout << "  timing=" << (timing ? "enabled" : "disabled") << endl;
        cout << "  verbose=" << (verbose ? "enabled" : "disabled") << endl;
        cout << "  cpu=" << (cpu ? "enabled" : "disabled") << endl;
        cout << "  gpu=" << (gpu ? "enabled" : "disabled") << endl;
    }

    // Sanity checks
    if (a >= b) {
        cerr << "Error: Invalid interval [" << a << "," << b << "]" << endl;
        return 1;
    }
    if (n <= 0) {
        cerr << "Error: Invalid order n=" << n << endl;
        return 1;
    }
    if (numberOfSamples <= 0) {
        cerr << "Error: Invalid number of samples=" << numberOfSamples << endl;
        return 1;
    }

    vector<vector<float>> resultsFloatCpu, resultsFloatGpu;
    vector<vector<double>> resultsDoubleCpu, resultsDoubleGpu;
    double timeTotalCpu = 0.0, timeTotalGpu = 0.0;
    double timeFloatKernel = 0.0, timeDoubleKernel = 0.0;

    // CPU execution
    if (cpu) {
        try {
            resultsFloatCpu.resize(n, vector<float>(numberOfSamples));
            resultsDoubleCpu.resize(n, vector<double>(numberOfSamples));
        } catch (const bad_alloc& e) {
            cerr << "Memory allocation failed: " << e.what() << endl;
            return 1;
        }
        
        struct timeval start, end;
        gettimeofday(&start, NULL);
        
        double division = (b - a) / numberOfSamples;
        for (unsigned int ui = 1; ui <= n; ui++) {
            for (unsigned int uj = 1; uj <= numberOfSamples; uj++) {
                double x = a + uj * division;
                resultsFloatCpu[ui-1][uj-1] = exponentialIntegralFloat(ui, static_cast<float>(x));
                resultsDoubleCpu[ui-1][uj-1] = exponentialIntegralDouble(ui, x);
            }
        }
        
        gettimeofday(&end, NULL);
        timeTotalCpu = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    }

    // GPU execution
    if (gpu) {
        try {
            resultsFloatGpu.resize(n, vector<float>(numberOfSamples));
            resultsDoubleGpu.resize(n, vector<double>(numberOfSamples));
        } catch (const bad_alloc& e) {
            cerr << "Memory allocation failed: " << e.what() << endl;
            return 1;
        }
        
        runGPU(resultsFloatGpu, resultsDoubleGpu, n, numberOfSamples, a, b, 
               maxIterations, timeTotalGpu, timeFloatKernel, timeDoubleKernel, verbose);
    }

    // Timing results
    if (timing) {
        if (cpu) {
            printf("CPU execution time: %.6f seconds\n", timeTotalCpu);
        }
        if (gpu) {
            printf("GPU execution time: %.6f seconds\n", timeTotalGpu);
            printf("  Float kernel time: %.6f seconds\n", timeFloatKernel);
            printf("  Double kernel time: %.6f seconds\n", timeDoubleKernel);
            if (cpu) {
                printf("Speedup (CPU/GPU): %.2fx\n", timeTotalCpu / timeTotalGpu);
            }
        }
    }

    // Numerical validation
    if (cpu && gpu) {
        double maxDiffFloat = 0.0, maxDiffDouble = 0.0;
        int countDiffFloat = 0, countDiffDouble = 0;
        const double tolerance = 1e-5;
        
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < numberOfSamples; j++) {
                float diffFloat = fabs(resultsFloatCpu[i][j] - resultsFloatGpu[i][j]);
                double diffDouble = fabs(resultsDoubleCpu[i][j] - resultsDoubleGpu[i][j]);
                
                if (diffFloat > maxDiffFloat) maxDiffFloat = diffFloat;
                if (diffDouble > maxDiffDouble) maxDiffDouble = diffDouble;
                
                if (diffFloat > tolerance) countDiffFloat++;
                if (diffDouble > tolerance) countDiffDouble++;
            }
        }
        
        printf("\nNumerical Validation:\n");
        printf("Max difference (float):  %e\n", maxDiffFloat);
        printf("Max difference (double): %e\n", maxDiffDouble);
        printf("Values exceeding tolerance (1e-5):\n");
        printf("  Float:  %d/%d\n", countDiffFloat, n * numberOfSamples);
        printf("  Double: %d/%d\n", countDiffDouble, n * numberOfSamples);
        
        if (countDiffFloat > 0 || countDiffDouble > 0) {
            printf("WARNING: Numerical differences detected between CPU and GPU!\n");
        } else {
            printf("SUCCESS: All values match within tolerance.\n");
        }
    }

    // Output results if verbose
    if (verbose) {
        if (cpu) {
            cout << "\n===== CPU Results =====" << endl;
            outputResults(resultsFloatCpu, resultsDoubleCpu);
        }
        if (gpu) {
            cout << "\n===== GPU Results =====" << endl;
            outputResults(resultsFloatGpu, resultsDoubleGpu, false);
        }
    }

    return 0;
}

float exponentialIntegralFloat(const int n, const float x) {
    static const float eulerConstant = 0.5772156649015329f;
    const float epsilon = 1.E-30f;
    const float bigfloat = numeric_limits<float>::max();
    const int nm1 = n - 1;
    float ans = 0.0f;

    if (n < 0 || x < 0.0f || (x == 0.0f && (n == 0 || n == 1))) {
        cerr << "Bad arguments passed to exponentialIntegralFloat: n=" << n << ", x=" << x << endl;
        exit(1);
    }

    if (n == 0) {
        ans = expf(-x)/x;
    } else {
        if (x > 1.0f) {
            float b = x + n;
            float c = bigfloat;
            float d = 1.0f/b;
            float h = d;
            
            for (int i = 1; i <= maxIterations; i++) {
                const float a = -i*(nm1 + i);
                b += 2.0f;
                d = 1.0f/(a*d + b);
                c = b + a/c;
                const float del = c*d;
                h *= del;
                if (fabsf(del - 1.0f) <= epsilon) {
                    return h*expf(-x);
                }
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
                    for (int ii = 1; ii <= nm1; ii++) {
                        psi += 1.0f/ii;
                    }
                    del = fact*(-logf(x) + psi);
                }
                
                ans += del;
                if (fabsf(del) < fabsf(ans)*epsilon) return ans;
            }
        }
    }
    return ans;
}

double exponentialIntegralDouble(const int n, const double x) {
    static const double eulerConstant = 0.5772156649015329;
    const double epsilon = 1.E-30;
    const double bigDouble = numeric_limits<double>::max();
    const int nm1 = n - 1;
    double ans = 0.0;

    if (n < 0 || x < 0.0 || (x == 0.0 && (n == 0 || n == 1))) {
        cerr << "Bad arguments passed to exponentialIntegralDouble: n=" << n << ", x=" << x << endl;
        exit(1);
    }

    if (n == 0) {
        ans = exp(-x)/x;
    } else {
        if (x > 1.0) {
            double b = x + n;
            double c = bigDouble;
            double d = 1.0/b;
            double h = d;
            
            for (int i = 1; i <= maxIterations; i++) {
                const double a = -i*(nm1 + i);
                b += 2.0;
                d = 1.0/(a*d + b);
                c = b + a/c;
                const double del = c*d;
                h *= del;
                if (fabs(del - 1.0) <= epsilon) {
                    return h*exp(-x);
                }
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
                    for (int ii = 1; ii <= nm1; ii++) {
                        psi += 1.0/ii;
                    }
                    del = fact*(-log(x) + psi);
                }
                
                ans += del;
                if (fabs(del) < fabs(ans)*epsilon) return ans;
            }
        }
    }
    return ans;
}

void outputResults(const vector<vector<float>>& resultsFloat, 
                 const vector<vector<double>>& resultsDouble,
                 bool isCpu) {
    double division = (b - a) / numberOfSamples;
    for (unsigned int ui = 1; ui <= n; ui++) {
        for (unsigned int uj = 1; uj <= numberOfSamples; uj++) {
            double x = a + uj * division;
            cout << (isCpu ? "CPU" : "GPU") << " ==> ";
            cout << "E_" << ui << "(" << x << ") float: " << resultsFloat[ui-1][uj-1];
            cout << ", double: " << resultsDouble[ui-1][uj-1] << endl;
        }
    }
}

int parseArguments(int argc, char *argv[]) {
    int c;
    while ((c = getopt(argc, argv, "cghn:m:a:b:i:tv")) != -1) {
        switch(c) {
            case 'c': cpu = false; break;
            case 'g': gpu = false; break;
            case 'h': printUsage(); exit(0);
            case 'i': maxIterations = atoi(optarg); break;
            case 'n': n = atoi(optarg); break;
            case 'm': numberOfSamples = atoi(optarg); break;
            case 'a': a = atof(optarg); break;
            case 'b': b = atof(optarg); break;
            case 't': timing = true; break;
            case 'v': verbose = true; break;
            default:
                fprintf(stderr, "Invalid option given\n");
                printUsage();
                return -1;
        }
    }
    return 0;
}

void printUsage() {
    printf("Exponential Integral Calculator\n");
    printf("Usage: exponentialIntegral [options]\n");
    printf("Options:\n");
    printf("  -a <value>   Start of interval (default: 0.0)\n");
    printf("  -b <value>   End of interval (default: 10.0)\n");
    printf("  -c           Disable CPU computation\n");
    printf("  -g           Disable GPU computation\n");
    printf("  -h           Show this help message\n");
    printf("  -i <value>   Max iterations (default: 2000000000)\n");
    printf("  -n <value>   Maximum order n (default: 10)\n");
    printf("  -m <value>   Number of samples (default: 10)\n");
    printf("  -t           Enable timing output\n");
    printf("  -v           Enable verbose output\n");
    printf("\nExample:\n");
    printf("  ./exponentialIntegral -n 5000 -m 5000 -t\n");
}
