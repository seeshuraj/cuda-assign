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

// Function declarations
float exponentialIntegralFloat(const int n, const float x);
double exponentialIntegralDouble(const int n, const double x);
void outputResults(const vector<vector<float>>& resultsFloat, 
                 const vector<vector<double>>& resultsDouble,
                 bool isCpu = true);
int parseArguments(int argc, char **argv);
void printUsage(void);
void validateResults(const vector<vector<float>>& cpuFloat,
                   const vector<vector<double>>& cpuDouble,
                   const vector<vector<float>>& gpuFloat,
                   const vector<vector<double>>& gpuDouble);

// Global configuration
bool verbose = false;
bool timing = false;
bool cpu = true;
bool gpu = true;
int maxIterations = 2000000000;
unsigned int n = 10;
unsigned int numberOfSamples = 10;
double a = 0.0;
double b = 10.0;

int main(int argc, char *argv[]) {
    // Parse command line arguments
    if (parseArguments(argc, argv) != 0) {
        return 1;
    }

    // Display configuration
    if (verbose) {
        cout << "\nConfiguration:\n";
        cout << "n: " << n << endl;
        cout << "samples: " << numberOfSamples << endl;
        cout << "interval: [" << a << ", " << b << "]" << endl;
        cout << "CPU: " << (cpu ? "enabled" : "disabled") << endl;
        cout << "GPU: " << (gpu ? "enabled" : "disabled") << endl;
    }

    // Validate input
    if (a >= b || n <= 0 || numberOfSamples <= 0) {
        cerr << "Invalid parameters" << endl;
        return 1;
    }

    // Initialize results
    vector<vector<float>> resultsFloatCpu, resultsFloatGpu;
    vector<vector<double>> resultsDoubleCpu, resultsDoubleGpu;
    double timeCpu = 0.0, timeGpu = 0.0;
    double timeFloatKernel = 0.0, timeDoubleKernel = 0.0;

    // CPU computation
    if (cpu) {
        try {
            resultsFloatCpu.resize(n, vector<float>(numberOfSamples));
            resultsDoubleCpu.resize(n, vector<double>(numberOfSamples));
        } catch (const bad_alloc& e) {
            cerr << "CPU memory error: " << e.what() << endl;
            return 1;
        }

        struct timeval start, end;
        gettimeofday(&start, NULL);

        double division = (b - a) / numberOfSamples;
        for (unsigned int ui = 1; ui <= n; ui++) {
            for (unsigned int uj = 1; uj <= numberOfSamples; uj++) {
                double x = a + uj * division;
                resultsFloatCpu[ui-1][uj-1] = exponentialIntegralFloat(ui, x);
                resultsDoubleCpu[ui-1][uj-1] = exponentialIntegralDouble(ui, x);
            }
        }

        gettimeofday(&end, NULL);
        timeCpu = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    }

    // GPU computation
    if (gpu) {
        try {
            resultsFloatGpu.resize(n, vector<float>(numberOfSamples));
            resultsDoubleGpu.resize(n, vector<double>(numberOfSamples));
        } catch (const bad_alloc& e) {
            cerr << "GPU memory error: " << e.what() << endl;
            return 1;
        }

        runGPU(resultsFloatGpu, resultsDoubleGpu, n, numberOfSamples, a, b, 
              maxIterations, timeGpu, timeFloatKernel, timeDoubleKernel, verbose);
    }

    // Output results
    if (timing) {
        cout << "\nTiming:" << endl;
        if (cpu) cout << "CPU: " << timeCpu << " s" << endl;
        if (gpu) {
            cout << "GPU: " << timeGpu << " s" << endl;
            cout << "  Float kernel: " << timeFloatKernel << " s" << endl;
            cout << "  Double kernel: " << timeDoubleKernel << " s" << endl;
            if (cpu) cout << "Speedup: " << timeCpu/timeGpu << "x" << endl;
        }
    }

    if (cpu && gpu) {
        validateResults(resultsFloatCpu, resultsDoubleCpu, resultsFloatGpu, resultsDoubleGpu);
    }

    if (verbose) {
        if (cpu) {
            cout << "\nCPU Results:" << endl;
            outputResults(resultsFloatCpu, resultsDoubleCpu);
        }
        if (gpu) {
            cout << "\nGPU Results:" << endl;
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
        cerr << "Bad arguments to exponentialIntegralFloat: n=" << n << ", x=" << x << endl;
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
        cerr << "Bad arguments to exponentialIntegralDouble: n=" << n << ", x=" << x << endl;
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
    }
    return ans;
}

void outputResults(const vector<vector<float>>& resultsFloat, 
                 const vector<vector<double>>& resultsDouble,
                 bool isCpu) {
    const int max_display = 5;
    double division = (b - a) / numberOfSamples;
    
    for (unsigned int ui = 1; ui <= min(n, (unsigned int)max_display); ui++) {
        for (unsigned int uj = 1; uj <= min(numberOfSamples, (unsigned int)max_display); uj++) {
            double x = a + uj * division;
            cout << (isCpu ? "CPU" : "GPU") << " ==> ";
            cout << "E_" << ui << "(" << x << ") float: " << resultsFloat[ui-1][uj-1];
            cout << ", double: " << resultsDouble[ui-1][uj-1] << endl;
        }
    }
    if (n > max_display || numberOfSamples > max_display) {
        cout << "... (showing first " << max_display << "x" << max_display << " results)" << endl;
    }
}

void validateResults(const vector<vector<float>>& cpuFloat,
                   const vector<vector<double>>& cpuDouble,
                   const vector<vector<float>>& gpuFloat,
                   const vector<vector<double>>& gpuDouble) {
    double maxDiffFloat = 0.0, maxDiffDouble = 0.0;
    int countDiffFloat = 0, countDiffDouble = 0;
    const double tolerance = 1e-5;
    
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < numberOfSamples; j++) {
            float diffFloat = fabs(cpuFloat[i][j] - gpuFloat[i][j]);
            double diffDouble = fabs(cpuDouble[i][j] - gpuDouble[i][j]);
            
            maxDiffFloat = max(maxDiffFloat, (double)diffFloat);
            maxDiffDouble = max(maxDiffDouble, diffDouble);
            
            if (diffFloat > tolerance) countDiffFloat++;
            if (diffDouble > tolerance) countDiffDouble++;
        }
    }
    
    cout << "\nValidation Results:" << endl;
    cout << "Max difference (float): " << maxDiffFloat << endl;
    cout << "Max difference (double): " << maxDiffDouble << endl;
    cout << "Values exceeding 1e-5 tolerance:" << endl;
    cout << "  Float: " << countDiffFloat << "/" << n*numberOfSamples << endl;
    cout << "  Double: " << countDiffDouble << "/" << n*numberOfSamples << endl;
    
    if (countDiffFloat > 0 || countDiffDouble > 0) {
        cout << "WARNING: Numerical differences detected!" << endl;
    } else {
        cout << "SUCCESS: All results match within tolerance" << endl;
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
                fprintf(stderr, "Invalid option\n");
                printUsage();
                return -1;
        }
    }
    return 0;
}

void printUsage() {
    printf("\nExponential Integral Calculator\n");
    printf("Usage: exponentialIntegral [options]\n\n");
    printf("Options:\n");
    printf("  -a <value>   Interval start (default: 0.0)\n");
    printf("  -b <value>   Interval end (default: 10.0)\n");
    printf("  -c           Disable CPU computation\n");
    printf("  -g           Disable GPU computation\n");
    printf("  -h           Show this help\n");
    printf("  -i <value>   Max iterations (default: 2000000000)\n");
    printf("  -n <value>   Maximum order n (default: 10)\n");
    printf("  -m <value>   Number of samples (default: 10)\n");
    printf("  -t           Enable timing output\n");
    printf("  -v           Enable verbose output\n\n");
    printf("Example:\n");
    printf("  ./exponentialIntegral -n 20000 -m 20000 -t -v\n");
}
