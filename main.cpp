#include <time.h>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <unistd.h>
#include "exponentialIntegralCuda.h"

using namespace std;

float   exponentialIntegralFloat          (const int n,const float x);
double  exponentialIntegralDouble         (const int n,const double x);
void    outputResults                    (const std::vector<std::vector<float>> &resultsFloat, 
                                         const std::vector<std::vector<double>> &resultsDouble,
                                         bool isCpu = true);
int     parseArguments                   (int argc, char **argv);
void    printUsage                       (void);

bool verbose, timing, cpu, gpu;
int maxIterations;
unsigned int n, numberOfSamples;
double a, b;    // The interval that we are going to use

int main(int argc, char *argv[]) {
    unsigned int ui, uj;
    cpu = true;
    gpu = true;
    verbose = false;
    timing = false;
    // n is the maximum order of the exponential integral that we are going to test
    // numberOfSamples is the number of samples in the interval [0,10] that we are going to calculate
    n = 10;
    numberOfSamples = 10;
    a = 0.0;
    b = 10.0;
    maxIterations = 2000000000;

    parseArguments(argc, argv);

    if (verbose) {
        cout << "n=" << n << endl;
        cout << "numberOfSamples=" << numberOfSamples << endl;
        cout << "a=" << a << endl;
        cout << "b=" << b << endl;
        cout << "timing=" << timing << endl;
        cout << "verbose=" << verbose << endl;
        cout << "cpu=" << cpu << endl;
        cout << "gpu=" << gpu << endl;
    }

    // Sanity checks
    if (a >= b) {
        cout << "Incorrect interval (" << a << "," << b << ") has been stated!" << endl;
        return 0;
    }
    if (n <= 0) {
        cout << "Incorrect orders (" << n << ") have been stated!" << endl;
        return 0;
    }
    if (numberOfSamples <= 0) {
        cout << "Incorrect number of samples (" << numberOfSamples << ") have been stated!" << endl;
        return 0;
    }

    std::vector<std::vector<float>> resultsFloatCpu, resultsFloatGpu;
    std::vector<std::vector<double>> resultsDoubleCpu, resultsDoubleGpu;
    double timeTotalCpu = 0.0, timeTotalGpu = 0.0;
    double timeFloatKernel = 0.0, timeDoubleKernel = 0.0;

    // CPU execution
    if (cpu) {
        struct timeval expoStart, expoEnd;
        try {
            resultsFloatCpu.resize(n, vector<float>(numberOfSamples));
            resultsDoubleCpu.resize(n, vector<double>(numberOfSamples));
        } catch (std::bad_alloc const&) {
            cout << "CPU results memory allocation fail!" << endl; 
            exit(1);
        }
        
        gettimeofday(&expoStart, NULL);
        double division = (b - a) / static_cast<double>(numberOfSamples);
        for (ui = 1; ui <= n; ui++) {
            for (uj = 1; uj <= numberOfSamples; uj++) {
                double x = a + uj * division;
                resultsFloatCpu[ui-1][uj-1] = exponentialIntegralFloat(ui, static_cast<float>(x));
                resultsDoubleCpu[ui-1][uj-1] = exponentialIntegralDouble(ui, x);
            }
        }
        gettimeofday(&expoEnd, NULL);
        timeTotalCpu = ((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - 
                       (expoStart.tv_sec + expoStart.tv_usec*0.000001));
    }

    // GPU execution
    if (gpu) {
        try {
            resultsFloatGpu.resize(n, vector<float>(numberOfSamples));
            resultsDoubleGpu.resize(n, vector<double>(numberOfSamples));
        } catch (std::bad_alloc const&) {
            cout << "GPU results memory allocation fail!" << endl; 
            exit(1);
        }
        runGPU(resultsFloatGpu, resultsDoubleGpu, n, numberOfSamples, a, b, 
               maxIterations, timeTotalGpu, timeFloatKernel, timeDoubleKernel);
    }

    // Timing results
    if (timing) {
        if (cpu) {
            printf("CPU execution time: %f seconds\n", timeTotalCpu);
        }
        if (gpu) {
            printf("GPU total time: %f seconds (includes memory transfers)\n", timeTotalGpu);
            printf("  - Float kernel time: %f seconds\n", timeFloatKernel);
            printf("  - Double kernel time: %f seconds\n", timeDoubleKernel);
            if (cpu) {
                printf("Speedup (CPU vs GPU total time): %.2fx\n", timeTotalCpu / timeTotalGpu);
            }
        }
    }

    // Numerical validation if both CPU and GPU were executed
    if (cpu && gpu) {
        double maxDiffFloat = 0.0, maxDiffDouble = 0.0;
        int countDiffFloat = 0, countDiffDouble = 0;
        double absTolerance = 1e-5;
        
        for (ui = 0; ui < n; ui++) {
            for (uj = 0; uj < numberOfSamples; uj++) {
                // Compare float results
                float cpuFloat = resultsFloatCpu[ui][uj];
                float gpuFloat = resultsFloatGpu[ui][uj];
                float diffFloat = fabs(cpuFloat - gpuFloat);
                
                // Compare double results
                double cpuDouble = resultsDoubleCpu[ui][uj];
                double gpuDouble = resultsDoubleGpu[ui][uj];
                double diffDouble = fabs(cpuDouble - gpuDouble);
                
                // Track maximum differences
                if (diffFloat > maxDiffFloat) maxDiffFloat = diffFloat;
                if (diffDouble > maxDiffDouble) maxDiffDouble = diffDouble;
                
                // Count values with significant differences
                if (diffFloat > absTolerance) countDiffFloat++;
                if (diffDouble > absTolerance) countDiffDouble++;
            }
        }
        
        printf("\nNumerical validation results:\n");
        printf("Max difference (float):  %e\n", maxDiffFloat);
        printf("Max difference (double): %e\n", maxDiffDouble);
        printf("Values exceeding tolerance (1e-5):\n");
        printf("  - Float:  %d/%d\n", countDiffFloat, n * numberOfSamples);
        printf("  - Double: %d/%d\n", countDiffDouble, n * numberOfSamples);
        
        if (countDiffFloat > 0 || countDiffDouble > 0) {
            printf("WARNING: Significant numerical differences detected between CPU and GPU!\n");
        } else {
            printf("SUCCESS: All values match within tolerance.\n");
        }
    }

    // Output results if verbose mode is enabled
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

void outputResults(const vector<vector<float>> &resultsFloat, 
                  const vector<vector<double>> &resultsDouble,
                  bool isCpu) {
    double division = (b - a) / static_cast<double>(numberOfSamples);
    for (unsigned int ui = 1; ui <= n; ui++) {
        for (unsigned int uj = 1; uj <= numberOfSamples; uj++) {
            double x = a + uj * division;
            cout << (isCpu ? "CPU" : "GPU") << " ==> ";
            cout << "E_" << ui << "(" << x << ") float: " << resultsFloat[ui-1][uj-1];
            cout << ", double: " << resultsDouble[ui-1][uj-1] << endl;
        }
    }
}

double exponentialIntegralDouble (const int n,const double x) {
    static const double eulerConstant=0.5772156649015329;
    double epsilon=1.E-30;
    double bigDouble=std::numeric_limits<double>::max();
    int i,ii,nm1=n-1;
    double a,b,c,d,del,fact,h,psi,ans=0.0;


    if (n<0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
        cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
        exit(1);
    }
    if (n==0) {
        ans=exp(-x)/x;
    } else {
        if (x>1.0) {
            b=x+n;
            c=bigDouble;
            d=1.0/b;
            h=d;
            for (i=1;i<=maxIterations;i++) {
                a=-i*(nm1+i);
                b+=2.0;
                d=1.0/(a*d+b);
                c=b+a/c;
                del=c*d;
                h*=del;
                if (fabs(del-1.0)<=epsilon) {
                    ans=h*exp(-x);
                    return ans;
                }
            }
            ans=h*exp(-x);
            return ans;
        } else { // Evaluate series
            ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);    // First term
            fact=1.0;
            for (i=1;i<=maxIterations;i++) {
                fact*=-x/i;
                if (i != nm1) {
                    del = -fact/(i-nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii=1;ii<=nm1;ii++) {
                        psi += 1.0/ii;
                    }
                    del=fact*(-log(x)+psi);
                }
                ans+=del;
                if (fabs(del)<fabs(ans)*epsilon) return ans;
            }
            //cout << "Series failed in exponentialIntegral" << endl;
            return ans;
        }
    }
    return ans;
}

float exponentialIntegralFloat (const int n,const float x) {
    static const float eulerConstant=0.5772156649015329;
    float epsilon=1.E-30;
    float bigfloat=std::numeric_limits<float>::max();
    int i,ii,nm1=n-1;
    float a,b,c,d,del,fact,h,psi,ans=0.0;

    if (n<0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
        cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
        exit(1);
    }
    if (n==0) {
        ans=exp(-x)/x;
    } else {
        if (x>1.0) {
            b=x+n;
            c=bigfloat;
            d=1.0/b;
            h=d;
            for (i=1;i<=maxIterations;i++) {
                a=-i*(nm1+i);
                b+=2.0;
                d=1.0/(a*d+b);
                c=b+a/c;
                del=c*d;
                h*=del;
                if (fabs(del-1.0)<=epsilon) {
                    ans=h*exp(-x);
                    return ans;
                }
            }
            ans=h*exp(-x);
            return ans;
        } else { // Evaluate series
            ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);    // First term
            fact=1.0;
            for (i=1;i<=maxIterations;i++) {
                fact*=-x/i;
                if (i != nm1) {
                    del = -fact/(i-nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii=1;ii<=nm1;ii++) {
                        psi += 1.0/ii;
                    }
                    del=fact*(-log(x)+psi);
                }
                ans+=del;
                if (fabs(del)<fabs(ans)*epsilon) return ans;
            }
            return ans;
        }
    }
    return ans;
}

int parseArguments (int argc, char *argv[]) {
    int c;

    while ((c = getopt (argc, argv, "cghn:m:a:b:i:tv")) != -1) {
        switch(c) {
            case 'c':
                cpu = false; break;     // Skip CPU test
            case 'g':
                gpu = false; break;     // Skip GPU test
            case 'h':
                printUsage(); exit(0); break;
            case 'i':
                maxIterations = atoi(optarg); break;
            case 'n':
                n = atoi(optarg); break;
            case 'm':
                numberOfSamples = atoi(optarg); break;
            case 'a':
                a = atof(optarg); break;
            case 'b':
                b = atof(optarg); break;
            case 't':
                timing = true; break;
            case 'v':
                verbose = true; break;
            default:
                fprintf(stderr, "Invalid option given\n");
                printUsage();
                return -1;
        }
    }
    return 0;
}

void printUsage () {
    printf("exponentialIntegral program - CUDA Accelerated Version\n");
    printf("by: Jose Mauricio Refojo <refojoj@tcd.ie>\n");
    printf("Modified for CUDA acceleration\n");
    printf("usage:\n");
    printf("exponentialIntegral.out [options]\n");
    printf("      -a   value   : Set the start of interval (default: 0.0)\n");
    printf("      -b   value   : Set the end of interval (default: 10.0)\n");
    printf("      -c           : Skip CPU execution\n");
    printf("      -g           : Skip GPU execution\n");
    printf("      -h           : Show usage\n");
    printf("      -i   size    : Set maximum iterations (default: 2000000000)\n");
    printf("      -n   size    : Set maximum order n (default: 10)\n");
    printf("      -m   size    : Set number of samples (default: 10)\n");
    printf("      -t           : Enable timing output\n");
    printf("      -v           : Enable verbose output\n");
    printf("\nBenchmark examples:\n");
    printf("  ./exponentialIntegral.out -n 5000 -m 5000 -t\n");
    printf("  ./exponentialIntegral.out -n 8192 -m 8192 -t\n");
    printf("  ./exponentialIntegral.out -n 16384 -m 16384 -t\n");
    printf("  ./exponentialIntegral.out -n 20000 -m 20000 -t\n");
}
