CC = g++
NVCC = nvcc
CFLAGS = -O3 -Wall
NVCCFLAGS = -arch=sm_61 -O3 --expt-relaxed-constexpr
LDFLAGS = -lcudart -L/usr/local/cuda/lib64
TARGET = exponentialIntegral
OBJS = main.o exponentialIntegralCuda.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

main.o: main.cpp exponentialIntegralCuda.h
	$(CC) $(CFLAGS) -c $<

exponentialIntegralCuda.o: exponentialIntegralCuda.cu exponentialIntegralCuda.h
	$(NVCC) $(NVCCFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean
