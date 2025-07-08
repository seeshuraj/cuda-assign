CC = g++
NVCC = nvcc
CFLAGS = -O3 -Wall
NVCCFLAGS = -arch=sm_61 -O3 -Wno-deprecated-gpu-targets --expt-relaxed-constexpr
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
	rm -f $(TARGET) *.o

run: all
	./$(TARGET) -n 5000 -m 5000 -t
	./$(TARGET) -n 8192 -m 8192 -t
	./$(TARGET) -n 16384 -m 16384 -t
	./$(TARGET) -n 20000 -m 20000 -t
