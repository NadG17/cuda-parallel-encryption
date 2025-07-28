NVCC = nvcc
INCLUDES = -I include
SRCS = src/main.cu src/aes_kernels.cu src/rc4_kernels.cu src/file_processor.cu src/key_management.cu src/performance_monitor.cu src/aes_rounds.cu
TARGET = cuda_encrypt

all:
	$(NVCC) $(SRCS) $(INCLUDES) -o $(TARGET) -O3

clean:
	rm -f $(TARGET)
