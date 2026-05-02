CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
TARGET := main
SRC := main.cu

GPU_ARCH ?= sm_86
NVCCFLAGS ?= -O2 -std=c++17 -arch=$(GPU_ARCH)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
