# Makefile for segment_sum kernel benchmarks (single GPU, no MPI/NCCL)

TARGET       = segment_sum_bench
TARGET_VEC   = segment_sum_bench_vec
TARGET_SAG   = sag_bench
TARGET_DEDUP = sag_bench_dedup
TARGET_GESPM = ge_spmm_bench
TARGET_PUSH  = ge_spmm_push_bench
TARGET_CMP   = kernel_compare
SRC          = segment_sum_bench.cu
SRC_VEC      = segment_sum_bench_vec.cu
SRC_SAG      = sag_bench.cu
SRC_DEDUP    = sag_bench_dedup.cu
SRC_GESPM    = ge_spmm_bench.cu
SRC_PUSH     = ge_spmm_push_bench.cu
SRC_CMP      = kernel_compare.cu

NVCC      = nvcc
NVCCFLAGS = -std=c++17 -O3 -lineinfo -arch=sm_70
CUDA_PATH ?= /usr/local/cuda

all: $(TARGET) $(TARGET_VEC) $(TARGET_SAG) $(TARGET_DEDUP) $(TARGET_GESPM) $(TARGET_PUSH) $(TARGET_CMP)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) \
		-o $(TARGET) $(SRC) \
		-L$(CUDA_PATH)/lib64 -lcudart

$(TARGET_VEC): $(SRC_VEC)
	$(NVCC) $(NVCCFLAGS) \
		-o $(TARGET_VEC) $(SRC_VEC) \
		-L$(CUDA_PATH)/lib64 -lcudart

$(TARGET_SAG): $(SRC_SAG)
	$(NVCC) $(NVCCFLAGS) \
		-o $(TARGET_SAG) $(SRC_SAG) \
		-L$(CUDA_PATH)/lib64 -lcudart

$(TARGET_DEDUP): $(SRC_DEDUP)
	$(NVCC) $(NVCCFLAGS) \
		-o $(TARGET_DEDUP) $(SRC_DEDUP) \
		-L$(CUDA_PATH)/lib64 -lcudart

$(TARGET_GESPM): $(SRC_GESPM)
	$(NVCC) $(NVCCFLAGS) \
		-o $(TARGET_GESPM) $(SRC_GESPM) \
		-L$(CUDA_PATH)/lib64 -lcudart -lcuda

$(TARGET_PUSH): $(SRC_PUSH)
	$(NVCC) $(NVCCFLAGS) \
		-o $(TARGET_PUSH) $(SRC_PUSH) \
		-L$(CUDA_PATH)/lib64 -lcudart

$(TARGET_CMP): $(SRC_CMP)
	$(NVCC) $(NVCCFLAGS) \
		-o $(TARGET_CMP) $(SRC_CMP) \
		-L$(CUDA_PATH)/lib64 -lcudart

clean:
	rm -f $(TARGET) $(TARGET_VEC) $(TARGET_SAG) $(TARGET_DEDUP) $(TARGET_GESPM) $(TARGET_PUSH) $(TARGET_CMP)

.PHONY: clean all
