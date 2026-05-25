NC ?= /home/zyl/.conda/envs/torch2.8/bin/nvcc
CXXFLAGS ?= -std=c++11 -O3
NFLAGS ?= -arch=sm_70

TARGET := pagerank_convergence
SRC := pagerank_convergence.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NC) $(SRC) -o $(TARGET) $(CXXFLAGS) $(NFLAGS)

clean:
	rm -f $(TARGET)
