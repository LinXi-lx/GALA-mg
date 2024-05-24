CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3
NVCCFLAGS = -arch=sm_80 -O3
MPI_FLAGS := -I /home/xlin/openmpi/include -L /home/xlin/openmpi/lib -lmpi 
INCLUDE_DIRS = ./

all: preprocess gala_main

preprocess: preprocess.cpp graph/graph.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

gala_main: gala_main.cu graph/graph.cpp louvain_gpu/louvain.cu louvain_gpu/louvain_main_process.cu louvain_gpu/build_compressed_graph.cu louvain_gpu/kernel_functions.cu
	$(NVCC) $^ -I $(INCLUDE_DIRS) -I /home/xlin/nccl/include/ $(NVCCFLAGS) $(MPI_FLAGS) -L /home/xlin/nccl/lib -lnccl --extended-lambda -o $@

clean:
	rm -f preprocess gala_main


