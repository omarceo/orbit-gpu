NVCC ?= nvcc
CXX  ?= g++
ARCH ?= native   # e.g. ARCH=sm_75 for a Colab T4

all: orbit_gpu test

orbit_gpu: src/main.cu include/orbit.hpp
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -o $@ src/main.cu

test: tests/test_physics.cpp include/orbit.hpp
	$(CXX) -O2 -std=c++17 -o test_physics tests/test_physics.cpp
	./test_physics

clean:
	rm -f orbit_gpu test_physics
