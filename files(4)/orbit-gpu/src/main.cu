// orbit-gpu: batched J2 orbit propagation, one GPU thread per satellite.
// Validates against a CPU reference and reports throughput.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include "../include/orbit.hpp"

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",                     \
                         cudaGetErrorString(err_), __FILE__, __LINE__);            \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

// Grid-stride loop: correct for any n, independent of launch size.
__global__ void propagate_kernel(const State* in, State* out, int n, double dt, int steps, bool useJ2) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = propagate(in[i], dt, steps, useJ2);
    }
}

struct Options {
    int n = 100000;        // number of satellites
    int steps = 8640;      // RK4 steps per satellite
    double dt = 10.0;      // step size [s]
    int block = 128;       // threads per block
    int check = 200;       // satellites verified against the CPU reference
    unsigned long long seed = 42;
    bool j2 = true;
};

static Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char* k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
        if (is("--n")) o.n = std::atoi(argv[++i]);
        else if (is("--steps")) o.steps = std::atoi(argv[++i]);
        else if (is("--dt")) o.dt = std::atof(argv[++i]);
        else if (is("--block")) o.block = std::atoi(argv[++i]);
        else if (is("--check")) o.check = std::atoi(argv[++i]);
        else if (is("--seed")) o.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--no-j2") == 0) o.j2 = false;
        else {
            std::printf("usage: %s [--n N] [--steps S] [--dt SEC] [--block B] [--check K] [--seed X] [--no-j2]\n", argv[0]);
            std::exit(std::strcmp(argv[i], "--help") == 0 ? 0 : 1);
        }
    }
    o.check = std::min(o.check, o.n);
    return o;
}

static std::vector<State> make_fleet(int n, unsigned long long seed) {
    const double PI = std::acos(-1.0);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<State> v(n);
    for (int i = 0; i < n; ++i) {
        double a = 6.8e6 + U(rng) * 1.0e6;       // LEO altitudes
        double e = U(rng) * 0.02;
        double inc = U(rng) * 98.0 * PI / 180.0;
        v[i] = from_elements(a, e, inc, U(rng) * 2 * PI, U(rng) * 2 * PI, U(rng) * 2 * PI);
    }
    return v;
}

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s (SMs=%d)\n", prop.name, prop.multiProcessorCount);
    std::printf("fleet: %d satellites, %d RK4 steps of %.1f s (%.2f h simulated), J2 %s\n",
                o.n, o.steps, o.dt, o.steps * o.dt / 3600.0, o.j2 ? "on" : "off");

    std::vector<State> h_in = make_fleet(o.n, o.seed), h_out(o.n);

    State *d_in = nullptr, *d_out = nullptr;
    const size_t bytes = sizeof(State) * (size_t)o.n;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    const int grid = (o.n + o.block - 1) / o.block;

    // Warm-up launch (context creation, clocks), then timed launch.
    propagate_kernel<<<grid, o.block>>>(d_in, d_out, o.n, o.dt, o.steps, o.j2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    propagate_kernel<<<grid, o.block>>>(d_in, d_out, o.n, o.dt, o.steps, o.j2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float gpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t0, t1));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // CPU reference on the first `check` satellites.
    auto c0 = std::chrono::steady_clock::now();
    double max_pos_err = 0.0, max_vel_err = 0.0;
    for (int i = 0; i < o.check; ++i) {
        State ref = propagate(h_in[i], o.dt, o.steps, o.j2);
        const State& g = h_out[i];
        double dp = std::sqrt((ref.x - g.x) * (ref.x - g.x) + (ref.y - g.y) * (ref.y - g.y) + (ref.z - g.z) * (ref.z - g.z));
        double dv = std::sqrt((ref.vx - g.vx) * (ref.vx - g.vx) + (ref.vy - g.vy) * (ref.vy - g.vy) + (ref.vz - g.vz) * (ref.vz - g.vz));
        max_pos_err = std::max(max_pos_err, dp);
        max_vel_err = std::max(max_vel_err, dv);
    }
    auto c1 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

    const double gpu_rate = o.n / (gpu_ms / 1000.0);
    const double cpu_rate = o.check > 0 ? o.check / (cpu_ms / 1000.0) : 0.0;

    std::printf("\nGPU kernel time : %.3f ms  (%.3e satellites/s)\n", gpu_ms, gpu_rate);
    if (o.check > 0) {
        std::printf("CPU 1 thread    : %.3f ms for %d sats (%.3e satellites/s)\n", cpu_ms, o.check, cpu_rate);
        std::printf("speedup vs 1 CPU thread: %.1fx\n", gpu_rate / cpu_rate);
        std::printf("max |dr| GPU vs CPU: %.3e m   max |dv|: %.3e m/s  -> %s\n",
                    max_pos_err, max_vel_err, max_pos_err < 1e-2 ? "OK" : "MISMATCH");
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return (o.check > 0 && max_pos_err >= 1e-2) ? 2 : 0;
}
