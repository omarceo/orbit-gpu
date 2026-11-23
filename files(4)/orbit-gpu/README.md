# orbit-gpu

Batched satellite orbit propagation on the GPU (CUDA C++). Each GPU thread integrates one
satellite with an RK4 integrator under two-body gravity plus the J2 oblateness perturbation.
The same dynamics code is compiled for both host and device, so the GPU output is checked
against a CPU reference.

## Why this problem
Propagating a large fleet (constellations, debris fields, Monte Carlo runs) is embarrassingly
parallel: every satellite is independent, with heavy arithmetic per thread and almost no memory
traffic. It is a good fit for the GPU.

## Layout
```
include/orbit.hpp      dynamics (HD = __host__ __device__), RK4, element conversion
src/main.cu            kernel, CLI, CUDA event timing, CPU verification
tests/test_physics.cpp CPU-only physics checks (no GPU required)
Makefile
```

## Build and run
Requires an NVIDIA GPU and the CUDA toolkit.
```
make ARCH=sm_75            # pick your GPU architecture (sm_75 = Tesla T4)
./orbit_gpu --n 100000 --steps 8640 --dt 10
```
Options: `--n`, `--steps`, `--dt`, `--block`, `--check`, `--seed`, `--no-j2`.

No local GPU? Use a free Google Colab GPU runtime:
```
!git clone https://github.com/<your-user>/orbit-gpu && cd orbit-gpu && make ARCH=sm_75 && ./orbit_gpu --n 100000
```

## Physics validation (CPU, `make test`)
| Check | Result |
|---|---|
| Two-body energy drift over 1 day | 3.8e-10 (relative) |
| Closure of a circular orbit after one period | 1.6 mm |
| J2 nodal regression vs analytic secular rate | -4.520 vs -4.469 deg/day (1.1 % off, expected from short-period terms) |

## Results
Fill these in from your own run (`./orbit_gpu`) and keep the GPU name:

| GPU | Satellites | Steps | GPU time (ms) | Speedup vs 1 CPU thread | max position error vs CPU |
|---|---|---|---|---|---|
| | | | | | |

## Design notes
- One thread per satellite with a grid-stride loop, so any `n` works with any launch size.
- All state lives in registers during the integration; global memory is touched once to read and once to write.
- Double precision is used for accuracy. Consumer GPUs run FP64 at a fraction of FP32 speed; a mixed-precision
  variant would be faster but needs error analysis.
- Small differences between GPU and CPU come from fused multiply-add contraction on the device.

## Ideas to extend
- Structure-of-arrays layout and a block-size sweep, profiled with Nsight Compute (`ncu ./orbit_gpu`).
- FP32 / mixed-precision variant with an error study.
- Add atmospheric drag and compare cost per step.
- WebGPU (WGSL) port of the kernel.
