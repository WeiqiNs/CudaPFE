# CUDA Functional Encryption Library (LibCuFE)

LibCuFE is a C++20 and CUDA library for BLS12-381 pairings and function-hiding inner-product encryption. Batch
operations run on a CPU engine or a CUDA engine; both execute the same `__host__ __device__` arithmetic, which is built
on [sppark](https://github.com/supranational/sppark)'s Montgomery fields and checked bit for bit against
[blst](https://github.com/supranational/blst). LibCuFE makes no constant-time guarantees and is meant for research
prototypes.

## Building

LibCuFE builds on Linux with CMake 3.25 or newer, a C++20 compiler, the CUDA toolkit (13.x) and git. blst and
GoogleTest are fetched at pinned commits; sppark is vendored under `third_party/sppark`.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build
```

Device code is compiled for `CMAKE_CUDA_ARCHITECTURES`, which defaults to `120`; pass `-DCMAKE_CUDA_ARCHITECTURES=<list>`
for other GPUs. `-DCUFE_PTXAS_VERBOSE=ON` prints the register and spill report of every kernel.

A GPU is not required to build or test: without a CUDA device the GPU test cases report as skipped. Setting
`CUDA_VISIBLE_DEVICES=` reproduces that on a machine that has one.

## Benchmarks

[`bench/bench_core.cpp`](bench/bench_core.cpp) builds as `cufe_bench` when `-DCUFE_BUILD_BENCH=ON` is set, and needs a
CUDA device to run:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCUFE_BUILD_BENCH=ON
cmake --build build --target cufe_bench
./build/bench/cufe_bench            # full sizes
./build/bench/cufe_bench --quick    # small sizes, a few seconds
```

It prints Markdown tables for pairing throughput, multi-pairing latency, batch shapes, fixed-base multiplication,
matrix inversion and product, and discrete-log tables. Each table compares the Gpu engine with whichever of the Cpu
engine, blst on one core and blst on every hardware thread its columns name. The header names the GPU, its SM count,
the CPU thread count, the build type and the placement thresholds compiled in. Every timing is the median of several
runs after a warm-up, except that a case whose warm-up exceeds the cut-off stated in the header reports that single run.
Every table checks a result of the Gpu engine against blst (or, for matrices and discrete logs, against an independent
answer) and aborts on a mismatch. Each table's caption states when a baseline is timed on a sample and scaled.

### Calibrating the host placement

On the Gpu engine, `pair_segments` moves tiny workloads to the host: shapes below `CUFE_HOST_MILLER_BELOW` pairs run
entirely on the host, and shapes below `CUFE_HOST_FINAL_EXP_BELOW` segments run only their final exponentiations there.
The defaults are the cache variables in `CMakeLists.txt`, chosen from measurements on one GPU; recalibrate for another:

1. Build a device-only copy with `-DCUFE_HOST_MILLER_BELOW=0 -DCUFE_HOST_FINAL_EXP_BELOW=0` and run
   `cufe_bench --placement-sweep`. It times every shape of S segments of n pairs (powers of two up to 256, at most 1024
   pairs) on the Gpu and Cpu engines. The smallest pair count at which the Gpu engine wins is the Miller threshold.
2. Build a second copy with `-DCUFE_HOST_MILLER_BELOW=0` and a `CUFE_HOST_FINAL_EXP_BELOW` above 256, so the Gpu
   engine always finishes on the host, and run the sweep again. Comparing its Gpu column with the device-only one gives
   the segment count at which device final exponentiation starts to win.
3. Configure the real build with the two measured values.

A kernel's block size is its functor's optional `threads` constant, and `threads_per_block` in
`src/support/for_each.cuh` supplies the default for functors without one. To tune the pairing kernels for a new GPU,
give `MillerItemOp` and `FinalExpOp` in `src/pairing/multi_pair.cu` a `threads` constant and compare tables 1 and 2
across candidate values.

## License

LibCuFE is licensed under the Apache License 2.0.
