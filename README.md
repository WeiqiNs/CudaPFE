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

## License

LibCuFE is licensed under the Apache License 2.0. See `NOTICE` for the sppark and blst attributions.
