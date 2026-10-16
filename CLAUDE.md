# CudaPFE for agents

C++20 and CUDA library for BLS12-381 pairings and pairing-based functional encryption: the public API in
`include/cudapfe/` (`Vec<T, E>`, `msm`, `pair_segments`, `DlogTable`, and the FE schemes in `fe/`), the arithmetic and
kernels in `src/`. It is the GPU counterpart of LibPFE. `README.md` is the user-facing overview.

## Commands

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCUDAPFE_BUILD_BENCH=ON
cmake --build build
ctest --test-dir build --output-on-failure
./build/bench/cudapfe_bench --quick && ./build/bench/cudapfe_bench_fe --quick
cuobjdump --dump-resource-usage build/libcudapfe_core.a
```

A full build takes minutes, dominated by device code generation for `src/vec/vec.cu` and
`test/internal/test_curve.cu`. `-DCUDAPFE_PTXAS_VERBOSE=ON` prints every kernel's register and spill report. CI builds
and tests in a CUDA container without a GPU, then builds the demo in `demo/` against the installed package.

## Invariants

- Every operation is a `CUDAPFE_HD` functor that `detail::for_each<E>` (`src/support/for_each.cuh`) runs with OpenMP on
  `Cpu` or as a kernel on `Gpu`. Op code has no `__CUDA_ARCH__` branches, and both engines return identical results.
- Tests are typed over both engines and compare against blst (`test/support/oracle.hpp`) or plain integers. Gpu cases
  skip through `CUDAPFE_REQUIRE_GPU` when `gpu_available()` is false, so a run with skipped Gpu cases proves nothing
  about device code: check the skip count. CUDA also needs the `nvidia_uvm` module (`/dev/nvidia-uvm`).
- Gpu calls below the `CUDAPFE_HOST_*_BELOW` cache variables in `CMakeLists.txt` reroute to the Cpu engine
  (`detail::points_on_host`, `detail::place`), so a Gpu test or benchmark shape must exceed them to reach the device.
- The Gpu engine is one process-wide stream (`GpuRuntime::require()`), and each operation synchronizes before
  returning; nothing overlaps across calls.
- Kernels select on data without branching where lanes of a warp disagree, such as the sign in `signed_scalar` and
  `straus_sum`. `msm` routes by the widest scalar against `kStrausMaxBits`; change it only with the width table of
  `cudapfe_bench` section 7 in view.
- A performance change states before and after numbers from `cudapfe_bench` or `cudapfe_bench_fe` on an idle GPU, both
  builds made with the same toolkit. Small Cpu-engine timings that land on exact multiples of 3 ms are scheduler noise.
- `third_party/sppark` is vendored and not edited. LibPFE, GoPFE, RustPFE and CudaPFE keep the same constructions;
  change a scheme in one only with the others in view. No comments in code.
