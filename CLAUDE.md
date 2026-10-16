# CudaPFE for agents

C++20 and CUDA library for BLS12-381 pairings and pairing-based functional encryption, the GPU counterpart of LibPFE.
`README.md` is the user-facing overview.

## Commands

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCUDAPFE_BUILD_BENCH=ON
cmake --build build
ctest --test-dir build --output-on-failure
./build/bench/cudapfe_bench --quick && ./build/bench/cudapfe_bench_fe --quick
```

A full build takes minutes, dominated by device code generation for `src/vec/vec.cu` and
`test/internal/test_curve.cu`. `-DCUDAPFE_PTXAS_VERBOSE=ON` prints every kernel's register and spill report, and
`cuobjdump --dump-resource-usage build/libcudapfe_core.a` reads it from a built library. CI
(`.github/workflows/ci.yml`) runs two jobs in a CUDA container without a GPU: the test suite in Release with Gpu cases
skipped, and a tests-off, bench-on build that installs the package and builds and runs `demo/` against it, so the
benchmarks must compile even though CI never runs them.

## Layout

- `include/cudapfe/`: the public API (`Vec<T, E>`, `msm`, `pair_segments`, `DlogTable`), with the header-only FE
  schemes in `fe/`.
- `src/{field,curve,pairing,linalg,dlog,vec,support}`: arithmetic and kernels, the private `cudapfe_internal`
  include root. `third_party/sppark` is vendored and not edited.
- `test/fe/` for the schemes, `test/internal/` for `src/`-level tests, `test/support/` for engines and the blst
  oracle; `bench/` holds `cudapfe_bench` and `cudapfe_bench_fe`.

## Invariants

- Every operation is a `CUDAPFE_HD` functor that `detail::for_each<E>` (`src/support/for_each.cuh`) runs as a kernel
  on `Gpu` or with OpenMP on `Cpu`. An op may declare `threads` (block size) and `host_parallel_from` (the OpenMP
  cutoff); `for_each.cuh` supplies the defaults. `-Werror=cross-execution-space-call` makes a host-only call inside
  an op a compile error. Execution-space branches (`__CUDA_ARCH__`) live only in `Field` (`src/field/field.hpp`:
  sppark's storage layout and inverse) and `MillerItemOp` (`src/pairing/multi_pair.cu`); every other op is written
  once, and both engines return identical results.
- The Gpu engine is one process-wide non-blocking stream (`GpuRuntime::require()`, `src/support/runtime.hpp`). Every
  op ends with `detail::finish<E>()` (`src/vec/storage.hpp`), so nothing overlaps across calls, and a new op must call
  it too.
- Tests are typed over both engines and compare against blst (`test/support/oracle.hpp`) or plain integers. A new
  scheme joins through one adapter in `test/fe/schemes.hpp` and the `OnEveryEngine` lists in
  `test/fe/scheme_types.hpp`; engine-typed cases derive from `EngineTest`/`CaseTest` (`test/support/engines.hpp`).
  Tests that include `src/` headers are declared `INTERNAL` in `test/CMakeLists.txt`.
- Gpu cases skip through `CUDAPFE_REQUIRE_GPU` when `gpu_available()` is false, so a run with skipped Gpu cases proves
  nothing about device code: check the skip count. CUDA also needs the `nvidia_uvm` module (`/dev/nvidia-uvm`).
- Gpu calls below the `CUDAPFE_HOST_*_BELOW` cache variables in `CMakeLists.txt` reroute to the Cpu engine
  (`detail::points_on_host`, `detail::place`). A test or benchmark that must reach the device sizes its shapes from
  `detail::kHostMillerBelow`/`kHostPointsBelow` and asserts the placement, as
  `ThreadsShareOneKeyThroughTheDevicePointAndPairingKernels` in `test/fe/test_concurrency.cpp` does; never a literal.
- Keys, prepared keys and `DlogTable` are immutable and shared across threads; `test/fe/test_concurrency.cpp` is the
  contract. There is no TSAN configure option: a hand-made `-fsanitize=thread` build must run with
  `OMP_NUM_THREADS=1`, because libgomp is uninstrumented and every parallel region otherwise reports a false race.
- Kernels select on data without branching where lanes of a warp disagree, such as the sign in `signed_scalar` and
  `straus_sum`. `msm` routes by the widest scalar against `kStrausMaxBits` (`src/curve/straus.hpp`); change it only
  with `cudapfe_bench`'s "Single MSM by scalar width" table in view.
- A performance change states before and after numbers from `cudapfe_bench` or `cudapfe_bench_fe` on an idle GPU,
  both builds made with the same toolkit.
- LibPFE, GoPFE, RustPFE and CudaPFE keep the same constructions; change a scheme in one only with the others in view.
  No comments in code.
