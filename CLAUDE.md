# CudaPFE for agents

C++20/CUDA library for BLS12-381 pairings and eight pairing-based functional encryption schemes. Every operation is
written once as `__host__ __device__` code and run by the `Gpu` engine (CUDA kernels) or the `Cpu` engine (OpenMP).
It is the GPU sibling of LibPFE (C++ on RELIC), GoPFE and RustPFE: all four implement the same constructions under the
same scheme names (`IPFE::BJK`, `QFE::SGP`, ...), so a scheme change here has counterparts there. `README.md` is the
user-facing overview and holds the scheme table.

## Commands

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDAPFE_BUILD_BENCH=ON
cmake --build build                            # minutes; src/vec/vec.cu and test/internal/test_curve.cu dominate
ctest --test-dir build --output-on-failure     # Gpu cases report "Skipped" without a usable device
ctest --test-dir build -R MsmTest              # or ./build/test/test_msm --gtest_filter='MsmTest/*'
./build/bench/cudapfe_bench --quick            # pairing, fixed-base, matrix, dlog, MSM tables vs blst
./build/bench/cudapfe_bench_fe --quick         # Setup/KeyGen/Enc/Dec per scheme, Gpu vs Cpu engine
./build/bench/cudapfe_bench --placement-sweep  # timings for retuning the CUDAPFE_HOST_*_BELOW thresholds
cmake --install build --prefix /tmp/cudapfe    # then, as CI does:
cmake -S demo -B demo/build -G Ninja -DCMAKE_PREFIX_PATH=/tmp/cudapfe && cmake --build demo/build && ./demo/build/demo
```

Both benchmarks exit early without a GPU; ctest names carry the engine type (`...<cudapfe::Gpu>`). Configure fetches
blst (`CUDAPFE_BLST_GIT_TAG`, `cmake/CudaPFEDependencies.cmake`) and, absent a system copy, GoogleTest. Options are the
cache variables at the top of `CMakeLists.txt`; `CUDAPFE_PTXAS_VERBOSE=ON` prints per-kernel registers and spills
(`cuobjdump --dump-resource-usage build/libcudapfe_core.a` reads them from a built library). A build lacking the
device's `CMAKE_CUDA_ARCHITECTURES` makes `GpuRuntime` throw a `DeviceError` naming the value to configure.

## Layout

- `include/cudapfe/`: public API. `core.hpp` (host values `Zp`, `G1`/`G2`, `Gt`, single `pair`), `engine.hpp`,
  `vec.hpp` (`Vec<T, E>`, `mul_generator`, `msm`, `concat`), `pairing.hpp` (`prepare` → `G2Lines`, `pair_segments`),
  `linalg.hpp` (`Matrix<E>`), `dlog.hpp` (`DlogTable`, `DlogTables`), `vector.hpp`, `io.hpp`, `errors.hpp`.
- `include/cudapfe/fe/`: header-only schemes, one per file; shared helpers in `ipfe.hpp` and `qfe.hpp`.
- `src/field/` (`Fp`, `Fr`, tower to `Fp12`, constants, SHA-256 counter RNG in `random.cpp`), `src/curve/` (affine and
  Jacobian arithmetic, `SignedScalar`, `straus_sum`, generator table, encoding), `src/pairing/` (Miller loop, final
  exponentiation, placement), `src/vec/` (`Storage`, `Layout`, `Chunks`, segmented reduction, `Vec` ops),
  `src/linalg/` (Gauss-Jordan), `src/dlog/` (baby-step giant-step), `src/support/` (`for_each`, `GpuRuntime`,
  `DeviceArray`, `Access`, `CUDAPFE_HD`).
- `third_party/sppark/`: sppark's BLS12-381 field headers. `test/`: `fe/` scheme suites, `internal/` tests of
  `src/`, `support/` (engine fixtures, blst oracle), public-API suites at the top. `demo/`: installed-package consumer.
- Targets: `CudaPFE::core`, `CudaPFE::fe` (interface over core), private `cudapfe_internal` (`src/`, sppark, blst).

## Data flow and where changes go

- `Vec<T, E>` is an immutable shared handle to `detail::Storage` (`src/vec/storage.hpp`): a `std::vector` on `Cpu`, a
  `DeviceArray` on `Gpu`, of device elements (`Fr`, affine points, `Fp12`). `upload`, `download`, `at` and `to<E>()`
  cross between host values and engines; `detail::Access` bit-casts between public types and elements.
- An operation is a `CUDAPFE_HD` functor (`struct ...Op` in its `.cu`'s anonymous namespace) passed to
  `detail::for_each<E>` (`src/support/for_each.cuh`): a `run_each` kernel on `Gpu`, an OpenMP loop on `Cpu`. It may set
  `threads` (block size) and `host_parallel_from` (OpenMP cutoff). The caller then runs `detail::finish<E>()`.
- Public templates are declared in `include/cudapfe/`, defined in `src/**/*.cu` and explicitly instantiated for both
  engines at the bottom of that file. New sources go in `add_library(cudapfe_core ...)`; new tests in
  `cudapfe_add_test` (`test/CMakeLists.txt`), marked `INTERNAL` when they include `src/` headers.
- `msm` takes `straus_sum` when the widest signed scalar fits `kStrausMaxBits` (`src/curve/straus.hpp`), else per-term
  `mul` plus `reduce_segments`. `pair_segments` chunks each segment over resident threads (at most `kMaxPairsPerItem`
  pairs per Gpu item), reduces the Miller products per segment, then final-exponentiates each.
- Schemes build exponents as host `Vector`s or `Matrix<E>` products, lift them with `mul_generator`
  (`IPFE::detail::lift`), and decrypt via `pair_segments` and a `DlogTable`/`DlogTables` lookup. A new scheme needs its
  header, an adapter in `test/fe/schemes.hpp`, an entry in `InnerProductSchemes` or `QuadraticSchemes`
  (`test/fe/scheme_types.hpp`, feeding `test_ipfe`, `test_qfe`, `test_concurrency`), and a `footprint<>` plus `table<>`
  entry in `bench/bench_fe.cpp`.

## Design facts

- Engines share one implementation and return identical bytes. `__CUDA_ARCH__` branches exist only in `detail::Field`
  (`src/field/field.hpp`: host `Base` is blst's `blst_t` with 64-bit limbs, device is sppark's `mont_t` with 32-bit
  limbs, and the device inverse is a `pow`) and `MillerItemOp` (`src/pairing/multi_pair.cu`: the device packs pair
  liveness into `MaskedPairs`, sized for `kMaxPairsPerItem`). `-Werror=cross-execution-space-call` rejects host-only
  calls in ops. Pinned by the `DeviceMatchesHost` and `...MatchesBlst` cases under `test/internal/`.
- `third_party/sppark` stays byte-identical to upstream so it can be refreshed with the blst pin: on the host its
  `ff/bls12-381.hpp` includes blst's `blst_t.hpp`. `FieldTest.ConstantsMatchVendoredSppark` pins
  `src/field/constants.hpp` to it. blst also backs `Zp::random`/`seed` and is the tests' oracle.
- `Gpu` work runs on one process-wide non-blocking stream (`GpuRuntime::require()`, `src/support/runtime.cu`), and
  `finish<E>()` synchronizes it, so calls never overlap. Keys, `G2Lines` and `DlogTable` are immutable and shared
  across threads; `test/fe/test_concurrency.cpp` pins this.
- Small `Gpu` calls run on the host: point addition, scaling, `mul_generator` and `msm` below `kHostPointsBelow`
  (`src/vec/placement.hpp`), pairings per `detail::place` (`src/pairing/plan.hpp`), all from the
  `CUDAPFE_HOST_*_BELOW` cache variables. Tests that must reach the device size shapes from these constants and assert
  the placement (`MultiPairTest.EveryPlacementAgrees`, `ConcurrentDeviceTest.*`); `PlanTest` pins `place`.
- Kernels select on data without warp-divergent branches (`signed_scalar`, `straus_sum`; README "Performance" has the
  measurements). `MsmTest.ScalarsOfZeroAndBoundaryWidthsSumCorrectly` hard-codes the widths at and just past
  `kStrausMaxBits`, and `cudapfe_bench`'s "Single MSM by scalar width" table measures that crossover.

## Tests and CI

Suites are typed over both engines via `EngineTest`/`CaseTest` (`test/support/engines.hpp`) and `OnEveryEngine`, and
check against blst (`test/support/oracle.hpp`) or plain integers. Device cases call `CUDAPFE_REQUIRE_GPU()`, which
skips when `gpu_available()` is false, including when `/dev/nvidia-uvm` is missing; a run with skips proves nothing
about device code. `.github/workflows/ci.yml` runs on GPU-less `nvidia/cuda` containers: `test` builds and runs the
suite with every device case skipped; `package` builds tests-off, bench-on (benchmarks must compile), installs, and
runs `demo/`, which falls back to `Cpu`. Device correctness and benchmarks are verified only locally.
