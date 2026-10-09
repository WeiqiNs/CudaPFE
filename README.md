# Pairing-based Functional Encryption in CUDA (CudaPFE)

[![CudaPFE CI](https://github.com/WeiqiNs/CudaPFE/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/WeiqiNs/CudaPFE/actions/workflows/ci.yml)

CudaPFE is a C++20 and CUDA library for BLS12-381 pairings and pairing-based functional encryption. Batch operations
run on a CPU engine or a CUDA engine; both execute the same `__host__ __device__` arithmetic, which is built on
[sppark](https://github.com/supranational/sppark)'s Montgomery fields and checked bit for bit against
[blst](https://github.com/supranational/blst). CudaPFE makes no constant-time guarantees and is meant for research
prototypes.

## Core API

`CudaPFE::core` (`<cudapfe/cudapfe.hpp>`) holds the scalar and group types `Zp`, `G1`, `G2` and `Gt` and their batch
form `Vec<T, E>` on the `Cpu` or `Gpu` engine:

```cpp
#include <cudapfe/cudapfe.hpp>

using namespace cudapfe;
const auto x = Vec<Zp, Gpu>::upload(Vector{1, 2, 3, 4});
const auto y = Vec<Zp, Gpu>::upload(Vector{5, 6, 7, 8});
const auto ps = mul_generator<G1>(x);
const auto qs = mul_generator<G2>(y);
const auto products = pair_segments(ps, qs, PairShape{1, 4});
const auto exponents = DlogTable<Gpu>(Gt::generator(), {0, 1000}).find(products);   // {70}
const auto sum = msm(ps, Vec<Zp, Gpu>::upload(Vector{1, 1, 1, 1}), MsmShape{1, {4, 1}});
const bool ok = pair(sum.to<Cpu>().download().front(), G2::generator()) == Gt::generator().pow(10);   // true
```

## Functional encryption

`CudaPFE::fe` is a header-only target in `include/cudapfe/fe/` with batch-first ports of the schemes in
[LibPFE](https://github.com/WeiqiNs/LibPFE): the function-hiding inner-product schemes `cudapfe::IPFE::{BJK, TAO, KIM,
LIN, KKS, OPT}` (`ipfe_<scheme>.hpp`) and the quadratic schemes `cudapfe::QFE::{BCFG, SGP}` (`qfe_<scheme>.hpp`). Every
scheme is a template over the engine, chosen once at `setup<Cpu>(n)` or `setup<Gpu>(n)`; LibPFE's README describes
the schemes and their papers. Consumers link `CudaPFE::fe`, which brings in `CudaPFE::core`. The Gpu engine pays off on
batches: it runs calls below the [placement thresholds](#calibrating-the-host-placement) on the host, but its setup
inversions and its discrete-log searches stay on the device, so for single keys and ciphertexts the Cpu engine is
usually faster. `cudapfe_bench_fe` shows where the two cross on a given machine.

```cpp
#include <cudapfe/fe/ipfe_opt.hpp>

using namespace cudapfe;
const auto msk = IPFE::OPT::setup<Gpu>(3);
const DlogTable<Gpu> table(IPFE::OPT::base(), {-1000, 1000});
const auto keys = IPFE::OPT::keygen(msk, IPFE::IntMatrix{{1, 2, 3}, {0, 1, 0}});
const auto ct = IPFE::OPT::enc(msk, IPFE::IntVec{4, -5, 6});
const auto results = IPFE::OPT::dec(table, IPFE::OPT::prepare(keys), ct);   // {12, -5}
```

- `keygen` and `enc` take a batch: an `IntMatrix` of rows for IPFE, a vector of `IntMatrix` functions and two
  `IntMatrix` of left and right vectors for QFE. The single-vector overloads make a batch of one. Each batch runs a
  fixed number of engine calls, whatever its size.
- `dec` pairs keys with ciphertexts: equal counts decrypt pairwise, one key decrypts every ciphertext, and one
  ciphertext decrypts under every key; any other combination throws `ShapeError`. It returns one
  `std::optional<std::int64_t>` per pair, empty when the result falls outside the searched range.
- Schemes that decrypt against a fixed base take a `DlogTable` over `msk.base` (TAO) or `<SCHEME>::base()`, so one
  table serves every decryption; BCFG's `dec` also takes the public key. BJK and KIM derive a base per pair and take a
  `Range` instead, searching it with `DlogTables` on every call.
- Every IPFE scheme has `prepare(sk)`, which precomputes the keys' pairing lines once (`G2Lines`, about 20 KB per G2
  point) for keys that decrypt many ciphertexts.
- BJK, TAO and KIM invert a random matrix at setup. On the Gpu engine the inversion must fit in free device memory;
  a larger setup throws `DeviceError` before it samples the matrix.

## Building

CudaPFE builds on Linux with CMake 3.25 or newer, a C++20 compiler with OpenMP, the CUDA toolkit (13.x) and git. blst
and GoogleTest are fetched at pinned commits; sppark is vendored under `third_party/sppark`. The Cpu engine runs each
batch call on every hardware thread; `OMP_NUM_THREADS` caps it. The installed package depends on OpenMP, so a project
that calls `find_package(CudaPFE)` must enable the CXX language.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build
```

Device code is compiled for `CMAKE_CUDA_ARCHITECTURES`, which defaults to `120`; pass `-DCMAKE_CUDA_ARCHITECTURES=<list>`
for other GPUs. On a GPU the build has no kernel image for, `gpu_available()` returns false, the Gpu engine throws a
`DeviceError` naming `-DCMAKE_CUDA_ARCHITECTURES`, and the GPU tests report as skipped.
`-DCUDAPFE_PTXAS_VERBOSE=ON` prints the register and spill report of every kernel.

A GPU is not required to build or test: without a CUDA device the GPU test cases report as skipped. Setting
`CUDA_VISIBLE_DEVICES=` reproduces that on a machine that has one.

## Benchmarks

[`bench/bench_core.cpp`](bench/bench_core.cpp) builds as `cudapfe_bench` and [`bench/bench_fe.cpp`](bench/bench_fe.cpp)
as `cudapfe_bench_fe` when `-DCUDAPFE_BUILD_BENCH=ON` is set; both need a CUDA device to run:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCUDAPFE_BUILD_BENCH=ON
cmake --build build --target cudapfe_bench cudapfe_bench_fe
./build/bench/cudapfe_bench            # full sizes
./build/bench/cudapfe_bench --quick    # small sizes, a few seconds
./build/bench/cudapfe_bench_fe         # every FE scheme; --quick for small sizes
```

`cudapfe_bench` prints Markdown tables for multi-pairing latency, batch shapes (single pairs among them), fixed-base
multiplication, matrix inversion and product, and discrete-log tables. Each table compares the Gpu engine with whichever
of the Cpu engine, blst on one core and blst on every hardware thread its columns name. The header names the GPU, its SM
count, the CPU thread count, the build type and the placement thresholds compiled in. Every timing is the median of
several runs after a warm-up, except that a case whose warm-up exceeds the cut-off stated in the header reports that
single run. Every table checks a result of the Gpu engine against blst (or, for matrices and discrete logs, against an
independent answer) and aborts on a mismatch. Each table's caption states when a baseline is timed on a sample and
scaled.

`cudapfe_bench_fe` prints LibPFE's table (Setup, KeyGen, Enc, Dec, Prepare and Prepared Dec, in ms per call) for every
scheme on both engines, once for single key/ciphertext pairs at growing n and once for batches. Each call handles the
whole batch, and every Dec is checked against the result computed in plain integers. A case the Cpu engine would take
too long on, or that does not fit on the device, prints the reason in place of its times.

### Calibrating the host placement

On the Gpu engine, `pair_segments` moves tiny workloads to the host: shapes below `CUDAPFE_HOST_MILLER_BELOW` pairs run
entirely on the host, and shapes below `CUDAPFE_HOST_FINAL_EXP_BELOW` segments run only their final exponentiations there.
Point addition, scaling, `mul_generator` and `msm` likewise run on the host below `CUDAPFE_HOST_POINTS_BELOW` results
(terms, for `msm`): a single GPU thread takes milliseconds per scalar multiplication, so small calls cost the same
fixed latency as large ones. The defaults are the cache variables in `CMakeLists.txt`, chosen from measurements on one
GPU; recalibrate for another:

1. Build a device-only copy with `-DCUDAPFE_HOST_MILLER_BELOW=0 -DCUDAPFE_HOST_FINAL_EXP_BELOW=0
   -DCUDAPFE_HOST_POINTS_BELOW=0` and run `cudapfe_bench --placement-sweep`. It times every shape of S segments of n pairs
   (powers of two up to 256, at most 1024 pairs) on the Gpu and Cpu engines. The smallest pair count at which the Gpu
   engine wins is the Miller threshold. Its second table times scaling and `mul_generator` in both groups at doubling N;
   the N at which the Gpu engine starts to win, interpolated between the last Cpu row and the first Gpu row, is the
   point threshold.
2. Build a second copy with `-DCUDAPFE_HOST_MILLER_BELOW=0` and a `CUDAPFE_HOST_FINAL_EXP_BELOW` above 256, so the Gpu
   engine always finishes on the host, and run the sweep again. Comparing its Gpu column with the device-only one gives
   the segment count at which device final exponentiation starts to win.
3. Configure the real build with the three measured values.

A kernel's block size is its functor's optional `threads` constant, and `threads_per_block` in
`src/support/for_each.cuh` supplies the default for functors without one. To tune the pairing kernels for a new GPU,
give `MillerItemOp` and `FinalExpOp` in `src/pairing/multi_pair.cu` a `threads` constant and compare tables 1 and 2
across candidate values.

## License

CudaPFE is licensed under the Apache License 2.0.
