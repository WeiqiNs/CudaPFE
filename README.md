# Pairing-based Functional Encryption in CUDA (CudaPFE)

[![CudaPFE CI](https://github.com/WeiqiNs/CudaPFE/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/WeiqiNs/CudaPFE/actions/workflows/ci.yml)

Functional encryption lets the holder of a key learn one function of encrypted data, such as an inner product, and
nothing else. CudaPFE is a C++20 and CUDA library that runs eight published pairing-based schemes, six function-hiding
inner-product functional encryption (IPFE) schemes and two quadratic functional encryption (QFE) schemes, in batches
on an NVIDIA GPU, on BLS12-381 pairing arithmetic written as CUDA kernels. It is the GPU counterpart of
[LibPFE](https://github.com/WeiqiNs/LibPFE), with the same schemes. CudaPFE makes no constant-time guarantees and is
meant for research prototypes.

```cpp
#include <cudapfe/fe/ipfe_opt.hpp>

using namespace cudapfe;
const auto msk = IPFE::OPT::setup<Gpu>(3);
const DlogTable<Gpu> table(IPFE::OPT::base(), {-1000, 1000});
const auto keys = IPFE::OPT::keygen(msk, IPFE::IntMatrix{{1, 2, 3}, {0, 1, 0}});
const auto ct = IPFE::OPT::enc(msk, IPFE::IntVec{4, -5, 6});
const auto results = IPFE::OPT::dec(table, IPFE::OPT::prepare(keys), ct);   // {12, -5}
```

`results` holds the inner product of the message with each key's vector, and the keys reveal nothing else about it.

## Highlights

- **BLS12-381 on the GPU.** The field tower, curve arithmetic, Miller loops, final exponentiation, multi-scalar
  multiplication, Gauss–Jordan inversion and baby-step giant-step discrete logs run as CUDA kernels over
  [sppark](https://github.com/supranational/sppark)'s Montgomery fields, checked bit for bit against
  [blst](https://github.com/supranational/blst).
- **One code base, two engines.** Every operation is a `__host__ __device__` functor. The `Gpu` engine launches it as
  a kernel and the `Cpu` engine runs it on every core with OpenMP, so the same arithmetic is the reference
  implementation and the fallback for calls too small to be worth a launch.
- **Batch-first functional encryption.** Every scheme takes whole batches of keys and ciphertexts, and each batch
  costs a fixed number of engine calls whatever its size.
- **Tuned from profiles, tested on both engines.** Nsight Systems timelines and the kernels' register reports drove
  the two changes in [Performance](#performance). The test suite runs its cases on both engines against blst or
  plain-integer answers, and CI runs it in a CUDA container without a GPU, where the Gpu cases report as skipped.

## Performance

Against [LibPFE](https://github.com/WeiqiNs/LibPFE), the same schemes on RELIC, for one key and one ciphertext of
length n with LibPFE's benchmark inputs. BLS12-381 on an RTX 5080 and an AMD Ryzen 7 9800X3D, Release builds, best of
several runs:

| Scheme | Implementation | Setup (ms) | KeyGen (ms) | Enc (ms) | Dec (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Ojaswi et al., n = 10,000 | LibPFE, RELIC | 18.9 | 1,405 | 493 | 2,036 |
| | CudaPFE, GPU | 0.32 | 2.5 | 1.9 | 9.9 |
| | **Speedup** | **59×** | **566×** | **265×** | **207×** |
| Kim et al., n = 1,000 | LibPFE, RELIC | 88,051 | 203 | 110 | 207 |
| | CudaPFE, GPU | 166 | 1.8 | 1.1 | 10.2 |
| | **Speedup** | **531×** | **115×** | **97×** | **20×** |

LibPFE runs each operation on a single core, one group operation after another; CudaPFE runs its n independent pieces
(n group elements, one multi-pairing of about n points, one n × n inversion) on thousands of GPU threads at once, so
the gain grows with n. Decryption gains less because its final exponentiation and discrete-log search are fixed work.

The core operations against blst on all 16 threads, medians from `cudapfe_bench`:

| Workload | CudaPFE, GPU | blst, 16 threads | Speedup |
| --- | ---: | ---: | ---: |
| One multi-pairing of 10,000 pairs | 8.5 ms | 130 ms | 15× |
| 100,000 independent pairings | 121 ms | 4,168 ms | 34× |
| 1,000 multi-pairings of 208 pairs each | 113 ms | 2,509 ms | 22× |

On a batch of 1,000 key–ciphertext pairs of length 100, `cudapfe_bench_fe` decrypts Ojaswi et al.'s inner-product
scheme in 70 ms (55 ms with prepared keys, about 18,000 decryptions per second) and the quadratic schemes of Baltico
et al. and Dufour-Sans et al. in 208 and 171 ms. Two changes, both found by profiling, account for most of the
quadratic-FE speed:

1. **Negative entries as small scalars, selected without a branch.** `Zp(-3)` is r − 3, a full 255-bit scalar, so
   negative entries made `msm` about 6× slower. `signed_scalar` multiplies a scalar above r/2 as its negation and
   negates the point; a first version that branched on the sign split each warp and slowed random scalars by up to
   50%, so the choice is made without a branch. Decryption with entries of either sign (Baltico et al., 1,000 pairs of
   length 100) went from 2,102 to 546 ms.
2. **A Straus sum for small scalars.** Each `msm` term ran a full scalar multiplication with its own 15-point table:
   255 registers per thread and 2.2 KB spilled to local memory (G1). FE scalars have 1–3 bits, so for scalars up to 32
   bits each thread now sums a chunk of an output's terms bit by bit, one doubling per bit for the chunk and one mixed
   addition per set bit: 186 registers, no spill. The FE-shaped sum of 1,000 × 100 × 100 terms went from 131 to 8 ms,
   and decryption of 1,000 pairs of length 100 from 575 to 195 ms (Baltico et al.) and 421 to 170 ms (Dufour-Sans et
   al.).

## Schemes

Inner-product FE, where a key for y decrypts a ciphertext of x to ⟨x, y⟩:

| Scheme | Namespace | Function hiding | Security | Paper |
| --- | --- | :---: | --- | --- |
| Bishop et al. | `IPFE::BJK` | weak | SXDH | [ASIACRYPT 2015](https://doi.org/10.1007/978-3-662-48797-6_20) |
| Tomida et al. | `IPFE::TAO` | full | XDLIN | [ISC 2016](https://doi.org/10.1007/978-3-319-45871-7_24) |
| Kim et al. | `IPFE::KIM` | full | generic group model | [SCN 2018](https://doi.org/10.1007/978-3-319-98113-0_29) |
| Lin | `IPFE::LIN` | full | SXDH | [CRYPTO 2017](https://doi.org/10.1007/978-3-319-63688-7_20) |
| Kim, Kim and Seo | `IPFE::KKS` | full | SXDH | [TCS 2019](https://doi.org/10.1016/j.tcs.2019.03.016) |
| Ojaswi et al. | `IPFE::OPT` | full | generic group model | [CiC 2025](https://doi.org/10.62056/abe0zo-3y) |

Quadratic FE, where a key for a matrix F decrypts a ciphertext of (x, y) to xᵀFy and anyone with the public key can
encrypt:

| Scheme | Namespace | Security | Paper |
| --- | --- | --- | --- |
| Baltico et al. | `QFE::BCFG` | adaptive, generic group model | [CRYPTO 2017](https://doi.org/10.1007/978-3-319-63688-7_3) |
| Dufour-Sans et al. | `QFE::SGP` | generic group model | [NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/9d28de8ff9bb6a3fa41fddfdc28f3bc1-Abstract.html) |

Each scheme is one header in `include/cudapfe/fe/` (`ipfe_<scheme>.hpp`, `qfe_<scheme>.hpp`) under `cudapfe::`, a
template over the engine chosen once at `setup<Cpu>(n)` or `setup<Gpu>(n)`. Consumers link `CudaPFE::fe`, which
brings in `CudaPFE::core`.

- `keygen` and `enc` take a batch: an `IntMatrix` of rows for IPFE, a vector of `IntMatrix` functions and two
  `IntMatrix` of left and right vectors for QFE. The single-vector overloads make a batch of one.
- `dec` pairs keys with ciphertexts: equal counts decrypt pairwise, one key decrypts every ciphertext, and one
  ciphertext decrypts under every key; any other combination throws `ShapeError`. It returns one
  `std::optional<std::int64_t>` per pair, empty when the result falls outside the searched range. Most schemes take a
  `DlogTable` over `<SCHEME>::base()` (TAO: `msk.base`) that serves every decryption; BCFG's `dec` also takes the
  public key; BJK and KIM derive a base per pair and take a `Range` instead.
- Every IPFE scheme has `prepare(sk)`, which precomputes the keys' pairing lines once for keys that decrypt many
  ciphertexts. Keys, prepared keys and `DlogTable`s never change after they are built, so threads can share them.
- BJK, TAO and KIM invert a random matrix at setup; on the Gpu engine a matrix that does not fit in free device memory
  throws `DeviceError` before it is sampled.

**When the GPU pays off.** A single GPU thread takes milliseconds per scalar multiplication, so a small call costs the
same fixed latency as a large one. The Gpu engine runs calls below the `CUDAPFE_HOST_*_BELOW` thresholds in
`CMakeLists.txt` on the host, but its setup inversions and discrete-log searches stay on the device, so for single
short keys and ciphertexts the Cpu engine is usually faster; batches and long vectors are where the Gpu engine wins.
`cudapfe_bench_fe` shows where the two cross on a given machine.

The schemes sit on a core API that is usable on its own: `CudaPFE::core` (`<cudapfe/cudapfe.hpp>`) holds `Zp`, `G1`,
`G2`, `Gt` and their batch form `Vec<T, E>`, with `mul_generator`, `msm`, `pair_segments` and `DlogTable`; [demo](demo)
is a complete example built against the installed package.

## Build and test

Linux, CMake 3.25 or newer, a C++20 compiler with OpenMP, the CUDA toolkit (13.x) and git. blst and GoogleTest are
fetched at pinned commits; sppark is vendored under `third_party/sppark`.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build
```

- Device code is compiled for `CMAKE_CUDA_ARCHITECTURES`, default `120`; on a GPU the build has no kernel image for,
  `gpu_available()` returns false and the Gpu engine throws a `DeviceError` naming the variable. A GPU is not required
  to build or test: without one the Gpu test cases report as skipped. `OMP_NUM_THREADS` caps the Cpu engine.
- `-DCUDAPFE_BUILD_BENCH=ON` builds `cudapfe_bench` (core operations against the Cpu engine and blst) and
  `cudapfe_bench_fe` (every scheme on both engines). Both need a CUDA device, take `--quick` for small sizes and check
  every result.
- The host-placement thresholds were measured on one GPU. To recalibrate for another, run
  `cudapfe_bench --placement-sweep` on a build with the `CUDAPFE_HOST_*_BELOW` cache variables set to 0 and configure
  with the sizes at which the Gpu engine starts to win. `-DCUDAPFE_PTXAS_VERBOSE=ON` prints every kernel's register
  and spill report.

Licensed under Apache-2.0.
