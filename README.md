# Pairing-based Functional Encryption in CUDA (CudaPFE)

[![CudaPFE CI](https://github.com/WeiqiNs/CudaPFE/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/WeiqiNs/CudaPFE/actions/workflows/ci.yml)

Functional encryption lets the holder of a key learn one function of encrypted data, such as an inner product, and
nothing else. CudaPFE runs eight published pairing-based schemes in batches on an NVIDIA GPU, on BLS12-381 arithmetic
written as CUDA kernels. It is the GPU counterpart of [LibPFE](https://github.com/WeiqiNs/LibPFE), makes no
constant-time guarantees, and is meant for research prototypes.

```cpp
#include <cudapfe/fe/ipfe_opt.hpp>

using namespace cudapfe;
const auto msk = IPFE::OPT::setup<Gpu>(3);
const DlogTable<Gpu> table(IPFE::OPT::base(), {-1000, 1000});
const auto keys = IPFE::OPT::keygen(msk, IPFE::IntMatrix{{1, 2, 3}, {0, 1, 0}});
const auto ct = IPFE::OPT::enc(msk, IPFE::IntVec{4, -5, 6});
const auto results = IPFE::OPT::dec(table, IPFE::OPT::prepare(keys), ct);   // {12, -5}
```

`results` holds the inner product of the message with each key's vector; the keys reveal nothing else about it.

## Highlights

- **BLS12-381 in CUDA**: fields, curves, Miller loops, final exponentiation, multi-scalar multiplication (MSM), matrix
  inversion and discrete logs as kernels, with field, curve and pairing results checked bit for bit against
  [blst](https://github.com/supranational/blst).
- **One code base, two engines**: every operation is a `__host__ __device__` functor that runs as a kernel on the GPU
  or with OpenMP on every CPU core, and the test suite runs on both.
- **Tuned from profiles**: Nsight Systems timelines and register reports led to branch-free signed scalars and a
  spill-free G1 MSM kernel, both measured under [Performance](#performance).

## Performance

Against LibPFE (the same schemes on RELIC, single-threaded), one key and one ciphertext of length n, BLS12-381 on an
RTX 5080 and a Ryzen 7 9800X3D, best of several runs:

| Scheme | Implementation | Setup (ms) | KeyGen (ms) | Enc (ms) | Dec (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Acharya et al., n = 10,000 | LibPFE, RELIC | 18.9 | 1,405 | 493 | 2,036 |
| | CudaPFE, GPU | 0.32 | 2.5 | 1.9 | 9.9 |
| | **Speedup** | **59×** | **566×** | **265×** | **207×** |
| Kim et al., n = 1,000 | LibPFE, RELIC | 88,051 | 203 | 110 | 207 |
| | CudaPFE, GPU | 166 | 1.8 | 1.1 | 10.2 |
| | **Speedup** | **531×** | **115×** | **97×** | **20×** |

Each operation splits into n independent pieces (n group elements, the n pairs of a multi-pairing, the rows of an
n × n inversion) that CudaPFE spreads over thousands of GPU threads, so the gain grows with n. Against blst on all 16
CPU threads, a 10,000-pair multi-pairing takes 8.5 ms against 130 ms (15×), and 100,000 independent pairings 121 ms
against 4,168 ms (34×). Two profiling fixes, timed on Baltico et al. decrypting 1,000 pairs of length 100:

- **Branch-free signed scalars.** `Zp(-3)` is a 255-bit scalar, so negative entries made `msm` about 6× slower;
  multiplying the negated point by the small magnitude fixed it, and selecting without a branch avoided warp
  divergence that slowed random scalars by up to 50%. Decryption with signed entries: 2,102 → 546 ms.
- **A Straus sum for small scalars.** A per-term 15-point table cost 255 registers per thread and 2.2 KB of spills;
  summing each output's terms bit by bit needs 186 registers and no spill. Decryption: 575 → 195 ms.

## Schemes

| Scheme | Namespace | Security | Paper |
| --- | --- | --- | --- |
| Bishop et al. (weakly function hiding) | `IPFE::BJK` | SXDH | [ASIACRYPT 2015](https://doi.org/10.1007/978-3-662-48797-6_20) |
| Tomida et al. | `IPFE::TAO` | XDLIN | [ISC 2016](https://doi.org/10.1007/978-3-319-45871-7_24) |
| Kim et al. | `IPFE::KIM` | generic group model | [SCN 2018](https://doi.org/10.1007/978-3-319-98113-0_29) |
| Lin | `IPFE::LIN` | SXDH | [CRYPTO 2017](https://doi.org/10.1007/978-3-319-63688-7_20) |
| Kim, Kim and Seo | `IPFE::KKS` | SXDH | [TCS 2019](https://doi.org/10.1016/j.tcs.2019.03.016) |
| Acharya et al. | `IPFE::OPT` | generic group model | [CiC 2025](https://doi.org/10.62056/abe0zo-3y) |
| Baltico et al. | `QFE::BCFG` | adaptive, generic group model | [CRYPTO 2017](https://doi.org/10.1007/978-3-319-63688-7_3) |
| Ryffel et al. | `QFE::SGP` | generic group model | [NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/9d28de8ff9bb6a3fa41fddfdc28f3bc1-Abstract.html) |

`IPFE` keys for y decrypt ciphertexts of x to ⟨x, y⟩ with function hiding; `QFE` keys for F decrypt (x, y) to xᵀFy.
Each scheme is one header in `include/cudapfe/fe/`, templated on the engine (`setup<Gpu>(n)` or `setup<Cpu>(n)`);
every call takes a batch (a single vector is a batch of one), and keys and discrete-log tables can be shared across
threads. The GPU wins on batches and
long vectors; for single short ones the Cpu engine is usually faster, and `cudapfe_bench_fe` shows the crossover. The
pairing layer underneath (`<cudapfe/cudapfe.hpp>`) is usable on its own; see [demo](demo).

## Build and test

Linux, CMake 3.25+, a C++20 compiler with OpenMP and the CUDA toolkit (13.x):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build
ctest --test-dir build --output-on-failure
```

Device code targets `CMAKE_CUDA_ARCHITECTURES` (default `120`); without a matching GPU the Gpu tests report as skipped.
`-DCUDAPFE_BUILD_BENCH=ON` builds the benchmarks `cudapfe_bench` and `cudapfe_bench_fe`. After `cmake --install build`,
projects use `find_package(CudaPFE REQUIRED)` and link `CudaPFE::fe` (or `CudaPFE::core` for the pairing layer alone).
