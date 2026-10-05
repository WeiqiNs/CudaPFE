# sppark (vendored)

These files are copied unmodified from [sppark](https://github.com/supranational/sppark) at commit
`9e5c7951d4ff4992f78af26f48d3c9230b8c4136`:

- `LICENSE`
- `ff/bls12-381.hpp`
- `ff/mont_t.cuh`
- `ff/pow.hpp`

On the host, `ff/bls12-381.hpp` includes `<blst_t.hpp>`, which LibCuFE takes from the blst checkout that
`cmake/CuFEDependencies.cmake` fetches. Refresh the copies only together with that blst pin, and keep them byte-identical
to upstream.
