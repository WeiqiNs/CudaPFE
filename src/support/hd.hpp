#ifndef CUFE_SUPPORT_HD_HPP
#define CUFE_SUPPORT_HD_HPP

#include <array>
#include <cstddef>

#ifdef __CUDACC__
#define CUFE_HD __host__ __device__ __forceinline__
#define CUFE_HD_CALL __host__ __device__ __noinline__ inline
#else
#define CUFE_HD inline
#define CUFE_HD_CALL inline
#endif

namespace cufe::detail{
    using Word = unsigned long long;

    template <std::size_t N>
    using Words = std::array<Word, N>;
}

#endif
