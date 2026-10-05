#ifndef CUFE_SUPPORT_HD_HPP
#define CUFE_SUPPORT_HD_HPP

#include <array>
#include <cstddef>

#ifdef __CUDACC__
#define CUFE_HD __host__ __device__ __forceinline__
#else
#define CUFE_HD inline
#endif

namespace cufe::detail{
    using Word = unsigned long long;

    template <std::size_t N>
    using Words = std::array<Word, N>;
}

#endif
