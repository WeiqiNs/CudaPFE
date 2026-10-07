#ifndef CUDAPFE_SUPPORT_HD_HPP
#define CUDAPFE_SUPPORT_HD_HPP

#include <array>
#include <cstddef>

#ifdef __CUDACC__
#define CUDAPFE_HD __host__ __device__ __forceinline__
#define CUDAPFE_HD_CALL __host__ __device__ __noinline__ inline
#else
#define CUDAPFE_HD inline
#define CUDAPFE_HD_CALL inline
#endif

namespace cudapfe::detail{
    using Word = unsigned long long;

    template <std::size_t N>
    using Words = std::array<Word, N>;

    [[nodiscard]] CUDAPFE_HD constexpr std::size_t ceil_div(const std::size_t n, const std::size_t d){
        return (n + d - 1) / d;
    }
}

#endif
