#ifndef CUDAPFE_VEC_REDUCTION_HPP
#define CUDAPFE_VEC_REDUCTION_HPP

#include <cstddef>
#include "support/hd.hpp"

namespace cudapfe::detail{
    inline constexpr std::size_t kReduceFanIn = 4;

    struct Group{
        std::size_t first;
        std::size_t count;
    };

    [[nodiscard]] CUDAPFE_HD constexpr std::size_t ceil_div(const std::size_t n, const std::size_t d){
        return (n + d - 1) / d;
    }

    struct Reduction{
        std::size_t segments;
        std::size_t width;

        [[nodiscard]] CUDAPFE_HD constexpr std::size_t groups_per_segment() const{ return ceil_div(width, kReduceFanIn); }

        [[nodiscard]] CUDAPFE_HD constexpr std::size_t group_count() const{ return segments * groups_per_segment(); }

        [[nodiscard]] CUDAPFE_HD constexpr Group group(const std::size_t index) const{
            const auto per_segment = groups_per_segment();
            const auto offset = index % per_segment * kReduceFanIn;
            const auto remaining = width - offset;
            return {index / per_segment * width + offset, remaining < kReduceFanIn ? remaining : kReduceFanIn};
        }

        [[nodiscard]] constexpr Reduction next() const{ return {segments, groups_per_segment()}; }
    };
}

#endif
