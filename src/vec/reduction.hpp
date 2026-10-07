#ifndef CUDAPFE_VEC_REDUCTION_HPP
#define CUDAPFE_VEC_REDUCTION_HPP

#include <cstddef>
#include "support/hd.hpp"

namespace cudapfe::detail{
    inline constexpr std::size_t kReduceFanIn = 4;

    struct Chunk{
        std::size_t segment;
        std::size_t first;
        std::size_t count;
    };

    struct Chunks{
        std::size_t segments;
        std::size_t length;
        std::size_t size;

        [[nodiscard]] CUDAPFE_HD constexpr std::size_t per_segment() const{ return ceil_div(length, size); }

        [[nodiscard]] CUDAPFE_HD constexpr std::size_t count() const{ return segments * per_segment(); }

        [[nodiscard]] CUDAPFE_HD constexpr Chunk at(const std::size_t index) const{
            const auto per = per_segment();
            const auto first = index % per * size;
            const auto remaining = length - first;
            return {index / per, first, remaining < size ? remaining : size};
        }
    };
}

#endif
