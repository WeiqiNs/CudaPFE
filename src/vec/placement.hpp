#ifndef CUDAPFE_VEC_PLACEMENT_HPP
#define CUDAPFE_VEC_PLACEMENT_HPP

#include <cstddef>

namespace cudapfe::detail{
    inline constexpr std::size_t kHostPointsBelow = CUDAPFE_HOST_POINTS_BELOW;

    [[nodiscard]] constexpr bool points_on_host(const std::size_t count){
        return count < kHostPointsBelow;
    }
}

#endif
