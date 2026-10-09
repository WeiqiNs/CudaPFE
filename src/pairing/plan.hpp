#ifndef CUDAPFE_PAIRING_PLAN_HPP
#define CUDAPFE_PAIRING_PLAN_HPP

#include <cstddef>
#include <cudapfe/pairing.hpp>
#include "vec/spread.hpp"

namespace cudapfe::detail{
    enum class Placement{ host, host_final_exp, device };

    inline constexpr std::size_t kMaxPairsPerItem = 8;
    inline constexpr std::size_t kHostMillerBelow = CUDAPFE_HOST_MILLER_BELOW;
    inline constexpr std::size_t kHostFinalExpBelow = CUDAPFE_HOST_FINAL_EXP_BELOW;

    [[nodiscard]] constexpr Placement place(const PairShape& shape){
        if (shape.segments * shape.length < kHostMillerBelow) return Placement::host;
        if (shape.segments < kHostFinalExpBelow) return Placement::host_final_exp;
        return Placement::device;
    }

    [[nodiscard]] constexpr Layout layout(const PairShape& shape, const Spread spread){
        return {shape.segments, shape.length, spread};
    }
}

#endif
