#ifndef CUDAPFE_PAIRING_PLAN_HPP
#define CUDAPFE_PAIRING_PLAN_HPP

#include <algorithm>
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

    // Prepared G2 lines dominate an unprepared pair_segments' memory (kLineCount Fp6 values, about 19 KB, per Q).
    // A per-segment G2 side whose lines would pass this budget (the same as DlogTables') is paired a chunk of
    // segments at a time.
    inline constexpr std::size_t kLineBudgetBytes = std::size_t{2} << 30;

    // Segments per chunk when each Q's lines take `q_bytes`: all of them if they fit `budget` (a shared G2 side is
    // prepared once), else the fewest chunks that fit, sized evenly so no small trailing chunk idles the device.
    [[nodiscard]] constexpr std::size_t segments_per_line_chunk(
        const PairShape& shape, const std::size_t q_bytes, const std::size_t budget
    ){
        if (shape.q == Spread::shared || shape.length == 0) return shape.segments;
        const auto fit = std::max<std::size_t>(budget / (shape.length * q_bytes), 1);
        if (fit >= shape.segments) return shape.segments;
        return ceil_div(shape.segments, ceil_div(shape.segments, fit));
    }

    // pair_segments' unprepared path, `chunk_segments` segments at a time (defined in multi_pair.cu). Exposed so
    // tests can force chunking at sizes far below kLineBudgetBytes.
    template <Engine E>
    [[nodiscard]] Vec<Gt, E> pair_in_chunks(
        const Vec<G1, E>& ps, const Vec<G2, E>& qs, const PairShape& shape, std::size_t chunk_segments
    );
}

#endif
