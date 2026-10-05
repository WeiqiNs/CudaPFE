#ifndef CUFE_PAIRING_PLAN_HPP
#define CUFE_PAIRING_PLAN_HPP

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <cufe/errors.hpp>
#include <cufe/pairing.hpp>
#include "support/hd.hpp"
#include "vec/reduction.hpp"

namespace cufe::detail{
    enum class Placement{ host, host_final_exp, device };

    struct Item{
        std::size_t segment;
        std::size_t first;
        std::size_t count;
    };

    inline constexpr std::size_t kMaxPairsPerItem = 8;
    inline constexpr std::size_t kHostMillerBelow = CUFE_HOST_MILLER_BELOW;
    inline constexpr std::size_t kHostFinalExpBelow = CUFE_HOST_FINAL_EXP_BELOW;

    [[nodiscard]] CUFE_HD constexpr std::size_t segment_offset(
        const PairShape& shape, const Spread spread, const std::size_t segment
    ){
        return spread == Spread::shared ? 0 : segment * shape.length;
    }

    struct ItemPlan{
        PairShape shape;
        std::size_t pairs_per_item;

        [[nodiscard]] CUFE_HD constexpr std::size_t items_per_segment() const{
            return ceil_div(shape.length, pairs_per_item);
        }

        [[nodiscard]] CUFE_HD constexpr std::size_t item_count() const{ return shape.segments * items_per_segment(); }

        [[nodiscard]] CUFE_HD constexpr Item item(const std::size_t index) const{
            const auto per_segment = items_per_segment();
            const auto first = index % per_segment * pairs_per_item;
            const auto remaining = shape.length - first;
            return {index / per_segment, first, remaining < pairs_per_item ? remaining : pairs_per_item};
        }

        [[nodiscard]] constexpr Reduction products() const{ return {shape.segments, items_per_segment()}; }
    };

    [[nodiscard]] constexpr std::size_t pairs_per_item(const PairShape& shape, const std::size_t resident_threads){
        const auto pairs = shape.segments * shape.length;
        return std::clamp<std::size_t>(pairs / std::max<std::size_t>(resident_threads, 1), 1, kMaxPairsPerItem);
    }

    [[nodiscard]] constexpr Placement place(const PairShape& shape){
        if (shape.segments * shape.length < kHostMillerBelow) return Placement::host;
        if (shape.segments < kHostFinalExpBelow) return Placement::host_final_exp;
        return Placement::device;
    }

    [[nodiscard]] constexpr std::size_t required_size(const PairShape& shape, const Spread spread){
        return spread == Spread::shared ? shape.length : shape.segments * shape.length;
    }

    inline void require_shape(const PairShape& shape){
        if (shape.length == 0) throw ShapeError("pair_segments needs a positive segment length");
        if (shape.segments > std::numeric_limits<std::size_t>::max() / shape.length){
            throw ShapeError("pair_segments shape of " + std::to_string(shape.segments) + " segments of "
                + std::to_string(shape.length) + " pairs overflows");
        }
    }

    inline void require_side(const PairShape& shape, const Spread spread, const std::size_t size, const char* side){
        const auto expected = required_size(shape, spread);
        if (size != expected){
            throw ShapeError("pair_segments needs " + std::to_string(expected) + " " + side
                + " points for its shape, got " + std::to_string(size));
        }
    }
}

#endif
