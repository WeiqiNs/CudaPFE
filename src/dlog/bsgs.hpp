#ifndef CUFE_DLOG_BSGS_HPP
#define CUFE_DLOG_BSGS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cufe/dlog.hpp>
#include <cufe/errors.hpp>
#include "support/hd.hpp"
#include "vec/reduction.hpp"

namespace cufe::detail{
    struct Steps{
        std::int64_t lo;
        std::uint64_t span;
        std::uint64_t baby;
        std::uint64_t giant;
    };

    struct Ladder{
        std::size_t entry;
        std::uint64_t first;
        std::uint64_t count;
    };

    inline constexpr std::size_t kTableBudgetBytes = std::size_t{2} << 30;
    inline constexpr std::size_t kBabyStepBytes = 2 * (sizeof(std::uint64_t) + sizeof(std::uint32_t));

    [[nodiscard]] constexpr std::uint64_t integer_sqrt(const std::uint64_t n){
        std::uint64_t root = 0;
        for (std::uint64_t bit = std::uint64_t{1} << 31; bit != 0; bit >>= 1){
            const auto candidate = root | bit;
            if (candidate <= n / candidate) root = candidate;
        }
        return root;
    }

    [[nodiscard]] constexpr Steps plan_steps(const Range& range){
        if (range.lo > range.hi) throw ShapeError("dlog needs lo <= hi");
        const auto span = static_cast<std::uint64_t>(range.hi) - static_cast<std::uint64_t>(range.lo);
        const auto baby = integer_sqrt(span) + 1;
        return {range.lo, span, baby, span / baby + 1};
    }

    [[nodiscard]] CUFE_HD constexpr std::int64_t offset(const std::int64_t lo, const std::uint64_t k){
        return static_cast<std::int64_t>(static_cast<std::uint64_t>(lo) + k);
    }

    struct LadderPlan{
        std::size_t entries;
        std::uint64_t length;
        std::uint64_t per_ladder;

        [[nodiscard]] CUFE_HD constexpr std::size_t ladders_per_entry() const{ return ceil_div(length, per_ladder); }

        [[nodiscard]] CUFE_HD constexpr std::size_t ladder_count() const{ return entries * ladders_per_entry(); }

        [[nodiscard]] CUFE_HD constexpr Ladder ladder(const std::size_t index) const{
            const auto per_entry = ladders_per_entry();
            const auto first = index % per_entry * per_ladder;
            const auto remaining = length - first;
            return {index / per_entry, first, remaining < per_ladder ? remaining : per_ladder};
        }
    };

    [[nodiscard]] constexpr LadderPlan plan_ladders(
        const std::size_t entries, const std::uint64_t length, const std::size_t resident_threads
    ){
        const auto steps = ceil_div(entries * length, std::max<std::size_t>(resident_threads, 1));
        return {entries, length, std::clamp<std::uint64_t>(steps, 1, std::max<std::uint64_t>(length, 1))};
    }

    [[nodiscard]] constexpr std::size_t entries_per_chunk(const std::uint64_t baby, const std::size_t budget_bytes){
        return std::max<std::size_t>(budget_bytes / (baby * kBabyStepBytes), 1);
    }
}

#endif
