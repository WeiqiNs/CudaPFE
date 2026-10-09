#ifndef CUDAPFE_DLOG_BSGS_HPP
#define CUDAPFE_DLOG_BSGS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cudapfe/dlog.hpp>
#include <cudapfe/errors.hpp>
#include "support/hd.hpp"
#include "vec/reduction.hpp"

namespace cudapfe::detail{
    struct Steps{
        std::int64_t lo;
        std::uint64_t span;
        std::uint64_t baby;
        std::uint64_t giant;
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

    [[nodiscard]] CUDAPFE_HD constexpr std::int64_t offset(const std::int64_t lo, const std::uint64_t k){
        return static_cast<std::int64_t>(static_cast<std::uint64_t>(lo) + k);
    }

    [[nodiscard]] constexpr Chunks plan_ladders(
        const std::size_t entries, const std::uint64_t length, const std::size_t resident_threads
    ){
        return {entries, length, chunk_size(entries * length, resident_threads, length)};
    }

    [[nodiscard]] constexpr std::size_t entries_per_chunk(const std::uint64_t baby, const std::size_t budget_bytes){
        return std::max<std::size_t>(budget_bytes / (baby * kBabyStepBytes), 1);
    }
}

#endif
