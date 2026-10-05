#ifndef CUFE_TEST_FE_SCHEME_TYPES_HPP
#define CUFE_TEST_FE_SCHEME_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <support/engines.hpp>
#include "schemes.hpp"

using Results = std::vector<std::optional<std::int64_t>>;

inline constexpr cufe::Range kRange{-100, 100};

template <template <cufe::Engine> class... Schemes>
using OnEveryEngine = ::testing::Types<Schemes<cufe::Cpu>..., Schemes<cufe::Gpu>...>;

using InnerProductSchemes = OnEveryEngine<Bjk, Tao, Kim, Lin, Kks, Opt>;
using QuadraticSchemes = OnEveryEngine<Bcfg, Sgp>;

inline cufe::IPFE::IntMatrix random_rows(const std::size_t count, const std::size_t length, const std::int64_t bound){
    static std::mt19937_64 generator(20261005);
    std::uniform_int_distribution<std::int64_t> entry(-bound, bound);
    cufe::IPFE::IntMatrix rows(count, cufe::IPFE::IntVec(length));
    for (auto& row : rows){
        for (auto& value : row) value = entry(generator);
    }
    return rows;
}

#endif
