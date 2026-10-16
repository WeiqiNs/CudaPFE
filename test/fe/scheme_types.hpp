#ifndef CUDAPFE_TEST_FE_SCHEME_TYPES_HPP
#define CUDAPFE_TEST_FE_SCHEME_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <support/engines.hpp>
#include "schemes.hpp"

using cudapfe::IPFE::Results;

inline constexpr cudapfe::Range kRange{-100, 100};

template <template <cudapfe::Engine> class... Schemes>
using OnEveryEngine = ::testing::Types<Schemes<cudapfe::Cpu>..., Schemes<cudapfe::Gpu>...>;

using InnerProductSchemes = OnEveryEngine<Bjk, Tao, Kim, Lin, Kks, Opt>;
using QuadraticSchemes = OnEveryEngine<Bcfg, Sgp>;

inline cudapfe::IPFE::IntMatrix random_rows(const std::size_t count, const std::size_t length, const std::int64_t bound){
    static std::mt19937_64 generator(20261005);
    std::uniform_int_distribution<std::int64_t> entry(-bound, bound);
    cudapfe::IPFE::IntMatrix rows(count, cudapfe::IPFE::IntVec(length));
    for (auto& row : rows){
        for (auto& value : row) value = entry(generator);
    }
    return rows;
}

inline std::int64_t inner_product(const cudapfe::IPFE::IntVec& x, const cudapfe::IPFE::IntVec& y){
    std::int64_t total = 0;
    for (std::size_t i = 0; i < x.size(); ++i) total += x[i] * y[i];
    return total;
}

inline std::int64_t quadratic_form(
    const cudapfe::IPFE::IntVec& x, const cudapfe::IPFE::IntMatrix& f, const cudapfe::IPFE::IntVec& y
){
    std::int64_t total = 0;
    for (std::size_t i = 0; i < x.size(); ++i){
        for (std::size_t j = 0; j < y.size(); ++j) total += x[i] * f[i][j] * y[j];
    }
    return total;
}

#endif
