#ifndef CUDAPFE_CURVE_GENERATOR_TABLE_HPP
#define CUDAPFE_CURVE_GENERATOR_TABLE_HPP

#include <cstddef>
#include <vector>
#include "curve/curve.hpp"
#include "field/field.hpp"
#include "support/hd.hpp"

namespace cudapfe::detail{
    inline constexpr std::size_t kWindowBits = 8;
    inline constexpr std::size_t kWindows = 32;
    inline constexpr std::size_t kWindowDigits = (std::size_t{1} << kWindowBits) - 1;

    static_assert(kWindows * kWindowBits == 64 * Fr::N && 64 % kWindowBits == 0);

    template <class F>
    [[nodiscard]] std::vector<Affine<F>> build_generator_table(){
        std::vector<Jacobian<F>> multiples;
        multiples.reserve(kWindows * kWindowDigits);
        auto base = from_affine(CurveParams<F>::generator());
        for (std::size_t window = 0; window < kWindows; ++window){
            auto multiple = base;
            for (std::size_t digit = 1; digit <= kWindowDigits; ++digit){
                multiples.push_back(multiple);
                multiple = add(multiple, base);
            }
            base = multiple;
        }
        std::vector<Affine<F>> table(multiples.size());
        to_affine(multiples.data(), multiples.size(), table.data());
        return table;
    }

    template <class F>
    [[nodiscard]] const std::vector<Affine<F>>& generator_table(){
        static const auto table = build_generator_table<F>();
        return table;
    }

    template <class F>
    [[nodiscard]] CUDAPFE_HD Jacobian<F> fixed_base_mul(const Affine<F>* table, const Fr& scalar){
        constexpr Word mask = kWindowDigits;
        const auto words = scalar.canonical();
        auto result = Jacobian<F>::identity();
#pragma unroll 1
        for (std::size_t window = 0; window < kWindows; ++window){
            const auto bit = window * kWindowBits;
            const auto digit = (words[bit / 64] >> (bit % 64)) & mask;
            if (digit != 0) result = add_mixed(result, table[window * kWindowDigits + digit - 1]);
        }
        return result;
    }
}

#endif
