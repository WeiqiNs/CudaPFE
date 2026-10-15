#ifndef CUDAPFE_CURVE_STRAUS_HPP
#define CUDAPFE_CURVE_STRAUS_HPP

#include <cstddef>
#include "curve/curve.hpp"
#include "support/hd.hpp"
#include "support/words.hpp"

namespace cudapfe::detail{
    inline constexpr std::size_t kStrausMaxBits = 32;

    template <class F>
    struct MsmTerms{
        const Affine<F>* bases;
        const SignedScalar* scalars;
        std::size_t scalar_stride;
        std::size_t count;
        std::size_t bits;
    };

    template <class F>
    [[nodiscard]] CUDAPFE_HD Jacobian<F> straus_sum(const MsmTerms<F>& terms){
        auto sum = Jacobian<F>::identity();
#pragma unroll 1
        for (auto i = terms.bits; i-- > 0;){
            sum = dbl(sum);
#pragma unroll 1
            for (std::size_t k = 0; k < terms.count; ++k){
                const auto& scalar = terms.scalars[k * terms.scalar_stride];
                if (!bit(scalar.magnitude, i)) continue;
                const auto& base = terms.bases[k];
                const auto negated = -base.y;
                sum = add_mixed(sum, {base.x, scalar.negative ? negated : base.y});
            }
        }
        return sum;
    }
}

#endif
