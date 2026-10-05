#include <algorithm>
#include <cstddef>
#include <vector>
#include <cufe/core.hpp>
#include "curve/curve.hpp"
#include "field/tower.hpp"
#include "pairing/miller.hpp"
#include "support/access.hpp"

namespace cufe{
    using detail::Access;

    namespace{
        constexpr std::size_t kPairsPerMillerLoop = 256;

        template <class F, class T>
        std::vector<detail::Affine<F>> affine(const std::vector<T>& points){
            std::vector<detail::Jacobian<F>> jacobians;
            jacobians.reserve(points.size());
            for (const auto& p : points) jacobians.push_back(Access::element<detail::Jacobian<F>>(p));
            std::vector<detail::Affine<F>> out(points.size());
            detail::to_affine<F>(jacobians, out);
            return out;
        }
    }

    Gt pair(const G1& p, const G2& q){
        return pair(std::vector{p}, std::vector{q});
    }

    Gt pair(const std::vector<G1>& ps, const std::vector<G2>& qs){
        if (ps.size() != qs.size()) throw ShapeError("multi-pairing needs one G2 point per G1 point");
        const auto p_affine = affine<detail::Fp>(ps);
        const auto q_affine = affine<detail::Fp2>(qs);

        auto product = detail::Fp12::one();
        std::vector<detail::Line> lines(kPairsPerMillerLoop * detail::kLineCount);
        for (std::size_t first = 0; first < ps.size(); first += kPairsPerMillerLoop){
            const auto count = std::min(kPairsPerMillerLoop, ps.size() - first);
            for (std::size_t k = 0; k < count; ++k){
                detail::prepare_lines(q_affine[first + k], lines.data() + k * detail::kLineCount);
            }
            const detail::PreparedPairs pairs{p_affine.data() + first, q_affine.data() + first, lines.data(), count};
            product = product * detail::miller(pairs);
        }
        return Access::value<Gt>(detail::final_exp(product));
    }
}
