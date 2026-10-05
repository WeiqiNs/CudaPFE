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
        std::vector<std::size_t> live;
        for (std::size_t i = 0; i < ps.size(); ++i){
            if (!p_affine[i].is_identity() && !q_affine[i].is_identity()) live.push_back(i);
        }

        auto product = detail::Fp12::one();
        for (std::size_t first = 0; first < live.size(); first += kPairsPerMillerLoop){
            const auto count = std::min(kPairsPerMillerLoop, live.size() - first);
            std::vector<detail::G1Affine> px2s;
            std::vector<detail::Line> lines(count * detail::kLineCount);
            for (std::size_t k = 0; k < count; ++k){
                const auto i = live[first + k];
                px2s.push_back(detail::px2(p_affine[i]));
                detail::prepare_lines(q_affine[i], lines.data() + k * detail::kLineCount);
            }
            product = product * detail::miller(detail::PreparedPairs{px2s.data(), lines.data(), count});
        }
        return Access::value<Gt>(detail::final_exp(product));
    }
}
