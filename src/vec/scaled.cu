#include <cstddef>
#include <cudapfe/core.hpp>
#include <cudapfe/vec.hpp>
#include "curve/curve.hpp"
#include "support/for_each.cuh"
#include "support/hd.hpp"
#include "vec/points.cuh"
#include "vec/storage.hpp"

// Vec::scaled's device work, apart from vec.cu so its variable-base multiplication compiles in parallel.
namespace cudapfe{
    namespace{
        using detail::data;
        using detail::FieldOf;

        template <class F>
        struct ScalarMulOp{
            enum : std::size_t{ host_parallel_from = 2 };

            const detail::Affine<F>* x;
            const detail::Fr* k;
            detail::Jacobian<F>* out;

            CUDAPFE_HD void operator()(const std::size_t i) const{ out[i] = detail::mul(detail::from_affine(x[i]), k[i]); }
        };
    }

    template <class G, Engine E>
    Vec<G, E> detail::scale_points(const Vec<G, E>& points, const Vec<Zp, E>& k){
        return generate_points<G, E>(points.size(), [&](auto* out){
            return ScalarMulOp<FieldOf<G>>{data(points), data(k), out};
        });
    }

    template Vec<G1, Cpu> detail::scale_points(const Vec<G1, Cpu>&, const Vec<Zp, Cpu>&);
    template Vec<G1, Gpu> detail::scale_points(const Vec<G1, Gpu>&, const Vec<Zp, Gpu>&);
    template Vec<G2, Cpu> detail::scale_points(const Vec<G2, Cpu>&, const Vec<Zp, Cpu>&);
    template Vec<G2, Gpu> detail::scale_points(const Vec<G2, Gpu>&, const Vec<Zp, Gpu>&);
}
